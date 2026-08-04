"""M9 tests: delete_file (ledgered/undoable, permission-gated) and fetch_url
(web docs, HTML-stripped, read-only)."""

import httpx
import pytest

from forge.safety.guard import SafetyViolation
from forge.safety.permissions import PermissionPolicy
from forge.tools.base import ToolRegistry
from forge.tools.changes import ChangeLedger
from forge.tools.filesystem import DeleteFileTool
from forge.tools.web import FetchUrlTool

# -- delete_file -----------------------------------------------------------------


def test_delete_removes_file_and_backs_up(guard, ledger, workspace):
    target = workspace / "gone.py"
    target.write_text("keep me\n", encoding="utf-8")
    result = DeleteFileTool(guard, ledger).run(path="gone.py")
    assert result.ok
    assert not target.exists()
    # ledger recorded it, so /undo restores it
    assert "gone.py" in ledger.changed_files
    ledger.restore_all()
    assert target.read_text(encoding="utf-8") == "keep me\n"


def test_delete_missing_file(guard, ledger):
    result = DeleteFileTool(guard, ledger).run(path="nope.py")
    assert not result.ok
    assert "not found" in result.error.lower()


def test_delete_rejects_directory(guard, ledger, workspace):
    (workspace / "adir").mkdir()
    result = DeleteFileTool(guard, ledger).run(path="adir")
    assert not result.ok
    assert "directory" in result.error.lower()


def test_delete_is_mutating_and_permission_gated(guard, workspace):
    (workspace / "x.py").write_text("data", encoding="utf-8")
    ledger = ChangeLedger(workspace, "r")
    policy = PermissionPolicy("ask", approver=lambda name, detail: False)
    registry = ToolRegistry([DeleteFileTool(guard, ledger)], policy=policy)
    result = registry.execute("delete_file", {"path": "x.py"})
    assert not result.ok
    assert "denied" in result.error.lower()
    assert (workspace / "x.py").exists()  # denial blocked the delete


def test_delete_escapes_workspace_blocked(guard, ledger):
    # direct .run() surfaces the guard's SafetyViolation; the registry is what
    # converts it into an error ToolResult (see test_delete_is_mutating_...).
    with pytest.raises(SafetyViolation):
        DeleteFileTool(guard, ledger).run(path="../outside.py")


# -- fetch_url -------------------------------------------------------------------


def _fetch(handler) -> FetchUrlTool:
    return FetchUrlTool(transport=httpx.MockTransport(handler))


def test_fetch_strips_html_to_text():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "docs.example.com"
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html><head><style>b{color:red}</style>"
            "<script>alert(1)</script></head><body>"
            "<h1>Title</h1><p>Hello &amp; welcome</p></body></html>",
        )

    result = _fetch(handler).run(url="https://docs.example.com/guide")
    assert result.ok
    assert "Title" in result.output
    assert "Hello & welcome" in result.output
    assert "alert(1)" not in result.output  # script stripped
    assert "color:red" not in result.output  # style stripped


def test_fetch_plain_text_passthrough():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="raw docs body")

    result = _fetch(handler).run(url="https://example.com/readme.txt")
    assert result.ok
    assert "raw docs body" in result.output


def test_fetch_rejects_non_http_scheme():
    result = FetchUrlTool().run(url="file:///etc/passwd")
    assert not result.ok
    assert "http" in result.error.lower()


def test_fetch_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="nope")

    result = _fetch(handler).run(url="https://example.com/missing")
    assert not result.ok
    assert "404" in result.error


def test_fetch_is_read_only():
    assert FetchUrlTool().mutating is False


@pytest.mark.parametrize("bad", ["ftp://x", "javascript:alert(1)", "not a url"])
def test_fetch_various_bad_urls(bad):
    assert not FetchUrlTool().run(url=bad).ok


def test_bad_tool_arguments_name_what_is_missing(workspace):
    """A raw TypeError leaks a Python signature — "run() missing 1 required
    positional argument: 'path'" — which the model cannot act on. Observed
    live on t4-add-cli-flag. Every other failure in Forge grounds the model;
    this one should too."""
    from forge.chat.session import ChatSession
    from forge.llm.mock import MockLLMClient

    session = ChatSession(workspace, MockLLMClient([]), session_id="args")
    result = session.registry.execute("edit_file", {"old_string": "a", "new_string": "b"})
    assert not result.ok
    assert "missing required argument(s): path" in result.error
    assert "It takes: path, old_string, new_string" in result.error
    assert "run()" not in result.error  # no Python signature leak


def test_unexpected_tool_arguments_are_named(workspace):
    from forge.chat.session import ChatSession
    from forge.llm.mock import MockLLMClient

    session = ChatSession(workspace, MockLLMClient([]), session_id="args2")
    result = session.registry.execute(
        "read_file", {"path": "x.py", "encoding": "utf-8", "mode": "r"}
    )
    assert not result.ok
    assert "unexpected argument(s)" in result.error
    assert "encoding" in result.error and "mode" in result.error
