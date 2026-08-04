"""append_to_file — the missing affordance for the commonest edit there is.

Found by the benchmark: on multi-requirement tasks the model reached for
edit_file with an empty old_string (its way of saying "put this at the end"),
was refused, and deflected instead of rewriting the file. mutated stayed
False and the task failed with nothing written."""

import pytest

from forge.config import ForgeSettings
from forge.llm.mock import MockLLMClient


def _session(workspace, **kwargs):
    from forge.chat.session import ChatSession

    return ChatSession(workspace, MockLLMClient([]), ForgeSettings(**kwargs), session_id="ap")


def test_appends_and_keeps_existing_content(workspace):
    (workspace / "m.py").write_text("def one():\n    return 1\n", encoding="utf-8")
    session = _session(workspace)
    result = session.registry.execute(
        "append_to_file", {"path": "m.py", "content": "def two():\n    return 2\n"}
    )
    assert result.ok
    text = (workspace / "m.py").read_text(encoding="utf-8")
    assert "def one():" in text and "def two():" in text
    assert text.index("def one") < text.index("def two")


def test_separates_with_a_blank_line(workspace):
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")
    _session(workspace).registry.execute(
        "append_to_file", {"path": "m.py", "content": "y = 2\n"}
    )
    assert (workspace / "m.py").read_text(encoding="utf-8") == "x = 1\n\ny = 2\n"


def test_does_not_stack_blank_lines(workspace):
    (workspace / "m.py").write_text("x = 1\n\n", encoding="utf-8")
    _session(workspace).registry.execute(
        "append_to_file", {"path": "m.py", "content": "\n\ny = 2"}
    )
    assert (workspace / "m.py").read_text(encoding="utf-8") == "x = 1\n\ny = 2\n"


def test_missing_file_points_at_write_file(workspace):
    result = _session(workspace).registry.execute(
        "append_to_file", {"path": "ghost.py", "content": "x = 1"}
    )
    assert not result.ok
    assert "write_file" in result.error


def test_syntax_gate_still_applies(workspace):
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")
    result = _session(workspace).registry.execute(
        "append_to_file", {"path": "m.py", "content": "def broken(:\n"}
    )
    assert not result.ok
    assert (workspace / "m.py").read_text(encoding="utf-8") == "x = 1\n"  # untouched


def test_resolution_rung_still_applies(workspace):
    (workspace / "util.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")
    result = _session(workspace).registry.execute(
        "append_to_file", {"path": "m.py", "content": "from util import ghost\n"}
    )
    assert not result.ok
    assert "does not define 'ghost'" in result.error


def test_append_is_reversible(workspace):
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")
    session = _session(workspace)
    session.registry.execute("append_to_file", {"path": "m.py", "content": "y = 2\n"})
    assert session.ledger.changed_files == ["m.py"]
    session.undo()
    assert (workspace / "m.py").read_text(encoding="utf-8") == "x = 1\n"


def test_empty_old_string_now_recommends_appending(workspace):
    """The exact call the model made in the failing benchmark run."""
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")
    result = _session(workspace).registry.execute(
        "edit_file", {"path": "m.py", "old_string": "", "new_string": "def f(): pass"}
    )
    assert not result.ok
    assert "append_to_file" in result.error


@pytest.mark.parametrize("tool", ["append_to_file"])
def test_registered_and_permission_gated(workspace, tool):
    from forge.tools.filesystem import AppendFileTool

    assert AppendFileTool.mutating is True
    assert tool in _session(workspace).registry.names()
