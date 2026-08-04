"""run_tests — detecting the project's runner instead of guessing.

Observed live on t4-make-suite-pass: the fixture holds pytest-style test
functions, the model ran `python -m unittest discover`, unittest reported
0 tests because there is no TestCase subclass, and the model concluded the
project had no tests and started writing its own — abandoning the request."""

from forge.chat.session import ChatSession
from forge.config import ForgeSettings
from forge.llm.mock import MockLLMClient
from forge.tools.testing import detect_runner, python_test_files

PYTEST_STYLE = "def test_one():\n    assert 1 == 1\n"
UNITTEST_STYLE = (
    "import unittest\n\n\n"
    "class T(unittest.TestCase):\n"
    "    def test_one(self):\n"
    "        self.assertEqual(1, 1)\n"
)


def _session(workspace):
    return ChatSession(workspace, MockLLMClient([]), ForgeSettings(), session_id="rt")


def test_finds_test_files_by_either_convention(workspace):
    (workspace / "test_a.py").write_text(PYTEST_STYLE, encoding="utf-8")
    (workspace / "b_test.py").write_text(PYTEST_STYLE, encoding="utf-8")
    (workspace / "notatest.py").write_text("x = 1\n", encoding="utf-8")
    names = {p.name for p in python_test_files(workspace)}
    assert names == {"test_a.py", "b_test.py"}


def test_ignores_noise_directories(workspace):
    junk = workspace / "node_modules" / "pkg"
    junk.mkdir(parents=True)
    (junk / "test_dep.py").write_text(PYTEST_STYLE, encoding="utf-8")
    assert python_test_files(workspace) == []


def test_plain_function_tests_choose_pytest(workspace):
    """The exact case that broke: unittest cannot run these."""
    (workspace / "test_a.py").write_text(PYTEST_STYLE, encoding="utf-8")
    command, why = detect_runner(workspace)
    assert command is not None and "pytest" in " ".join(command)
    assert "pytest" in why


def test_no_tests_is_reported_not_guessed(workspace):
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    command, why = detect_runner(workspace)
    assert command is None
    assert "no test files" in why


def test_tool_runs_a_passing_suite(workspace):
    (workspace / "test_ok.py").write_text(PYTEST_STYLE, encoding="utf-8")
    result = _session(workspace).registry.execute("run_tests", {})
    assert result.ok
    assert "PASSED" in result.output


def test_tool_reports_a_failing_suite(workspace):
    (workspace / "test_bad.py").write_text("def test_one():\n    assert 1 == 2\n", "utf-8")
    result = _session(workspace).registry.execute("run_tests", {})
    assert result.ok  # the tool worked; the SUITE failed
    assert "FAILED" in result.output
    assert "test_one" in result.output


def test_tool_says_so_when_there_is_nothing_to_run(workspace):
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    result = _session(workspace).registry.execute("run_tests", {})
    assert result.ok
    assert "No tests were run" in result.output
    assert "py_compile" in result.output  # offers a real alternative


def test_unittest_classes_still_work_without_pytest(workspace, monkeypatch):
    (workspace / "test_c.py").write_text(UNITTEST_STYLE, encoding="utf-8")
    import builtins

    real_import = builtins.__import__

    def no_pytest(name, *args, **kwargs):
        if name == "pytest":
            raise ImportError("no pytest")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pytest)
    command, why = detect_runner(workspace)
    assert command is not None and "unittest" in " ".join(command)
    assert "TestCase" in why


def test_tool_is_registered_and_permission_gated(workspace):
    from forge.tools.testing import RunTestsTool

    assert RunTestsTool.mutating is True  # it executes project code
    assert "run_tests" in _session(workspace).registry.names()


def test_running_tests_counts_as_verification(workspace):
    """run_tests must satisfy the false-verification gate — it IS running the
    checks, so a reply that says so afterwards is telling the truth."""
    from forge.llm.base import ChatMessage, ToolCall

    (workspace / "test_ok.py").write_text(PYTEST_STYLE, encoding="utf-8")
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    llm = MockLLMClient([
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file", arguments={"path": "app.py", "content": "x = 2\n"})]),
        ChatMessage(role="assistant", tool_calls=[ToolCall(name="run_tests", arguments={})]),
        ChatMessage(role="assistant", content="Updated app.py; ran the tests, all passed."),
    ])
    session = ChatSession(workspace, llm, ForgeSettings(), session_id="rtv")
    reply = session.send("change x to 2 in app.py")
    assert reply == "Updated app.py; ran the tests, all passed."
    assert len(llm.requests) == 3  # not bounced as a false claim
