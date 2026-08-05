"""L4 — importing what was just written.

Every rung below this one reads the code. All of them pass on a package
whose __init__.py exports nothing, because nothing in the source is wrong:
the mistake is in what the source does not do. This rung stops reading and
runs it, which is the cheapest possible execution — no test to write, no
model judgement, no output to interpret.
"""

from pathlib import Path

from forge.chat.session import ChatSession
from forge.config import ForgeSettings
from forge.llm.base import ChatMessage, ToolCall
from forge.llm.mock import MockLLMClient
from forge.verify.runtime import import_errors, module_name


def test_module_names_map_from_paths(tmp_path):
    assert module_name(tmp_path, tmp_path / "cli.py") == "cli"
    assert module_name(tmp_path, tmp_path / "pkg" / "thing.py") == "pkg.thing"
    # importing the PACKAGE is what exercises its exports
    assert module_name(tmp_path, tmp_path / "pkg" / "__init__.py") == "pkg"


def test_files_that_should_not_be_imported_are_skipped(tmp_path):
    assert module_name(tmp_path, tmp_path / "test_thing.py") is None
    assert module_name(tmp_path, tmp_path / "setup.py") is None
    assert module_name(tmp_path, tmp_path / "notes.md") is None
    assert module_name(tmp_path, tmp_path / "my-dir" / "x.py") is None  # not importable
    assert module_name(tmp_path, Path("/elsewhere/x.py")) is None


def test_a_package_that_exports_nothing_is_caught(tmp_path):
    """The exact failure this rung was built for, and the one every static
    check passes: `from mathkit import is_prime` raises ImportError while
    every file parses and every internal import resolves."""
    pkg = tmp_path / "mathkit"
    pkg.mkdir()
    (pkg / "primes.py").write_text("def is_prime(n):\n    return n > 1\n", encoding="utf-8")
    (pkg / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    consumer = tmp_path / "app.py"
    consumer.write_text("from mathkit import is_prime\n", encoding="utf-8")

    problems = import_errors(tmp_path, [consumer])
    assert len(problems) == 1
    assert "import app" in problems[0]
    assert "is_prime" in problems[0]


def test_a_working_package_is_silent(tmp_path):
    pkg = tmp_path / "mathkit"
    pkg.mkdir()
    (pkg / "primes.py").write_text("def is_prime(n):\n    return n > 1\n", encoding="utf-8")
    (pkg / "__init__.py").write_text(
        "from mathkit.primes import is_prime\n", encoding="utf-8"
    )
    assert import_errors(tmp_path, [pkg / "__init__.py"]) == []


def test_a_missing_import_is_caught(tmp_path):
    module = tmp_path / "signup.py"
    module.write_text("import nonexistent_package_xyz\n", encoding="utf-8")
    problems = import_errors(tmp_path, [module])
    assert problems and "nonexistent_package_xyz" in problems[0]


def test_a_module_that_cannot_run_is_never_invented(tmp_path):
    """A rung that cannot run must say nothing, never something wrong."""
    assert import_errors(tmp_path, []) == []
    assert import_errors(tmp_path, [tmp_path / "gone.py"]) == []


def test_the_turn_sends_the_model_back_when_the_import_fails(workspace):
    """A module whose top level raises is the class L4 owns outright: it
    parses, every name in it resolves, and it explodes the moment anyone
    imports it. The static rungs are all green."""
    llm = MockLLMClient([
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file",
            arguments={"path": "config.py",
                       "content": "TIMEOUT = 30\nRATIO = 100 / 0\n"})]),
        ChatMessage(role="assistant", content="Added the config."),
        # bounced, and fixed
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file",
            arguments={"path": "config.py",
                       "content": "TIMEOUT = 30\nRATIO = 100 / 4\n"})]),
        ChatMessage(role="assistant", content="Fixed the division."),
    ] + [ChatMessage(role="assistant", content="Done.")] * 6)
    session = ChatSession(workspace, llm, ForgeSettings(), session_id="imp")
    session.send("add a config.py with a TIMEOUT and a RATIO")
    assert any("import config" in m.content for m in session.history)
    assert "100 / 4" in (workspace / "config.py").read_text(encoding="utf-8")


def test_the_rung_can_be_switched_off(workspace):
    """Every gate here is ablatable, and a subprocess per changed file is
    the most expensive one."""
    (workspace / "broken.py").write_text("x = 1\n", encoding="utf-8")
    llm = MockLLMClient([
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file",
            arguments={"path": "broken.py", "content": "import no_such_module_xyz\n"})]),
        ChatMessage(role="assistant", content="Wrote it."),
    ])
    settings = ForgeSettings(gate_import_check=False, gate_resolution=False)
    session = ChatSession(workspace, llm, settings, session_id="off")
    assert session.send("write broken.py") == "Wrote it."
