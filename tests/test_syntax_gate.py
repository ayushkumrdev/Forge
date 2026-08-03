"""The syntax gate: writes/edits that would introduce a syntax error are
refused before touching disk; files that were already broken are never
blocked (only NEW errors are gated)."""

from forge.tools.filesystem import EditFileTool, WriteFileTool
from forge.tools.syntax_check import gate_edit, syntax_error


def test_python_error_detected_with_line():
    error = syntax_error("app.py", "def broken(:\n    pass\n")
    assert error is not None
    assert "line 1" in error


def test_python_valid_passes():
    assert syntax_error("app.py", "def fine():\n    return 1\n") is None


def test_json_and_toml_checked():
    assert syntax_error("cfg.json", '{"a": 1}') is None
    assert "line 1" in syntax_error("cfg.json", '{"a": }')
    assert syntax_error("cfg.toml", 'a = 1\n') is None
    assert syntax_error("cfg.toml", 'a = = 1\n') is not None


def test_unknown_language_never_blocks():
    assert syntax_error("notes.txt", "anything at all {{{") is None
    assert syntax_error("Makefile", "\t$(error") is None


def test_gate_only_blocks_new_errors():
    # file was already broken -> gate stays open even for broken output
    assert gate_edit("old.py", "def broken(:\n", "def still(:\n") is None
    # file was fine -> broken result is blocked
    assert gate_edit("old.py", "x = 1\n", "def broken(:\n") is not None
    # new file (no original) -> broken content is blocked
    assert gate_edit("new.py", None, "def broken(:\n") is not None


def test_write_file_refuses_broken_python(guard, ledger, workspace):
    write = WriteFileTool(guard, ledger)
    result = write.run(path="bad.py", content="def broken(:\n    pass\n")
    assert not result.ok
    assert "syntax error" in result.error
    assert not (workspace / "bad.py").exists()  # nothing touched disk


def test_edit_file_refuses_edit_that_breaks_syntax(guard, ledger, workspace):
    (workspace / "code.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    edit = EditFileTool(guard, ledger)
    result = edit.run(path="code.py", old_string="return 1", new_string="return (1")
    assert not result.ok
    assert "NOT modified" in result.error
    # the file is untouched
    assert (workspace / "code.py").read_text(encoding="utf-8") == "def f():\n    return 1\n"


def test_edit_file_allows_valid_edit(guard, ledger, workspace):
    (workspace / "code.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    result = EditFileTool(guard, ledger).run(
        path="code.py", old_string="return 1", new_string="return 2"
    )
    assert result.ok


def test_edit_on_already_broken_file_not_trapped(guard, ledger, workspace):
    """A pre-existing broken file must remain editable — the gate only rejects
    errors the edit itself introduces."""
    (workspace / "wip.py").write_text("def broken(:\n    x = 1\n", encoding="utf-8")
    result = EditFileTool(guard, ledger).run(
        path="wip.py", old_string="x = 1", new_string="x = 2"
    )
    assert result.ok


def test_gate_can_be_disabled(guard, ledger, workspace):
    write = WriteFileTool(guard, ledger, syntax_gate=False)
    result = write.run(path="bad.py", content="def broken(:\n")
    assert result.ok
    assert (workspace / "bad.py").exists()


def test_tree_sitter_language_gated_when_available(guard, ledger, workspace):
    from forge.tools.syntax_check import TREE_SITTER_AVAILABLE

    if not TREE_SITTER_AVAILABLE:
        return  # environment without native wheels: gate is open by design
    write = WriteFileTool(guard, ledger)
    result = write.run(path="app.js", content="function f( {\n")
    assert not result.ok
    result = write.run(path="app.js", content="function f() { return 1; }\n")
    assert result.ok
