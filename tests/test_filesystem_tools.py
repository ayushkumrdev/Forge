from forge.tools.changes import ChangeLedger
from forge.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool


def test_write_then_read_roundtrip(guard, ledger, workspace):
    write = WriteFileTool(guard, ledger)
    read = ReadFileTool(guard)

    result = write.run(path="pkg/hello.py", content="print('hi')\n")
    assert result.ok
    assert (workspace / "pkg" / "hello.py").exists()

    result = read.run(path="pkg/hello.py")
    assert result.ok
    # content must be verbatim (no line-number prefixes) with an info header
    assert "print('hi')" in result.output
    assert result.output.startswith("[pkg/hello.py: lines 1-1 of 1]")


def test_overwrite_creates_backup(guard, ledger, workspace):
    write = WriteFileTool(guard, ledger)
    write.run(path="a.txt", content="original")
    # a second run overwriting the existing file must back up the original
    write2 = WriteFileTool(guard, ChangeLedger(workspace, "run2"))
    write2.run(path="a.txt", content="replaced")
    backup = workspace / ".forge" / "backups" / "run2" / "a.txt"
    assert backup.read_text(encoding="utf-8") == "original"
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "replaced"


def test_edit_requires_unique_match(guard, ledger, workspace):
    (workspace / "code.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    edit = EditFileTool(guard, ledger)

    result = edit.run(path="code.py", old_string="x = 1", new_string="x = 2")
    assert not result.ok
    assert "2 places" in result.error  # ambiguous match reported

    result = edit.run(path="code.py", old_string="missing", new_string="y")
    assert not result.ok
    assert "not found" in result.error


def test_edit_rejects_empty_old_string(guard, ledger, workspace):
    (workspace / "code.py").write_text("x = 1\n", encoding="utf-8")
    result = EditFileTool(guard, ledger).run(path="code.py", old_string="", new_string="y = 2")
    assert not result.ok
    assert "must not be empty" in result.error


def test_edit_applies_single_match(guard, ledger, workspace):
    (workspace / "code.py").write_text("value = 10\n", encoding="utf-8")
    edit = EditFileTool(guard, ledger)
    result = edit.run(path="code.py", old_string="value = 10", new_string="value = 42")
    assert result.ok
    assert (workspace / "code.py").read_text(encoding="utf-8") == "value = 42\n"


def test_ledger_diff_reports_changes(guard, ledger, workspace):
    write = WriteFileTool(guard, ledger)
    write.run(path="new.py", content="a = 1\n")
    diff = ledger.unified_diff()
    assert "+a = 1" in diff
    assert ledger.changed_files == ["new.py"]


def test_list_dir(guard, workspace):
    (workspace / "sub").mkdir()
    (workspace / "file.txt").write_text("x", encoding="utf-8")
    result = ListDirTool(guard).run(path=".")
    assert result.ok
    assert "sub/" in result.output
    assert "file.txt" in result.output


def test_bom_files_read_and_edited_cleanly(guard, ledger, workspace):
    """Windows tools often write UTF-8 with a BOM; the model must never see
    the \\ufeff artifact and exact-match edits must still work."""
    (workspace / "bom.py").write_bytes(b"\xef\xbb\xbf" + b"x = 1\n")

    result = ReadFileTool(guard).run(path="bom.py")
    assert result.ok
    assert "﻿" not in result.output

    result = EditFileTool(guard, ledger).run(path="bom.py", old_string="x = 1", new_string="x = 2")
    assert result.ok
    assert "x = 2" in (workspace / "bom.py").read_text(encoding="utf-8-sig")


def test_read_missing_file(guard):
    result = ReadFileTool(guard).run(path="nope.txt")
    assert not result.ok
    assert "not found" in result.error.lower()
