"""Line endings must survive a round trip.

Path.write_text defaults to newline=None, which rewrites every "\n" as
os.linesep. On Windows that silently converted every LF file Forge touched
into CRLF: editing one line of an ordinary repository produced a diff
touching every line — unreviewable, and a merge conflict against every other
checkout. These tests pin both directions."""

import pytest

from forge.chat.session import ChatSession
from forge.config import ForgeSettings
from forge.llm.mock import MockLLMClient

LF = b"def one():\n    return 1\n"
CRLF = b"def one():\r\n    return 1\r\n"


@pytest.fixture
def session(workspace):
    return ChatSession(workspace, MockLLMClient([]), ForgeSettings(), session_id="nl")


def test_edit_preserves_lf(workspace, session):
    (workspace / "m.py").write_bytes(LF)
    result = session.registry.execute(
        "edit_file", {"path": "m.py", "old_string": "return 1", "new_string": "return 9"}
    )
    assert result.ok
    raw = (workspace / "m.py").read_bytes()
    assert b"\r\n" not in raw
    assert raw == b"def one():\n    return 9\n"


def test_edit_preserves_crlf(workspace, session):
    (workspace / "m.py").write_bytes(CRLF)
    session.registry.execute(
        "edit_file", {"path": "m.py", "old_string": "return 1", "new_string": "return 9"}
    )
    assert (workspace / "m.py").read_bytes() == b"def one():\r\n    return 9\r\n"


def test_append_preserves_lf(workspace, session):
    (workspace / "m.py").write_bytes(LF)
    session.registry.execute(
        "append_to_file", {"path": "m.py", "content": "def two():\n    return 2\n"}
    )
    assert b"\r\n" not in (workspace / "m.py").read_bytes()


def test_append_preserves_crlf(workspace, session):
    (workspace / "m.py").write_bytes(CRLF)
    session.registry.execute(
        "append_to_file", {"path": "m.py", "content": "def two():\n    return 2\n"}
    )
    raw = (workspace / "m.py").read_bytes()
    assert b"def two():" in raw
    # every LF in the file is part of a CRLF — no stray bare newlines
    assert raw.count(b"\r\n") == raw.count(b"\n")


def test_write_file_preserves_lf(workspace, session):
    (workspace / "m.py").write_bytes(LF)
    session.registry.execute(
        "write_file", {"path": "m.py", "content": "a = 1\nb = 2\n"}
    )
    assert (workspace / "m.py").read_bytes() == b"a = 1\nb = 2\n"


def test_new_files_use_lf(workspace, session):
    session.registry.execute(
        "write_file", {"path": "new.py", "content": "a = 1\nb = 2\n"}
    )
    assert (workspace / "new.py").read_bytes() == b"a = 1\nb = 2\n"


def test_undo_restores_bytes_exactly(workspace, session):
    (workspace / "m.py").write_bytes(LF)
    session.registry.execute("write_file", {"path": "m.py", "content": "gone\n"})
    session.undo()
    assert (workspace / "m.py").read_bytes() == LF


def test_a_one_line_edit_touches_one_line(workspace, session):
    """The symptom that made this bug expensive: a whole-file diff."""
    original = b"".join(f"line_{i} = {i}\n".encode() for i in range(40))
    (workspace / "big.py").write_bytes(original)
    session.registry.execute(
        "edit_file",
        {"path": "big.py", "old_string": "line_7 = 7", "new_string": "line_7 = 99"},
    )
    after = (workspace / "big.py").read_bytes()
    changed = [
        (a, b) for a, b in zip(original.split(b"\n"), after.split(b"\n"), strict=False)
        if a != b
    ]
    assert len(changed) == 1, f"{len(changed)} lines changed, expected 1"
