"""rename_symbol — renaming as one correct AST operation.

Text substitution is the wrong instrument for a rename, and the benchmark
proved it repeatedly: asked to rename `pop` to `dequeue`, the model issued
edit_file with old_string "pop()", which matched `self.pop()` — the CALL —
instead of `def pop(self)`. It never recovered, even when told precisely what
was broken with the file in front of it."""

import ast

from forge.chat.session import ChatSession
from forge.config import ForgeSettings
from forge.llm.mock import MockLLMClient
from forge.tools.rename import apply_rename, occurrences

QUEUE = (
    "class Queue:\n"
    "    def __init__(self):\n"
    "        self._items = []\n\n"
    "    def push(self, item):\n"
    "        self._items.append(item)\n\n"
    "    def pop(self):\n"
    "        return self._items.pop(0)\n\n"
    "    def drain(self):\n"
    "        out = []\n"
    "        while self._items:\n"
    "            out.append(self.pop())\n"
    "        return out\n\n\n"
    "def fill(queue, values):\n"
    "    for value in values:\n"
    "        queue.push(value)\n"
    "    return queue\n"
)


def _session(workspace, **kw):
    return ChatSession(workspace, MockLLMClient([]), ForgeSettings(**kw), session_id="rn")


# -- the analysis -----------------------------------------------------------------


def test_rename_updates_definition_and_every_reference():
    out, count = apply_rename(QUEUE, "push", "enqueue")
    assert "def enqueue(self, item):" in out
    assert "queue.enqueue(value)" in out
    assert "push" not in out
    assert count == 2
    ast.parse(out)


def test_a_list_method_sharing_the_name_is_left_alone():
    """The bug this tool hit on its very first real run: renaming the method
    `pop` also rewrote `self._items.pop(0)`, which is a LIST pop."""
    out, _ = apply_rename(QUEUE, "pop", "dequeue")
    assert "def dequeue(self):" in out
    assert "self._items.pop(0)" in out  # untouched
    assert "out.append(self.dequeue())" in out
    ast.parse(out)


def test_both_renames_compose_into_the_correct_file():
    out, _ = apply_rename(QUEUE, "push", "enqueue")
    out, _ = apply_rename(out, "pop", "dequeue")
    assert "def enqueue(self, item)" in out
    assert "def dequeue(self)" in out
    assert "queue.enqueue(value)" in out
    assert "self._items.pop(0)" in out
    assert "self._items.append(item)" in out
    ast.parse(out)


def test_comments_and_strings_are_not_touched():
    source = 'def go():\n    """calls go elsewhere"""\n    # go go go\n    return "go"\n'
    out, count = apply_rename(source, "go", "run")
    assert "def run():" in out
    assert '"""calls go elsewhere"""' in out
    assert "# go go go" in out
    assert 'return "go"' in out
    assert count == 1


def test_longer_identifiers_sharing_a_prefix_survive():
    source = "def go():\n    return 1\n\n\ndef going():\n    return go()\n"
    out, _ = apply_rename(source, "go", "run")
    assert "def going():" in out
    assert "return run()" in out


def test_module_level_function_and_its_callers():
    source = "def helper(x):\n    return x\n\n\ndef main():\n    return helper(1)\n"
    out, count = apply_rename(source, "helper", "assist")
    assert "def assist(x):" in out and "return assist(1)" in out
    assert count == 2


def test_unparseable_source_yields_nothing():
    assert occurrences("def broken(:\n", "broken") == []


def test_missing_name_changes_nothing():
    out, count = apply_rename(QUEUE, "nonexistent", "x")
    assert count == 0 and out == QUEUE


# -- the tool ---------------------------------------------------------------------


def test_tool_renames_and_reports(workspace):
    (workspace / "q.py").write_text(QUEUE, encoding="utf-8")
    result = _session(workspace).registry.execute(
        "rename_symbol", {"path": "q.py", "old_name": "push", "new_name": "enqueue"}
    )
    assert result.ok
    assert "2 occurrences" in result.output
    assert "def enqueue(self, item)" in (workspace / "q.py").read_text(encoding="utf-8")


def test_tool_refuses_a_name_collision(workspace):
    (workspace / "q.py").write_text(QUEUE, encoding="utf-8")
    result = _session(workspace).registry.execute(
        "rename_symbol", {"path": "q.py", "old_name": "push", "new_name": "drain"}
    )
    assert not result.ok
    assert "collide" in result.error
    assert "def push" in (workspace / "q.py").read_text(encoding="utf-8")  # untouched


def test_tool_refuses_an_unknown_name(workspace):
    (workspace / "q.py").write_text(QUEUE, encoding="utf-8")
    result = _session(workspace).registry.execute(
        "rename_symbol", {"path": "q.py", "old_name": "ghost", "new_name": "spirit"}
    )
    assert not result.ok
    assert "not defined or referenced" in result.error


def test_tool_refuses_a_broken_file(workspace):
    (workspace / "bad.py").write_text("def oops(:\n", encoding="utf-8")
    result = _session(workspace).registry.execute(
        "rename_symbol", {"path": "bad.py", "old_name": "oops", "new_name": "fine"}
    )
    assert not result.ok
    assert "does not parse" in result.error


def test_tool_refuses_non_python(workspace):
    (workspace / "notes.md").write_text("# push\n", encoding="utf-8")
    result = _session(workspace).registry.execute(
        "rename_symbol", {"path": "notes.md", "old_name": "push", "new_name": "enqueue"}
    )
    assert not result.ok
    assert "Python only" in result.error


def test_tool_validates_identifiers(workspace):
    (workspace / "q.py").write_text(QUEUE, encoding="utf-8")
    ex = _session(workspace).registry.execute
    assert not ex("rename_symbol", {"path": "q.py", "old_name": "push", "new_name": "2bad"}).ok
    assert not ex("rename_symbol", {"path": "q.py", "old_name": "push", "new_name": "push"}).ok


def test_rename_is_reversible(workspace):
    (workspace / "q.py").write_text(QUEUE, encoding="utf-8")
    session = _session(workspace)
    session.registry.execute(
        "rename_symbol", {"path": "q.py", "old_name": "push", "new_name": "enqueue"}
    )
    session.undo()
    assert (workspace / "q.py").read_text(encoding="utf-8") == QUEUE


def test_rename_preserves_line_endings(workspace):
    (workspace / "q.py").write_bytes(b"def go():\r\n    return go\r\n")
    _session(workspace).registry.execute(
        "rename_symbol", {"path": "q.py", "old_name": "go", "new_name": "run"}
    )
    raw = (workspace / "q.py").read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n")


def test_tool_is_registered(workspace):
    assert "rename_symbol" in _session(workspace).registry.names()
