"""rename_symbol — renaming as one correct operation instead of many risky ones.

Renaming is among the commonest edits there is, and text substitution is the
wrong instrument for it. Observed repeatedly on the benchmark: asked to
rename `pop` to `dequeue`, the model issued edit_file with old_string
"pop()", which matched `self.pop()` — the CALL — instead of `def pop(self)`.
The method kept its old name, the call pointed at nothing, and the model
never recovered even when told precisely what was wrong, twice, with the file
in front of it.

The fix is not a better nudge. It is an operation the model cannot get wrong:
find every binding of the name in the parse tree — the definition and each
reference — and rewrite exactly those spans. Comments, formatting and
unrelated identifiers that merely share the name are untouched, because the
edit is driven by the AST rather than by string matching.
"""

from __future__ import annotations

import ast
from typing import Any

from forge.safety.guard import SafetyGuard
from forge.tools.base import Tool, ToolResult
from forge.tools.changes import ChangeLedger
from forge.tools.filesystem import _write_exact
from forge.verify.ladder import Ladder

_IDENTIFIER_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _is_identifier(name: str) -> bool:
    return bool(name) and name.isidentifier() and not name[0].isdigit()


def occurrences(source: str, old: str) -> list[tuple[int, int, int]]:
    """Every place `old` is bound or referenced, as (line, start_col, end_col).

    Driven by the parse tree, so a comment mentioning the name, a string
    containing it, or a different object's attribute of the same name are all
    correctly left alone."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    spots: list[tuple[int, int, int]] = []

    def add_definition(node: ast.AST) -> None:
        # ast gives the node's position, not its name's; the name follows the
        # keyword on that line, so find it there.
        line_index = node.lineno - 1
        if line_index >= len(lines):
            return
        text = lines[line_index]
        column = text.find(old, node.col_offset)
        if column == -1:
            return
        before = text[column - 1] if column > 0 else " "
        after_index = column + len(old)
        after = text[after_index] if after_index < len(text) else " "
        if before in _IDENTIFIER_OK or after in _IDENTIFIER_OK:
            return  # part of a longer identifier
        spots.append((node.lineno, column, column + len(old)))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.name == old:
                add_definition(node)
        elif isinstance(node, ast.Name) and node.id == old:
            spots.append((node.lineno, node.col_offset, node.end_col_offset))
        elif isinstance(node, ast.Attribute) and node.attr == old:
            # Only through a PLAIN receiver: `self.pop()` and `queue.pop()`
            # may be the symbol being renamed, `self._items.pop(0)` is a list
            # method that merely shares the name. Renaming that broke the
            # file the first time this tool was run.
            if not isinstance(node.value, ast.Name):
                continue
            end = node.end_col_offset
            spots.append((node.end_lineno, end - len(old), end))
        elif isinstance(node, ast.arg) and node.arg == old:
            spots.append((node.lineno, node.col_offset, node.col_offset + len(old)))
        elif isinstance(node, ast.keyword) and node.arg == old:
            continue  # a keyword argument belongs to the callee's signature
    return sorted(set(spots))


def apply_rename(source: str, old: str, new: str) -> tuple[str, int]:
    """Rewrite every occurrence found in the parse tree. Returns the new
    source and how many places changed."""
    spots = occurrences(source, old)
    if not spots:
        return source, 0
    lines = source.splitlines(keepends=True)
    by_line: dict[int, list[tuple[int, int]]] = {}
    for line, start, end in spots:
        by_line.setdefault(line, []).append((start, end))
    for line, spans in by_line.items():
        index = line - 1
        if index >= len(lines):
            continue
        text = lines[index]
        for start, end in sorted(spans, reverse=True):  # right to left
            if text[start:end] != old:
                continue  # positions shifted or mismatched: never guess
            text = text[:start] + new + text[end:]
        lines[index] = text
    return "".join(lines), len(spots)


class RenameSymbolTool(Tool):
    name = "rename_symbol"
    mutating = True
    description = (
        "Rename a function, class, method or variable throughout a Python "
        "file, updating its definition AND every reference in one correct "
        "operation. ALWAYS use this for a rename instead of edit_file — "
        "edit_file matches text and will hit a call instead of the definition."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Python file, relative to the repo."},
            "old_name": {"type": "string", "description": "Current name."},
            "new_name": {"type": "string", "description": "New name."},
        },
        "required": ["path", "old_name", "new_name"],
    }

    def __init__(
        self,
        guard: SafetyGuard,
        ledger: ChangeLedger,
        syntax_gate: bool = True,
        ladder: Ladder | None = None,
    ) -> None:
        self._guard = guard
        self._ledger = ledger
        self._syntax_gate = syntax_gate
        self._ladder = ladder

    def run(self, path: str, old_name: str, new_name: str) -> ToolResult:
        resolved = self._guard.check_write_path(path)
        if not resolved.is_file():
            return ToolResult(ok=False, error=f"File not found: {path}")
        if resolved.suffix not in (".py", ".pyw"):
            return ToolResult(
                ok=False,
                error=f"rename_symbol understands Python only; {path} is not a .py file.",
            )
        if not _is_identifier(old_name) or not _is_identifier(new_name):
            return ToolResult(
                ok=False, error="old_name and new_name must both be valid identifiers."
            )
        if old_name == new_name:
            return ToolResult(ok=False, error="old_name and new_name are the same.")

        original = resolved.read_text(encoding="utf-8-sig", errors="replace")
        try:
            ast.parse(original)
        except SyntaxError as exc:
            return ToolResult(
                ok=False,
                error=f"{path} does not parse ({exc.msg} at line {exc.lineno}), so a "
                "rename cannot be done safely. Fix the syntax first.",
            )
        if occurrences(original, new_name):
            return ToolResult(
                ok=False,
                error=f"'{new_name}' is already used in {path}; renaming "
                f"'{old_name}' to it would collide. Pick another name.",
            )

        updated, count = apply_rename(original, old_name, new_name)
        if count == 0:
            return ToolResult(
                ok=False,
                error=f"'{old_name}' is not defined or referenced in {path}. "
                "Check the spelling, or the file.",
            )

        from forge.tools.filesystem import _verify

        refusal, advisory = _verify(
            self._ladder, self._syntax_gate, resolved, original, updated
        )
        if refusal:
            return ToolResult(
                ok=False,
                error=f"Rejected — {refusal} {path} was NOT modified.",
            )
        self._ledger.record_before_write(resolved)
        _write_exact(resolved, updated)
        return ToolResult(
            ok=True,
            output=f"Renamed {old_name} to {new_name} in {path} "
            f"({count} occurrence{'s' if count != 1 else ''}: the definition and "
            f"every reference).{advisory}",
        )
