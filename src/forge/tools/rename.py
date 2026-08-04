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
import re
from pathlib import Path
from typing import Any

from forge.safety.guard import SafetyGuard
from forge.tools.base import Tool, ToolResult
from forge.tools.changes import ChangeLedger
from forge.tools.filesystem import _verify, _write_exact
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
        elif isinstance(node, ast.ImportFrom):
            # `from engine import calc` binds the name here too
            for alias in node.names:
                if alias.name == old and alias.asname is None:
                    add_definition(node)
        elif isinstance(node, ast.keyword) and node.arg == old:
            continue  # a keyword argument belongs to the callee's signature
    return sorted(set(spots))


def _defines(source: str, name: str) -> bool:
    """True when `name` is bound by a definition — not merely referenced."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        and node.name == name
        for node in ast.walk(tree)
    )


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


_SKIP_DIRS = frozenset(
    {".git", ".forge", ".venv", "venv", "node_modules", "__pycache__", "dist", "build"}
)
_MAX_SCANNED_FILES = 300


def importers_of(workspace: Path, module: str) -> list[Path]:
    """Repository files that import `module`, and so may reference the symbol.

    A rename that stops at one file is not a rename: observed live, the model
    correctly renamed `calc` in engine.py while cart.py and report.py kept
    calling it, and nothing noticed because the structural check only inspects
    files that were written to."""
    found: list[Path] = []
    scanned = 0
    stack = [workspace]
    while stack and scanned < _MAX_SCANNED_FILES:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name in _SKIP_DIRS:
                continue
            if entry.is_dir():
                stack.append(entry)
                continue
            if entry.suffix not in (".py", ".pyw"):
                continue
            scanned += 1
            try:
                text = entry.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                continue
            if re.search(rf"^\s*(?:import\s+{module}\b|from\s+{module}\s+import\b)",
                         text, re.MULTILINE):
                found.append(entry)
    return sorted(found)


def rename_module_references(source: str, module: str, old: str, new: str) -> tuple[str, int]:
    """Rename `old` in a file that IMPORTS it, covering both shapes:
    `from mod import old` / bare `old(...)`, and `mod.old(...)`."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, 0
    lines = source.splitlines(keepends=True)
    spots: list[tuple[int, int, int]] = []
    imported_by_name = any(
        isinstance(node, ast.ImportFrom)
        and node.module == module
        and any(a.name == old and a.asname is None for a in node.names)
        for node in ast.walk(tree)
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            for alias in node.names:
                if alias.name == old and alias.asname is None:
                    index = node.lineno - 1
                    if index < len(lines):
                        column = lines[index].find(old, node.col_offset)
                        if column != -1:
                            spots.append((node.lineno, column, column + len(old)))
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == old
            and isinstance(node.value, ast.Name)
            and node.value.id == module
        ):
            end = node.end_col_offset
            spots.append((node.end_lineno, end - len(old), end))
        elif imported_by_name and isinstance(node, ast.Name) and node.id == old:
            spots.append((node.lineno, node.col_offset, node.end_col_offset))

    if not spots:
        return source, 0
    by_line: dict[int, list[tuple[int, int]]] = {}
    for line, start, end in sorted(set(spots)):
        by_line.setdefault(line, []).append((start, end))
    for line, spans in by_line.items():
        index = line - 1
        if index >= len(lines):
            continue
        text = lines[index]
        for start, end in sorted(spans, reverse=True):
            if text[start:end] == old:
                text = text[:start] + new + text[end:]
        lines[index] = text
    return "".join(lines), len(set(spots))


class RenameSymbolTool(Tool):
    name = "rename_symbol"
    mutating = True
    description = (
        "Rename a function, class, method or variable: updates its definition, "
        "every reference in that file, AND every other file in the repository "
        "that imports it — one correct operation. ALWAYS use this for a rename "
        "instead of edit_file, which matches text and will hit a call site "
        "instead of the definition."
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
        # Only a DEFINITION collides. A mere reference to `new_name` is very
        # often a half-finished rename — the model renamed a call site by
        # hand and is now asking for the definition to follow. Refusing that
        # blocks the exact recovery the tool exists to provide, which is what
        # happened live: an earlier edit had produced `self.dequeue()`, so
        # renaming `pop` to `dequeue` was rejected as a collision.
        if _defines(original, new_name):
            return ToolResult(
                ok=False,
                error=f"'{new_name}' is already defined in {path}; renaming "
                f"'{old_name}' to it would collide. Pick another name.",
            )

        updated, count = apply_rename(original, old_name, new_name)
        if count == 0:
            return ToolResult(
                ok=False,
                error=f"'{old_name}' is not defined or referenced in {path}. "
                "Check the spelling, or the file.",
            )

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

        # A rename that stops at one file is not a rename. Propagate to every
        # repository file that imports this module.
        workspace = self._guard.workspace
        module = resolved.stem
        propagated: list[str] = []
        for other in importers_of(workspace, module):
            if other == resolved:
                continue
            try:
                text = other.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                continue
            rewritten, hits = rename_module_references(text, module, old_name, new_name)
            if not hits:
                continue
            refusal, _ = _verify(self._ladder, self._syntax_gate, other, text, rewritten)
            if refusal:
                continue  # never leave a dependent file worse than it was
            self._ledger.record_before_write(other)
            _write_exact(other, rewritten)
            propagated.append(
                str(other.relative_to(workspace)).replace("\\", "/")
            )

        also = (
            f" Also updated {len(propagated)} importing file(s): "
            + ", ".join(propagated)
            if propagated
            else ""
        )
        return ToolResult(
            ok=True,
            output=f"Renamed {old_name} to {new_name} in {path} "
            f"({count} occurrence{'s' if count != 1 else ''}: the definition and "
            f"every reference).{also}{advisory}",
        )
