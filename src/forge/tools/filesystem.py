"""Filesystem tools: read, write, edit, list. Writes are confined to the
workspace, backed up via the ChangeLedger, and edits require an exact unique
match so the model can never clobber a file it has not actually read."""

from __future__ import annotations

from typing import Any

from forge.safety.guard import SafetyGuard
from forge.tools.base import Tool, ToolResult
from forge.tools.changes import ChangeLedger
from forge.tools.edit_repair import MatchOutcome, compute_edit
from forge.tools.syntax_check import gate_edit


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a file from the repository. Returns the exact file content. "
        "Use offset/limit for large files."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root."},
            "offset": {"type": "integer", "description": "1-based first line to read."},
            "limit": {"type": "integer", "description": "Maximum number of lines to return."},
        },
        "required": ["path"],
    }

    def __init__(self, guard: SafetyGuard) -> None:
        self._guard = guard

    def run(self, path: str, offset: int = 1, limit: int = 2000) -> ToolResult:
        resolved = self._guard.resolve_path(path)
        if not resolved.exists():
            return ToolResult(ok=False, error=f"File not found: {path}")
        if resolved.is_dir():
            return ToolResult(ok=False, error=f"{path} is a directory; use list_dir.")
        # utf-8-sig strips Windows BOMs so the model never sees ﻿ artifacts
        lines = resolved.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        start = max(offset, 1)
        selected = lines[start - 1 : start - 1 + max(limit, 1)]
        if not selected and lines:
            return ToolResult(
                ok=False, error=f"Offset {offset} is past end of file ({len(lines)} lines)."
            )
        # Content is returned verbatim (no line-number prefixes) so the model
        # can copy exact snippets into edit_file's old_string reliably.
        header = (
            f"[{path}: lines {start}-{start + len(selected) - 1} of {len(lines)}]\n"
            if lines
            else f"[{path}: empty file]\n"
        )
        return ToolResult(ok=True, output=header + "\n".join(selected))


class WriteFileTool(Tool):
    name = "write_file"
    mutating = True
    description = (
        "Create a new file or fully replace an existing one. The previous "
        "content is automatically backed up. Prefer edit_file for small changes."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root."},
            "content": {"type": "string", "description": "Complete file content."},
        },
        "required": ["path", "content"],
    }

    def __init__(
        self, guard: SafetyGuard, ledger: ChangeLedger, syntax_gate: bool = True
    ) -> None:
        self._guard = guard
        self._ledger = ledger
        self._syntax_gate = syntax_gate

    def run(self, path: str, content: str) -> ToolResult:
        resolved = self._guard.check_write_path(path)
        if self._syntax_gate:
            original = (
                resolved.read_text(encoding="utf-8-sig", errors="replace")
                if resolved.is_file()
                else None
            )
            gate = gate_edit(resolved.name, original, content)
            if gate:
                return ToolResult(
                    ok=False,
                    error=f"Rejected — this content has a syntax error ({gate}). "
                    f"Nothing was written to {path}. Fix the syntax and call "
                    "write_file again with the corrected complete content.",
                )
        self._ledger.record_before_write(resolved)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return ToolResult(ok=True, output=f"Wrote {len(content)} chars to {path}.")


class EditFileTool(Tool):
    name = "edit_file"
    mutating = True
    description = (
        "Replace an exact text snippet in a file. old_string must appear "
        "exactly once — read the file first to copy it precisely."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root."},
            "old_string": {
                "type": "string",
                "description": "Exact text to replace (must be unique).",
            },
            "new_string": {"type": "string", "description": "Replacement text."},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def __init__(
        self, guard: SafetyGuard, ledger: ChangeLedger, syntax_gate: bool = True
    ) -> None:
        self._guard = guard
        self._ledger = ledger
        self._syntax_gate = syntax_gate

    def run(self, path: str, old_string: str, new_string: str) -> ToolResult:
        resolved = self._guard.check_write_path(path)
        if not old_string:
            return ToolResult(
                ok=False,
                error="old_string must not be empty. Instead: call read_file on this "
                "file, then call write_file with the COMPLETE updated file content "
                "(existing content plus your change).",
            )
        if not resolved.exists():
            return ToolResult(ok=False, error=f"File not found: {path}")
        content = resolved.read_text(encoding="utf-8-sig", errors="replace")

        # Self-repairing match: exact, else whitespace-tolerant, else grounded
        # correction. Kills the "old_string not found" retry death-spiral.
        result = compute_edit(content, old_string, new_string)

        if result.outcome == MatchOutcome.APPLIED:
            if self._syntax_gate:
                gate = gate_edit(resolved.name, content, result.new_content)
                if gate:
                    return ToolResult(
                        ok=False,
                        error=f"Rejected — this edit would introduce a syntax error "
                        f"({gate}). The file was NOT modified. Fix new_string and "
                        "call edit_file again.",
                    )
            self._ledger.record_before_write(resolved)
            resolved.write_text(result.new_content, encoding="utf-8")
            note = (
                " (matched despite whitespace differences)"
                if result.tier == "whitespace"
                else ""
            )
            return ToolResult(ok=True, output=f"Edited {path}.{note}")

        if result.outcome == MatchOutcome.AMBIGUOUS:
            return ToolResult(
                ok=False,
                error=f"old_string matches {result.occurrences} places in {path}; "
                "include more surrounding lines so it is unique.",
            )

        # NOT_FOUND — ground the model with the file's real text when we found
        # a close span, instead of letting it retry the same hallucinated string.
        if result.suggestion is not None:
            return ToolResult(
                ok=False,
                error=f"old_string not found in {path}. The closest ACTUAL text in "
                f"the file is:\n----- copy this exactly -----\n{result.suggestion}\n"
                "-----------------------------\nRe-issue edit_file using that exact "
                "text as old_string.",
            )
        return ToolResult(
            ok=False,
            error=f"old_string not found in {path} and nothing similar is present. "
            "Call read_file to see the current content before editing.",
        )


class DeleteFileTool(Tool):
    name = "delete_file"
    mutating = True
    description = (
        "Delete a file from the repository. The content is backed up first, so "
        "the deletion is reversible with /undo. Only deletes files, not directories."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root."},
        },
        "required": ["path"],
    }

    def __init__(self, guard: SafetyGuard, ledger: ChangeLedger) -> None:
        self._guard = guard
        self._ledger = ledger

    def run(self, path: str) -> ToolResult:
        resolved = self._guard.check_write_path(path)
        if not resolved.exists():
            return ToolResult(ok=False, error=f"File not found: {path}")
        if resolved.is_dir():
            return ToolResult(
                ok=False, error=f"{path} is a directory; delete_file only removes files."
            )
        self._ledger.record_before_write(resolved)  # backup for /undo
        resolved.unlink()
        return ToolResult(ok=True, output=f"Deleted {path} (recoverable with /undo).")


class ListDirTool(Tool):
    name = "list_dir"
    description = "List files and directories at a path in the repository."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path, defaults to repo root."},
        },
        "required": [],
    }

    def __init__(self, guard: SafetyGuard) -> None:
        self._guard = guard

    def run(self, path: str = ".") -> ToolResult:
        resolved = self._guard.resolve_path(path)
        if not resolved.is_dir():
            return ToolResult(ok=False, error=f"Not a directory: {path}")
        entries = sorted(
            resolved.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
        )
        lines = [f"{entry.name}/" if entry.is_dir() else entry.name for entry in entries]
        return ToolResult(ok=True, output="\n".join(lines) or "(empty directory)")
