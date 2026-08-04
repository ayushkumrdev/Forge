"""Filesystem tools: read, write, edit, list. Writes are confined to the
workspace, backed up via the ChangeLedger, and edits require an exact unique
match so the model can never clobber a file it has not actually read."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.safety.guard import SafetyGuard
from forge.tools.base import Tool, ToolResult
from forge.tools.changes import ChangeLedger
from forge.tools.edit_repair import (
    EditResult,
    MatchOutcome,
    compute_edit,
    reindent_replacement,
)
from forge.tools.syntax_check import gate_edit
from forge.verify.ladder import Ladder


def _write_exact(path: Path, content: str) -> None:
    r"""Write content without translating newlines, keeping the file's own
    line-ending style.

    Path.write_text defaults to newline=None, which rewrites every "\n" as
    os.linesep. On Windows that silently turned LF files into CRLF, so
    editing one line of an ordinary repository produced a diff touching every
    line — unreviewable, and a merge conflict against every other checkout.

    Reads normalise CRLF to "\n" so the model can match text reliably, so a
    genuinely CRLF file is converted back here. New files get LF.
    """
    if path.exists():
        try:
            raw = path.read_bytes()
        except OSError:
            raw = b""
        if b"\r\n" in raw and "\r\n" not in content:
            content = content.replace("\n", "\r\n")
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)


def _verify(
    ladder: Ladder | None,
    syntax_gate: bool,
    resolved: Path,
    original: str | None,
    new_content: str,
) -> tuple[str | None, str]:
    """Run the strongest verification configured for this tool.

    Returns (refusal, advisory). `refusal` is a reason to reject the write;
    `advisory` reports a problem that was detected but deliberately not
    enforced, so an ablation still records what it stopped blocking."""
    if ladder is not None:
        verdict = ladder.check(resolved, original, new_content)
        if not verdict.ok:
            return f"the {verdict.failed_rung} check failed: {verdict.diagnostic}.", ""
        unenforced = verdict.unenforced_failures
        if unenforced:
            detail = "; ".join(f"{r.rung} check failed: {r.diagnostic}" for r in unenforced)
            return None, f" [unenforced: {detail}]"
        return None, ""
    if syntax_gate:
        error = gate_edit(resolved.name, original, new_content)
        if error:
            return f"this content has a syntax error ({error}).", ""
    return None, ""


def _exact_only_edit(content: str, old_string: str, new_string: str) -> EditResult:
    """Baseline matcher for ablations: byte-exact unique match or nothing —
    the behaviour of a conventional edit tool, with no repair."""
    occurrences = content.count(old_string)
    if occurrences == 1:
        return EditResult(
            outcome=MatchOutcome.APPLIED,
            new_content=content.replace(old_string, new_string, 1),
            tier="exact",
        )
    if occurrences > 1:
        return EditResult(outcome=MatchOutcome.AMBIGUOUS, occurrences=occurrences)
    return EditResult(outcome=MatchOutcome.NOT_FOUND)


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

    def run(self, path: str, content: str) -> ToolResult:
        resolved = self._guard.check_write_path(path)
        if self._syntax_gate or self._ladder is not None:
            original = (
                resolved.read_text(encoding="utf-8-sig", errors="replace")
                if resolved.is_file()
                else None
            )
            refusal, advisory = _verify(
                self._ladder, self._syntax_gate, resolved, original, content
            )
            if refusal:
                return ToolResult(
                    ok=False,
                    error=f"Rejected — {refusal} Nothing was written to {path}. "
                    "Fix it and call write_file again with the corrected "
                    "complete content.",
                )
        else:
            advisory = ""
        self._ledger.record_before_write(resolved)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        _write_exact(resolved, content)
        return ToolResult(
            ok=True, output=f"Wrote {len(content)} chars to {path}.{advisory}"
        )


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
        self,
        guard: SafetyGuard,
        ledger: ChangeLedger,
        syntax_gate: bool = True,
        edit_repair: bool = True,
        ladder: Ladder | None = None,
    ) -> None:
        self._guard = guard
        self._ledger = ledger
        self._syntax_gate = syntax_gate
        self._edit_repair = edit_repair
        self._ladder = ladder

    def run(self, path: str, old_string: str, new_string: str) -> ToolResult:
        resolved = self._guard.check_write_path(path)
        if not old_string:
            # The right advice depends on whether the file exists at all:
            # append_to_file cannot create one, and pointing the model at it
            # for a missing file just produces a second failure.
            if not resolved.exists():
                return ToolResult(
                    ok=False,
                    error=f"old_string must not be empty, and {path} does not "
                    "exist yet. Call write_file with the complete content to "
                    "create it.",
                )
            return ToolResult(
                ok=False,
                error="old_string must not be empty. To ADD new code to the end "
                "of this file, call append_to_file with just the new content — "
                "that is almost certainly what you want here. To change text "
                "that already exists, copy it exactly into old_string.",
            )
        if not resolved.exists():
            return ToolResult(ok=False, error=f"File not found: {path}")
        content = resolved.read_text(encoding="utf-8-sig", errors="replace")

        # Self-repairing match: exact, else whitespace-tolerant, else grounded
        # correction. Kills the "old_string not found" retry death-spiral.
        # With repair disabled (ablation), only an exact unique match applies.
        result = (
            compute_edit(content, old_string, new_string)
            if self._edit_repair
            else _exact_only_edit(content, old_string, new_string)
        )

        if result.outcome == MatchOutcome.APPLIED:
            refusal, advisory = _verify(
                self._ladder, self._syntax_gate, resolved, content, result.new_content
            )
            repaired_indent = False
            if refusal and self._edit_repair:
                # The replacement probably lost its indentation. Try lifting it
                # to the anchor's depth — and keep the result ONLY if it now
                # verifies, so a guess can never make things worse.
                lifted = reindent_replacement(content, old_string, new_string)
                if lifted != new_string:
                    candidate = compute_edit(content, old_string, lifted)
                    if candidate.outcome == MatchOutcome.APPLIED:
                        retry, retry_advisory = _verify(
                            self._ladder, self._syntax_gate, resolved,
                            content, candidate.new_content,
                        )
                        if not retry:
                            result, refusal, advisory = candidate, None, retry_advisory
                            repaired_indent = True
            if refusal:
                return ToolResult(
                    ok=False,
                    error=f"Rejected — {refusal} The file was NOT modified. "
                    "Fix new_string and call edit_file again.",
                )
            self._ledger.record_before_write(resolved)
            _write_exact(resolved, result.new_content)
            note = (
                " (matched despite whitespace differences)"
                if result.tier == "whitespace"
                else ""
            )
            if repaired_indent:
                note += " (re-indented to match the surrounding block)"
            # the tier is carried in the output so traces reveal HOW the edit
            # landed (exact vs repaired) — the grounded-edit metric reads it
            return ToolResult(
                ok=True, output=f"Edited {path} [match:{result.tier}].{note}{advisory}"
            )

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


class AppendFileTool(Tool):
    """Adding a function to an existing file is the commonest edit there is,
    and until this tool existed Forge had no way to express it: the model
    reached for edit_file with an empty old_string (meaning "put this at the
    end"), was refused, and deflected instead of rewriting the whole file.
    Observed as the dominant tier-2 failure in the benchmark."""

    name = "append_to_file"
    mutating = True
    description = (
        "Add content to the END of an existing file — the right tool for "
        "adding a new function, class or constant without touching what is "
        "already there. Use edit_file to change existing text, write_file "
        "only for a new file or a full rewrite."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root."},
            "content": {"type": "string", "description": "Text to add at the end of the file."},
        },
        "required": ["path", "content"],
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

    def run(self, path: str, content: str) -> ToolResult:
        resolved = self._guard.check_write_path(path)
        if not resolved.exists():
            return ToolResult(
                ok=False,
                error=f"File not found: {path}. Use write_file to create it.",
            )
        if resolved.is_dir():
            return ToolResult(ok=False, error=f"{path} is a directory.")
        original = resolved.read_text(encoding="utf-8-sig", errors="replace")
        # keep exactly one blank line between the old tail and the new block
        separator = "" if original.endswith("\n\n") or not original else (
            "\n" if original.endswith("\n") else "\n\n"
        )
        updated = original + separator + content.lstrip("\n")
        if not updated.endswith("\n"):
            updated += "\n"

        refusal, advisory = _verify(
            self._ladder, self._syntax_gate, resolved, original, updated
        )
        if refusal:
            return ToolResult(
                ok=False,
                error=f"Rejected — {refusal} {path} was NOT modified. Fix the "
                "content and call append_to_file again.",
            )
        self._ledger.record_before_write(resolved)
        _write_exact(resolved, updated)
        return ToolResult(
            ok=True, output=f"Appended {len(content)} chars to {path}.{advisory}"
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
