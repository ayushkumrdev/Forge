"""Git tool: a curated, allowlisted subset of git. Read operations plus safe
write operations (add/commit/new branch). Destructive git (force push, hard
reset, clean) is blocked both here and by the safety guard."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from forge.tools.base import Tool, ToolResult

_ALLOWED_SUBCOMMANDS = {
    "status", "diff", "log", "show", "branch", "add", "commit", "stash",
    "ls-files", "blame", "rev-parse",
}


class GitTool(Tool):
    name = "git"
    mutating = True
    description = (
        "Run a git command in the repository. Allowed subcommands: "
        "status, diff, log, show, branch, add, commit, stash, ls-files, "
        "blame, rev-parse. Example args: 'status --short' or "
        "'commit -m \"message\"'."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "args": {"type": "string", "description": "Arguments after 'git', e.g. 'diff --stat'."},
        },
        "required": ["args"],
    }

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    def run(self, args: str) -> ToolResult:
        tokens = args.strip().split()
        if not tokens:
            return ToolResult(ok=False, error="Empty git command.")
        subcommand = tokens[0]
        if subcommand not in _ALLOWED_SUBCOMMANDS:
            return ToolResult(
                ok=False,
                error=f"git {subcommand} is not allowed. "
                f"Allowed: {', '.join(sorted(_ALLOWED_SUBCOMMANDS))}",
            )
        try:
            completed = subprocess.run(
                ["git", *tokens],
                cwd=self._workspace,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
        except FileNotFoundError:
            return ToolResult(ok=False, error="git is not installed or not on PATH.")
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error=f"git {subcommand} timed out.")

        output = (completed.stdout + completed.stderr).strip() or "(no output)"
        return ToolResult(ok=True, output=f"exit code: {completed.returncode}\n{output}")
