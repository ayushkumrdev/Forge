"""Safety guard: every command and file path the agents touch goes through
here. Blocks destructive commands, confines file access to the workspace,
and protects VCS internals. Tools must not bypass this module."""

from __future__ import annotations

import re
from pathlib import Path


class SafetyViolation(Exception):
    pass


# Case-insensitive patterns for commands that are never allowed, regardless of
# what the model asks for. Covers POSIX and Windows destructive idioms.
_BLOCKED_COMMANDS: list[tuple[str, str]] = [
    (r"\brm\s+(-[a-z]*[rf][a-z]*\s+)+([\"']?)(/|~|\\|[a-z]:)", "recursive delete of a root path"),
    (r"\bdel\b.*\s/s\b", "recursive Windows delete"),
    (r"\b(rd|rmdir)\b.*\s/s\b", "recursive Windows directory removal"),
    (r"\bmkfs\b|\bformat\s+[a-z]:", "filesystem format"),
    (r"\bdd\s+if=", "raw disk write"),
    (r"\bshutdown\b|\breboot\b", "system shutdown/reboot"),
    (r"\bgit\s+push\b.*(--force|-f\b)", "force push"),
    (r"\bgit\s+reset\s+--hard\b", "hard reset discards user work"),
    (r"\bgit\s+clean\b.*-[a-z]*f", "git clean deletes untracked files"),
    (r"\bgit\s+checkout\s+--\s", "checkout -- discards uncommitted changes"),
    (r"\bsudo\b", "privilege escalation"),
    (r"(curl|wget|iwr|invoke-webrequest)\b.*\|\s*(sh|bash|powershell|iex)\b",
     "piping downloads into a shell"),
    (r"\breg\s+(add|delete)\b|\bregedit\b", "registry modification"),
    (r":\(\)\s*\{.*\};\s*:", "fork bomb"),
]

# Directories inside the workspace that tools may never write to.
_PROTECTED_DIRS = (".git",)


class SafetyGuard:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    # -- commands ------------------------------------------------------------

    def check_command(self, command: str) -> None:
        """Raise SafetyViolation if the command matches a destructive pattern."""
        lowered = command.lower()
        for pattern, reason in _BLOCKED_COMMANDS:
            if re.search(pattern, lowered):
                raise SafetyViolation(
                    f"Command blocked ({reason}): {command!r}. "
                    "Forge never executes destructive commands."
                )

    # -- paths ---------------------------------------------------------------

    def resolve_path(self, path: str) -> Path:
        """Resolve a path and ensure it stays inside the workspace."""
        candidate = Path(path)
        resolved = (candidate if candidate.is_absolute() else self.workspace / candidate).resolve()
        if resolved != self.workspace and self.workspace not in resolved.parents:
            raise SafetyViolation(
                f"Path {path!r} escapes the workspace {self.workspace}. "
                "Forge only touches files inside the target repository."
            )
        return resolved

    def check_write_path(self, path: str) -> Path:
        """Like resolve_path, but additionally protects VCS internals."""
        resolved = self.resolve_path(path)
        relative_parts = resolved.relative_to(self.workspace).parts
        if relative_parts and relative_parts[0] in _PROTECTED_DIRS:
            raise SafetyViolation(
                f"Writing into {relative_parts[0]}/ is not allowed (path: {path!r})."
            )
        return resolved
