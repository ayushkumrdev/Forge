"""Search tools: regex grep and filename glob across the repository, skipping
VCS internals, virtualenvs and other noise directories."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from forge.tools.base import Tool, ToolResult

IGNORED_DIRS = {
    ".git",
    ".forge",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".next",
    "coverage",
    "htmlcov",
}

_MAX_MATCHES = 200
_MAX_FILE_BYTES = 2_000_000


def iter_source_files(root: Path, glob: str | None = None):
    """Yield non-binary files under root, skipping ignored directories."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name not in IGNORED_DIRS:
                    stack.append(entry)
                continue
            relative = str(entry.relative_to(root)).replace("\\", "/")
            if glob and not (
                fnmatch.fnmatch(relative, glob) or fnmatch.fnmatch(entry.name, glob)
            ):
                continue
            yield entry, relative


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return b"\0" in handle.read(1024)
    except OSError:
        return True


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents with a regular expression. Returns "
        "path:line: match lines. Optionally filter files with a glob like *.py."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for."},
            "glob": {"type": "string", "description": "Optional filename filter, e.g. *.py."},
        },
        "required": ["pattern"],
    }

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    def run(self, pattern: str, glob: str | None = None) -> ToolResult:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult(ok=False, error=f"Invalid regex {pattern!r}: {exc}")

        matches: list[str] = []
        for path, relative in iter_source_files(self._workspace, glob):
            if path.stat().st_size > _MAX_FILE_BYTES or _is_binary(path):
                continue
            try:
                text = path.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{relative}:{line_number}: {line.strip()}")
                    if len(matches) >= _MAX_MATCHES:
                        matches.append(f"... stopped at {_MAX_MATCHES} matches")
                        return ToolResult(ok=True, output="\n".join(matches))
        if not matches:
            return ToolResult(ok=True, output=f"No matches for {pattern!r}.")
        return ToolResult(ok=True, output="\n".join(matches))


class GlobTool(Tool):
    name = "find_files"
    description = "Find files by name pattern, e.g. *.py or test_*.py or src/**/*.ts."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "glob": {"type": "string", "description": "Filename or path glob pattern."},
        },
        "required": ["glob"],
    }

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    def run(self, glob: str) -> ToolResult:
        paths = [relative for _, relative in iter_source_files(self._workspace, glob)]
        if not paths:
            return ToolResult(ok=True, output=f"No files match {glob!r}.")
        shown = paths[:_MAX_MATCHES]
        suffix = f"\n... and {len(paths) - len(shown)} more" if len(paths) > len(shown) else ""
        return ToolResult(ok=True, output="\n".join(shown) + suffix)
