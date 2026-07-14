"""Code-intelligence tools backed by the repository snapshot: symbol lookup
and import-graph queries. The snapshot is produced by the orchestrator at run
start, so these tools answer instantly without re-scanning."""

from __future__ import annotations

from typing import Any

from forge.repo.scanner import RepoSnapshot
from forge.tools.base import Tool, ToolResult

_MAX_RESULTS = 40


class FindSymbolTool(Tool):
    name = "find_symbol"
    description = (
        "Look up a function, class or method by name across the whole "
        "repository. Returns file, line and signature. Much faster and more "
        "precise than grep for finding definitions."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Symbol name, e.g. 'UserService' or 'greet'.",
            },
        },
        "required": ["name"],
    }

    def __init__(self, snapshot: RepoSnapshot) -> None:
        self._snapshot = snapshot

    def run(self, name: str) -> ToolResult:
        matches = self._snapshot.find_symbol(name.strip())
        if not matches:
            return ToolResult(
                ok=True, output=f"No symbol named {name!r} found in the repository index."
            )
        lines = [
            f"{file.path}:{symbol.line}  {symbol.kind} {symbol.signature}"
            for file, symbol in matches[:_MAX_RESULTS]
        ]
        if len(matches) > _MAX_RESULTS:
            lines.append(f"... and {len(matches) - _MAX_RESULTS} more matches")
        return ToolResult(ok=True, output="\n".join(lines))


class WhoImportsTool(Tool):
    name = "who_imports"
    description = (
        "Show which files import a given file, and which files it imports. "
        "Use this to understand the blast radius before changing a module."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repo-relative file path, e.g. 'src/app/utils.py'.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, snapshot: RepoSnapshot) -> None:
        self._snapshot = snapshot

    def run(self, path: str) -> ToolResult:
        normalized = path.replace("\\", "/").strip()
        if self._snapshot.file(normalized) is None:
            return ToolResult(
                ok=False,
                error=f"{normalized!r} is not in the repository index. "
                "Use find_files to locate the correct path.",
            )
        importers = self._snapshot.graph.importers_of(normalized)
        imports = self._snapshot.graph.imports_of(normalized)
        parts = [f"Files that import {normalized} ({len(importers)}):"]
        parts.extend(_listing(importers))
        parts.append(f"Files that {normalized} imports ({len(imports)}):")
        parts.extend(_listing(imports))
        return ToolResult(ok=True, output="\n".join(parts))


def _listing(paths: list[str]) -> list[str]:
    return [f"  {p}" for p in paths] if paths else ["  (none found)"]
