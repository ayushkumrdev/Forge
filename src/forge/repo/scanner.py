"""Repository intelligence: recursive scan, language stats, a prompt-friendly
file tree, multi-language symbol extraction (ast + tree-sitter), an import
graph, and a persistent per-file cache keyed by mtime+size so re-scans only
parse what changed."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from forge.repo.graph import ImportGraph, build_import_graph
from forge.repo.symbols import Symbol, parse_file
from forge.tools.search import IGNORED_DIRS

_CACHE_VERSION = 2

_LANGUAGE_BY_EXTENSION = {
    ".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript", ".js": "JavaScript",
    ".jsx": "JavaScript", ".go": "Go", ".rs": "Rust", ".java": "Java", ".rb": "Ruby",
    ".c": "C", ".h": "C", ".cpp": "C++", ".hpp": "C++", ".cs": "C#", ".php": "PHP",
    ".swift": "Swift", ".kt": "Kotlin", ".sh": "Shell", ".ps1": "PowerShell",
    ".sql": "SQL", ".html": "HTML", ".css": "CSS", ".scss": "CSS", ".md": "Markdown",
    ".json": "JSON", ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML",
}


class FileInfo(BaseModel):
    path: str  # forward-slash relative path
    language: str
    lines: int
    symbols: list[Symbol] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)  # raw specs
    mtime_ns: int = 0
    size: int = 0


class RepoSnapshot(BaseModel):
    root: str
    files: list[FileInfo]
    tree: str
    language_stats: dict[str, int]  # language -> line count
    graph: ImportGraph = Field(default_factory=ImportGraph)

    @property
    def is_empty(self) -> bool:
        """No source files yet — a project to build rather than change."""
        return not self.files

    def file(self, path: str) -> FileInfo | None:
        return next((f for f in self.files if f.path == path), None)

    def find_symbol(self, name: str) -> list[tuple[FileInfo, Symbol]]:
        """Exact matches first, then case-insensitive substring matches."""
        needle = name.lower()
        exact: list[tuple[FileInfo, Symbol]] = []
        fuzzy: list[tuple[FileInfo, Symbol]] = []
        for file in self.files:
            for symbol in file.symbols:
                if symbol.name == name or symbol.name.endswith(f".{name}"):
                    exact.append((file, symbol))
                elif needle in symbol.name.lower():
                    fuzzy.append((file, symbol))
        return exact + fuzzy

    def summary(self, max_chars: int = 12_000) -> str:
        """Compact textual overview used as planner/coder context."""
        stats = ", ".join(
            f"{lang}: {lines} lines"
            for lang, lines in sorted(self.language_stats.items(), key=lambda kv: -kv[1])
        ) or "no source files detected"
        parts = [
            f"Repository root: {self.root}",
            f"Files: {len(self.files)} | Languages: {stats}",
            "",
            "File tree:",
            self.tree,
        ]
        symbol_lines: list[str] = []
        for file in self.files:
            if not file.symbols:
                continue
            symbol_lines.append(f"{file.path}:")
            symbol_lines.extend(
                f"  {symbol.kind} {symbol.signature}  (line {symbol.line})"
                for symbol in file.symbols
            )
        if symbol_lines:
            parts += ["", "Symbols:", *symbol_lines]
        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [summary truncated]"
        return text


class RepoScanner:
    def __init__(
        self,
        root: Path,
        max_tree_entries: int = 400,
        cache_path: Path | None = None,
    ) -> None:
        self.root = root.resolve()
        self.max_tree_entries = max_tree_entries
        self.cache_path = cache_path
        self._cache: dict[str, dict] = self._load_cache()

    def scan(self) -> RepoSnapshot:
        files: list[FileInfo] = []
        language_stats: dict[str, int] = {}
        tree_lines: list[str] = []
        self._walk(self.root, prefix="", files=files, stats=language_stats, tree=tree_lines)
        if len(tree_lines) > self.max_tree_entries:
            hidden = len(tree_lines) - self.max_tree_entries
            tree_lines = tree_lines[: self.max_tree_entries] + [f"... and {hidden} more entries"]
        # every file must be present so import-less files can still be targets
        graph = build_import_graph({f.path: f.imports for f in files})
        self._save_cache(files)
        return RepoSnapshot(
            root=str(self.root),
            files=files,
            tree="\n".join(tree_lines) or "(empty repository)",
            language_stats=language_stats,
            graph=graph,
        )

    def _walk(
        self,
        directory: Path,
        prefix: str,
        files: list[FileInfo],
        stats: dict[str, int],
        tree: list[str],
    ) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return
        for entry in entries:
            if entry.is_dir():
                if entry.name in IGNORED_DIRS:
                    continue
                tree.append(f"{prefix}{entry.name}/")
                self._walk(entry, prefix + "  ", files, stats, tree)
                continue
            tree.append(f"{prefix}{entry.name}")
            info = self._file_info(entry)
            if info is None:
                continue
            if info.language != "Other":
                stats[info.language] = stats.get(info.language, 0) + info.lines
            files.append(info)

    def _file_info(self, entry: Path) -> FileInfo | None:
        language = _LANGUAGE_BY_EXTENSION.get(entry.suffix.lower(), "Other")
        relative = str(entry.relative_to(self.root)).replace("\\", "/")
        try:
            stat = entry.stat()
        except OSError:
            return None

        cached = self._cache.get(relative)
        if cached and cached["mtime_ns"] == stat.st_mtime_ns and cached["size"] == stat.st_size:
            return FileInfo.model_validate(cached)

        try:
            text = entry.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            return None
        line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        facts = parse_file(language, text)
        return FileInfo(
            path=relative,
            language=language,
            lines=line_count,
            symbols=facts.symbols,
            imports=facts.imports,
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
        )

    # -- cache -----------------------------------------------------------------

    def _load_cache(self) -> dict[str, dict]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if data.get("version") != _CACHE_VERSION:
            return {}
        return data.get("files", {})

    def _save_cache(self, files: list[FileInfo]) -> None:
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(
                    {
                        "version": _CACHE_VERSION,
                        "files": {f.path: f.model_dump() for f in files},
                    }
                ),
                encoding="utf-8",
            )
        except OSError:  # cache is an optimization, never a failure
            pass
