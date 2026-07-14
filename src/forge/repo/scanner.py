"""Repository intelligence, milestone 1: recursive scan, language stats, a
prompt-friendly file tree, and Python symbol extraction via the stdlib `ast`
module. Tree-sitter multi-language parsing and graph storage arrive in a later
milestone behind this same interface."""

from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel, Field

from forge.tools.search import IGNORED_DIRS

_LANGUAGE_BY_EXTENSION = {
    ".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript", ".js": "JavaScript",
    ".jsx": "JavaScript", ".go": "Go", ".rs": "Rust", ".java": "Java", ".rb": "Ruby",
    ".c": "C", ".h": "C", ".cpp": "C++", ".hpp": "C++", ".cs": "C#", ".php": "PHP",
    ".swift": "Swift", ".kt": "Kotlin", ".sh": "Shell", ".ps1": "PowerShell",
    ".sql": "SQL", ".html": "HTML", ".css": "CSS", ".scss": "CSS", ".md": "Markdown",
    ".json": "JSON", ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML",
}


class Symbol(BaseModel):
    kind: str  # "function" | "class" | "method"
    name: str
    line: int
    signature: str


class FileInfo(BaseModel):
    path: str  # forward-slash relative path
    language: str
    lines: int
    symbols: list[Symbol] = Field(default_factory=list)


class RepoSnapshot(BaseModel):
    root: str
    files: list[FileInfo]
    tree: str
    language_stats: dict[str, int]  # language -> line count

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
            parts += ["", "Symbols (Python):", *symbol_lines]
        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [summary truncated]"
        return text


class RepoScanner:
    def __init__(self, root: Path, max_tree_entries: int = 400) -> None:
        self.root = root.resolve()
        self.max_tree_entries = max_tree_entries

    def scan(self) -> RepoSnapshot:
        files: list[FileInfo] = []
        language_stats: dict[str, int] = {}
        tree_lines: list[str] = []
        self._walk(self.root, prefix="", files=files, stats=language_stats, tree=tree_lines)
        if len(tree_lines) > self.max_tree_entries:
            hidden = len(tree_lines) - self.max_tree_entries
            tree_lines = tree_lines[: self.max_tree_entries] + [f"... and {hidden} more entries"]
        return RepoSnapshot(
            root=str(self.root),
            files=files,
            tree="\n".join(tree_lines) or "(empty repository)",
            language_stats=language_stats,
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
            language = _LANGUAGE_BY_EXTENSION.get(entry.suffix.lower(), "Other")
            try:
                text = entry.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                continue
            line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
            if language != "Other":
                stats[language] = stats.get(language, 0) + line_count
            relative = str(entry.relative_to(self.root)).replace("\\", "/")
            symbols = _python_symbols(text) if language == "Python" else []
            files.append(
                FileInfo(path=relative, language=language, lines=line_count, symbols=symbols)
            )


def _python_symbols(source: str) -> list[Symbol]:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return []
    symbols: list[Symbol] = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            symbols.append(_function_symbol(node, kind="function"))
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(base) for base in node.bases)
            signature = f"{node.name}({bases})" if bases else node.name
            symbols.append(
                Symbol(kind="class", name=node.name, line=node.lineno, signature=signature)
            )
            symbols.extend(
                _function_symbol(child, kind="method", owner=node.name)
                for child in node.body
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
            )
    return symbols


def _function_symbol(
    node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str, owner: str | None = None
) -> Symbol:
    args = ", ".join(arg.arg for arg in node.args.args)
    name = f"{owner}.{node.name}" if owner else node.name
    return Symbol(kind=kind, name=name, line=node.lineno, signature=f"{name}({args})")
