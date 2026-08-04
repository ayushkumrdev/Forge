"""Syntax gate — the strongest grounding signal available before running code.

Every write/edit is parsed BEFORE it touches disk: Python with the stdlib ast
(compiler-grade messages), JSON/TOML with the stdlib parsers, and other
languages with tree-sitter when available. A change that would introduce a
syntax error is refused with the parser's diagnosis, so the repository never
enters a state the model believes is fine but the compiler rejects.

One rule keeps the gate honest: it only blocks NEW errors. If the file was
already unparseable before the change (templates, fixtures, mid-refactor
files), the gate stays open — Forge never traps the model in a file it did
not break."""

from __future__ import annotations

import json
import tomllib

try:
    from tree_sitter_language_pack import get_parser as _get_ts_parser

    TREE_SITTER_AVAILABLE = True
except Exception:  # pragma: no cover - environment without native wheels
    _get_ts_parser = None
    TREE_SITTER_AVAILABLE = False

_PYTHON_EXTS = (".py", ".pyw")

_TS_BY_EXT = {
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
}


def syntax_error(filename: str, content: str) -> str | None:
    """Parse `content` as the language `filename` implies. Returns a
    human-readable error, or None when it parses or the language is unknown."""
    lowered = filename.lower()
    dot = lowered.rfind(".")
    ext = lowered[dot:] if dot != -1 else ""

    if ext in _PYTHON_EXTS:
        return _python_error(content)
    if ext == ".json":
        return _json_error(content)
    if ext == ".toml":
        return _toml_error(content)
    language = _TS_BY_EXT.get(ext)
    if language and TREE_SITTER_AVAILABLE:
        return _tree_sitter_error(language, content)
    return None


def gate_edit(filename: str, original: str | None, new_content: str) -> str | None:
    """The gate: an error message when `new_content` introduces a syntax error
    the `original` did not have; None means the change may be applied."""
    error = syntax_error(filename, new_content)
    if error is None:
        return None
    if original is not None and syntax_error(filename, original) is not None:
        return None  # the file was already broken; never trap the model
    return error


def _python_error(content: str) -> str | None:
    # compile(), not ast.parse(): the parser accepts `return`, `break` and
    # `yield` outside their enclosing block — those are rejected later, when
    # the symbol table is built. ast.parse alone let a broken file through
    # that Python itself refuses to run, which is exactly what happened when
    # a multi-line edit lost its indentation and dedented a return statement
    # out of its function.
    try:
        compile(content, "<forge>", "exec", dont_inherit=True)
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno else "unknown line"
        snippet = f": {exc.text.strip()!r}" if exc.text and exc.text.strip() else ""
        return f"Python {exc.msg} at {location}{snippet}"
    return None


def _json_error(content: str) -> str | None:
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        return f"JSON invalid: {exc.msg} at line {exc.lineno} column {exc.colno}"
    return None


def _toml_error(content: str) -> str | None:
    try:
        tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        return f"TOML invalid: {exc}"
    return None


def _tree_sitter_error(language: str, content: str) -> str | None:
    try:
        parser = _get_ts_parser(language)
        tree = parser.parse(content.encode("utf-8"))
    except Exception:  # pragma: no cover - parser failure on odd input
        return None  # can't judge -> never block
    root = tree.root_node
    if not root.has_error:
        return None
    line = _first_error_line(root) or root.start_point[0] + 1
    return f"{language} syntax error near line {line}"


def _first_error_line(node) -> int | None:
    if node.type == "ERROR" or node.is_missing:
        return node.start_point[0] + 1
    if not node.has_error:
        return None
    for child in node.children:
        line = _first_error_line(child)
        if line is not None:
            return line
    return node.start_point[0] + 1
