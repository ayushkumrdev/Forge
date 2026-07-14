"""Multi-language symbol and import extraction.

Python uses the stdlib ast (richest signal). Other languages use tree-sitter
when available and degrade to regex heuristics when it is not, so the scanner
never hard-fails on a missing native dependency."""

from __future__ import annotations

import ast
import re

from pydantic import BaseModel

try:
    from tree_sitter_language_pack import get_parser as _get_ts_parser

    TREE_SITTER_AVAILABLE = True
except Exception:  # pragma: no cover - environment without native wheels
    _get_ts_parser = None
    TREE_SITTER_AVAILABLE = False


class Symbol(BaseModel):
    kind: str  # "function" | "class" | "method"
    name: str
    line: int
    signature: str


class FileFacts(BaseModel):
    symbols: list[Symbol] = []
    imports: list[str] = []  # raw import specs, resolved later by the graph


_MAX_SYMBOLS_PER_FILE = 200

_TS_LANGUAGE_IDS = {
    "TypeScript": "typescript",
    "JavaScript": "javascript",
    "Go": "go",
    "Rust": "rust",
    "Java": "java",
    "C": "c",
    "C++": "cpp",
    "C#": "csharp",
    "Ruby": "ruby",
    "PHP": "php",
}

_FUNCTION_NODES = {
    "function_declaration",
    "function_definition",
    "function_item",
    "function",
}
_METHOD_NODES = {"method_definition", "method_declaration"}
_CLASS_NODES = {
    "class_declaration",
    "class_specifier",
    "class_definition",
    "struct_item",
    "enum_item",
    "enum_declaration",
    "trait_item",
    "interface_declaration",
    "type_spec",
}

_JS_IMPORT_RE = re.compile(
    r"""(?:import\s[^'"]*?from\s*|import\s*\(\s*|require\s*\(\s*|export\s[^'"]*?from\s*)
        ['"]([^'"]+)['"]""",
    re.VERBOSE,
)

# Regex fallback per language when tree-sitter is unavailable.
_REGEX_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "JavaScript": [
        (r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
        (r"^\s*(?:export\s+)?class\s+(\w+)", "class"),
        (r"^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(", "function"),
    ],
    "TypeScript": [
        (r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
        (r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)", "class"),
        (r"^\s*(?:export\s+)?interface\s+(\w+)", "class"),
        (r"^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(", "function"),
    ],
    "Go": [
        (r"^func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(", "function"),
        (r"^type\s+(\w+)\s+(?:struct|interface)\b", "class"),
    ],
    "Rust": [
        (r"^\s*(?:pub\s+)?fn\s+(\w+)", "function"),
        (r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)", "class"),
    ],
    "Java": [
        (
            r"^\s*(?:public|private|protected)?\s*(?:abstract\s+|final\s+)?"
            r"(?:class|interface|enum)\s+(\w+)",
            "class",
        ),
    ],
}


def parse_file(language: str, source: str) -> FileFacts:
    """Extract symbols and raw import specs for one file."""
    if language == "Python":
        return _python_facts(source)
    facts = FileFacts()
    if language in ("JavaScript", "TypeScript"):
        facts.imports = _JS_IMPORT_RE.findall(source)
    if language in _TS_LANGUAGE_IDS and TREE_SITTER_AVAILABLE:
        facts.symbols = _tree_sitter_symbols(language, source)
    elif language in _REGEX_PATTERNS:
        facts.symbols = _regex_symbols(language, source)
    return facts


# -- Python (stdlib ast) -------------------------------------------------------


def _python_facts(source: str) -> FileFacts:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return FileFacts()
    symbols: list[Symbol] = []
    imports: list[str] = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            symbols.append(_py_function(node, kind="function"))
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(base) for base in node.bases)
            signature = f"{node.name}({bases})" if bases else node.name
            symbols.append(
                Symbol(kind="class", name=node.name, line=node.lineno, signature=signature)
            )
            symbols.extend(
                _py_function(child, kind="method", owner=node.name)
                for child in node.body
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
            )
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # keep relative level as leading dots: ".utils", "..pkg.mod"
            imports.append("." * node.level + (node.module or ""))
    return FileFacts(symbols=symbols[:_MAX_SYMBOLS_PER_FILE], imports=imports)


def _py_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str, owner: str | None = None
) -> Symbol:
    args = ", ".join(arg.arg for arg in node.args.args)
    name = f"{owner}.{node.name}" if owner else node.name
    return Symbol(kind=kind, name=name, line=node.lineno, signature=f"{name}({args})")


# -- Other languages (tree-sitter) ---------------------------------------------


def _tree_sitter_symbols(language: str, source: str) -> list[Symbol]:
    try:
        parser = _get_ts_parser(_TS_LANGUAGE_IDS[language])
        tree = parser.parse(source.encode("utf-8"))
    except Exception:  # pragma: no cover - parser failure on odd input
        return _regex_symbols(language, source)

    symbols: list[Symbol] = []

    def visit(node) -> None:
        if len(symbols) >= _MAX_SYMBOLS_PER_FILE:
            return
        kind: str | None = None
        if node.type in _FUNCTION_NODES:
            kind = "function"
        elif node.type in _METHOD_NODES:
            kind = "method"
        elif node.type in _CLASS_NODES:
            kind = "class"
        if kind:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = name_node.text.decode("utf-8", errors="replace")
                params_node = node.child_by_field_name("parameters")
                params = (
                    params_node.text.decode("utf-8", errors="replace")[:60]
                    if params_node is not None
                    else ""
                )
                symbols.append(
                    Symbol(
                        kind=kind,
                        name=name,
                        line=node.start_point[0] + 1,
                        signature=f"{name}{params}" if params else name,
                    )
                )
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return symbols


def _regex_symbols(language: str, source: str) -> list[Symbol]:
    patterns = [
        (re.compile(pattern, re.MULTILINE), kind)
        for pattern, kind in _REGEX_PATTERNS.get(language, [])
    ]
    symbols: list[Symbol] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        for regex, kind in patterns:
            match = regex.match(line)
            if match:
                name = match.group(1)
                symbols.append(Symbol(kind=kind, name=name, line=line_number, signature=name))
                break
        if len(symbols) >= _MAX_SYMBOLS_PER_FILE:
            break
    return symbols
