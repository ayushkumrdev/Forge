"""Milestone 2 tests: multi-language symbols, import graph, index cache,
and the code-intelligence tools."""

import json

from forge.repo import symbols as symbols_module
from forge.repo.graph import build_import_graph
from forge.repo.scanner import RepoScanner
from forge.repo.symbols import parse_file
from forge.tools.code_intel import FindSymbolTool, WhoImportsTool

# -- symbol extraction ---------------------------------------------------------


def test_python_symbols_and_imports():
    source = (
        "import os\n"
        "from collections import OrderedDict\n"
        "from .sibling import helper\n"
        "\n"
        "class Store:\n"
        "    def save(self, item):\n"
        "        return item\n"
        "\n"
        "def main():\n"
        "    pass\n"
    )
    facts = parse_file("Python", source)
    names = {s.name for s in facts.symbols}
    assert {"Store", "Store.save", "main"} <= names
    assert "os" in facts.imports
    assert "collections" in facts.imports
    assert ".sibling" in facts.imports


def test_typescript_symbols_via_tree_sitter():
    source = (
        "import { helper } from './util';\n"
        "export function processOrder(id: number): void {}\n"
        "export class OrderService {\n"
        "  submit(order: Order) { return order; }\n"
        "}\n"
        "export interface Order { id: number }\n"
    )
    facts = parse_file("TypeScript", source)
    names = {s.name for s in facts.symbols}
    assert "processOrder" in names
    assert "OrderService" in names
    assert "Order" in names
    assert "./util" in facts.imports


def test_go_symbols():
    source = (
        "package main\n"
        "\n"
        "type Server struct{}\n"
        "\n"
        "func (s *Server) Start() error { return nil }\n"
        "\n"
        "func main() {}\n"
    )
    facts = parse_file("Go", source)
    names = {s.name for s in facts.symbols}
    assert "main" in names
    assert "Server" in names


def test_regex_fallback_when_tree_sitter_unavailable(monkeypatch):
    monkeypatch.setattr(symbols_module, "TREE_SITTER_AVAILABLE", False)
    source = "export function fallbackFn(a) {}\nclass FallbackClass {}\n"
    facts = parse_file("JavaScript", source)
    names = {s.name for s in facts.symbols}
    assert "fallbackFn" in names
    assert "FallbackClass" in names


# -- import graph --------------------------------------------------------------


def test_python_import_graph_absolute_relative_and_src_layout():
    graph = build_import_graph(
        {
            "src/pkg/__init__.py": [],
            "src/pkg/a.py": ["pkg.b", ".c"],
            "src/pkg/b.py": ["os"],
            "src/pkg/c.py": [],
            "main.py": ["pkg.a"],
        }
    )
    assert graph.imports_of("src/pkg/a.py") == ["src/pkg/b.py", "src/pkg/c.py"]
    assert graph.imports_of("main.py") == ["src/pkg/a.py"]
    assert graph.importers_of("src/pkg/a.py") == ["main.py"]
    # external imports (os) do not create edges
    assert graph.imports_of("src/pkg/b.py") == []


def test_js_import_graph_relative_resolution():
    graph = build_import_graph(
        {
            "src/app.ts": ["./util", "../shared/types", "react"],
            "src/util.ts": [],
            "shared/types.ts": [],
        }
    )
    assert graph.imports_of("src/app.ts") == ["shared/types.ts", "src/util.ts"]
    assert graph.importers_of("src/util.ts") == ["src/app.ts"]


# -- scanner integration and cache ---------------------------------------------


def _make_repo(workspace):
    (workspace / "src" / "pkg").mkdir(parents=True)
    (workspace / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "src" / "pkg" / "core.py").write_text(
        "from pkg.helpers import util\n\ndef run():\n    pass\n", encoding="utf-8"
    )
    (workspace / "src" / "pkg" / "helpers.py").write_text(
        "def util():\n    pass\n", encoding="utf-8"
    )


def test_snapshot_has_graph_and_find_symbol(workspace):
    _make_repo(workspace)
    snapshot = RepoScanner(workspace).scan()

    matches = snapshot.find_symbol("run")
    assert matches and matches[0][0].path == "src/pkg/core.py"

    assert snapshot.graph.imports_of("src/pkg/core.py") == ["src/pkg/helpers.py"]
    assert "Symbols:" in snapshot.summary()


def test_scan_cache_skips_unchanged_files(workspace, monkeypatch):
    _make_repo(workspace)
    cache_path = workspace / ".forge" / "repo_index.json"

    RepoScanner(workspace, cache_path=cache_path).scan()
    assert cache_path.exists()
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "src/pkg/core.py" in cached["files"]

    calls = {"n": 0}
    original = symbols_module.parse_file

    def counting_parse(language, source):
        calls["n"] += 1
        return original(language, source)

    import forge.repo.scanner as scanner_module

    monkeypatch.setattr(scanner_module, "parse_file", counting_parse)
    snapshot = RepoScanner(workspace, cache_path=cache_path).scan()
    assert calls["n"] == 0  # everything served from cache
    assert snapshot.find_symbol("run")  # cached symbols still usable

    # touching a file invalidates only that entry
    (workspace / "src" / "pkg" / "core.py").write_text(
        "def run():\n    pass\n\ndef extra():\n    pass\n", encoding="utf-8"
    )
    RepoScanner(workspace, cache_path=cache_path).scan()
    assert calls["n"] == 1


# -- tools ----------------------------------------------------------------------


def test_find_symbol_tool(workspace):
    _make_repo(workspace)
    snapshot = RepoScanner(workspace).scan()
    result = FindSymbolTool(snapshot).run(name="util")
    assert result.ok
    assert "src/pkg/helpers.py:1" in result.output

    result = FindSymbolTool(snapshot).run(name="does_not_exist")
    assert result.ok
    assert "No symbol" in result.output


def test_who_imports_tool(workspace):
    _make_repo(workspace)
    snapshot = RepoScanner(workspace).scan()
    result = WhoImportsTool(snapshot).run(path="src/pkg/helpers.py")
    assert result.ok
    assert "src/pkg/core.py" in result.output

    result = WhoImportsTool(snapshot).run(path="nope.py")
    assert not result.ok
