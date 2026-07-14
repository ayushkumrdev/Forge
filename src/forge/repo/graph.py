"""Import graph: resolves the raw import specs collected during scanning into
file→file edges, entirely from in-memory data (no extra file reads).

Supported resolution:
- Python: absolute dotted modules and relative imports (leading dots), with
  awareness of src/-style layouts.
- JavaScript/TypeScript: relative specifiers ('./x', '../y') with the usual
  extension and index-file conventions.
External/third-party imports simply do not resolve and are omitted."""

from __future__ import annotations

from pathlib import PurePosixPath

from pydantic import BaseModel, Field

_PY_STRIP_PREFIXES = ("src", "lib")
_JS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


class ImportGraph(BaseModel):
    # file -> files it imports (repo-relative posix paths, sorted)
    edges: dict[str, list[str]] = Field(default_factory=dict)

    def imports_of(self, path: str) -> list[str]:
        return self.edges.get(path, [])

    def importers_of(self, path: str) -> list[str]:
        return sorted(source for source, targets in self.edges.items() if path in targets)


def build_import_graph(imports_by_file: dict[str, list[str]]) -> ImportGraph:
    """imports_by_file maps repo-relative posix path -> raw import specs."""
    known = set(imports_by_file)
    module_map = _python_module_map(known)
    edges: dict[str, list[str]] = {}
    for path, specs in imports_by_file.items():
        targets: set[str] = set()
        for spec in specs:
            if path.endswith(".py"):
                resolved = _resolve_python(path, spec, module_map)
            else:
                resolved = _resolve_js(path, spec, known)
            if resolved and resolved != path:
                targets.add(resolved)
        if targets:
            edges[path] = sorted(targets)
    return ImportGraph(edges=edges)


def _python_module_map(known_files: set[str]) -> dict[str, str]:
    """Dotted module name -> file path, for every .py file in the repo."""
    module_map: dict[str, str] = {}
    for path in known_files:
        if not path.endswith(".py"):
            continue
        parts = list(PurePosixPath(path).parts)
        parts[-1] = parts[-1][: -len(".py")]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        candidates = [parts]
        if parts and parts[0] in _PY_STRIP_PREFIXES and len(parts) > 1:
            candidates.append(parts[1:])
        for candidate in candidates:
            if candidate:
                module_map.setdefault(".".join(candidate), path)
    return module_map


def _resolve_python(path: str, spec: str, module_map: dict[str, str]) -> str | None:
    if spec.startswith("."):
        level = len(spec) - len(spec.lstrip("."))
        remainder = spec.lstrip(".")
        base_parts = list(PurePosixPath(path).parts[:-1])
        # each level beyond the first walks one package up
        base_parts = base_parts[: len(base_parts) - (level - 1)] if level > 1 else base_parts
        dotted_parts = [p for p in base_parts if p not in _PY_STRIP_PREFIXES] or base_parts
        dotted = ".".join(dotted_parts)
        spec = f"{dotted}.{remainder}" if remainder else dotted
    # try the full spec, then progressively drop trailing components
    # ("from pkg.mod import symbol" -> pkg.mod)
    parts = spec.split(".")
    while parts:
        hit = module_map.get(".".join(parts))
        if hit:
            return hit
        parts.pop()
    return None


def _resolve_js(path: str, spec: str, known_files: set[str]) -> str | None:
    if not spec.startswith("."):
        return None
    base = PurePosixPath(path).parent
    target = base
    for part in PurePosixPath(spec).parts:
        if part == ".":
            continue
        target = target.parent if part == ".." else target / part
    candidates = [str(target)]
    candidates += [str(target) + ext for ext in _JS_EXTENSIONS]
    candidates += [str(target / f"index{ext}") for ext in _JS_EXTENSIONS]
    for candidate in candidates:
        normalized = candidate.lstrip("./")
        if normalized in known_files:
            return normalized
    return None
