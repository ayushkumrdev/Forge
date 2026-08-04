"""L2 — resolution: do the names this change references actually exist?

The syntax gate proves a file parses; it says nothing about whether
`from utils import helper` refers to anything real. Inventing plausible
imports is the single most characteristic hallucination of a code model, and
it survives every cheaper check: the file parses, the diff looks clean, and
the failure only appears at import time.

This rung resolves every import in the changed file against three sources of
truth — the standard library, installed distributions, and the repository
itself — and for repository modules it goes further: a `from module import
name` must name something that module actually defines.

Python only for now. The ladder is language-parametric by design; other
languages simply skip this rung until an analyser exists for them."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path


@dataclass(frozen=True)
class ImportRef:
    module: str  # dotted module, "" for `from . import x`
    names: tuple[str, ...]  # imported names; empty for plain `import x`
    level: int = 0  # 0 absolute, >0 relative dots
    line: int = 0


def collect_imports(tree: ast.AST) -> list[ImportRef]:
    refs: list[ImportRef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                refs.append(ImportRef(module=alias.name, names=(), line=node.lineno))
        elif isinstance(node, ast.ImportFrom):
            refs.append(
                ImportRef(
                    module=node.module or "",
                    names=tuple(a.name for a in node.names),
                    level=node.level,
                    line=node.lineno,
                )
            )
    return refs


def public_names(source: str) -> set[str]:
    """Top-level names a module provides: defs, classes, assignments, and
    whatever it re-exports through its own imports."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


@lru_cache(maxsize=512)
def _is_available(module: str) -> bool:
    """Standard library or an installed distribution."""
    root = module.split(".")[0]
    if root in sys.stdlib_module_names or root in sys.builtin_module_names:
        return True
    try:
        return find_spec(root) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


class RepoIndex:
    """Maps dotted module paths to files inside the workspace, honouring both
    flat and src/ layouts."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self._roots = [workspace]
        src = workspace / "src"
        if src.is_dir():
            self._roots.append(src)

    def find(self, module: str) -> Path | None:
        if not module:
            return None
        parts = module.split(".")
        for root in self._roots:
            base = root.joinpath(*parts)
            for candidate in (base.with_suffix(".py"), base / "__init__.py"):
                if candidate.is_file():
                    return candidate
            if base.is_dir():
                return base  # namespace package
        return None

    def resolve_relative(self, current: Path, level: int, module: str) -> Path | None:
        anchor = current.parent
        for _ in range(level - 1):
            anchor = anchor.parent
        if not module:
            return anchor if anchor.is_dir() else None
        base = anchor.joinpath(*module.split("."))
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file():
                return candidate
        return base if base.is_dir() else None


def _defined_names(source: str) -> set[str]:
    """Every function/class name defined anywhere in the file, including
    methods — a rename has to update calls to those too."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }


def _called_names(source: str) -> dict[str, int]:
    """Names that are CALLED in the file, mapped to their line number."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    calls: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute)
            else None
        )
        if name is not None:
            calls.setdefault(name, node.lineno)
    return calls


def dangling_reference_errors(original: str, new_content: str) -> list[str]:
    """A rename that missed a caller.

    When a change removes a definition, nothing in the same file may still
    call it. Observed live: the model renamed `push` to `enqueue`, updated
    the call inside the class, and left `queue.push(value)` in a module-level
    helper — valid Python, resolvable imports, and broken at runtime. This is
    decided mechanically from the AST, with no model judgement involved."""
    removed = _defined_names(original) - _defined_names(new_content)
    if not removed:
        return []
    still_called = _called_names(new_content)
    return [
        f"line {still_called[name]}: '{name}' is still called here, but its "
        f"definition was removed or renamed in this change"
        for name in sorted(removed)
        if name in still_called
    ]


def resolution_errors(
    file_path: Path, source: str, workspace: Path, max_reported: int = 4
) -> list[str]:
    """Every import in `source` that cannot be resolved, described so the
    model can act on it. Empty list means the rung passes.

    Deliberately conservative: anything it cannot judge with confidence is
    allowed through. A false rejection would block correct work, which costs
    far more than letting a rare bad import reach the next rung."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # not this rung's job — L1 reports syntax
    index = RepoIndex(workspace)
    problems: list[str] = []

    for ref in collect_imports(tree):
        if ref.level > 0:
            target = index.resolve_relative(file_path, ref.level, ref.module)
            if target is None:
                dots = "." * ref.level
                problems.append(
                    f"line {ref.line}: relative import '{dots}{ref.module}' does not "
                    f"resolve to a module in this repository"
                )
                continue
            problems.extend(_check_names(ref, target, index))
            continue

        if _is_available(ref.module):
            continue
        target = index.find(ref.module)
        if target is None:
            problems.append(
                f"line {ref.line}: module '{ref.module}' is not installed and does "
                f"not exist in this repository"
            )
            continue
        problems.extend(_check_names(ref, target, index))

    return problems[:max_reported]


def _check_names(ref: ImportRef, target: Path, index: RepoIndex) -> list[str]:
    """For `from <repo module> import a, b`, each name must exist there."""
    if not ref.names or not target.is_file():
        return []
    if "*" in ref.names:
        return []
    try:
        provided = public_names(target.read_text(encoding="utf-8-sig", errors="replace"))
    except OSError:
        return []
    if not provided:
        return []  # unparseable target — don't guess
    problems = []
    for name in ref.names:
        # a submodule import (`from pkg import mod`) is valid too
        if name in provided or index.find(f"{ref.module}.{name}") is not None:
            continue
        near = _closest(name, provided)
        hint = f" (did you mean '{near}'?)" if near else ""
        problems.append(
            f"line {ref.line}: '{ref.module}' does not define '{name}'{hint}"
        )
    return problems


def _closest(name: str, options: set[str]) -> str | None:
    import difflib

    matches = difflib.get_close_matches(name, sorted(options), n=1, cutoff=0.75)
    return matches[0] if matches else None
