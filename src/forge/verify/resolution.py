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


# Method names that belong to builtin containers and strings. A call to one
# of these says nothing about a renamed method of the same name, and matching
# them produced a false positive on the very first real rename: after
# `pop` was renamed to `dequeue`, the body's own `self._items.pop(0)` — a
# LIST pop — was reported as a caller left behind.
_BUILTIN_METHOD_NAMES = frozenset(
    {
        "append", "extend", "insert", "remove", "pop", "clear", "index",
        "count", "sort", "reverse", "copy", "keys", "values", "items", "get",
        "setdefault", "update", "add", "discard", "union", "join", "split",
        "strip", "lstrip", "rstrip", "replace", "format", "encode", "decode",
        "startswith", "endswith", "lower", "upper", "read", "write", "close",
    }
)


def _called_names(source: str) -> dict[str, int]:
    """Names CALLED in the file, mapped to a line number.

    Only calls whose receiver is a plain name are counted — `queue.push(v)`
    yes, `self._items.pop(0)` no. A call through an attribute chain is being
    made on some other object, not on the thing that was renamed."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    calls: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            calls.setdefault(func.id, node.lineno)
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.attr not in _BUILTIN_METHOD_NAMES
        ):
            calls.setdefault(func.attr, node.lineno)
    return calls


_SELFLESS_OK = frozenset({"staticmethod", "classmethod"})


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = set()
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def broken_method_signature_errors(source: str) -> list[str]:
    """A method that lost its `self`.

    Observed live: renaming `push` to `enqueue`, the model rewrote
    `def push(self, item)` as `def enqueue(item)` and dropped the receiver.
    The file parses, every name resolves, the method exists — and every call
    on an instance raises TypeError. Nothing else catches it.

    Only plain methods are judged: staticmethod and classmethod legitimately
    have no `self`, and a nested function inside a method is not a method."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    problems: list[str] = []
    for klass in ast.walk(tree):
        if not isinstance(klass, ast.ClassDef):
            continue
        for item in klass.body:  # body, not walk: nested defs are not methods
            if not isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if _decorator_names(item) & _SELFLESS_OK:
                continue
            args = item.args
            first = (args.posonlyargs + args.args)[:1]
            if not first:
                problems.append(
                    f"line {item.lineno}: method '{klass.name}.{item.name}' takes no "
                    "arguments at all — it is missing 'self'"
                )
            elif first[0].arg not in ("self", "cls"):
                problems.append(
                    f"line {item.lineno}: method '{klass.name}.{item.name}' has "
                    f"'{first[0].arg}' as its first parameter, not 'self' — "
                    "calling it on an instance will fail"
                )
    return problems


def undefined_self_call_errors(source: str) -> list[str]:
    """`self.foo()` where the class has no `foo`.

    The mirror of a missed caller, and just as invisible to every other
    check. Observed live: renaming `pop` to `dequeue`, the model's edit
    matched `self.pop()` (the call) instead of `def pop` (the definition),
    leaving the method defined under its old name and the call pointing at a
    method that never existed. Valid Python, resolvable imports, AttributeError
    at runtime.

    Conservative: only classes with no base class are judged, since a base
    could supply the method, and only calls are checked, not attribute reads.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.bases:
            continue
        defined = {
            item.name
            for item in ast.walk(node)
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        # attributes assigned on self are legitimate callees too (callbacks)
        defined |= {
            target.attr
            for item in ast.walk(node)
            if isinstance(item, ast.Assign)
            for target in item.targets
            if isinstance(target, ast.Attribute)
        }
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
                and func.attr not in defined
            ):
                near = _closest(func.attr, defined)
                hint = f" (did you mean '{near}'?)" if near else ""
                problems.append(
                    f"line {call.lineno}: self.{func.attr}() is called but "
                    f"'{node.name}' does not define {func.attr}{hint}"
                )
    return problems


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
