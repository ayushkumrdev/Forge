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
import builtins
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


_INFINITE = float("inf")


def _arity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, float]:
    """(minimum, maximum) positional arguments this definition accepts."""
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    minimum = len(positional) - len(args.defaults)
    maximum = _INFINITE if args.vararg is not None else float(len(positional))
    # a keyword-only argument with no default is required at every call site
    minimum += sum(1 for default in args.kw_defaults if default is None)
    return minimum, maximum


def _signatures(source: str) -> dict[str, tuple[int, float, int]]:
    """Qualified function name -> (min args, max args, line)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    found: dict[str, tuple[int, float, int]] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                minimum, maximum = _arity(child)
                found[f"{prefix}{child.name}"] = (minimum, maximum, child.lineno)
                walk(child, f"{prefix}{child.name}.")

    walk(tree, "")
    return found


def _describe(count: float) -> str:
    return "any number of" if count == _INFINITE else str(int(count))


def narrowed_signature_errors(original: str, new_content: str) -> list[str]:
    """A public function that no longer accepts the calls it used to.

    Observed live: asked to validate the address inside `register(name,
    email)`, the model rewrote it as `register(user)` — the change it was
    asked for was made, and every existing caller broke silently. Nothing
    cheaper catches this: the file parses, the imports resolve, and the
    signature is internally consistent. It is only wrong relative to what was
    there before, which is exactly what a diff-aware check can see.

    Narrowing only. Widening a signature (a new argument with a default) keeps
    every existing call valid, so it is never reported."""
    old_defs = _signatures(original)
    if not old_defs:
        return []
    new_defs = _signatures(new_content)
    problems: list[str] = []
    for qualname, (old_min, old_max, _) in old_defs.items():
        if qualname.rpartition(".")[2].startswith("_"):
            continue  # private helper: callers are all in this repo's control
        current = new_defs.get(qualname)
        if current is None:
            continue  # removed or renamed — dangling_reference_errors owns that
        new_min, new_max, line = current
        if new_max < old_max:
            problems.append(
                f"line {line}: '{qualname}()' used to accept {_describe(old_max)} "
                f"positional arguments and now accepts {_describe(new_max)}; every "
                f"existing call passing more than that breaks"
            )
        elif new_min > old_min:
            problems.append(
                f"line {line}: '{qualname}()' now requires {_describe(new_min)} "
                f"arguments where {_describe(old_min)} used to be enough; every "
                f"existing call breaks"
            )
    return problems


# calls that are known to evaluate to a real boolean
_BOOLEAN_CALLS = frozenset(
    {"bool", "isinstance", "issubclass", "callable", "hasattr", "any", "all"}
)
_BOOLEAN_METHODS = frozenset(
    {
        "startswith", "endswith", "isdigit", "isalpha", "isalnum", "isspace",
        "islower", "isupper", "istitle", "isidentifier", "isnumeric",
        "is_file", "is_dir", "exists", "issubset", "issuperset", "isdisjoint",
    }
)


def _is_boolean_expr(node: ast.expr) -> bool:
    """Whether this expression already evaluates to a real True/False.

    `a or b` is only a bug when an operand can be something other than a
    boolean. `x.startswith('{') or '"name"' in x` is correct code and must
    never be reported — a false rejection costs far more than a missed one."""
    if isinstance(node, ast.Compare):
        return True  # includes `in`, `is`, and the ordering operators
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return True
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return True
    if isinstance(node, ast.BoolOp):
        return all(_is_boolean_expr(value) for value in node.values)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id in _BOOLEAN_CALLS
        if isinstance(func, ast.Attribute):
            return func.attr in _BOOLEAN_METHODS
    return False


def inconsistent_boolean_return_errors(source: str) -> list[str]:
    """A predicate that returns `a and b` instead of a real boolean.

    Observed live: `validate_email('@b.com')` returned `''` rather than
    False, because `and` evaluates to one of its operands, not to a boolean.

    Narrow on two counts, both needed to keep it silent on correct code: the
    function must also return a literal True or False somewhere — declaring
    that it deals in booleans — and at least one operand must be something
    other than a boolean expression."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        returns = [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Return) and child.value is not None
        ]
        literal = any(
            isinstance(r.value, ast.Constant) and isinstance(r.value.value, bool)
            for r in returns
        )
        if not literal:
            continue
        problems += [
            f"line {r.lineno}: '{node.name}()' returns True or False elsewhere but "
            f"returns a bare 'and'/'or' expression here, which evaluates to one of "
            f"the operands (for example '' or None) rather than a boolean — wrap it "
            f"in bool(...)"
            for r in returns
            if isinstance(r.value, ast.BoolOp) and not _is_boolean_expr(r.value)
        ]
    return problems


def self_recursive_errors(source: str) -> list[str]:
    """A function whose whole body is a call to itself.

    Observed live: renaming `pop` to `dequeue` by hand turned
    `return self._items.pop(0)` into `return self.dequeue()`, so the method
    called itself forever. It parses, it resolves, and it destroys the
    process at the first call — a RecursionError in the hidden check was the
    only sign.

    Narrow on purpose: only a body that is exactly one statement, and that
    statement a bare call to the function itself. Real recursion has a base
    case, which means more than one statement."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = [s for s in node.body if not isinstance(s, ast.Expr | ast.Pass)] or node.body
        if len(body) != 1:
            continue
        statement = body[0]
        value = (
            statement.value
            if isinstance(statement, ast.Return | ast.Expr)
            else None
        )
        if not isinstance(value, ast.Call):
            continue
        func = value.func
        calls_itself = (isinstance(func, ast.Name) and func.id == node.name) or (
            isinstance(func, ast.Attribute)
            and func.attr == node.name
            and isinstance(func.value, ast.Name)
            and func.value.id in {"self", "cls"}
        )
        if calls_itself:
            problems.append(
                f"line {statement.lineno}: '{node.name}()' does nothing but call "
                f"itself, so it recurses forever — this is not what the original "
                f"code did"
            )
    return problems


def _bound_names(tree: ast.AST) -> set[str]:
    """Every name this module binds anywhere: definitions, imports,
    assignments, parameters, and the various statement targets."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound.add(node.name)
            args = getattr(node, "args", None)
            if args is not None:
                bound.update(
                    a.arg
                    for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]
                )
                for extra in (args.vararg, args.kwarg):
                    if extra is not None:
                        bound.add(extra.arg)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            bound.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.Global | ast.Nonlocal):
            bound.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Lambda):
            a = node.args
            bound.update(x.arg for x in [*a.posonlyargs, *a.args, *a.kwonlyargs])
    return bound


def undefined_call_errors(source: str) -> list[str]:
    """A function called by bare name that this module never defines or imports.

    Observed live: told to use `validate_email` inside `signup.register`, the
    model added the call and no import. The file parses, every import in it
    resolves, and it raises NameError the moment it runs — the module-level
    counterpart to `undefined_self_call_errors`, and the commonest shape of
    a cross-file edit that only half-lands.

    Only bare-name calls are considered, and a star import makes the module
    unanalysable, so it is skipped entirely."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    if any(
        isinstance(node, ast.ImportFrom)
        and any(alias.name == "*" for alias in node.names)
        for node in ast.walk(tree)
    ):
        return []
    known = _bound_names(tree) | set(dir(builtins))
    seen: dict[str, int] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id not in known
        ):
            seen.setdefault(node.func.id, node.lineno)
    return [
        f"line {line}: '{name}' is called here but this file never defines or "
        f"imports it"
        for name, line in sorted(seen.items(), key=lambda kv: kv[1])
    ]


def _public_definitions(source: str) -> list[str]:
    """Top-level functions and classes a module offers to importers."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        and not node.name.startswith("_")
    ]


def unexported_package_errors(init_path: Path, source: str) -> list[str]:
    """A package whose __init__.py exports nothing its modules define.

    Observed live, on both seeds of a build-from-scratch task: Forge created
    `mathkit/primes.py` with is_prime and primes_up_to, and a `__init__.py`
    containing `__all__ = []` and no imports at all. Everything parses, every
    import inside the package resolves, and `from mathkit import is_prime` —
    the only way anyone will actually use it — raises ImportError.

    Deliberately only for an __init__.py this session CREATED. A package that
    was already in the repository is entitled to be a namespace whose callers
    import submodules directly; that is a real and common style, and judging
    it would be inventing work nobody asked for."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    if imported:
        return []

    package = init_path.parent
    offered: list[str] = []
    for sibling in sorted(package.glob("*.py")):
        if sibling.name == "__init__.py":
            continue
        try:
            offered += _public_definitions(sibling.read_text(encoding="utf-8-sig"))
        except OSError:
            continue
    if not offered:
        return []
    names = ", ".join(offered[:6])
    return [
        f"line 1: '{package.name}/__init__.py' imports nothing, so "
        f"'from {package.name} import {offered[0]}' raises ImportError — "
        f"{package.name} defines {names} in its modules but the package "
        f"exports none of them"
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
