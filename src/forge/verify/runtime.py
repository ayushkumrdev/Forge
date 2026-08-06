"""L4 — runtime: does the code Forge just wrote actually import?

The rungs below this one read the code. L1 proves a file parses, L2 that the
names it references exist, L3 that the types agree. All three are static, and
all three pass happily on a package that raises ImportError the moment anyone
uses it:

    mathkit/primes.py   def is_prime(n): ...
    mathkit/__init__.py __all__ = []

Every file parses. Every import inside the package resolves. `from mathkit
import is_prime` — the only way the package is ever used — fails. Observed on
both seeds of a build-from-scratch task, and no static check can see it,
because nothing in the source is wrong. The mistake is in what the source
does not do.

So this rung stops reading and runs it. Importing a module is the cheapest
possible execution: no test to write, no model judgement, no output to
interpret — either the interpreter loads it or it tells you exactly why not.

It runs in a subprocess with a timeout, because importing a module executes
its top level, and a module Forge just wrote is not code anyone has vetted.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from forge import process

_TIMEOUT_SECONDS = 20
_MAX_REPORTED = 3
# Files whose import is meaningless or actively unhelpful to run: a test
# module imports a framework, a setup script may build a package.
_SKIP_STEMS = frozenset({"setup", "conftest"})


def module_name(workspace: Path, path: Path) -> str | None:
    """Dotted module name for a file inside the workspace, or None.

    `pkg/thing.py` is `pkg.thing`, and `pkg/__init__.py` is `pkg` — importing
    the package is what exercises its exports, which is the whole point.
    """
    try:
        relative = path.resolve().relative_to(workspace.resolve())
    except ValueError:
        return None
    if relative.suffix != ".py":
        return None
    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
        if not parts:
            return None
    else:
        parts[-1] = relative.stem
    if any(not part.isidentifier() for part in parts):
        return None  # a directory that is not importable as a package
    if parts[-1].startswith("test_") or parts[-1] in _SKIP_STEMS:
        return None
    return ".".join(parts)


def _describe(module: str, stderr: str) -> str:
    """The last, most specific line of a traceback — the actual error."""
    lines = [line.strip() for line in stderr.strip().splitlines() if line.strip()]
    detail = lines[-1] if lines else "import failed"
    return f"'import {module}' fails: {detail}"


def import_errors(
    workspace: Path, paths: list[Path], timeout: float = _TIMEOUT_SECONDS
) -> list[str]:
    """Import each module in a subprocess; report the ones that fail.

    Only what Forge wrote is checked, and a module that cannot be named is
    skipped rather than guessed at. Anything that goes wrong with the
    subprocess itself is silence: a verification rung that cannot run must
    never invent a defect.
    """
    modules: list[str] = []
    for path in paths:
        if not path.is_file():
            continue  # deleted, or never written — not this rung's business
        name = module_name(workspace, path)
        if name and name not in modules:
            modules.append(name)
    problems: list[str] = []
    for module in modules:
        try:
            completed = process.run(
                [sys.executable, "-c", f"import {module}"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue  # cannot run it — say nothing rather than something wrong
        if completed.returncode != 0:
            problems.append(_describe(module, completed.stderr))
        if len(problems) >= _MAX_REPORTED:
            break
    return problems
