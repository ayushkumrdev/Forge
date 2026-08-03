"""The verification ladder.

Every mutation climbs an ordered, cost-aware sequence of checks before it
counts as progress. Cheap rungs run first and stop the climb early, so the
common case costs microseconds and only plausible changes pay for expensive
verification:

    L1 syntax      the file parses                        ~10 ms
    L2 resolution  its imports and names resolve          ~50 ms
    L3 types       a type checker accepts it              ~1-5 s   (opt-in)

Two rules keep the ladder honest, both inherited from the syntax gate:

  * Only NEW failures block. If the file already failed a rung before the
    change, the change is not blamed for it — the agent is never trapped in
    a file it did not break.
  * A failure returns the rung's own diagnostic, not a generic refusal. The
    parser's message, the unresolved name, the type error: grounding the
    model in the specific truth is what makes it recover instead of retry.

L0 (schema) is enforced upstream by constrained decoding, and L4 (tests) /
L5 (review) run at the orchestrator level where a test subset is meaningful;
this module owns the per-file rungs.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from forge.tools.syntax_check import syntax_error
from forge.verify.resolution import resolution_errors

SYNTAX = "syntax"
RESOLUTION = "resolution"
TYPES = "types"

_TYPE_CHECK_TIMEOUT_S = 30.0


@dataclass
class RungResult:
    rung: str
    passed: bool
    diagnostic: str = ""
    skipped: bool = False
    pre_existing: bool = False  # the failure was already there before the change
    enforced: bool = True  # False => detected but allowed through (ablation)


@dataclass
class LadderVerdict:
    ok: bool
    results: list[RungResult] = field(default_factory=list)
    failed_rung: str | None = None
    diagnostic: str = ""

    @property
    def highest_passed(self) -> str | None:
        passed = [r.rung for r in self.results if r.passed and not r.skipped]
        return passed[-1] if passed else None

    @property
    def unenforced_failures(self) -> list[RungResult]:
        """Rungs that detected a real problem but were configured not to
        block — the signal an ablation needs to stay measurable."""
        return [r for r in self.results if not r.passed and not r.enforced]

    def summary(self) -> str:
        parts = []
        for r in self.results:
            mark = "skip" if r.skipped else ("ok" if r.passed else "FAIL")
            parts.append(f"{r.rung}:{mark}")
        return " ".join(parts)


class Ladder:
    """Per-file verification. `workspace` anchors repository-relative
    resolution; rungs above `max_rung` are not attempted."""

    def __init__(
        self,
        workspace: Path,
        resolution: bool = True,
        types: bool = False,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.resolution = resolution
        self.types = types

    def check(
        self, path: Path, original: str | None, new_content: str
    ) -> LadderVerdict:
        """Climb the ladder for one changed file. `original` is the content
        before the change (None for a new file) and is used to suppress
        pre-existing failures."""
        results: list[RungResult] = []

        # L1 — syntax
        error = syntax_error(path.name, new_content)
        if error is not None:
            if original is not None and syntax_error(path.name, original) is not None:
                results.append(
                    RungResult(SYNTAX, passed=True, pre_existing=True,
                               diagnostic="file was already unparseable")
                )
            else:
                results.append(RungResult(SYNTAX, passed=False, diagnostic=error))
                return LadderVerdict(False, results, SYNTAX, error)
        else:
            results.append(RungResult(SYNTAX, passed=True))

        # L2 — resolution. Always evaluated, even when enforcement is off:
        # an ablation must still measure the hallucinations it stops blocking.
        problems = resolution_errors(path, new_content, self.workspace)
        if problems:
            before = (
                resolution_errors(path, original, self.workspace)
                if original is not None
                else []
            )
            new_problems = [p for p in problems if _without_line(p) not in
                            {_without_line(b) for b in before}]
            if not new_problems:
                results.append(
                    RungResult(RESOLUTION, passed=True, pre_existing=True,
                               diagnostic="unresolved imports already present")
                )
            else:
                diagnostic = "; ".join(new_problems)
                results.append(
                    RungResult(
                        RESOLUTION, passed=False, diagnostic=diagnostic,
                        enforced=self.resolution,
                    )
                )
                if self.resolution:
                    return LadderVerdict(False, results, RESOLUTION, diagnostic)
        else:
            results.append(RungResult(RESOLUTION, passed=True))

        # L3 — types (opt-in; skipped when no checker is installed)
        if not self.types:
            results.append(RungResult(TYPES, passed=True, skipped=True))
        else:
            result = self._type_check(path, new_content)
            results.append(result)
            if not result.passed:
                return LadderVerdict(False, results, TYPES, result.diagnostic)

        return LadderVerdict(True, results)

    def _type_check(self, path: Path, new_content: str) -> RungResult:
        checker = shutil.which("pyright") or shutil.which("mypy")
        if checker is None or path.suffix != ".py":
            return RungResult(TYPES, passed=True, skipped=True)
        scratch = self.workspace / ".forge" / "typecheck"
        scratch.mkdir(parents=True, exist_ok=True)
        target = scratch / path.name
        try:
            target.write_text(new_content, encoding="utf-8")
            completed = subprocess.run(
                [checker, str(target)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_TYPE_CHECK_TIMEOUT_S,
                cwd=self.workspace,
            )
            if completed.returncode == 0:
                return RungResult(TYPES, passed=True)
            output = (completed.stdout + completed.stderr).strip()
            return RungResult(TYPES, passed=False, diagnostic=output[:800])
        except (OSError, subprocess.TimeoutExpired):
            return RungResult(TYPES, passed=True, skipped=True)
        finally:
            target.unlink(missing_ok=True)


def _without_line(problem: str) -> str:
    """Compare problems ignoring line numbers — an edit shifts lines, and a
    pre-existing bad import must still be recognised after it moves."""
    _, _, rest = problem.partition(": ")
    return rest or problem
