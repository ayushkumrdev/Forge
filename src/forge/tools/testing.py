"""run_tests — find and run the project's test suite, correctly.

"Run the tests" is the commonest verification step there is, and the model
has to guess the runner. Observed live on a repo-level task: the fixture
holds pytest-style functions, the model ran `python -m unittest discover`,
unittest found 0 tests because there is no TestCase subclass, and the model
concluded the project had no tests and began writing its own — abandoning
the actual request.

Detection belongs in a tool, not in the model's memory. This one looks at
what the repository actually contains and picks accordingly, then reports
what it ran so the answer is auditable.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from forge.safety.guard import SafetyGuard
from forge.tools.base import Tool, ToolResult

_SKIP_DIRS = frozenset(
    {".git", ".forge", ".venv", "venv", "node_modules", "__pycache__", "dist", "build"}
)
_MAX_SCAN = 400
_UNITTEST_CLASS_RE = re.compile(r"^\s*class\s+\w+\((?:unittest\.)?TestCase\)", re.MULTILINE)


def python_test_files(workspace: Path) -> list[Path]:
    found: list[Path] = []
    stack = [workspace]
    scanned = 0
    while stack and scanned < _MAX_SCAN:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name in _SKIP_DIRS:
                continue
            if entry.is_dir():
                stack.append(entry)
            elif entry.suffix == ".py" and (
                entry.name.startswith("test_") or entry.name.endswith("_test.py")
            ):
                scanned += 1
                found.append(entry)
    return sorted(found)


def detect_runner(workspace: Path) -> tuple[list[str] | None, str]:
    """Return (command, why). command is None when there is nothing to run."""
    test_files = python_test_files(workspace)
    if test_files:
        # pytest runs unittest classes too, so prefer it whenever available;
        # unittest does NOT run bare pytest-style functions, which is exactly
        # how a real suite got reported as "no tests".
        try:
            import pytest  # noqa: F401

            has_pytest = True
        except ImportError:
            has_pytest = False
        if has_pytest:
            return (
                [sys.executable, "-m", "pytest", "-q", "--no-header",
                 "-p", "no:cacheprovider"],
                f"pytest ({len(test_files)} test file(s) found)",
            )
        uses_classes = any(
            _UNITTEST_CLASS_RE.search(f.read_text(encoding="utf-8-sig", errors="replace"))
            for f in test_files[:20]
        )
        if uses_classes:
            return (
                [sys.executable, "-m", "unittest", "discover"],
                "unittest (TestCase classes found, pytest not installed)",
            )
        return (
            None,
            f"{len(test_files)} test file(s) use plain test functions, which "
            "need pytest — it is not installed in this environment",
        )
    if (workspace / "package.json").is_file():
        text = (workspace / "package.json").read_text(encoding="utf-8", errors="replace")
        if '"test"' in text and shutil.which("npm"):
            return (["npm", "test"], "npm test (package.json defines a test script)")
    return None, "no test files found in this repository"


class RunTestsTool(Tool):
    name = "run_tests"
    mutating = True  # executes project code, so it is permission-gated
    description = (
        "Find and run this project's test suite, choosing the right runner "
        "automatically. Use this to verify your work instead of guessing at "
        "pytest or unittest — guessing wrong reports 'no tests' on a suite "
        "that exists."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "timeout_s": {
                "type": "number",
                "description": "Seconds before the run is killed (default 300).",
            },
        },
        "required": [],
    }

    def __init__(
        self, guard: SafetyGuard, workspace: Path, default_timeout_s: float = 300.0
    ) -> None:
        self._guard = guard
        self._workspace = workspace
        self._default_timeout_s = default_timeout_s

    def run(self, timeout_s: float | None = None) -> ToolResult:
        command, why = detect_runner(self._workspace)
        if command is None:
            return ToolResult(
                ok=True,
                output=f"No tests were run: {why}. Verify another way — for "
                "example `python -m py_compile <file>` on what you changed.",
            )
        timeout = min(timeout_s or self._default_timeout_s, 600.0)
        try:
            completed = subprocess.run(
                command,
                cwd=self._workspace,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error=f"Tests timed out after {timeout:.0f}s.")
        except (OSError, FileNotFoundError) as exc:
            return ToolResult(ok=False, error=f"Could not run the tests: {exc}")

        output = (completed.stdout + completed.stderr).strip()
        verdict = "PASSED" if completed.returncode == 0 else "FAILED"
        return ToolResult(
            ok=True,
            output=f"[{why}] tests {verdict} (exit code {completed.returncode})\n{output}",
        )
