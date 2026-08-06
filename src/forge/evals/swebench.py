"""SWE-bench: external calibration for the SWE-micro numbers.

SWE-micro exists because SWE-bench is uninformative at this model size — a
floor tells you nothing about which mechanism helped. But a benchmark we
wrote ourselves cannot be the only thing we report, so this runs Forge
against real SWE-bench instances and reports whatever it gets, including
zero.

The official harness imports `resource` and does not run on Windows, so the
work is split the way the benchmark itself is split:

  patch generation   locally — clone the repo at base_commit, let Forge work,
                     take `git diff` as the prediction
  evaluation         inside the official per-instance Docker image, which
                     already carries the repo and its installed environment;
                     apply the prediction and the held-out test patch, run
                     FAIL_TO_PASS and PASS_TO_PASS, and read the outcome

An instance counts as resolved only when every FAIL_TO_PASS test passes AND
every PASS_TO_PASS test still passes — the official criterion, so the numbers
mean what they mean elsewhere.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from forge import process
from forge.config import ForgeSettings

DATASET = "princeton-nlp/SWE-bench_Lite"
IMAGE = "swebench/sweb.eval.x86_64.{key}:latest"
_CLONE_TIMEOUT = 900
_EVAL_TIMEOUT = 1800
_MAX_PROBLEM_CHARS = 6_000


@dataclass
class Instance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    test_patch: str

    @property
    def image(self) -> str:
        # swebench mangles the id: django__django-11099 -> django_1776_django-11099
        return IMAGE.format(key=self.instance_id.replace("__", "_1776_").lower())


@dataclass
class InstanceResult:
    instance_id: str
    resolved: bool = False
    patch: str = ""
    error: str = ""
    duration_s: float = 0.0
    fail_to_pass_passed: int = 0
    fail_to_pass_total: int = 0
    pass_to_pass_broken: int = 0


@dataclass
class SWEBenchReport:
    results: list[InstanceResult] = field(default_factory=list)

    def resolved_rate(self) -> float:
        return (
            sum(1 for r in self.results if r.resolved) / len(self.results)
            if self.results
            else 0.0
        )

    def summary(self) -> dict:
        attempted = [r for r in self.results if r.patch.strip()]
        return {
            "dataset": DATASET,
            "instances": len(self.results),
            "resolved": sum(1 for r in self.results if r.resolved),
            "resolved_rate": round(self.resolved_rate(), 4),
            # a patch that is empty means the agent produced nothing at all,
            # which is a different failure from producing a wrong patch
            "produced_a_patch": len(attempted),
            "duration_s": round(sum(r.duration_s for r in self.results), 1),
        }


def load_instances(limit: int | None = None, ids: list[str] | None = None) -> list[Instance]:
    from datasets import load_dataset

    rows = load_dataset(DATASET, split="test")
    wanted = set(ids or [])
    out: list[Instance] = []
    for row in rows:
        if wanted and row["instance_id"] not in wanted:
            continue
        out.append(
            Instance(
                instance_id=row["instance_id"],
                repo=row["repo"],
                base_commit=row["base_commit"],
                problem_statement=row["problem_statement"],
                fail_to_pass=json.loads(row["FAIL_TO_PASS"]),
                pass_to_pass=json.loads(row["PASS_TO_PASS"]),
                test_patch=row["test_patch"],
            )
        )
        if limit and len(out) >= limit and not wanted:
            break
    return out


def prepare_workspace(instance: Instance, root: Path, cache: Path) -> Path:
    """Repository at base_commit, ready to edit.

    Clones are cached per repo — django is a quarter of a gigabyte and the
    benchmark has 114 django instances.
    """
    mirror = cache / instance.repo.replace("/", "__")
    if not mirror.exists():
        mirror.parent.mkdir(parents=True, exist_ok=True)
        process.run(
            ["git", "clone", f"https://github.com/{instance.repo}.git", str(mirror)],
            check=True, timeout=_CLONE_TIMEOUT, capture_output=True,
        )
    workspace = root / instance.instance_id
    if workspace.exists():
        process.run(["git", "-C", str(workspace), "clean", "-fdx"], capture_output=True)
        process.run(["git", "-C", str(workspace), "reset", "--hard"], capture_output=True)
    else:
        workspace.parent.mkdir(parents=True, exist_ok=True)
        process.run(
            ["git", "clone", "--shared", str(mirror), str(workspace)],
            check=True, timeout=_CLONE_TIMEOUT, capture_output=True,
        )
    process.run(
        ["git", "-C", str(workspace), "checkout", "--force", instance.base_commit],
        check=True, timeout=_CLONE_TIMEOUT, capture_output=True,
    )
    return workspace


def _request(instance: Instance) -> str:
    """The issue, framed as work rather than as a bug report to discuss."""
    problem = instance.problem_statement[:_MAX_PROBLEM_CHARS]
    return (
        "Fix this issue in the repository. Find the source file responsible, "
        "read it, and make the smallest change that fixes the described "
        "behaviour. Do not write tests — they already exist.\n\n"
        f"{problem}"
    )


def generate_patch(
    instance: Instance,
    workspace: Path,
    llm_factory,
    settings: ForgeSettings,
) -> tuple[str, str]:
    """Run Forge on the issue; return (patch, error)."""
    from forge.chat.session import ChatSession
    from forge.safety.permissions import PermissionPolicy

    try:
        session = ChatSession(
            workspace,
            llm_factory(),
            settings,
            policy=PermissionPolicy("auto"),
            session_id=f"swe-{instance.instance_id}",
        )
        session.send(_request(instance))
    except Exception as exc:  # noqa: BLE001 — a crashed run is a failed instance
        return "", f"{type(exc).__name__}: {exc}"
    diff = process.run(
        ["git", "-C", str(workspace), "diff"],
        capture_output=True, text=True, timeout=120,
    )
    return strip_test_changes(diff.stdout), ""


_FILE_HEADER = re.compile(r"^diff --git a/(\S+) b/\S+", re.MULTILINE)


def _is_test_path(path: str) -> bool:
    parts = path.split("/")
    name = parts[-1]
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
        or any(part in ("tests", "test", "testing") for part in parts[:-1])
    )


def strip_test_changes(patch: str) -> str:
    """Drop the agent's edits to test files.

    The held-out test patch is applied on top of the prediction, so a
    prediction that also touches the tests collides with it and the instance
    fails for a reason that has nothing to do with the fix. Every SWE-bench
    harness discards test changes for this reason.

    It is worth naming what this hides: the model was told the tests already
    exist and wrote one anyway, on the very first instance we ran. That is a
    real behaviour and it belongs in the discussion, not in the score.
    """
    if not patch.strip():
        return patch
    kept: list[str] = []
    keeping = False
    for line in patch.splitlines(keepends=True):
        header = _FILE_HEADER.match(line)
        if header:
            keeping = not _is_test_path(header.group(1))
        if keeping:
            kept.append(line)
    return "".join(kept)


_SUMMARY = re.compile(r"^(PASSED|FAILED|ERROR)\s+(\S+)", re.MULTILINE)


def _outcomes(output: str) -> dict[str, str]:
    """Per-test outcome from pytest's verbose report."""
    found: dict[str, str] = {}
    for status, test in _SUMMARY.findall(output):
        found[test] = status
    for line in output.splitlines():
        # "path::test PASSED" is the other shape pytest -rA emits
        parts = line.split()
        if len(parts) >= 2 and parts[-1] in ("PASSED", "FAILED", "ERROR"):
            found.setdefault(parts[0], parts[-1])
    return found


def evaluate(instance: Instance, patch: str, workdir: Path) -> InstanceResult:
    """Apply the prediction inside the official image and run the real tests."""
    result = InstanceResult(instance_id=instance.instance_id, patch=patch)
    if not patch.strip():
        result.error = "the agent produced no patch"
        return result

    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "model.patch").write_text(patch, encoding="utf-8", newline="\n")
    (workdir / "test.patch").write_text(instance.test_patch, encoding="utf-8", newline="\n")
    tests = " ".join(instance.fail_to_pass + instance.pass_to_pass)
    (workdir / "run.sh").write_text(
        "set -x\n"
        "cd /testbed\n"
        "git checkout -- .\n"
        "git apply -v /work/model.patch || exit 91\n"
        "git apply -v /work/test.patch || exit 92\n"
        f"python -m pytest -rA --no-header -q {tests}\n",
        encoding="utf-8", newline="\n",
    )
    try:
        completed = process.run(
            [
                "docker", "run", "--rm",
                "-v", f"{workdir.resolve()}:/work",
                instance.image, "bash", "/work/run.sh",
            ],
            capture_output=True, text=True, timeout=_EVAL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        result.error = "evaluation timed out"
        return result
    except OSError as exc:
        result.error = f"could not run docker: {exc}"
        return result

    output = completed.stdout + completed.stderr
    if completed.returncode == 91:
        result.error = "the patch did not apply"
        return result
    if completed.returncode == 92:
        result.error = "the held-out test patch did not apply"
        return result

    outcomes = _outcomes(output)
    result.fail_to_pass_total = len(instance.fail_to_pass)
    result.fail_to_pass_passed = sum(
        1 for test in instance.fail_to_pass if outcomes.get(test) == "PASSED"
    )
    result.pass_to_pass_broken = sum(
        1 for test in instance.pass_to_pass if outcomes.get(test) in ("FAILED", "ERROR")
    )
    # the official criterion: every FAIL_TO_PASS now passes and nothing that
    # passed before is broken
    result.resolved = (
        result.fail_to_pass_total > 0
        and result.fail_to_pass_passed == result.fail_to_pass_total
        and result.pass_to_pass_broken == 0
    )
    if not result.resolved and not result.error:
        result.error = (
            f"{result.fail_to_pass_passed}/{result.fail_to_pass_total} target tests pass"
            + (f", {result.pass_to_pass_broken} regressions" if result.pass_to_pass_broken else "")
        )
    return result


def run_swebench(
    llm_factory,
    settings: ForgeSettings,
    root: Path,
    instances: list[Instance],
    on_result=None,
) -> SWEBenchReport:
    report = SWEBenchReport()
    cache = root / "_repos"
    for instance in instances:
        started = time.monotonic()
        try:
            workspace = prepare_workspace(instance, root / "work", cache)
        except (subprocess.SubprocessError, OSError) as exc:
            outcome = InstanceResult(
                instance_id=instance.instance_id, error=f"could not prepare: {exc}"
            )
        else:
            patch, error = generate_patch(instance, workspace, llm_factory, settings)
            outcome = (
                InstanceResult(instance_id=instance.instance_id, error=error, patch=patch)
                if error
                else evaluate(instance, patch, root / "eval" / instance.instance_id)
            )
        outcome.duration_s = round(time.monotonic() - started, 1)
        report.results.append(outcome)
        if on_result is not None:
            on_result(outcome)
    return report


def write_report(report: SWEBenchReport, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": report.summary(),
        "results": [
            {
                "instance_id": r.instance_id,
                "resolved": r.resolved,
                "error": r.error,
                "duration_s": r.duration_s,
                "fail_to_pass": f"{r.fail_to_pass_passed}/{r.fail_to_pass_total}",
                "pass_to_pass_broken": r.pass_to_pass_broken,
                "patch_chars": len(r.patch),
            }
            for r in report.results
        ],
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "Instance",
    "InstanceResult",
    "SWEBenchReport",
    "evaluate",
    "generate_patch",
    "load_instances",
    "prepare_workspace",
    "run_swebench",
    "write_report",
]
