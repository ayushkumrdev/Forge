"""The benchmark runner: materialize a fixture repo, let the agent work,
score it with hidden checks, and record behavioural metrics from the trace.

Scoring discipline: the hidden test module is written into the workspace only
AFTER the agent has finished and is deleted before the next run, so it can
never be read, imported, or edited by the agent under test. A task counts as
solved only when the hidden suite passes on the agent's own output."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from forge import process
from forge.chat.session import ChatSession
from forge.config import ForgeSettings
from forge.evals.metrics import TrajectoryMetrics, aggregate, metrics_from_events
from forge.evals.suite import Task, tasks
from forge.llm.base import LLMClient
from forge.safety.permissions import PermissionPolicy
from forge.telemetry import Recorder

_HIDDEN_TEST = "test_hidden_checks.py"
_CHECK_TIMEOUT_S = 120.0

# Ablation presets: a name -> the gate flags to force off.
ABLATIONS: dict[str, dict[str, bool]] = {
    "all-gates": {},
    "no-syntax": {"syntax_gate": False},
    "no-edit-repair": {"gate_edit_repair": False},
    "no-action-gate": {"gate_action": False, "gate_false_verification": False},
    "no-preflight": {"gate_preflight": False},
    "no-resolution": {"gate_resolution": False},
    "no-coverage": {"gate_coverage": False},
    "no-plan-first": {"gate_plan_first": False},
    # reasoning-first is reported to HURT agents (NL2Repo-Bench: a
    # "self-reinforcing echo chamber", 49% early-stop). It ships behind
    # this switch precisely so the claim can be tested rather than assumed.
    "no-intent-brief": {"gate_intent_brief": False},
    "no-import-check": {"gate_import_check": False},
    "no-focused-retry": {"gate_focused_retry": False},
    "search-3": {"search_candidates": 3},
    "no-gates": {
        "syntax_gate": False,
        "gate_edit_repair": False,
        "gate_action": False,
        "gate_false_verification": False,
        "gate_preflight": False,
        "gate_constrained_retry": False,
        "gate_resolution": False,
        "gate_coverage": False,
        "gate_focused_retry": False,
        "gate_intent_brief": False,
        "gate_import_check": False,
        "gate_plan_first": False,
    },
}


class TaskResult(BaseModel):
    task_id: str
    tier: int
    solved: bool
    seed: int = 0
    duration_s: float = 0.0
    check_output: str = ""
    error: str | None = None
    metrics: TrajectoryMetrics = Field(default_factory=TrajectoryMetrics)
    reply: str = ""


class SuiteReport(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    results: list[TaskResult] = Field(default_factory=list)
    duration_s: float = 0.0

    @property
    def solved(self) -> int:
        return sum(1 for r in self.results if r.solved)

    def task_success_rate(self) -> float:
        return round(self.solved / len(self.results), 4) if self.results else 0.0

    def by_tier(self) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        for tier in sorted({r.tier for r in self.results}):
            rows = [r for r in self.results if r.tier == tier]
            out[tier] = {
                "solved": sum(1 for r in rows if r.solved),
                "total": len(rows),
                "rate": round(sum(1 for r in rows if r.solved) / len(rows), 4),
            }
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "task_success_rate": self.task_success_rate(),
            "solved": self.solved,
            "total": len(self.results),
            "by_tier": self.by_tier(),
            "metrics": aggregate([r.metrics for r in self.results]),
            "duration_s": round(self.duration_s, 1),
        }


def materialize(task: Task, root: Path) -> Path:
    """Create a clean fixture repository for one task."""
    workspace = root / task.id
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    for relative, content in task.files.items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return workspace


def score(task: Task, workspace: Path) -> tuple[bool, str]:
    """Run the hidden checks against whatever the agent left behind."""
    check_path = workspace / _HIDDEN_TEST
    check_path.write_text(task.check, encoding="utf-8")
    try:
        completed = process.run(
            [sys.executable, "-m", "pytest", _HIDDEN_TEST, "-q", "--no-header", "-p",
             "no:cacheprovider"],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_CHECK_TIMEOUT_S,
        )
        output = (completed.stdout + completed.stderr).strip()
        return completed.returncode == 0, output[-2000:]
    except subprocess.TimeoutExpired:
        return False, "hidden checks timed out"
    finally:
        check_path.unlink(missing_ok=True)


def run_task(
    task: Task,
    llm_factory: Callable[[], LLMClient],
    settings: ForgeSettings,
    root: Path,
    seed: int = 0,
) -> TaskResult:
    # one workspace per seed: they used to share a directory that was wiped
    # on each run, so only the last seed's trace survived — and the trace is
    # the only thing that says WHY a run failed. Twice now a diagnosis has
    # had to wait for a whole suite to be re-run because of it.
    workspace = materialize(task, root / f"seed-{seed}")
    started = time.monotonic()
    events: list[dict[str, Any]] = []
    result = TaskResult(task_id=task.id, tier=task.tier, solved=False, seed=seed)

    try:
        recorder = Recorder(
            f"eval-{task.id}-{seed}", workspace, console=None, sink=events.append
        )
        session = ChatSession(
            workspace,
            llm_factory(),
            settings,
            policy=PermissionPolicy("auto"),  # benchmarks never block on approval
            recorder=recorder,
            session_id=f"eval-{task.id}-{seed}",
        )
        result.reply = session.send(task.request)[:2000]
    except Exception as exc:  # noqa: BLE001 — a crashed run is a failed task, not a crashed suite
        result.error = f"{type(exc).__name__}: {exc}"

    result.metrics = metrics_from_events(events)
    result.solved, result.check_output = score(task, workspace)
    result.duration_s = round(time.monotonic() - started, 1)
    return result


def run_suite(
    llm_factory: Callable[[], LLMClient],
    settings: ForgeSettings | None = None,
    root: Path | None = None,
    tier: int | None = None,
    ids: list[str] | None = None,
    seeds: int = 1,
    ablation: str = "all-gates",
    on_result: Callable[[TaskResult], None] | None = None,
) -> SuiteReport:
    """Run the suite under one configuration. `seeds` repeats every task to
    expose run-to-run variance (small models are not deterministic)."""
    base = settings or ForgeSettings()
    overrides = ABLATIONS.get(ablation)
    if overrides is None:
        raise ValueError(f"Unknown ablation {ablation!r}. Options: {', '.join(ABLATIONS)}")
    run_settings = base.model_copy(update=overrides) if overrides else base

    root = root or Path.cwd() / ".forge" / "evals"
    root.mkdir(parents=True, exist_ok=True)
    selected = tasks(tier=tier, ids=ids)

    report = SuiteReport(
        config={
            "model": run_settings.model,
            "effort": run_settings.effort,
            "ablation": ablation,
            "thinker_model": run_settings.thinker_model or None,
            "seeds": seeds,
            "gates": {
                "syntax": run_settings.syntax_gate,
                "edit_repair": run_settings.gate_edit_repair,
                "action": run_settings.gate_action,
                "false_verification": run_settings.gate_false_verification,
                "preflight": run_settings.gate_preflight,
                "constrained_retry": run_settings.gate_constrained_retry,
                "resolution": run_settings.gate_resolution,
                "coverage": run_settings.gate_coverage,
                "plan_first": run_settings.gate_plan_first,
                "search_candidates": run_settings.search_candidates,
            },
        }
    )
    started = time.monotonic()
    for seed in range(seeds):
        for task in selected:
            scoped = replace(task, id=task.id if seeds == 1 else f"{task.id}#{seed}")
            result = run_task(
                replace(scoped, id=task.id), llm_factory, run_settings, root, seed=seed
            )
            report.results.append(result)
            if on_result is not None:
                on_result(result)
    report.duration_s = time.monotonic() - started
    return report


def write_report(report: SuiteReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"summary": report.summary(), "results": [r.model_dump() for r in report.results]},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path
