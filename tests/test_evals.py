"""The evaluation harness must be trustworthy before its numbers mean
anything: metrics computed correctly from traces, hidden checks that actually
discriminate solved from unsolved, and gate kill-switches that really disable
the mechanism they name."""

import json

from forge.config import ForgeSettings
from forge.evals.metrics import (
    TrajectoryMetrics,
    aggregate,
    metrics_from_events,
    metrics_from_trace,
)
from forge.evals.runner import ABLATIONS, materialize, run_suite, run_task, score
from forge.evals.suite import SUITE, tasks
from forge.llm.base import ChatMessage, ToolCall
from forge.llm.mock import MockLLMClient

# -- metrics ---------------------------------------------------------------------


def _turn(action=True, mutated=True, ran_command=True, unverified=False):
    return {
        "kind": "turn_finished",
        "action_turn": action,
        "mutated": mutated,
        "ran_command": ran_command,
        "unverified_claim": unverified,
    }


def test_act_dont_tell_measures_action_turns_that_changed_something():
    m = metrics_from_events([_turn(mutated=True), _turn(mutated=False), _turn(mutated=True)])
    assert m.action_turns == 3
    assert m.action_turns_mutated == 2
    assert m.act_dont_tell == round(2 / 3, 4)


def test_non_action_turns_excluded_from_adt():
    m = metrics_from_events([_turn(action=False, mutated=False), _turn(mutated=True)])
    assert m.action_turns == 1
    assert m.act_dont_tell == 1.0


def test_false_verification_rate():
    events = [
        _turn(ran_command=True, unverified=False),  # honest verification
        _turn(ran_command=False, unverified=True),  # claimed without running
    ]
    m = metrics_from_events(events)
    assert m.verification_claims == 2
    assert m.unverified_claims == 1
    assert m.false_verification == 0.5


def test_unobserved_metric_is_none_not_zero():
    """A rate with no observations must not masquerade as a perfect or
    failing score — that would corrupt any average across runs."""
    m = metrics_from_events([{"kind": "llm_response", "completion_tokens": 10}])
    assert m.act_dont_tell is None
    assert m.false_verification is None
    assert m.grounded_edit is None


def test_grounded_edit_and_repair_counting():
    events = [
        {"kind": "tool_call", "tool": "edit_file", "arguments": {"path": "a.py"}},
        {"kind": "tool_result", "tool": "edit_file", "ok": True,
         "output": "Edited a.py [match:exact]."},
        {"kind": "tool_call", "tool": "edit_file", "arguments": {"path": "b.py"}},
        {"kind": "tool_result", "tool": "edit_file", "ok": True,
         "output": "Edited b.py [match:whitespace]."},
        {"kind": "tool_call", "tool": "edit_file", "arguments": {"path": "c.py"}},
        {"kind": "tool_result", "tool": "edit_file", "ok": False, "error": "not found"},
    ]
    m = metrics_from_events(events)
    assert m.edits_attempted == 3
    assert m.edits_applied == 2
    assert m.edits_repaired == 1
    assert m.grounded_edit == round(2 / 3, 4)


def test_wasted_cycle_counts_identical_repeat_calls():
    call = {"kind": "tool_call", "tool": "read_file", "arguments": {"path": "a.py"}}
    other = {"kind": "tool_call", "tool": "read_file", "arguments": {"path": "b.py"}}
    m = metrics_from_events([call, dict(call), other])
    assert m.tool_calls == 3
    assert m.repeated_calls == 1
    assert m.wasted_cycle == round(1 / 3, 4)


def test_honesty_violations_tracked_even_when_uncorrected():
    events = [
        {"kind": "honesty_violation", "gate": "pasted_code", "corrected": True},
        {"kind": "honesty_violation", "gate": "false_verification", "corrected": False},
    ]
    m = metrics_from_events(events)
    assert m.honesty_violations == 2
    assert m.violations_corrected == 1


def test_metrics_from_trace_survives_truncated_line(tmp_path):
    trace = tmp_path / "t.jsonl"
    trace.write_text(
        json.dumps(_turn()) + "\n" + '{"kind": "tool_call", "tool": "grep"' + "\n",
        encoding="utf-8",
    )
    m = metrics_from_trace(trace)
    assert m.turns == 1  # the good line still counted


def test_aggregate_reports_n_and_skips_unobserved():
    runs = [
        TrajectoryMetrics(act_dont_tell=1.0),
        TrajectoryMetrics(act_dont_tell=0.0),
        TrajectoryMetrics(act_dont_tell=None, tool_calls=5),
    ]
    agg = aggregate(runs)
    assert agg["act_dont_tell"] == {"mean": 0.5, "n": 2}
    assert agg["grounded_edit"] == {"mean": None, "n": 0}
    assert agg["totals"]["tool_calls"] == 5


# -- suite & scoring --------------------------------------------------------------


def test_suite_is_tiered_and_selectable():
    assert len(SUITE) >= 10
    # four rungs: single edit, several requirements, cross-file, repo-level
    assert {t.tier for t in SUITE} == {1, 2, 3, 4}
    # every rung needs enough tasks for a rate to mean anything
    for tier in (1, 2, 3, 4):
        assert len(tasks(tier=tier)) >= 2
    assert all(t.id for t in SUITE)
    assert len({t.id for t in SUITE}) == len(SUITE)  # ids unique
    assert all(t.tier == 1 for t in tasks(tier=1))
    assert [t.id for t in tasks(ids=["t1-fix-offbyone"])] == ["t1-fix-offbyone"]


def test_hidden_checks_fail_on_untouched_fixture(tmp_path):
    """Every task must actually be unsolved before the agent works — a task
    whose checks already pass would inflate the success rate."""
    for task in SUITE:
        workspace = materialize(task, tmp_path)
        solved, _ = score(task, workspace)
        assert not solved, f"{task.id} passes without any agent work"


def test_hidden_checks_pass_on_correct_solution(tmp_path):
    """And they must be satisfiable — verified against a real solution."""
    task = next(t for t in SUITE if t.id == "t1-fix-offbyone")
    workspace = materialize(task, tmp_path)
    (workspace / "batching.py").write_text(
        "def chunk(items, size):\n"
        "    return [items[i:i + size] for i in range(0, len(items), size)]\n",
        encoding="utf-8",
    )
    solved, output = score(task, workspace)
    assert solved, output


def test_hidden_test_file_is_never_left_behind(tmp_path):
    task = SUITE[0]
    workspace = materialize(task, tmp_path)
    score(task, workspace)
    assert not (workspace / "test_hidden_checks.py").exists()


def test_materialize_gives_a_clean_repo_each_time(tmp_path):
    task = SUITE[0]
    workspace = materialize(task, tmp_path)
    (workspace / "garbage.py").write_text("junk", encoding="utf-8")
    workspace = materialize(task, tmp_path)
    assert not (workspace / "garbage.py").exists()


# -- runner ----------------------------------------------------------------------


def test_run_task_scores_a_solving_agent(tmp_path):
    task = next(t for t in SUITE if t.id == "t1-guard-divzero")
    fixed = (
        "def average(values):\n"
        '    """Arithmetic mean of a sequence of numbers."""\n'
        "    if not values:\n"
        "        return 0.0\n"
        "    return sum(values) / len(values)\n"
    )
    llm = MockLLMClient(
        [
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(name="write_file", arguments={"path": "stats.py", "content": fixed})
                ],
            ),
            ChatMessage(role="assistant", content="Guarded the empty case in stats.py."),
        ]
    )
    result = run_task(task, lambda: llm, ForgeSettings(), tmp_path)
    assert result.solved
    assert result.metrics.act_dont_tell == 1.0  # it acted, not talked


def test_run_task_marks_a_talking_agent_unsolved(tmp_path):
    """The agent that only describes the fix fails the task AND is measured
    as a deflection — the exact behaviour the honesty gates target."""
    task = next(t for t in SUITE if t.id == "t1-guard-divzero")
    llm = MockLLMClient(
        [
            ChatMessage(
                role="assistant",
                content="You should add:\n```python\nif not values:\n    return 0.0\n```",
            ),
            ChatMessage(role="assistant", content="Just add that guard clause."),
            ChatMessage(role="assistant", content="As described above."),
        ]
    )
    result = run_task(task, lambda: llm, ForgeSettings(), tmp_path)
    assert not result.solved
    assert result.metrics.act_dont_tell == 0.0
    assert result.metrics.honesty_violations >= 1


def test_run_suite_reports_rates_and_tiers(tmp_path):
    llm_calls = []

    def factory():
        client = MockLLMClient([ChatMessage(role="assistant", content="I cannot help.")])
        llm_calls.append(client)
        return client

    report = run_suite(
        llm_factory=factory, settings=ForgeSettings(), root=tmp_path, tier=1
    )
    assert len(report.results) == 3
    assert report.task_success_rate() == 0.0
    assert set(report.by_tier()) == {1}
    assert report.summary()["config"]["ablation"] == "all-gates"
    assert len(llm_calls) == 3  # a fresh client per task, no state leakage


def test_seeds_repeat_every_task(tmp_path):
    report = run_suite(
        llm_factory=lambda: MockLLMClient([ChatMessage(role="assistant", content="no")]),
        settings=ForgeSettings(),
        root=tmp_path,
        ids=["t1-fix-offbyone"],
        seeds=3,
    )
    assert len(report.results) == 3
    assert sorted(r.seed for r in report.results) == [0, 1, 2]


def test_unknown_ablation_is_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="Unknown ablation"):
        run_suite(
            llm_factory=lambda: MockLLMClient([]),
            root=tmp_path,
            ablation="does-not-exist",
        )


def test_ablation_presets_disable_real_settings(tmp_path):
    report = run_suite(
        llm_factory=lambda: MockLLMClient([ChatMessage(role="assistant", content="no")]),
        settings=ForgeSettings(),
        root=tmp_path,
        ids=["t1-fix-offbyone"],
        ablation="no-gates",
    )
    gates = report.summary()["config"]["gates"]
    # every boolean gate is off; search_candidates is a count, not a flag
    assert not any(v for v in gates.values() if isinstance(v, bool)), gates
    assert set(ABLATIONS) >= {"all-gates", "no-gates", "no-edit-repair"}


def test_every_task_is_actually_solvable(tmp_path):
    """The guard that was missing. t2-rename-in-file scored 0/3 across many
    runs and several rounds of diagnosis before the cause turned out to be
    its own hidden check importing the stdlib `queue` instead of the
    fixture's `taskqueue` — the task had been impossible the whole time, and
    every failure read as an agent failure.

    Each task now carries a reference solution (never shown to the agent).
    Applying it must make the hidden checks pass."""
    unsolvable = []
    for task in SUITE:
        assert task.solution, f"{task.id} has no reference solution"
        workspace = materialize(task, tmp_path)
        for relative, content in task.solution.items():
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        solved, output = score(task, workspace)
        if not solved:
            unsolvable.append(f"{task.id}:\n{output[:600]}")
    assert not unsolvable, "tasks whose checks cannot be satisfied:\n" + "\n\n".join(unsolvable)


def test_empty_step_rate_counts_focused_steps_that_did_nothing():
    """The metric that would have caught a silently skipped plan-first step:
    the run that exposed it had 100% tool reliability and 0% wasted cycles
    while doing half the work it was asked for."""
    events = [
        {"kind": "focused_pass_done", "ok": True},
        {"kind": "focused_pass_done", "ok": False},
    ]
    m = metrics_from_events(events)
    assert m.focused_steps == 2
    assert m.focused_steps_empty == 1
    assert m.empty_step == 0.5


def test_empty_step_is_unobserved_without_focused_steps():
    """Unobserved must stay None, never 0 — a turn with no plan-first steps
    has not demonstrated anything about them."""
    assert metrics_from_events([{"kind": "turn_finished"}]).empty_step is None


def test_each_seed_gets_its_own_workspace_so_traces_survive(tmp_path):
    """The trace is the only artifact that says WHY a run failed, and seeds
    used to share one directory that was wiped on every run."""
    task = tasks(ids=["t1-add-function"])[0]
    seen = set()
    for seed in (0, 1, 2):
        workspace = materialize(task, tmp_path / f"seed-{seed}")
        (workspace / "trace.jsonl").write_text(f"seed {seed}\n", encoding="utf-8")
        seen.add(workspace)
    assert len(seen) == 3
    for seed in (0, 1, 2):
        trace = tmp_path / f"seed-{seed}" / task.id / "trace.jsonl"
        assert trace.read_text(encoding="utf-8").strip() == f"seed {seed}"
