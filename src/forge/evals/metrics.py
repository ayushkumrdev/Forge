"""Behavioural metrics derived from an agent's event trace.

Task success rate says whether an agent worked; these say *how* it worked —
whether it acted instead of talking, whether it told the truth about
verifying, whether its edits landed on real text, and how much of its budget
it burned repeating itself. They are computed from the JSONL trace the
Recorder already writes, so any run (chat, orchestrator, benchmark) can be
scored after the fact with no extra instrumentation.

Definitions (all in [0,1], higher is better unless noted):

  ADT  act-don't-tell     action turns that actually changed something
                          / action turns
  FVR  false-verification turns claiming tests ran with no command executed
       (LOWER better)     / turns claiming verification
  GER  grounded-edit      edits that landed on provably-existing text
                          / attempted edits
  WCR  wasted-cycle       tool calls repeating an identical earlier call
       (LOWER better)     / tool calls
  TRR  tool-reliability   successful tool calls / tool calls
  HIR  hallucinated-      writes referencing a module or name that does not
       identifier         exist / write attempts
       (LOWER better)

HIR is measurable because the ladder's resolution rung already decides the
question authoritatively at write time; the metric just counts its verdicts.
It is recorded whether or not the rung is enabled to block — with
`gate_resolution=0` the write still lands, and the attempt is still counted.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_MUTATING_TOOLS = {
    "write_file",
    "edit_file",
    "delete_file",
    "run_command",
    "run_powershell",
    "git",
}


def _ratio(numerator: int, denominator: int) -> float | None:
    """None (not 0.0) when the denominator is zero — an unobserved metric must
    never be averaged in as if it were a perfect or failing score."""
    return None if denominator == 0 else round(numerator / denominator, 4)


class TrajectoryMetrics(BaseModel):
    # behavioural rates (None = not observed in this trace)
    act_dont_tell: float | None = None
    false_verification: float | None = None
    grounded_edit: float | None = None
    wasted_cycle: float | None = None
    tool_reliability: float | None = None
    hallucinated_identifier: float | None = None

    # raw counts — the evidence behind the rates
    turns: int = 0
    action_turns: int = 0
    action_turns_mutated: int = 0
    verification_claims: int = 0
    unverified_claims: int = 0
    edits_attempted: int = 0
    edits_applied: int = 0
    edits_repaired: int = 0  # landed via whitespace-tolerant repair
    tool_calls: int = 0
    tool_failures: int = 0
    repeated_calls: int = 0
    write_attempts: int = 0  # write_file + edit_file calls
    resolution_rejections: int = 0  # writes naming something that doesn't exist
    honesty_violations: int = 0
    violations_corrected: int = 0
    llm_calls: int = 0
    completion_tokens: int = 0

    def summary_line(self) -> str:
        def pct(value: float | None) -> str:
            return "n/a" if value is None else f"{value * 100:.0f}%"

        return (
            f"ADT {pct(self.act_dont_tell)} · FVR {pct(self.false_verification)} · "
            f"GER {pct(self.grounded_edit)} · WCR {pct(self.wasted_cycle)} · "
            f"tools {self.tool_calls} ({self.tool_failures} failed)"
        )


def metrics_from_events(events: Iterable[dict[str, Any]]) -> TrajectoryMetrics:
    """Score a trace. Accepts the Recorder's event dicts in emission order."""
    m = TrajectoryMetrics()
    seen_calls: set[str] = set()

    for event in events:
        kind = event.get("kind")

        if kind == "turn_finished":
            m.turns += 1
            if event.get("action_turn"):
                m.action_turns += 1
                if event.get("mutated"):
                    m.action_turns_mutated += 1
            # a verification claim is only counted when the reply asserts one;
            # `unverified_claim` marks the subset that had no command behind it
            if event.get("unverified_claim"):
                m.verification_claims += 1
                m.unverified_claims += 1
            elif event.get("ran_command"):
                m.verification_claims += 1

        elif kind == "honesty_violation":
            m.honesty_violations += 1
            if event.get("corrected"):
                m.violations_corrected += 1

        elif kind == "llm_response":
            m.llm_calls += 1
            m.completion_tokens += int(event.get("completion_tokens") or 0)

        elif kind == "tool_call":
            m.tool_calls += 1
            tool = event.get("tool", "")
            fingerprint = tool + json.dumps(event.get("arguments") or {}, sort_keys=True)
            if fingerprint in seen_calls:
                m.repeated_calls += 1
            seen_calls.add(fingerprint)
            if tool == "edit_file":
                m.edits_attempted += 1
            if tool in ("edit_file", "write_file"):
                m.write_attempts += 1

        elif kind == "tool_result":
            if not event.get("ok"):
                m.tool_failures += 1
                if "resolution check failed" in (event.get("error") or ""):
                    m.resolution_rejections += 1
            # a write that landed only because the rung was ablated still
            # counts as a hallucination — that is what keeps HIR comparable
            elif "resolution check failed" in (event.get("output") or ""):
                m.resolution_rejections += 1
            if event.get("tool") == "edit_file" and event.get("ok"):
                m.edits_applied += 1
                if "match:whitespace" in (event.get("output") or ""):
                    m.edits_repaired += 1

    m.act_dont_tell = _ratio(m.action_turns_mutated, m.action_turns)
    m.false_verification = _ratio(m.unverified_claims, m.verification_claims)
    m.grounded_edit = _ratio(m.edits_applied, m.edits_attempted)
    m.wasted_cycle = _ratio(m.repeated_calls, m.tool_calls)
    m.tool_reliability = _ratio(m.tool_calls - m.tool_failures, m.tool_calls)
    m.hallucinated_identifier = _ratio(m.resolution_rejections, m.write_attempts)
    return m


def metrics_from_trace(path: Path) -> TrajectoryMetrics:
    """Score a Recorder JSONL trace file."""
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a truncated trailing line must not lose the run
    return metrics_from_events(events)


def aggregate(runs: list[TrajectoryMetrics]) -> dict[str, Any]:
    """Mean of each rate across runs, ignoring runs where it was unobserved.

    Reported with n so a rate averaged over 2 runs is never mistaken for one
    averaged over 20."""
    out: dict[str, Any] = {}
    for field in (
        "act_dont_tell",
        "false_verification",
        "grounded_edit",
        "wasted_cycle",
        "tool_reliability",
        "hallucinated_identifier",
    ):
        values = [v for v in (getattr(r, field) for r in runs) if v is not None]
        out[field] = (
            {"mean": round(sum(values) / len(values), 4), "n": len(values)}
            if values
            else {"mean": None, "n": 0}
        )
    totals = ("tool_calls", "tool_failures", "llm_calls", "completion_tokens")
    out["totals"] = {f: sum(getattr(r, f) for r in runs) for f in totals}
    return out
