"""Requirement coverage — the gate that checks the REQUEST rather than the
action, targeting the partial-completion failure the benchmark exposed."""

import pytest

from forge.chat.session import ChatSession
from forge.config import ForgeSettings
from forge.llm.base import ChatMessage, ToolCall
from forge.llm.mock import MockLLMClient
from forge.verify.coverage import (
    CoverageItem,
    CoverageVerdict,
    Requirement,
    assess,
    build_evidence,
    coverage_nudge,
    decompose,
    looks_multi_requirement,
)

# -- the cheap pre-filter ---------------------------------------------------------


def test_multi_requirement_requests_are_detected():
    for request in [
        "Add validate_email to validators.py and use it in register()",
        "add a helper, then call it from checkout",
        "strip whitespace, raise on out of range, and return None for junk",
        "First add the flag. Second, document it.",
        "rename push to enqueue and update every use",
    ]:
        assert looks_multi_requirement(request), request


def test_single_requirement_requests_skip_the_check():
    """Decomposition costs an LLM call; a one-outcome request must not pay."""
    for request in [
        "fix the off-by-one in chunk()",
        "add a slugify function to strings.py",
        "make average() return 0.0 for an empty list",
        "delete the unused import",
    ]:
        assert not looks_multi_requirement(request), request


# -- evidence ---------------------------------------------------------------------


def test_evidence_reports_nothing_done():
    assert "No files were changed" in build_evidence("", [], [])


def test_evidence_carries_diff_files_and_commands():
    evidence = build_evidence("+ added line", ["a.py"], ["pytest -q"])
    assert "a.py" in evidence
    assert "$ pytest -q" in evidence
    assert "+ added line" in evidence


def test_evidence_truncates_a_huge_diff():
    evidence = build_evidence("x" * 10_000, ["a.py"], [])
    assert "[diff truncated]" in evidence
    assert len(evidence) < 6_000


# -- decompose / assess -----------------------------------------------------------


def test_decompose_returns_requirements():
    llm = MockLLMClient([
        ChatMessage(role="assistant", content=(
            '{"requirements": [{"id": 1, "text": "apply_discount exists"},'
            ' {"id": 2, "text": "checkout uses it"}]}'
        )),
    ])
    reqs = decompose(llm, "add apply_discount and use it in checkout")
    assert [r.text for r in reqs] == ["apply_discount exists", "checkout uses it"]


def test_decompose_survives_unusable_output():
    """A broken decomposition must not block the turn."""
    llm = MockLLMClient([ChatMessage(role="assistant", content="I'm not sure")] * 4)
    assert decompose(llm, "do two things") == []


def test_assess_flags_the_unmet_requirement():
    llm = MockLLMClient([
        ChatMessage(role="assistant", content=(
            '{"items": [{"id": 1, "met": true, "reason": "in diff"},'
            ' {"id": 2, "met": false, "reason": "checkout unchanged"}]}'
        )),
    ])
    reqs = [Requirement(id=1, text="helper exists"), Requirement(id=2, text="used")]
    verdict = assess(llm, reqs, "some diff")
    assert [r.id for r in verdict.unmet(reqs)] == [2]


def test_assess_failure_is_treated_as_cannot_tell_not_unmet():
    llm = MockLLMClient([ChatMessage(role="assistant", content="???")] * 4)
    reqs = [Requirement(id=1, text="x")]
    assert assess(llm, reqs, "evidence").unmet(reqs) == []


def test_no_requirements_means_no_llm_call():
    llm = MockLLMClient([])
    assert assess(llm, [], "evidence") == CoverageVerdict()
    assert llm.requests == []


def test_unmet_maps_ids_back_to_requirements():
    reqs = [Requirement(id=1, text="a"), Requirement(id=2, text="b")]
    verdict = CoverageVerdict(items=[
        CoverageItem(id=1, met=True), CoverageItem(id=2, met=False),
    ])
    assert [r.text for r in verdict.unmet(reqs)] == ["b"]


def test_nudge_names_only_the_missing_parts():
    nudge = coverage_nudge([Requirement(id=2, text="checkout must use it")])
    assert "checkout must use it" in nudge
    assert "NOT" in nudge
    assert "do not repeat work that is already done" in nudge


# -- the gate in the turn loop ----------------------------------------------------


def test_turn_cannot_end_while_a_requirement_is_missing(workspace):
    """The observed failure: part 1 done, part 2 dropped. The gate must send
    the model back rather than accept the reply."""
    (workspace / "prices.py").write_text("def checkout(items):\n    return 0\n", "utf-8")
    llm = MockLLMClient([
        # 1. does only the first half
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file",
            arguments={"path": "prices.py",
                       "content": "def apply_discount(a, p):\n    return a\n\n\n"
                                  "def checkout(items):\n    return 0\n"},
        )]),
        ChatMessage(role="assistant", content="Added apply_discount."),
        # 2. decompose
        ChatMessage(role="assistant", content=(
            '{"requirements": [{"id": 1, "text": "apply_discount exists"},'
            ' {"id": 2, "text": "checkout uses apply_discount"}]}'
        )),
        # 3. assess -> second requirement missing
        ChatMessage(role="assistant", content=(
            '{"items": [{"id": 1, "met": true, "reason": "in diff"},'
            ' {"id": 2, "met": false, "reason": "checkout untouched"}]}'
        )),
        # 4. after the nudge, finishes the job
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file",
            arguments={"path": "prices.py",
                       "content": "def apply_discount(a, p):\n    return a - a * p / 100\n\n\n"
                                  "def checkout(items):\n    return apply_discount(0, 0)\n"},
        )]),
        ChatMessage(role="assistant", content="Now checkout uses it."),
        # 5. decompose is cached; assess again -> all met
        ChatMessage(role="assistant", content=(
            '{"items": [{"id": 1, "met": true, "reason": "in diff"},'
            ' {"id": 2, "met": true, "reason": "checkout calls it"}]}'
        )),
    ])
    # this test covers the in-place nudge path; the focused path has its own
    settings = ForgeSettings(gate_focused_retry=False)
    session = ChatSession(workspace, llm, settings, session_id="cov")
    reply = session.send("add apply_discount to prices.py and use it in checkout")

    assert reply == "Now checkout uses it."
    assert "apply_discount(0, 0)" in (workspace / "prices.py").read_text(encoding="utf-8")
    assert any("checkout uses apply_discount" in m.content for m in session.history)


def test_gate_accepts_when_everything_is_covered(workspace):
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")
    llm = MockLLMClient([
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file", arguments={"path": "m.py", "content": "x = 2\ny = 3\n"})]),
        ChatMessage(role="assistant", content="Set x and added y."),
        ChatMessage(role="assistant", content=(
            '{"requirements": [{"id": 1, "text": "x is 2"}, {"id": 2, "text": "y exists"}]}'
        )),
        ChatMessage(role="assistant", content=(
            '{"items": [{"id": 1, "met": true, "reason": "diff"},'
            ' {"id": 2, "met": true, "reason": "diff"}]}'
        )),
    ])
    session = ChatSession(workspace, llm, ForgeSettings(), session_id="cov2")
    reply = session.send("set x to 2 in m.py and also add y = 3")
    assert reply == "Set x and added y."


def test_gate_can_be_ablated(workspace):
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")
    llm = MockLLMClient([
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file", arguments={"path": "m.py", "content": "x = 2\n"})]),
        ChatMessage(role="assistant", content="Half done."),
    ])
    settings = ForgeSettings(gate_coverage=False)
    session = ChatSession(workspace, llm, settings, session_id="cov3")
    assert session.send("set x to 2 and also add y") == "Half done."
    assert len(llm.requests) == 2  # no decompose, no assess


def test_fast_effort_skips_coverage(workspace):
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")
    llm = MockLLMClient([
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file", arguments={"path": "m.py", "content": "x = 2\n"})]),
        ChatMessage(role="assistant", content="Done."),
    ])
    session = ChatSession(workspace, llm, ForgeSettings(effort="fast"), session_id="cov4")
    assert session.send("set x to 2 and also add y") == "Done."
    assert len(llm.requests) == 2


def test_coverage_is_bounded(workspace):
    """A model that never finishes must still end the turn."""
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")
    write = ChatMessage(role="assistant", tool_calls=[ToolCall(
        name="write_file", arguments={"path": "m.py", "content": "x = 2\n"})])
    stubborn = ChatMessage(role="assistant", content="Half done.")
    unmet = ChatMessage(role="assistant", content=(
        '{"items": [{"id": 1, "met": true, "reason": "d"},'
        ' {"id": 2, "met": false, "reason": "missing"}]}'
    ))
    decomposition = ChatMessage(role="assistant", content=(
        '{"requirements": [{"id": 1, "text": "x is 2"}, {"id": 2, "text": "y exists"}]}'
    ))
    llm = MockLLMClient([
        write, stubborn, decomposition, unmet,
        stubborn, unmet,
        stubborn, unmet,
        stubborn,
    ])
    settings = ForgeSettings(gate_focused_retry=False)
    session = ChatSession(workspace, llm, settings, session_id="cov5")
    assert session.send("set x to 2 and also add y") == "Half done."
    gaps = [m for m in session.history if "does not yet cover" in m.content]
    assert len(gaps) == 2  # bounded at _MAX_COVERAGE_PASSES


def test_coverage_skipped_when_nothing_changed(workspace):
    """No mutation means the action gate owns the turn, not coverage."""
    llm = MockLLMClient([
        ChatMessage(role="assistant", content="I could not find the file."),
    ] * 4)
    session = ChatSession(workspace, llm, ForgeSettings(), session_id="cov6")
    session.send("add a thing and also another thing")
    assert not any("does not yet cover" in m.content for m in session.history)


@pytest.mark.parametrize("effort", ["smart", "genius"])
def test_coverage_runs_at_smart_and_genius(workspace, effort):
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")
    reply = ChatMessage(role="assistant", content="Done.")
    # genius self-briefs with the main model before acting, which consumes an
    # extra scripted response
    brief = [ChatMessage(role="assistant", content="INTENT: set x, add y.")]
    llm = MockLLMClient([
        *(brief if effort == "genius" else []),
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file", arguments={"path": "m.py", "content": "x = 2\ny = 3\n"})]),
        reply,
        ChatMessage(role="assistant", content=(
            '{"requirements": [{"id": 1, "text": "x"}, {"id": 2, "text": "y"}]}'
        )),
        ChatMessage(role="assistant", content=(
            '{"items": [{"id": 1, "met": true, "reason": "d"},'
            ' {"id": 2, "met": true, "reason": "d"}]}'
        )),
        reply, reply, reply,
    ])
    session = ChatSession(workspace, llm, ForgeSettings(effort=effort), session_id="cov7")
    session.send("set x to 2 and also add y = 3")
    # the decomposition request is identifiable by its system prompt
    systems = [m.content for req in llm.requests for m in req if m.role == "system"]
    assert any("You split a software request" in s for s in systems)


# -- focused retry: capability from attention, not model size ----------------------


def test_focused_pass_gets_a_clean_context(workspace):
    """The lever for fixed hardware: a missing requirement is re-issued as its
    OWN short task, not appended to a long polluted history. The observed
    failure was the model ignoring a correct instruction at the end of a
    twenty-message conversation."""
    (workspace / "prices.py").write_text("def checkout(i):\n    return 0\n", "utf-8")
    llm = MockLLMClient([
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file",
            arguments={"path": "prices.py", "content": "def checkout(i):\n    return 1\n"})]),
        ChatMessage(role="assistant", content="Tweaked checkout."),
        ChatMessage(role="assistant", content=(
            '{"requirements": [{"id": 1, "text": "checkout returns 1"},'
            ' {"id": 2, "text": "apply_discount exists"}]}'
        )),
        ChatMessage(role="assistant", content=(
            '{"items": [{"id": 1, "met": true, "reason": "d"},'
            ' {"id": 2, "met": false, "reason": "absent"}]}'
        )),
        # the focused pass — a fresh, single-requirement conversation
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="append_to_file",
            arguments={"path": "prices.py",
                       "content": "def apply_discount(a, p):\n    return a\n"})]),
        ChatMessage(role="assistant", content="Added it."),
        ChatMessage(role="assistant", content="Both parts are done."),
    ])
    session = ChatSession(workspace, llm, ForgeSettings(), session_id="foc")
    reply = session.send("set checkout to 1 and also add apply_discount")

    assert "apply_discount" in (workspace / "prices.py").read_text(encoding="utf-8")
    assert reply == "Both parts are done."
    # the focused request went out with a SHORT history, not the main thread
    focused = [
        req for req in llm.requests
        if any("Do exactly this one thing" in m.content for m in req)
    ]
    assert focused, "no focused pass was issued"
    assert len(focused[0]) <= 3, "focused pass must start from a clean context"


def test_focused_prompt_states_one_thing_and_protects_the_rest():
    from forge.verify.coverage import focused_prompt

    prompt = focused_prompt(
        Requirement(id=2, text="apply_discount exists"),
        [Requirement(id=1, text="checkout returns 1")],
    )
    assert "Do exactly this one thing" in prompt
    assert "apply_discount exists" in prompt
    assert "do NOT redo these" in prompt
    assert "checkout returns 1" in prompt
    assert "append_to_file" in prompt


def test_focused_retry_can_be_ablated(workspace):
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")
    llm = MockLLMClient([
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file", arguments={"path": "m.py", "content": "x = 2\n"})]),
        ChatMessage(role="assistant", content="Did one."),
        ChatMessage(role="assistant", content=(
            '{"requirements": [{"id": 1, "text": "x"}, {"id": 2, "text": "y"}]}')),
        ChatMessage(role="assistant", content=(
            '{"items": [{"id": 1, "met": true, "reason": "d"},'
            ' {"id": 2, "met": false, "reason": "no"}]}')),
        ChatMessage(role="assistant", content="Still one."),
        ChatMessage(role="assistant", content=(
            '{"items": [{"id": 1, "met": true, "reason": "d"},'
            ' {"id": 2, "met": false, "reason": "no"}]}')),
        ChatMessage(role="assistant", content="Still one."),
    ])
    settings = ForgeSettings(gate_focused_retry=False)
    session = ChatSession(workspace, llm, settings, session_id="foc2")
    session.send("set x to 2 and also add y")
    assert not any("Do exactly this one thing" in m.content for m in session.history)


def test_focused_pass_failure_does_not_break_the_turn(workspace):
    """An LLM error inside a focused pass must be swallowed."""
    class Flaky(MockLLMClient):
        def chat(self, messages, **kwargs):
            if any("Do exactly this one thing" in m.content for m in messages):
                raise RuntimeError("model died")
            return super().chat(messages, **kwargs)

    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")
    llm = Flaky([
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file", arguments={"path": "m.py", "content": "x = 2\n"})]),
        ChatMessage(role="assistant", content="Did one."),
        ChatMessage(role="assistant", content=(
            '{"requirements": [{"id": 1, "text": "x"}, {"id": 2, "text": "y"}]}')),
        ChatMessage(role="assistant", content=(
            '{"items": [{"id": 1, "met": true, "reason": "d"},'
            ' {"id": 2, "met": false, "reason": "no"}]}')),
        ChatMessage(role="assistant", content="Summary after the failed pass."),
    ])
    session = ChatSession(workspace, llm, ForgeSettings(), session_id="foc3")
    assert session.send("set x to 2 and also add y") == "Summary after the failed pass."
