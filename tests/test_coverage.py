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
    focused_prompt,
    looks_multi_requirement,
    rename_pair,
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
    settings = ForgeSettings(gate_focused_retry=False, gate_plan_first=False)
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
    session = ChatSession(workspace, llm, ForgeSettings(gate_plan_first=False), session_id="cov2")
    reply = session.send("set x to 2 in m.py and also add y = 3")
    assert reply == "Set x and added y."


def test_gate_can_be_ablated(workspace):
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")
    llm = MockLLMClient([
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file", arguments={"path": "m.py", "content": "x = 2\n"})]),
        ChatMessage(role="assistant", content="Half done."),
    ])
    settings = ForgeSettings(gate_coverage=False, gate_plan_first=False)
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
    settings = ForgeSettings(gate_focused_retry=False, gate_plan_first=False)
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
    settings = ForgeSettings(gate_focused_retry=False, gate_plan_first=False)
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
    session = ChatSession(workspace, llm, ForgeSettings(gate_plan_first=False), session_id="foc3")
    assert session.send("set x to 2 and also add y") == "Summary after the failed pass."


# -- decomposition quality: the bug that made search worse than no search ----------
# Observed live: "rename push to enqueue and pop to dequeue, update every use"
# was split into FIVE requirements — two renames, two overlapping "update all
# occurrences" duplicates of those renames, and "ensure that no functionality
# is broken". Candidate search then spent a full round re-doing work and a
# round chasing the platitude.


def test_meta_requirements_are_dropped():
    llm = MockLLMClient([
        ChatMessage(role="assistant", content=(
            '{"requirements": ['
            ' {"id": 1, "text": "push is renamed to enqueue"},'
            ' {"id": 2, "text": "Ensure that no functionality is broken"},'
            ' {"id": 3, "text": "Verify the change works correctly"},'
            ' {"id": 4, "text": "Test the renamed methods"}]}'
        )),
    ])
    reqs = decompose(llm, "rename push and also pop")
    assert [r.text for r in reqs] == ["push is renamed to enqueue"]


def test_real_requirements_survive_the_filter():
    llm = MockLLMClient([
        ChatMessage(role="assistant", content=(
            '{"requirements": ['
            ' {"id": 1, "text": "apply_discount(amount, percent) exists in prices.py"},'
            ' {"id": 2, "text": "checkout applies the discount to its total"}]}'
        )),
    ])
    assert len(decompose(llm, "add apply_discount and use it in checkout")) == 2


def test_decompose_prompt_forbids_overlap_and_platitudes():
    """The prompt itself carries the rules; if it is ever rewritten, these
    constraints must survive."""
    from forge.verify.coverage import _DECOMPOSE_SYSTEM

    assert "NOT OVERLAP" in _DECOMPOSE_SYSTEM
    assert "rename INCLUDES updating" in _DECOMPOSE_SYSTEM
    assert "only restates quality" in _DECOMPOSE_SYSTEM
    assert "verify it works" in _DECOMPOSE_SYSTEM


def test_multi_part_requests_without_a_conjunction_are_detected():
    """An explicit "and"/"then" is not the only shape. Observed live on
    t4-add-cli-flag: three requirements, no conjunction, so the coverage gate
    never armed and the model shipped two of the three — it added the
    uppercase logic and read args.upper without ever adding the flag to the
    parser, so every run raised AttributeError."""
    assert looks_multi_requirement(
        "Add a --upper flag to the CLI in app.py: when passed, greet() output "
        "is uppercased. Keep the default behaviour identical."
    )
    assert looks_multi_requirement(
        "Make chunk() handle an empty list. Keep the existing behaviour for "
        "non-empty input."
    )


def test_single_outcome_requests_still_skip_the_expensive_check():
    for request in [
        "fix the off-by-one in chunk()",
        "add a slugify function to strings.py",
        "delete the unused import",
        "make average() return 0.0 for an empty list",
    ]:
        assert not looks_multi_requirement(request), request


# -- evidence overrides the judge -------------------------------------------------
# Observed live on t3-wire-validator: validate_email was added correctly and
# signup.py was never touched. The coverage judge — an LLM reading the diff —
# declared the register() requirement met anyway, so no gap was reported and
# the task shipped half done.


def test_a_requirement_naming_an_untouched_file_is_unmet():
    from forge.verify.coverage import mechanically_unmet

    reqs = [
        Requirement(id=1, text="a validate_email helper is added to validators.py"),
        Requirement(id=2, text="register() in signup.py raises ValueError for a bad address"),
    ]
    assert mechanically_unmet(reqs, ["validators.py"], "") == {2}
    assert mechanically_unmet(reqs, ["validators.py", "signup.py"], "") == set()


def test_nested_paths_count_as_the_same_file():
    from forge.verify.coverage import mechanically_unmet

    reqs = [Requirement(id=1, text="signup.py rejects bad input")]
    assert mechanically_unmet(reqs, ["src/pkg/signup.py"], "") == set()


def test_requirements_naming_no_file_are_left_to_the_judge():
    from forge.verify.coverage import mechanically_unmet

    reqs = [Requirement(id=1, text="the returned total is discounted")]
    assert mechanically_unmet(reqs, [], "") == set()


def test_evidence_overrides_an_optimistic_judge():
    """The judge says met; the file was never changed. Evidence wins."""
    reqs = [Requirement(id=1, text="signup.py raises ValueError")]
    verdict = CoverageVerdict(items=[CoverageItem(id=1, met=True, reason="looks fine")])
    assert verdict.unmet(reqs) == []                      # judge alone
    assert [r.id for r in verdict.unmet(reqs, {1})] == [1]  # with evidence


def test_gate_catches_the_half_finished_cross_file_change(workspace):
    """End to end: the helper lands, the caller is never wired, and the judge
    is wrong about it. The turn must not end."""
    (workspace / "validators.py").write_text("def v(n):\n    return True\n", "utf-8")
    (workspace / "signup.py").write_text(
        "from validators import v\n\n\ndef register(u):\n    return u\n", "utf-8"
    )
    optimistic = ChatMessage(role="assistant", content=(
        '{"items": [{"id": 1, "met": true, "reason": "added"},'
        ' {"id": 2, "met": true, "reason": "looks wired"}]}'
    ))
    llm = MockLLMClient([
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="append_to_file",
            arguments={"path": "validators.py",
                       "content": "def validate_email(a):\n    return '@' in a\n"})]),
        ChatMessage(role="assistant", content="Added the helper."),
        ChatMessage(role="assistant", content=(
            '{"requirements": [{"id": 1, "text": "validate_email is added to validators.py"},'
            ' {"id": 2, "text": "register() in signup.py raises ValueError for a bad address"}]}'
        )),
        optimistic,
        # focused pass on the requirement the evidence flagged
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file",
            arguments={"path": "signup.py",
                       "content": "from validators import v, validate_email\n\n\n"
                                  "def register(u):\n"
                                  "    if not validate_email(u):\n"
                                  "        raise ValueError('invalid email')\n"
                                  "    return u\n"})]),
        ChatMessage(role="assistant", content="Wired it into register."),
        ChatMessage(role="assistant", content="Both parts done."),
    ])
    session = ChatSession(workspace, llm, ForgeSettings(gate_plan_first=False), session_id="mech")
    session.send("add validate_email to validators.py and use it in signup.py register()")
    assert "raise ValueError" in (workspace / "signup.py").read_text(encoding="utf-8")


# -- plan-first execution ---------------------------------------------------------
# A multi-part request attempted in one go is where this model comes apart:
# it does part one, botches part two, and every later step works from a
# context full of its own half-finished edits. The focused pass already
# worked — but only as REPAIR, after the damage. Plan-first runs the same
# mechanism BEFORE the mess.


def test_each_requirement_runs_in_its_own_focused_step(workspace):
    (workspace / "q.py").write_text("class Q:\n    def push(self, i):\n        pass\n", "utf-8")
    llm = MockLLMClient([
        # decomposition happens FIRST, before any attempt
        ChatMessage(role="assistant", content=(
            '{"requirements": [{"id": 1, "text": "push is renamed to enqueue"},'
            ' {"id": 2, "text": "pop is renamed to dequeue"}]}'
        )),
        # focused step 1
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file",
            arguments={"path": "q.py",
                       "content": "class Q:\n    def enqueue(self, i):\n        pass\n"})]),
        ChatMessage(role="assistant", content="Renamed push."),
        # focused step 2
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file",
            arguments={"path": "q.py",
                       "content": "class Q:\n    def enqueue(self, i):\n        pass\n\n"
                                  "    def dequeue(self):\n        pass\n"})]),
        ChatMessage(role="assistant", content="Renamed pop."),
        # the main loop then reviews and summarizes
        ChatMessage(role="assistant", content="Both renames are done."),
        ChatMessage(role="assistant", content=(
            '{"items": [{"id": 1, "met": true, "reason": "d"},'
            ' {"id": 2, "met": true, "reason": "d"}]}'
        )),
    ])
    session = ChatSession(workspace, llm, ForgeSettings(), session_id="pf")
    session.send("rename push to enqueue and rename pop to dequeue")
    text = (workspace / "q.py").read_text(encoding="utf-8")
    assert "def enqueue" in text and "def dequeue" in text
    # a pass STARTS with exactly [system, focused prompt] — nothing else in
    # front of the model, which is the entire point of the mechanism
    starts = [
        req for req in llm.requests
        if len(req) == 2 and "Do exactly this one thing" in req[-1].content
    ]
    assert len(starts) == 2  # one clean step per requirement
    assert "enqueue" in starts[0][-1].content
    assert "dequeue" in starts[1][-1].content


def test_plan_first_is_skipped_for_a_single_outcome_request(workspace):
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")
    llm = MockLLMClient([
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file", arguments={"path": "m.py", "content": "x = 2\n"})]),
        ChatMessage(role="assistant", content="Done."),
    ])
    session = ChatSession(workspace, llm, ForgeSettings(), session_id="pf2")
    assert session.send("fix the value of x in m.py") == "Done."
    assert len(llm.requests) == 2  # no decomposition, no focused steps


def test_plan_first_can_be_ablated(workspace):
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")
    llm = MockLLMClient([
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file", arguments={"path": "m.py", "content": "x = 2\ny = 3\n"})]),
        ChatMessage(role="assistant", content="Did both."),
        ChatMessage(role="assistant", content=(
            '{"requirements": [{"id": 1, "text": "x"}, {"id": 2, "text": "y"}]}')),
        ChatMessage(role="assistant", content=(
            '{"items": [{"id": 1, "met": true, "reason": "d"},'
            ' {"id": 2, "met": true, "reason": "d"}]}')),
    ])
    settings = ForgeSettings(gate_plan_first=False)
    session = ChatSession(workspace, llm, settings, session_id="pf3")
    session.send("set x to 2 and also add y")
    assert not any("Do exactly this one thing" in m.content for m in session.history)


def test_a_broken_decomposition_falls_back_to_one_attempt(workspace):
    (workspace / "m.py").write_text("x = 1\n", encoding="utf-8")
    llm = MockLLMClient(
        [ChatMessage(role="assistant", content="not json")] * 3
        + [
            ChatMessage(role="assistant", tool_calls=[ToolCall(
                name="write_file", arguments={"path": "m.py", "content": "x = 2\ny = 3\n"})]),
            ChatMessage(role="assistant", content="Did both."),
        ]
    )
    session = ChatSession(workspace, llm, ForgeSettings(), session_id="pf4")
    assert session.send("set x to 2 and also add y") == "Did both."


# -- requirement-shaped tool guidance ---------------------------------------------


def test_a_rename_requirement_is_pointed_at_rename_symbol():
    req = Requirement(id=1, text="push is renamed to enqueue, including every call to it")
    prompt = focused_prompt(req, [])
    assert "rename_symbol" in prompt
    assert '"old_name": "push"' in prompt and '"new_name": "enqueue"' in prompt
    # the old fixed guidance actively steered renames into hand-editing
    assert "make the change with append_to_file" not in prompt


def test_rename_pair_reads_both_phrasings():
    assert rename_pair("rename `push` to `enqueue`") == ("push", "enqueue")
    assert rename_pair("pop is renamed to dequeue") == ("pop", "dequeue")
    assert rename_pair("rename the pop method to dequeue") == ("pop", "dequeue")
    assert rename_pair("add a validate_email helper") is None
    assert rename_pair("keep the renamed function documented") is None


def test_non_rename_requirements_still_get_the_general_guidance():
    req = Requirement(id=1, text="validate_email is used in signup.register")
    prompt = focused_prompt(req, [])
    assert "append_to_file" in prompt and "edit_file" in prompt


def test_a_predicate_requirement_warns_about_and_returning_an_operand():
    req = Requirement(
        id=1,
        text="validate_email(address) returns True only when there is exactly one '@'",
    )
    prompt = focused_prompt(req, [])
    assert "bool(...)" in prompt


def test_a_behaviour_change_requirement_warns_against_touching_the_signature():
    req = Requirement(
        id=2,
        text="register() raises ValueError('invalid email') for a bad address",
    )
    assert "Keep the existing function signature" in focused_prompt(req, [])


def test_a_requirement_that_asks_for_a_new_argument_is_exempt():
    """The warning must not fight a signature change the user asked for."""
    req = Requirement(id=3, text="register() takes a new strict argument that validates input")
    assert "Keep the existing function signature" not in focused_prompt(req, [])
