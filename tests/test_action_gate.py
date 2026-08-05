"""Act-don't-tell enforcement: on a turn that asks for a change, a reply that
pastes code or promises future action — while no file was actually touched —
is bounced back with a corrective nudge instead of being accepted."""

from forge.chat.session import ChatSession
from forge.llm.base import ChatMessage, ToolCall
from forge.llm.mock import MockLLMClient
from forge.safety.guard import SafetyGuard
from forge.tools.terminal import PowerShellTool


def _session(workspace, llm) -> ChatSession:
    return ChatSession(workspace, llm, session_id="gate-test")


PASTE_REPLY = (
    "Sure! Add this function to app.py:\n\n"
    "```python\ndef greet(name):\n    return f'hello {name}'\n```\n"
    "That should do it."
)


def test_pasted_code_is_bounced_until_the_model_acts(workspace):
    llm = MockLLMClient(
        [
            ChatMessage(role="assistant", content=PASTE_REPLY),  # deflection
            ChatMessage(  # after the nudge: actually writes the file
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="write_file",
                        arguments={
                            "path": "app.py",
                            "content": "def greet(name):\n    return f'hello {name}'\n",
                        },
                    )
                ],
            ),
            ChatMessage(role="assistant", content="Added greet() to app.py."),
        ]
    )
    session = _session(workspace, llm)
    reply = session.send("add a greet function to app.py")

    assert (workspace / "app.py").exists()  # the change actually happened
    assert reply == "Added greet() to app.py."
    # the corrective nudge was injected between the attempts
    nudges = [m for m in session.history if "pasted code" in m.content]
    assert len(nudges) == 1


def test_promise_reply_is_bounced(workspace):
    llm = MockLLMClient(
        [
            ChatMessage(
                role="assistant", content="I will now create the config file for you."
            ),
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="write_file",
                        arguments={"path": "config.toml", "content": "debug = false\n"},
                    )
                ],
            ),
            ChatMessage(role="assistant", content="Created config.toml."),
        ]
    )
    session = _session(workspace, llm)
    reply = session.send("create a config file with debug off")
    assert (workspace / "config.toml").exists()
    assert reply == "Created config.toml."


def test_explanations_are_never_bounced(workspace):
    """A non-action question answered with a code example is legitimate."""
    example = "Decorators wrap functions:\n```python\n@wraps(f)\ndef inner(): ...\n```"
    llm = MockLLMClient([ChatMessage(role="assistant", content=example)])
    session = _session(workspace, llm)
    reply = session.send("how do python decorators work?")
    assert len(llm.requests) == 1  # accepted first time: no bounce
    assert "Decorators wrap functions:" in reply
    assert "@wraps(f)" in reply
    # the formatter separates the fence from the prose above it
    assert "functions:\n\n```python" in reply


def test_fences_fine_after_a_real_change(workspace):
    """Once a mutating tool succeeded, a summary containing code is accepted."""
    llm = MockLLMClient(
        [
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="write_file",
                        arguments={"path": "util.py", "content": "x = 1\n"},
                    )
                ],
            ),
            ChatMessage(
                role="assistant",
                content="Added util.py:\n```python\nx = 1\n```\nVerified.",
            ),
        ]
    )
    session = _session(workspace, llm)
    reply = session.send("add a util module")
    assert "Verified." in reply
    assert len(llm.requests) == 2  # no bounce


def test_gate_gives_up_after_budget(workspace):
    """Two nudges maximum — a stubborn reply is eventually surfaced to the
    user rather than looping forever."""
    stubborn = ChatMessage(role="assistant", content=PASTE_REPLY)
    llm = MockLLMClient([stubborn, stubborn, stubborn])
    session = _session(workspace, llm)
    reply = session.send("add a greet function to app.py")
    assert "greet" in reply  # final stubborn reply was accepted
    assert len(llm.requests) == 3  # initial + 2 nudged retries


def test_false_verification_claim_is_bounced(workspace):
    """Claiming 'tests passed' without having run any command is the purest
    hallucination — the gate forces a real run (or an honest report)."""
    llm = MockLLMClient(
        [
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="write_file",
                        arguments={"path": "m.py", "content": "x = 1\n"},
                    )
                ],
            ),
            ChatMessage(
                role="assistant",
                content="Added m.py. Ran the tests and all tests passed.",  # lie
            ),
            ChatMessage(
                role="assistant",
                tool_calls=[ToolCall(name="run_command", arguments={"command": "echo checked"})],
            ),
            ChatMessage(role="assistant", content="Added m.py; echo check exit code 0."),
        ]
    )
    session = _session(workspace, llm)
    reply = session.send("add a module m.py")
    assert reply == "Added m.py; echo check exit code 0."
    assert any("did NOT run any command" in m.content for m in session.history)


def test_honest_verified_reply_passes(workspace):
    """A verification claim after a REAL command run is accepted."""
    llm = MockLLMClient(
        [
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="write_file",
                        arguments={"path": "n.py", "content": "y = 2\n"},
                    )
                ],
            ),
            ChatMessage(
                role="assistant",
                tool_calls=[ToolCall(name="run_command", arguments={"command": "echo ok"})],
            ),
            ChatMessage(role="assistant", content="Added n.py; ran the checks, all passed."),
        ]
    )
    session = _session(workspace, llm)
    reply = session.send("add a module n.py")
    assert reply == "Added n.py; ran the checks, all passed."
    assert len(llm.requests) == 3  # no bounce


def test_template_tags_stripped_from_reply(workspace):
    llm = MockLLMClient(
        [
            ChatMessage(
                role="assistant",
                content="<tool_response>\nAll done here.\n</tool_response>",
            )
        ]
    )
    session = _session(workspace, llm)
    assert session.send("hi there") == "All done here."


def test_running_a_command_is_not_acting(workspace):
    """Measurement validity: a turn that only ran a command has not done the
    work. run_command is 'mutating' for permission purposes, but counting it
    as action inflated ADT and stopped the gate firing on a model that just
    re-ran the tests. (Observed live: 4x `unittest discover`, zero writes,
    turn recorded as mutated.)"""
    llm = MockLLMClient(
        [
            ChatMessage(
                role="assistant",
                tool_calls=[ToolCall(name="run_command", arguments={"command": "echo hi"})],
            ),
            ChatMessage(role="assistant", content=PASTE_REPLY),  # still deflecting
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="write_file",
                        arguments={"path": "app.py", "content": "def greet(n):\n    return n\n"},
                    )
                ],
            ),
            ChatMessage(role="assistant", content="Wrote greet() to app.py."),
        ]
    )
    session = _session(workspace, llm)
    reply = session.send("add a greet function to app.py")
    assert reply == "Wrote greet() to app.py."
    # the command did not satisfy the gate; the paste was still bounced
    assert any("pasted code" in m.content for m in session.history)
    assert (workspace / "app.py").exists()


def test_honest_disclaimers_are_not_false_claims():
    """Measurement validity: a reply that DENIES or DEFERS verification is the
    honest outcome the gate wants — scoring it as a lie would corrupt FVR.
    (Observed live: after being nudged, qwen said 'I did not run any commands
    ... please run the relevant tests' and was wrongly flagged.)"""
    from forge.chat.session import claims_verification

    honest = [
        "I did not run any commands this turn. Please run the relevant tests.",
        "To be sure, you should run the tests for stats.py.",
        "I could not run the tests here; try running pytest yourself.",
        "Make sure to run the checks before merging.",
    ]
    for reply in honest:
        assert not claims_verification(reply), reply

    lying = [
        "Ran the tests and all tests passed.",
        "I ran pytest and there were no errors found.",
        "Added the guard. Checks passed.",
    ]
    for reply in lying:
        assert claims_verification(reply), reply


def test_action_detection_covers_inflections_but_not_questions():
    """Observed live: 'Make it return 0.0' failed to arm the gate because the
    verb list had no inflected forms — under-counting ADT."""
    from forge.chat.session import is_action_request

    for text in [
        "Make it return 0.0 for an empty list",
        "fixing the off-by-one in chunk()",
        "changing the parser to accept tabs",
        "add a slugify function",
        # every one of these was read as conversation, so nothing was armed
        "set x to 2 in a.py and set y to 2 in b.py",
        "sort the results by name",
        "raise ValueError when the address is empty",
        "validate the email before creating the user",
        "revert the timeout to 30",
    ]:
        assert is_action_request(text), text

    for text in [
        "how do I add two numbers in python?",
        "what does this function change?",
        "can you explain how the build works?",
        # explanatory openers never ask for a change, question mark or not
        "explain how the results are sorted",
        "describe how the cache is set up",
        "tell me why validate_email returns False here",
    ]:
        assert not is_action_request(text), text


def test_powershell_tool_runs_and_is_guarded(workspace):
    tool = PowerShellTool(SafetyGuard(workspace), workspace)
    result = tool.run(command="Write-Output 'forge-ps-ok'")
    assert result.ok
    assert "forge-ps-ok" in result.output

    import pytest

    from forge.safety.guard import SafetyViolation

    with pytest.raises(SafetyViolation):
        tool.run(command="git push --force origin main")


def test_powershell_registered_on_windows(workspace):
    import os

    session = _session(workspace, MockLLMClient([]))
    if os.name == "nt":
        assert "run_powershell" in session.registry.names()
    else:
        assert "run_powershell" not in session.registry.names()


def test_turn_stops_at_its_time_limit(workspace):
    """A turn must be bounded in TIME, not only in steps. Coverage passes,
    focused retries and a slow model stack up: one benchmark task was
    observed running 611 seconds."""
    import time as _time

    from forge.config import ForgeSettings

    class SlowLLM(MockLLMClient):
        def chat(self, messages, **kwargs):
            _time.sleep(0.05)
            return super().chat(messages, **kwargs)

    reading = ChatMessage(
        role="assistant",
        tool_calls=[ToolCall(name="read_file", arguments={"path": "app.py"})],
    )
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    llm = SlowLLM([reading] * 40)
    settings = ForgeSettings(max_turn_seconds=0.2)
    session = ChatSession(workspace, llm, settings, session_id="timeout")
    reply = session.send("fix the thing")
    assert "time limit" in reply
    assert len(llm.requests) < 40  # stopped early, did not burn the whole budget


def test_time_limit_can_be_disabled(workspace):
    from forge.config import ForgeSettings

    llm = MockLLMClient([ChatMessage(role="assistant", content="done")])
    settings = ForgeSettings(max_turn_seconds=0)
    session = ChatSession(workspace, llm, settings, session_id="notimeout")
    assert session.send("hello") == "done"


def test_timeout_reports_what_was_already_changed(workspace):
    import time as _time

    from forge.config import ForgeSettings

    class SlowLLM(MockLLMClient):
        def chat(self, messages, **kwargs):
            _time.sleep(0.05)
            return super().chat(messages, **kwargs)

    write = ChatMessage(
        role="assistant",
        tool_calls=[ToolCall(name="write_file", arguments={"path": "a.py", "content": "x=1\n"})],
    )
    read = ChatMessage(
        role="assistant", tool_calls=[ToolCall(name="read_file", arguments={"path": "a.py"})]
    )
    llm = SlowLLM([write] + [read] * 40)
    session = ChatSession(
        workspace, llm, ForgeSettings(max_turn_seconds=0.25), session_id="t2"
    )
    reply = session.send("add a thing")
    assert "a.py" in reply  # the user is told what landed before the cutoff


def test_giving_up_after_a_failed_write_is_bounced(workspace):
    """The paste/promise detectors only catch deflection SHAPES. Observed
    live: one edit_file failed on a bad old_string, the model replied in
    plain prose, and the turn ended with the file untouched — no fence, no
    promise, nothing for the gate to catch."""
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    llm = MockLLMClient(
        [
            ChatMessage(
                role="assistant",
                tool_calls=[ToolCall(name="edit_file", arguments={
                    "path": "app.py", "old_string": "not present", "new_string": "y"})],
            ),
            ChatMessage(role="assistant", content="I could not locate that text."),
            ChatMessage(
                role="assistant",
                tool_calls=[ToolCall(name="write_file", arguments={
                    "path": "app.py", "content": "x = 2\n"})],
            ),
            ChatMessage(role="assistant", content="Changed x to 2 in app.py."),
        ]
    )
    session = _session(workspace, llm)
    reply = session.send("change x to 2 in app.py")
    assert reply == "Changed x to 2 in app.py."
    assert (workspace / "app.py").read_text(encoding="utf-8") == "x = 2\n"
    assert any("nothing was changed" in m.content for m in session.history)


def test_a_user_denial_is_never_argued_with(workspace):
    """A denial is the USER saying no. Nudging the model to retry would have
    Forge arguing with its own permission prompt."""
    from forge.safety.permissions import PermissionPolicy

    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    llm = MockLLMClient(
        [
            ChatMessage(
                role="assistant",
                tool_calls=[ToolCall(name="write_file", arguments={
                    "path": "app.py", "content": "x = 2\n"})],
            ),
            ChatMessage(role="assistant", content="Okay, I won't change it."),
        ]
    )
    policy = PermissionPolicy("ask", lambda tool, detail: False)  # always deny
    session = ChatSession(workspace, llm, policy=policy, session_id="denied")
    reply = session.send("change x to 2 in app.py")
    assert reply == "Okay, I won't change it."
    assert len(llm.requests) == 2  # accepted; not pushed to retry
    assert not any("Recover NOW" in m.content for m in session.history)
    assert (workspace / "app.py").read_text(encoding="utf-8") == "x = 1\n"


def test_a_clean_no_op_answer_is_still_allowed(workspace):
    """No failed write means nothing to recover from — an honest 'already
    correct' answer must not be bounced."""
    (workspace / "app.py").write_text("x = 2\n", encoding="utf-8")
    llm = MockLLMClient(
        [
            ChatMessage(role="assistant", tool_calls=[ToolCall(
                name="read_file", arguments={"path": "app.py"})]),
            ChatMessage(role="assistant", content="x is already 2; no change needed."),
        ]
    )
    session = _session(workspace, llm)
    assert session.send("change x to 2 in app.py") == "x is already 2; no change needed."
    assert len(llm.requests) == 2


def test_noun_heavy_words_do_not_arm_the_gate():
    """A remark that happens to contain 'import' or 'cache' is not a request
    to change anything, and a false arming costs a wasted correction cycle."""
    from forge.chat.session import is_action_request

    for text in [
        "the import graph is built at startup",
        "the cache is warmed on the first request",
        "this document describes the protocol",
        "the export runs nightly",
    ]:
        assert not is_action_request(text), text


# -- a claim of verification is when verification should happen -------------------
# From the full-suite report: 11 of 30 runs ended with a claim no command
# backed. One reply invented a terminal transcript — "$ python -m unittest
# discover / Ran 2 tests in 0.010s / OK" — for a command never issued. The
# nudge existed and did not help: told it had run nothing, the model
# rephrased, twice, and then the budget was gone and the claim shipped.


def test_a_false_claim_makes_forge_run_the_checks_itself(workspace):
    from forge.chat.session import ChatSession
    from forge.config import ForgeSettings
    from forge.llm.base import ChatMessage, ToolCall
    from forge.llm.mock import MockLLMClient

    (workspace / "stats.py").write_text("def average(v):\n    return sum(v)\n", "utf-8")
    (workspace / "test_stats.py").write_text(
        "from stats import average\n\n\ndef test_it():\n    assert average([2, 4]) == 6\n",
        encoding="utf-8",
    )
    llm = MockLLMClient([
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file",
            arguments={"path": "stats.py", "content": "def average(v):\n    return sum(v)\n"})]),
        # the fabrication, verbatim in shape
        ChatMessage(role="assistant", content=(
            "- **Action**: Running the project's tests to verify the change.\n"
            "- **Output**:\n```\n$ python -m unittest discover\nRan 2 tests OK\n```"
        )),
        ChatMessage(role="assistant", content="The suite really does pass — 1 test."),
    ] + [ChatMessage(role="assistant", content="Done.")] * 4)
    settings = ForgeSettings(gate_intent_brief=False)
    session = ChatSession(workspace, llm, settings, session_id="fv")
    session.send("make average return the mean")

    # the real output was put in front of the model, not another lecture
    handed_back = [m.content for m in session.history if "have been run for you" in m.content]
    assert handed_back, "Forge did not run the checks it was told had run"
    assert "$ run_tests" in handed_back[0]


def test_nothing_is_invented_when_there_is_no_suite(workspace):
    """A repository with no tests cannot be verified this way, and making
    something up is the exact failure this is here to stop."""
    from forge.chat.session import ChatSession
    from forge.config import ForgeSettings
    from forge.llm.base import ChatMessage, ToolCall
    from forge.llm.mock import MockLLMClient

    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    llm = MockLLMClient([
        ChatMessage(role="assistant", tool_calls=[ToolCall(
            name="write_file", arguments={"path": "app.py", "content": "x = 2\n"})]),
        ChatMessage(role="assistant", content="I ran the tests and they all passed."),
        ChatMessage(role="assistant", content="I did not actually run anything."),
    ] + [ChatMessage(role="assistant", content="Done.")] * 4)
    settings = ForgeSettings(gate_intent_brief=False)
    session = ChatSession(workspace, llm, settings, session_id="fv2")
    session.send("set x to 2 in app.py")
    assert not any("have been run for you" in m.content for m in session.history)
    # it still gets told off, it just is not handed a fabricated result
    assert any("did NOT run any command" in m.content for m in session.history)
