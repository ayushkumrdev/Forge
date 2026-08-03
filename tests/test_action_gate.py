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
    assert reply == example
    assert len(llm.requests) == 1  # no retry happened


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
