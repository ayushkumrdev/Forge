from forge.agents.base import ToolLoopAgent
from forge.llm.base import ChatMessage, ToolCall
from forge.llm.mock import MockLLMClient
from forge.safety.guard import SafetyGuard
from forge.tools.base import ToolRegistry
from forge.tools.changes import ChangeLedger
from forge.tools.filesystem import ReadFileTool, WriteFileTool


def _registry(workspace):
    guard = SafetyGuard(workspace)
    ledger = ChangeLedger(workspace, "testrun")
    return ToolRegistry([ReadFileTool(guard), WriteFileTool(guard, ledger)])


def test_agent_executes_tool_calls_then_finishes(workspace, recorder):
    llm = MockLLMClient(
        [
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(name="write_file", arguments={"path": "out.txt", "content": "done"})
                ],
            ),
            ChatMessage(role="assistant", content="I wrote out.txt with the requested content."),
        ]
    )
    agent = ToolLoopAgent("coder", llm, _registry(workspace), recorder, max_steps=5)
    outcome = agent.run("system", "write out.txt")

    assert outcome.final_text.startswith("I wrote out.txt")
    assert outcome.steps == 2
    assert not outcome.exhausted
    assert (workspace / "out.txt").read_text(encoding="utf-8") == "done"

    # the tool result must have been fed back to the model
    last_request = llm.requests[-1]
    tool_messages = [m for m in last_request if m.role == "tool"]
    assert tool_messages and "Wrote" in tool_messages[0].content


def test_agent_receives_error_for_unknown_tool(workspace, recorder):
    llm = MockLLMClient(
        [
            ChatMessage(
                role="assistant",
                tool_calls=[ToolCall(name="launch_rocket", arguments={})],
            ),
            ChatMessage(role="assistant", content="understood"),
        ]
    )
    agent = ToolLoopAgent("coder", llm, _registry(workspace), recorder, max_steps=5)
    agent.run("system", "do something")

    tool_messages = [m for m in llm.requests[-1] if m.role == "tool"]
    assert "Unknown tool" in tool_messages[0].content


def test_agent_recovers_inline_json_tool_call(workspace, recorder):
    """Models like qwen2.5-coder via Ollama emit tool calls as JSON content;
    the loop must recover and execute them."""
    llm = MockLLMClient(
        [
            ChatMessage(
                role="assistant",
                content='{"name": "write_file", "arguments": '
                '{"path": "inline.txt", "content": "recovered"}}',
            ),
            ChatMessage(role="assistant", content="done"),
        ]
    )
    agent = ToolLoopAgent("coder", llm, _registry(workspace), recorder, max_steps=5)
    outcome = agent.run("system", "write inline.txt")

    assert not outcome.exhausted
    assert (workspace / "inline.txt").read_text(encoding="utf-8") == "recovered"


def test_agent_treats_non_tool_json_as_final_answer(workspace, recorder):
    llm = MockLLMClient(
        [ChatMessage(role="assistant", content='Summary: {"files_changed": 2} all good.')]
    )
    agent = ToolLoopAgent("coder", llm, _registry(workspace), recorder, max_steps=5)
    outcome = agent.run("system", "task")
    assert outcome.steps == 1
    assert "all good" in outcome.final_text


def test_agent_stops_at_step_budget(workspace, recorder):
    endless = ChatMessage(
        role="assistant",
        tool_calls=[ToolCall(name="read_file", arguments={"path": "missing.txt"})],
    )
    llm = MockLLMClient([endless, endless, endless])
    agent = ToolLoopAgent("coder", llm, _registry(workspace), recorder, max_steps=3)
    outcome = agent.run("system", "loop forever")

    assert outcome.exhausted
    assert outcome.steps == 3
