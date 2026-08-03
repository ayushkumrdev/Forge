"""Grammar-constrained tool-call recovery: a mangled tool call triggers one
re-ask constrained to the tool-call JSON schema, so malformed output can't
derail the loop."""

from forge.agents.base import ToolLoopAgent
from forge.llm.base import ChatMessage
from forge.llm.json_utils import looks_like_tool_call, tool_call_schema
from forge.llm.mock import MockLLMClient
from forge.safety.guard import SafetyGuard
from forge.tools.base import ToolRegistry
from forge.tools.changes import ChangeLedger
from forge.tools.filesystem import ReadFileTool, WriteFileTool

TOOLS = ["read_file", "write_file", "edit_file"]


def _registry(workspace):
    guard = SafetyGuard(workspace)
    ledger = ChangeLedger(workspace, "testrun")
    return ToolRegistry([ReadFileTool(guard), WriteFileTool(guard, ledger)])


def test_looks_like_tool_call_detects_mangled_calls():
    # single quotes (invalid JSON) but clearly an attempted call
    assert looks_like_tool_call("{'name': 'read_file', 'arguments': {'path': 'a'}}", TOOLS)
    # qwen-style tag with broken body
    assert looks_like_tool_call('<tool_call>{"name": "read_file", oops', TOOLS)
    # truncated JSON naming a tool
    assert looks_like_tool_call('{"name": "edit_file", "arguments": {"path": "x.py", "old', TOOLS)


def test_looks_like_tool_call_ignores_answers():
    assert not looks_like_tool_call("I edited the file and all tests pass.", TOOLS)
    # mentions a tool but is prose with unrelated JSON
    assert not looks_like_tool_call('I used read_file to check. Result: {"lines": 3}', TOOLS)
    assert not looks_like_tool_call("", TOOLS)


def test_tool_call_schema_enumerates_tools():
    schema = tool_call_schema(TOOLS)
    assert schema["properties"]["name"]["enum"] == TOOLS
    assert schema["required"] == ["name", "arguments"]


def test_agent_retries_mangled_call_with_grammar(workspace, recorder):
    llm = MockLLMClient(
        [
            # step 1: mangled call (single quotes -> unparseable)
            ChatMessage(
                role="assistant",
                content="{'name': 'write_file', 'arguments': {'path': 'g.txt', 'content': 'ok'}}",
            ),
            # constrained retry: valid JSON under the grammar
            ChatMessage(
                role="assistant",
                content='{"name": "write_file", "arguments": {"path": "g.txt", "content": "ok"}}',
            ),
            ChatMessage(role="assistant", content="done"),
        ]
    )
    agent = ToolLoopAgent("coder", llm, _registry(workspace), recorder, max_steps=5)
    outcome = agent.run("system", "write g.txt")

    assert not outcome.exhausted
    assert (workspace / "g.txt").read_text(encoding="utf-8") == "ok"
    # the second request must have carried the constraining schema
    assert llm.formats[0] is None
    assert llm.formats[1] is not None
    assert "write_file" in llm.formats[1]["properties"]["name"]["enum"]


def test_agent_keeps_answer_when_retry_unusable(workspace, recorder):
    llm = MockLLMClient(
        [
            ChatMessage(role="assistant", content="{'name': 'read_file', broken"),
            # retry yields a non-call -> original content stands as the answer
            ChatMessage(role="assistant", content="cannot comply"),
        ]
    )
    agent = ToolLoopAgent("coder", llm, _registry(workspace), recorder, max_steps=5)
    outcome = agent.run("system", "task")
    assert outcome.steps == 1
    assert "read_file" in outcome.final_text  # the original reply was kept
