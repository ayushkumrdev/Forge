"""OpenAI-compatible client: wire format, tool calls, SSE streaming, the
json_schema graceful degrade, and the provider factory."""

import json

import httpx
import pytest

from forge.config import ForgeSettings
from forge.llm.base import ChatMessage, LLMError, ToolCall, ToolSpec
from forge.llm.factory import make_client
from forge.llm.ollama import OllamaClient
from forge.llm.openai_compat import OpenAICompatClient

TOOL = ToolSpec(
    name="read_file",
    description="Read a file.",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
)


def _client_with(handler) -> OpenAICompatClient:
    return OpenAICompatClient(transport=httpx.MockTransport(handler))


def test_chat_parses_tool_calls_and_usage():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "a.py"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 9},
            },
        )

    response = _client_with(handler).chat(
        [ChatMessage(role="user", content="read a.py")], tools=[TOOL]
    )
    assert captured["tools"][0]["function"]["name"] == "read_file"
    assert response.message.tool_calls[0].arguments == {"path": "a.py"}
    assert response.usage.prompt_tokens == 50


def test_tool_history_gets_synthesized_call_ids():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    history = [
        ChatMessage(role="user", content="go"),
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(name="read_file", arguments={"path": "a.py"})],
        ),
        ChatMessage(role="tool", content="file text", tool_name="read_file"),
    ]
    _client_with(handler).chat(history)
    wire = captured["messages"]
    assert wire[1]["tool_calls"][0]["id"] == wire[2]["tool_call_id"]
    assert json.loads(wire[1]["tool_calls"][0]["function"]["arguments"]) == {"path": "a.py"}


def test_streaming_assembles_content_and_tool_fragments():
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "read_file", "arguments": '{"pa'}}
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "a.py"}'}}]}}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        },
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode())

    tokens: list[str] = []
    response = _client_with(handler).chat(
        [ChatMessage(role="user", content="hi")], on_token=tokens.append
    )
    assert tokens == ["Hel", "lo"]
    assert response.message.content == "Hello"
    assert response.message.tool_calls[0].arguments == {"path": "a.py"}
    assert response.usage.completion_tokens == 4


def test_json_schema_rejection_degrades_gracefully():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        attempts.append("response_format" in payload)
        if "response_format" in payload:
            return httpx.Response(400, text="response_format not supported")
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "{}"}}]}
        )

    response = _client_with(handler).chat(
        [ChatMessage(role="user", content="go")], format={"type": "object"}
    )
    assert attempts == [True, False]  # retried once without the schema
    assert response.message.content == "{}"


def test_connect_error_is_friendly():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="Cannot reach"):
        _client_with(handler).chat([ChatMessage(role="user", content="hi")])


def test_list_models_reads_data_ids():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "m1"}, {"id": "m2"}]})

    assert _client_with(handler).list_models() == ["m1", "m2"]


def test_factory_selects_provider():
    assert isinstance(make_client(ForgeSettings(provider="ollama")), OllamaClient)
    openai = make_client(ForgeSettings(provider="openai"), model="custom")
    assert isinstance(openai, OpenAICompatClient)
    assert openai.model == "custom"
