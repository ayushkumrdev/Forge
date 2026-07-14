import json

import httpx
import pytest

from forge.llm.base import ChatMessage, LLMError, ToolSpec
from forge.llm.ollama import OllamaClient

TOOL = ToolSpec(
    name="read_file",
    description="Read a file.",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
)


def _client_with(handler) -> OllamaClient:
    return OllamaClient(transport=httpx.MockTransport(handler))


def test_chat_sends_tools_and_parses_tool_calls():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "read_file", "arguments": {"path": "a.py"}}}
                    ],
                },
                "prompt_eval_count": 100,
                "eval_count": 20,
                "total_duration": 2_000_000_000,
            },
        )

    response = _client_with(handler).chat(
        [ChatMessage(role="user", content="read a.py")], tools=[TOOL]
    )

    assert captured["model"] == "qwen2.5-coder:7b"
    assert captured["tools"][0]["function"]["name"] == "read_file"
    assert captured["stream"] is False

    assert response.message.tool_calls[0].name == "read_file"
    assert response.message.tool_calls[0].arguments == {"path": "a.py"}
    assert response.usage.prompt_tokens == 100
    assert response.usage.completion_tokens == 20
    assert response.usage.duration_ms == 2000


def test_chat_plain_text_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "All done."}}
        )

    response = _client_with(handler).chat([ChatMessage(role="user", content="hi")])
    assert response.message.content == "All done."
    assert response.message.tool_calls == []


def test_streaming_assembles_content_and_calls_on_token():
    ndjson = b"\n".join(
        [
            json.dumps({"message": {"role": "assistant", "content": "Hel"}}).encode(),
            json.dumps({"message": {"role": "assistant", "content": "lo"}}).encode(),
            json.dumps(
                {
                    "done": True,
                    "message": {"role": "assistant", "content": ""},
                    "prompt_eval_count": 7,
                    "eval_count": 5,
                    "total_duration": 2_000_000_000,
                }
            ).encode(),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=ndjson)

    tokens: list[str] = []
    response = _client_with(handler).chat(
        [ChatMessage(role="user", content="hi")], on_token=tokens.append
    )
    assert tokens == ["Hel", "lo"]
    assert response.message.content == "Hello"
    assert response.usage.completion_tokens == 5
    assert response.usage.duration_ms == 2000


def test_connect_error_raises_friendly_llmerror():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="Cannot reach Ollama"):
        _client_with(handler).chat([ChatMessage(role="user", content="hi")])


def test_http_error_raises_llmerror():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="model exploded")

    with pytest.raises(LLMError, match="HTTP 500"):
        _client_with(handler).chat([ChatMessage(role="user", content="hi")])
