"""OpenAI-compatible chat client — one implementation that unlocks every
runtime speaking the /v1/chat/completions dialect: LM Studio, llama.cpp
server, vLLM, text-generation-webui, OpenRouter, and cloud providers.

Supports native tool calling, SSE token streaming, and structured outputs
(json_schema response_format) where the backend implements them; a backend
that rejects the response_format is retried once without it rather than
failing the run."""

from __future__ import annotations

import json
from typing import Any

import httpx

from forge.llm.base import (
    ChatMessage,
    LLMClient,
    LLMError,
    LLMResponse,
    TokenCallback,
    ToolCall,
    ToolSpec,
    Usage,
)


class OpenAICompatClient(LLMClient):
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "qwen2.5-coder-7b-instruct",
        api_key: str = "",
        temperature: float = 0.2,
        timeout_s: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            headers=headers,
            transport=transport,
        )

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        on_token: TokenCallback | None = None,
        format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_wire(messages),
            "stream": on_token is not None,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = [t.to_wire() for t in tools]
        if format is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "constrained", "schema": format},
            }
        try:
            return self._request(payload, on_token)
        except httpx.HTTPStatusError as exc:
            if format is not None and exc.response.status_code in (400, 422):
                # backend doesn't implement json_schema: degrade gracefully
                payload.pop("response_format", None)
                try:
                    return self._request(payload, on_token)
                except httpx.HTTPError as retry_exc:
                    raise _wrap(retry_exc, self._client.base_url) from retry_exc
            raise _wrap(exc, self._client.base_url) from exc
        except httpx.HTTPError as exc:
            raise _wrap(exc, self._client.base_url) from exc

    def _request(self, payload: dict[str, Any], on_token: TokenCallback | None) -> LLMResponse:
        if on_token is not None:
            return self._request_streaming(payload, on_token)
        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return _parse(response.json())

    def _request_streaming(
        self, payload: dict[str, Any], on_token: TokenCallback
    ) -> LLMResponse:
        content_parts: list[str] = []
        # tool-call fragments accumulate by stream index
        names: dict[int, str] = {}
        args: dict[int, str] = {}
        usage = Usage()
        with self._client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = _usage(chunk["usage"])
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if text:
                    content_parts.append(text)
                    on_token(text)
                for tc in delta.get("tool_calls") or []:
                    index = tc.get("index", 0)
                    function = tc.get("function") or {}
                    if function.get("name"):
                        names[index] = names.get(index, "") + function["name"]
                    if function.get("arguments"):
                        args[index] = args.get(index, "") + function["arguments"]
        tool_calls = [
            ToolCall(name=names[i], arguments=_parse_args(args.get(i, "")))
            for i in sorted(names)
        ]
        return LLMResponse(
            message=ChatMessage(
                role="assistant", content="".join(content_parts), tool_calls=tool_calls
            ),
            usage=usage,
        )

    def ping(self) -> bool:
        try:
            return self._client.get("/models").status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        try:
            response = self._client.get("/models")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot list models: {exc}") from exc
        return [m["id"] for m in response.json().get("data", [])]


def _to_wire(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """OpenAI wire format. Tool results must reference the id of the call that
    produced them; Forge's history is ordered, so ids are synthesized per
    assistant message and consumed by the tool messages that follow."""
    wire: list[dict[str, Any]] = []
    pending_ids: list[str] = []
    counter = 0
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            ids = [f"call_{(counter := counter + 1)}" for _ in message.tool_calls]
            wire.append(
                {
                    "role": "assistant",
                    "content": message.content or None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for call_id, tc in zip(ids, message.tool_calls, strict=True)
                    ],
                }
            )
            pending_ids = list(ids)
        elif message.role == "tool":
            entry: dict[str, Any] = {
                "role": "tool",
                "content": message.content,
                "tool_call_id": pending_ids.pop(0) if pending_ids else "call_0",
            }
            if message.tool_name:
                entry["name"] = message.tool_name
            wire.append(entry)
        else:
            wire.append({"role": message.role, "content": message.content})
    return wire


def _parse(data: dict[str, Any]) -> LLMResponse:
    choices = data.get("choices") or []
    raw = (choices[0].get("message") if choices else None) or {}
    tool_calls = [
        ToolCall(
            name=tc["function"]["name"],
            arguments=_parse_args(tc["function"].get("arguments") or ""),
        )
        for tc in raw.get("tool_calls") or []
    ]
    return LLMResponse(
        message=ChatMessage(
            role="assistant", content=raw.get("content") or "", tool_calls=tool_calls
        ),
        usage=_usage(data.get("usage") or {}),
    )


def _parse_args(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _usage(raw: dict[str, Any]) -> Usage:
    return Usage(
        prompt_tokens=raw.get("prompt_tokens") or 0,
        completion_tokens=raw.get("completion_tokens") or 0,
    )


def _wrap(exc: httpx.HTTPError, base_url: httpx.URL) -> LLMError:
    if isinstance(exc, httpx.ConnectError):
        return LLMError(
            f"Cannot reach the OpenAI-compatible server at {base_url}. "
            "Check FORGE_OPENAI_BASE_URL and that the server is running."
        )
    if isinstance(exc, httpx.HTTPStatusError):
        return LLMError(
            f"Server returned HTTP {exc.response.status_code}: {exc.response.text[:500]}"
        )
    return LLMError(f"Request failed: {exc}")
