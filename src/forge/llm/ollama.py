"""Ollama chat client with native tool-calling (supported by qwen2.5-coder)."""

from __future__ import annotations

from typing import Any

import httpx

from forge.llm.base import (
    ChatMessage,
    LLMClient,
    LLMError,
    LLMResponse,
    ToolCall,
    ToolSpec,
    Usage,
)


class OllamaClient(LLMClient):
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:7b",
        temperature: float = 0.2,
        num_ctx: int = 16384,
        timeout_s: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._client = httpx.Client(base_url=host, timeout=timeout_s, transport=transport)

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._to_wire(m) for m in messages],
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_ctx": self.num_ctx,
            },
        }
        if tools:
            payload["tools"] = [t.to_wire() for t in tools]

        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self._client.base_url}. "
                "Start it with `ollama serve` and ensure the model is pulled."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Ollama returned HTTP {exc.response.status_code}: "
                           f"{exc.response.text[:500]}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        data = response.json()
        return self._parse(data)

    def ping(self) -> bool:
        try:
            return self._client.get("/api/tags").status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot list Ollama models: {exc}") from exc
        return [m["name"] for m in response.json().get("models", [])]

    @staticmethod
    def _to_wire(message: ChatMessage) -> dict[str, Any]:
        wire: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            wire["tool_calls"] = [
                {"function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in message.tool_calls
            ]
        if message.role == "tool" and message.tool_name:
            wire["tool_name"] = message.tool_name
        return wire

    @staticmethod
    def _parse(data: dict[str, Any]) -> LLMResponse:
        raw = data.get("message") or {}
        tool_calls = [
            ToolCall(
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments") or {},
            )
            for tc in raw.get("tool_calls") or []
        ]
        message = ChatMessage(
            role="assistant",
            content=raw.get("content") or "",
            tool_calls=tool_calls,
        )
        usage = Usage(
            prompt_tokens=data.get("prompt_eval_count") or 0,
            completion_tokens=data.get("eval_count") or 0,
            duration_ms=(data.get("total_duration") or 0) / 1_000_000,
        )
        return LLMResponse(message=message, usage=usage)
