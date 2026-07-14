"""Scripted LLM client for deterministic tests — no network, no model."""

from __future__ import annotations

from forge.llm.base import (
    ChatMessage,
    LLMClient,
    LLMError,
    LLMResponse,
    TokenCallback,
    ToolSpec,
)


class MockLLMClient(LLMClient):
    """Returns pre-scripted responses in order and records every request."""

    def __init__(self, responses: list[ChatMessage]) -> None:
        self._responses = list(responses)
        self.requests: list[list[ChatMessage]] = []

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        on_token: TokenCallback | None = None,
    ) -> LLMResponse:
        self.requests.append(list(messages))
        if not self._responses:
            raise LLMError("MockLLMClient ran out of scripted responses")
        message = self._responses.pop(0)
        if on_token is not None and message.content and not message.tool_calls:
            on_token(message.content)
        return LLMResponse(message=message)
