"""Scripted LLM client for deterministic tests — no network, no model."""

from __future__ import annotations

from forge.llm.base import ChatMessage, LLMClient, LLMError, LLMResponse, ToolSpec


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
    ) -> LLMResponse:
        self.requests.append(list(messages))
        if not self._responses:
            raise LLMError("MockLLMClient ran out of scripted responses")
        return LLMResponse(message=self._responses.pop(0))
