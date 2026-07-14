"""Provider-agnostic chat types. Agents depend only on this module, never on a
concrete provider, so backends (Ollama today, others later) are swappable."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

# receives each content delta as it is generated (streaming)
TokenCallback = Callable[[str], None]

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_name: str | None = None  # set on role="tool" messages


class ToolSpec(BaseModel):
    """A tool definition in JSON-schema form, convertible to the wire format."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: float = 0.0

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.duration_ms += other.duration_ms


class LLMResponse(BaseModel):
    message: ChatMessage
    usage: Usage = Field(default_factory=Usage)


class LLMError(RuntimeError):
    """Raised when the LLM backend is unreachable or returns an invalid reply."""


class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        on_token: TokenCallback | None = None,
    ) -> LLMResponse: ...
