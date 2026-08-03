"""The agentic tool loop: send messages + tool specs to the model, execute the
tool calls it returns, feed results back, repeat until the model answers in
plain text or the step budget runs out. This is the same core loop that
Claude Code-style agents are built on."""

from __future__ import annotations

from pydantic import BaseModel, Field

from forge.llm.base import ChatMessage, LLMClient, ToolSpec, Usage
from forge.llm.json_utils import extract_tool_call, looks_like_tool_call, tool_call_schema
from forge.telemetry import Recorder
from forge.tools.base import ToolRegistry


class AgentOutcome(BaseModel):
    final_text: str
    steps: int
    exhausted: bool = False  # True when the step budget ran out
    usage: Usage = Field(default_factory=Usage)


def recover_inline_tool_call(message: ChatMessage, known_tools: list[str]) -> ChatMessage:
    """Some local model templates emit tool calls as JSON text in the content
    instead of the native tool_calls field (qwen2.5-coder via Ollama does
    this). Recover them so the loop still works — but only when the JSON
    names a registered tool, so real answers that merely contain JSON are
    never misread as calls."""
    if message.tool_calls or not message.content:
        return message
    call = extract_tool_call(message.content)
    if call is None or call.name not in known_tools:
        return message
    return message.model_copy(update={"tool_calls": [call], "content": ""})


def constrained_tool_retry(
    llm: LLMClient,
    messages: list[ChatMessage],
    specs: list[ToolSpec],
    known_tools: list[str],
) -> tuple[ChatMessage, Usage] | None:
    """Grammar-constrained recovery: when a reply looks like a mangled tool
    call, re-ask with the response constrained to the tool-call JSON schema
    (Ollama structured outputs) — the model cannot emit malformed JSON under
    the grammar. Returns None when the retry did not yield a usable call."""
    try:
        response = llm.chat(messages, tools=specs, format=tool_call_schema(known_tools))
    except Exception:  # noqa: BLE001 — recovery must never break the main loop
        return None
    message = response.message
    if not message.tool_calls:
        call = extract_tool_call(message.content or "")
        if call is None or call.name not in known_tools:
            return None
        message = message.model_copy(update={"tool_calls": [call], "content": ""})
    return message, response.usage


class ToolLoopAgent:
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        registry: ToolRegistry,
        recorder: Recorder,
        max_steps: int = 25,
        max_tool_output_chars: int = 8_000,
    ) -> None:
        self.name = name
        self._llm = llm
        self._registry = registry
        self._recorder = recorder
        self._max_steps = max_steps
        self._max_tool_output_chars = max_tool_output_chars

    def run(
        self, system_prompt: str, user_message: str, temperature: float | None = None
    ) -> AgentOutcome:
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_message),
        ]
        total_usage = Usage()
        specs = self._registry.specs()

        for step in range(1, self._max_steps + 1):
            response = self._llm.chat(messages, tools=specs, temperature=temperature)
            total_usage.add(response.usage)
            message = self._recover_inline_tool_call(response.message)
            if (
                not message.tool_calls
                and message.content
                and looks_like_tool_call(message.content, self._registry.names())
            ):
                retried = constrained_tool_retry(
                    self._llm, messages, specs, self._registry.names()
                )
                if retried is not None:
                    message, retry_usage = retried
                    total_usage.add(retry_usage)
                    self._recorder.event(
                        self.name,
                        "constrained_tool_retry",
                        tool=message.tool_calls[0].name,
                    )
            self._recorder.event(
                self.name,
                "llm_response",
                step=step,
                completion_tokens=response.usage.completion_tokens,
                duration_ms=response.usage.duration_ms,
                tool_calls=[tc.name for tc in message.tool_calls],
            )

            if not message.tool_calls:
                return AgentOutcome(
                    final_text=message.content, steps=step, usage=total_usage
                )

            messages.append(message)
            for call in message.tool_calls:
                self._recorder.event(
                    self.name, "tool_call", tool=call.name, arguments=call.arguments
                )
                result = self._registry.execute(call.name, call.arguments)
                self._recorder.event(
                    self.name, "tool_result", tool=call.name, ok=result.ok, error=result.error
                )
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=result.render(self._max_tool_output_chars),
                        tool_name=call.name,
                    )
                )

        self._recorder.event(self.name, "step_budget_exhausted", max_steps=self._max_steps)
        return AgentOutcome(
            final_text="Stopped: step budget exhausted before the agent finished.",
            steps=self._max_steps,
            exhausted=True,
            usage=total_usage,
        )

    def _recover_inline_tool_call(self, message: ChatMessage) -> ChatMessage:
        recovered = recover_inline_tool_call(message, self._registry.names())
        if recovered is not message and recovered.tool_calls:
            self._recorder.event(
                self.name, "inline_tool_call_recovered", tool=recovered.tool_calls[0].name
            )
        return recovered
