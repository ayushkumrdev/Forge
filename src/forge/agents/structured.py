"""Helper for agents that must return validated JSON (Planner, Reviewer).
On invalid output the model gets the error back and one retry — local models
occasionally wrap JSON in prose, and this recovers most of those cases."""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from forge.llm.base import ChatMessage, LLMClient, Usage
from forge.llm.json_utils import JSONExtractionError, extract_json


class StructuredOutputError(RuntimeError):
    pass


def structured_call[T: BaseModel](
    llm: LLMClient,
    system_prompt: str,
    user_message: str,
    model_type: type[T],
    usage: Usage | None = None,
    max_attempts: int = 3,
) -> T:
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_message),
    ]
    last_error = ""
    for _ in range(max_attempts):
        response = llm.chat(messages, temperature=0.1)
        if usage is not None:
            usage.add(response.usage)
        text = response.message.content
        try:
            return model_type.model_validate(extract_json(text))
        except (JSONExtractionError, ValidationError) as exc:
            last_error = str(exc)
            messages.append(ChatMessage(role="assistant", content=text))
            messages.append(
                ChatMessage(
                    role="user",
                    content=f"Your previous reply was not valid: {last_error}\n"
                    "Respond again with ONLY the JSON object, no other text.",
                )
            )
    raise StructuredOutputError(
        f"Model failed to produce valid {model_type.__name__} JSON "
        f"after {max_attempts} attempts: {last_error}"
    )
