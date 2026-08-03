"""Provider factory: settings in, the right LLMClient out. Both providers
also expose ping() and list_models(), so callers can stay provider-blind."""

from __future__ import annotations

from forge.config import ForgeSettings
from forge.llm.ollama import OllamaClient
from forge.llm.openai_compat import OpenAICompatClient


def make_client(
    settings: ForgeSettings, model: str | None = None
) -> OllamaClient | OpenAICompatClient:
    if settings.provider == "openai":
        return OpenAICompatClient(
            base_url=settings.openai_base_url,
            model=model or settings.model,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
            timeout_s=settings.request_timeout_s,
        )
    return OllamaClient(
        host=settings.ollama_host,
        model=model or settings.model,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        timeout_s=settings.request_timeout_s,
    )
