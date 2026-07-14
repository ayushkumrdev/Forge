"""Optional dense embeddings via Ollama's /api/embed. When no embedding model
is configured or the model is not pulled, the retrieval engine silently runs
BM25-only — embeddings improve ranking but are never required."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import httpx


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder(Embedder):
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout_s: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self._client = httpx.Client(base_url=host, timeout=timeout_s, transport=transport)

    def available(self) -> bool:
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        models = [m["name"] for m in response.json().get("models", [])]
        return any(name == self.model or name.startswith(self.model + ":") for name in models)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post(
            "/api/embed", json={"model": self.model, "input": texts}
        )
        response.raise_for_status()
        return response.json().get("embeddings", [])


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
