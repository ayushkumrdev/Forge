"""search_code: semantic-ish code search for agents, backed by the hybrid
retrieval engine. Complements grep (exact patterns) with ranked, chunked
results for natural-language queries."""

from __future__ import annotations

from typing import Any

from forge.retrieval.engine import RetrievalEngine
from forge.tools.base import Tool, ToolResult

_MAX_CHUNK_PREVIEW_CHARS = 900


class SearchCodeTool(Tool):
    name = "search_code"
    description = (
        "Search the repository by meaning, not exact text: describe what you "
        "are looking for (e.g. 'where user sessions are validated') and get "
        "the most relevant code chunks with their locations. Use grep instead "
        "when you know the exact string."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What you are looking for."},
            "k": {"type": "integer", "description": "Number of results (default 5)."},
        },
        "required": ["query"],
    }

    def __init__(self, engine: RetrievalEngine) -> None:
        self._engine = engine

    def run(self, query: str, k: int = 5) -> ToolResult:
        results = self._engine.search(query, k=max(1, min(k, 15)))
        if not results:
            return ToolResult(
                ok=True, output=f"No relevant code found for {query!r}. Try grep."
            )
        sections: list[str] = []
        for chunk, _score in results:
            body = chunk.text
            if len(body) > _MAX_CHUNK_PREVIEW_CHARS:
                body = body[:_MAX_CHUNK_PREVIEW_CHARS] + "\n... [chunk truncated]"
            sections.append(f"### {chunk.location}\n{body}")
        return ToolResult(ok=True, output="\n\n".join(sections))
