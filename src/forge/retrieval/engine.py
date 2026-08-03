"""Hybrid retrieval engine.

Chunks every source file (at symbol boundaries when the index knows them,
fixed line windows otherwise), ranks with BM25, optionally re-ranks with dense
embeddings, and fuses the two orderings with reciprocal-rank fusion — which
needs no score normalization and is robust when either signal is weak."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from forge.repo.scanner import FileInfo, RepoSnapshot
from forge.retrieval.bm25 import BM25Index
from forge.retrieval.embeddings import Embedder, cosine

_WINDOW_LINES = 60
_WINDOW_OVERLAP = 10
_MAX_CHUNK_LINES = 120
_RRF_K = 60  # standard reciprocal-rank-fusion constant
_PREFLIGHT_MAX_CHARS = 2_400


class Chunk(BaseModel):
    path: str
    start_line: int
    end_line: int
    text: str

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


class RetrievalEngine:
    def __init__(self, workspace: Path, embedder: Embedder | None = None) -> None:
        self.workspace = workspace
        self.chunks: list[Chunk] = []
        self._bm25 = BM25Index()
        self._embedder = embedder
        self._vectors: list[list[float]] | None = None

    def build(self, snapshot: RepoSnapshot) -> int:
        """Chunk and index every scanned file. Returns the chunk count."""
        for file in snapshot.files:
            full_path = self.workspace / file.path
            try:
                lines = full_path.read_text(
                    encoding="utf-8-sig", errors="replace"
                ).splitlines()
            except OSError:
                continue
            self.chunks.extend(_chunk_file(file, lines))
        self._bm25.add_documents([chunk.text for chunk in self.chunks])
        if self._embedder is not None and self.chunks:
            self._vectors = self._embedder.embed([c.text for c in self.chunks])
        return len(self.chunks)

    def search(self, query: str, k: int = 8) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        rankings: list[list[int]] = []
        bm25_top = self._bm25.top(query, k * 4)
        rankings.append([index for index, _ in bm25_top])

        if self._embedder is not None and self._vectors:
            query_vector = self._embedder.embed([query])[0]
            similarity = [
                (i, cosine(query_vector, vector))
                for i, vector in enumerate(self._vectors)
            ]
            similarity.sort(key=lambda pair: -pair[1])
            rankings.append([index for index, _ in similarity[: k * 4]])

        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, index in enumerate(ranking):
                fused[index] = fused.get(index, 0.0) + 1.0 / (_RRF_K + rank + 1)
        ordered = sorted(fused.items(), key=lambda pair: -pair[1])
        return [(self.chunks[index], score) for index, score in ordered[:k]]

    def preflight(
        self, query: str, k: int = 3, max_chars: int = _PREFLIGHT_MAX_CHARS
    ) -> str:
        """Prompt-ready block of the code most relevant to `query`, injected
        BEFORE the model generates so it works from the repository's real APIs
        instead of remembered ones. Empty when nothing matches (BM25-gated, so
        small talk injects nothing)."""
        if not self.chunks or not self._bm25.top(query, 1):
            return ""
        blocks: list[str] = []
        used = 0
        for chunk, _score in self.search(query, k):
            block = f"### {chunk.location}\n```\n{chunk.text}\n```"
            if used + len(block) > max_chars:
                if blocks:
                    break
                # even the best chunk overflows: keep its head so the model
                # still sees real code rather than nothing
                keep = max_chars - len("### \n```\n\n``` [truncated]") - len(chunk.location)
                block = (
                    f"### {chunk.location}\n```\n{chunk.text[:keep]}\n``` [truncated]"
                )
            blocks.append(block)
            used += len(block)
        return (
            "## Relevant code from the repository (auto-retrieved; verify with "
            "read_file before editing)\n" + "\n".join(blocks)
        )


def _chunk_file(file: FileInfo, lines: list[str]) -> list[Chunk]:
    if not lines:
        return []
    boundaries = sorted({s.line for s in file.symbols if s.kind in ("function", "class")})
    if boundaries:
        return _symbol_chunks(file.path, lines, boundaries)
    return _window_chunks(file.path, lines)


def _symbol_chunks(path: str, lines: list[str], boundaries: list[int]) -> list[Chunk]:
    starts = [1] if boundaries[0] > 1 else []
    starts.extend(boundaries)
    chunks: list[Chunk] = []
    for i, start in enumerate(starts):
        end = (starts[i + 1] - 1) if i + 1 < len(starts) else len(lines)
        # oversized symbol bodies are windowed so no chunk explodes the context
        if end - start + 1 > _MAX_CHUNK_LINES:
            chunks.extend(_window_chunks(path, lines, start, end))
        else:
            chunks.append(_make_chunk(path, lines, start, end))
    return chunks


def _window_chunks(
    path: str, lines: list[str], start: int = 1, end: int | None = None
) -> list[Chunk]:
    end = end or len(lines)
    chunks: list[Chunk] = []
    position = start
    while position <= end:
        window_end = min(position + _WINDOW_LINES - 1, end)
        chunks.append(_make_chunk(path, lines, position, window_end))
        if window_end >= end:
            break
        position = window_end + 1 - _WINDOW_OVERLAP
    return chunks


def _make_chunk(path: str, lines: list[str], start: int, end: int) -> Chunk:
    return Chunk(
        path=path,
        start_line=start,
        end_line=end,
        text="\n".join(lines[start - 1 : end]),
    )
