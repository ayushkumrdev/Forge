"""Pure-Python BM25 (Okapi) with a code-aware tokenizer: identifiers are kept
whole AND split on camelCase/snake_case so a query for "order service" finds
`OrderService` and `order_service` alike."""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")

# common English noise words that would otherwise dominate natural-language
# queries like "where are user sessions validated"
_STOPWORDS = frozenset(
    ["the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on", "for", "and", "or", "not", "no", "with", "as", "at", "by", "it", "its", "this", "that", "these", "those", "from", "into", "how", "what", "when", "where", "which", "who", "whom", "why", "do", "does", "did", "done", "can", "could", "should", "would", "will", "i", "we", "you", "they", "he", "she", "them", "us", "our", "your", "my"]
)


def _stem(token: str) -> str:
    """Very light suffix stripping so 'sessions'/'session' and
    'validated'/'validate' land on the same term."""
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ed", "es")):
        return token[:-2] if token.endswith("ed") else token[:-1]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for word in _WORD_RE.findall(text):
        lowered = word.lower()
        raw = [lowered]
        parts = [p.lower() for p in _CAMEL_RE.findall(word) if len(p) > 1]
        if len(parts) > 1 or (parts and parts[0] != lowered.strip("_")):
            raw.extend(parts)
        elif "_" in word:
            raw.extend(p for p in lowered.split("_") if len(p) > 1)
        tokens.extend(_stem(t) for t in raw if t not in _STOPWORDS)
    return tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._df: Counter[str] = Counter()
        self._avg_len = 0.0

    def add_documents(self, texts: list[str]) -> None:
        for text in texts:
            counts = Counter(tokenize(text))
            self._doc_freqs.append(counts)
            self._doc_lens.append(sum(counts.values()))
            for token in counts:
                self._df[token] += 1
        self._avg_len = (
            sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0
        )

    def __len__(self) -> int:
        return len(self._doc_freqs)

    def scores(self, query: str) -> list[float]:
        n = len(self._doc_freqs)
        query_tokens = set(tokenize(query))
        scores = [0.0] * n
        for token in query_tokens:
            df = self._df.get(token)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, counts in enumerate(self._doc_freqs):
                tf = counts.get(token)
                if not tf:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self._doc_lens[i] / self._avg_len)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.scores(query)
        ranked = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0),
            key=lambda pair: -pair[1],
        )
        return ranked[:k]
