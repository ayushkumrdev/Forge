"""Self-repairing edit matching — a grounding layer that turns the most common
small-model failure (an ``old_string`` that doesn't byte-match the file) into a
successful edit or a precise, copy-pasteable correction.

Match tiers, tried in order:
  1. exact & unique                → apply
  2. whitespace-tolerant & unique  → apply to the file's REAL text (fixes the
     tab/space, trailing-space, and CRLF mismatches that plague Windows)
  3. fuzzy closest span            → don't guess; hand back the exact snippet
     that is actually in the file so the model can copy it verbatim

The philosophy: reality is the verifier. We never invent an edit location; we
only ever apply the model's replacement to text that provably exists in the
file, and when we can't be certain we ground the model with the truth."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

_FUZZY_THRESHOLD = 0.60  # below this, a "closest span" is too different to show
_WS_RE = re.compile(r"[ \t]+")

# Small models routinely over-escape when emitting JSON tool arguments: the
# newline in a snippet arrives as the two characters \ and n, and backslashes
# in regexes arrive doubled. The needle is then a single line that can never
# match a multi-line span, so every tier fails and the model loops forever
# "correcting" itself. Observed live on qwen2.5-coder:7b: 11 consecutive
# failed edits on one file. Decoding the escapes is not guessing — the
# repaired needle still has to match text that provably exists.
_ESCAPE_RE = re.compile(r"\\([nrt\\])")
_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\"}


def unescape_literals(text: str) -> str:
    """Decode literal \\n / \\t / \\r / \\\\ sequences in one pass."""
    return _ESCAPE_RE.sub(lambda m: _ESCAPES[m.group(1)], text)


class MatchOutcome:
    APPLIED = "applied"
    AMBIGUOUS = "ambiguous"  # matched in several places
    NOT_FOUND = "not_found"


@dataclass
class EditResult:
    outcome: str
    new_content: str | None = None  # set when outcome == APPLIED
    tier: str = ""  # "exact" | "whitespace" — how it matched
    occurrences: int = 0  # for AMBIGUOUS
    suggestion: str | None = None  # exact file text to copy, for NOT_FOUND


def _normalize(text: str) -> str:
    """Collapse runs of spaces/tabs and strip each line's trailing whitespace,
    so indentation style and stray spaces don't block a match — while the
    file's real bytes are what actually gets edited."""
    lines = [_WS_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def _find_whitespace_tolerant_spans(content: str, needle: str) -> list[str]:
    """Return every real substring of `content` whose normalized form equals
    the normalized `needle`, matched on line boundaries."""
    content_lines = content.splitlines(keepends=True)
    needle_norm = _normalize(needle)
    span_line_count = max(len(needle.splitlines()), 1)
    spans: list[str] = []
    for i in range(len(content_lines) - span_line_count + 1):
        window = "".join(content_lines[i : i + span_line_count])
        # drop a trailing newline the window may carry that the needle lacks
        candidates = {window, window.rstrip("\n"), window.rstrip("\r\n")}
        if any(_normalize(c) == needle_norm for c in candidates):
            # prefer the exact window without an extra trailing newline
            spans.append(window.rstrip("\n") if not needle.endswith("\n") else window)
    return spans


def _closest_span(content: str, needle: str) -> tuple[str, float]:
    """Best-matching real substring of the file (by line window), and its ratio."""
    content_lines = content.splitlines(keepends=True)
    span_line_count = max(len(needle.splitlines()), 1)
    best_text, best_ratio = "", 0.0
    matcher = difflib.SequenceMatcher()
    matcher.set_seq2(needle)
    # try windows of the needle's height, plus one line of slack each way
    for height in {max(span_line_count - 1, 1), span_line_count, span_line_count + 1}:
        for i in range(len(content_lines) - height + 1):
            window = "".join(content_lines[i : i + height])
            matcher.set_seq1(window)
            ratio = matcher.quick_ratio()
            if ratio > best_ratio:
                real_ratio = matcher.ratio()
                if real_ratio > best_ratio:
                    best_ratio, best_text = real_ratio, window.rstrip("\n")
    return best_text, best_ratio


def reindent_replacement(content: str, old_string: str, new_string: str) -> str:
    """Re-indent a multi-line replacement to the depth of the line it lands on.

    Models write `new_string` as if it started at column zero. When the match
    sits inside a function, every line after the first arrives under-indented
    and the block falls out of its enclosing scope — observed live: a three
    line guard turned `return port` into a module-level statement Python
    refuses to run. Returns the replacement unchanged when it cannot help."""
    index = content.find(old_string)
    if index == -1:
        return new_string
    line_start = content.rfind("\n", 0, index) + 1
    base = content[line_start:index]
    if base.strip():
        return new_string  # match starts mid-line; nothing reliable to infer
    if not base:
        # old_string carried its own indentation, so take the depth from it
        first = old_string.split("\n", 1)[0]
        base = first[: len(first) - len(first.lstrip())]
    if not base:
        return new_string  # genuinely at column zero
    lines = new_string.split("\n")
    if len(lines) < 2:
        return new_string
    # Shift the whole block by the anchor's depth, preserving each line's
    # RELATIVE indentation — a nested body must stay nested. Over-shooting is
    # safe: the caller keeps this only if the result actually verifies.
    return lines[0] + "\n" + "\n".join(
        (base + line if line.strip() else line) for line in lines[1:]
    )


def compute_edit(content: str, old_string: str, new_string: str) -> EditResult:
    """Resolve an (old_string -> new_string) edit against `content`, repairing
    near-misses. Pure function: returns what WOULD happen, applies nothing."""
    result = _match(content, old_string, new_string)
    if result.outcome == MatchOutcome.APPLIED:
        return result

    # Tier 2.5 — escape repair: the model may have sent literal \n instead of
    # newlines. Retry with the decoded needle (and decode the replacement to
    # match), but only accept it when it lands on real text.
    decoded_old = unescape_literals(old_string)
    if decoded_old != old_string:
        repaired = _match(content, decoded_old, unescape_literals(new_string))
        if repaired.outcome == MatchOutcome.APPLIED:
            return EditResult(
                outcome=MatchOutcome.APPLIED,
                new_content=repaired.new_content,
                tier="escaped",
            )
        if repaired.outcome == MatchOutcome.NOT_FOUND and repaired.suggestion:
            return repaired  # a better-grounded suggestion than the raw needle
    return result


def _match(content: str, old_string: str, new_string: str) -> EditResult:
    # Tier 1 — exact
    exact = content.count(old_string)
    if exact == 1:
        return EditResult(
            outcome=MatchOutcome.APPLIED,
            new_content=content.replace(old_string, new_string, 1),
            tier="exact",
        )
    if exact > 1:
        return EditResult(outcome=MatchOutcome.AMBIGUOUS, occurrences=exact)

    # Tier 2 — whitespace-tolerant, must be unique to auto-apply
    spans = _find_whitespace_tolerant_spans(content, old_string)
    unique_spans = list(dict.fromkeys(spans))
    if len(unique_spans) == 1:
        real = unique_spans[0]
        return EditResult(
            outcome=MatchOutcome.APPLIED,
            new_content=content.replace(real, new_string, 1),
            tier="whitespace",
        )
    if len(unique_spans) > 1:
        return EditResult(outcome=MatchOutcome.AMBIGUOUS, occurrences=len(unique_spans))

    # Tier 3 — fuzzy: ground the model with the real nearby text, never guess
    closest, ratio = _closest_span(content, old_string)
    suggestion = closest if ratio >= _FUZZY_THRESHOLD else None
    return EditResult(outcome=MatchOutcome.NOT_FOUND, suggestion=suggestion)
