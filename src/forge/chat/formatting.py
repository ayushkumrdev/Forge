"""Structural repair of a reply's Markdown.

Asking a 7B to format well works most of the time, which is another way of
saying it fails often enough to notice. The same lesson as every other gate
applies: fix it structurally rather than hoping the prompt held.

Only unambiguous damage is repaired, and every rule is conservative — a rule
that might mangle correct prose is not worth the formatting it fixes:

  * an unclosed ``` fence swallows the rest of the answer in any renderer
  * a numbered list emitted on one line ("1. do x 2. do y") reads as a wall
  * a list glued to the paragraph above it renders as one run-on block
  * template tags and trailing whitespace are noise
"""

from __future__ import annotations

import re

# "1. first 2. second 3. third" all on one line. Requires an ascending run
# starting at 1, so ordinary prose containing "... in 2. of the spec" is safe.
_RUN_ON_NUMBERED = re.compile(r"^\s*1[.)]\s+.*?\s2[.)]\s+")
_NUMBER_SPLIT = re.compile(r"\s+(?=\d+[.)]\s+[A-Z(`\"'])")

_LIST_LINE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s+")
_FENCE = re.compile(r"^\s*```")


def _split_run_on_lists(text: str) -> str:
    out = []
    for line in text.split("\n"):
        if _RUN_ON_NUMBERED.match(line) and len(_NUMBER_SPLIT.findall(line)) >= 1:
            out.extend(_NUMBER_SPLIT.split(line.strip()))
        else:
            out.append(line)
    return "\n".join(out)


def _separate_blocks(text: str) -> str:
    """Insert the blank line a list or heading needs when the model glued it
    to the paragraph above."""
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
            if not in_fence:
                out.append(line)
                continue
            # opening a fence directly under prose also needs separation
            if out and out[-1].strip() and not _FENCE.match(out[-1]):
                out.append("")
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        starts_block = _LIST_LINE.match(line) or _HEADING_LINE.match(line)
        if starts_block and out and out[-1].strip():
            previous = out[-1]
            # a list item following another list item is already fine
            if not (_LIST_LINE.match(line) and _LIST_LINE.match(previous)):
                out.append("")
        out.append(line)
    return "\n".join(out)


def _close_dangling_fence(text: str) -> str:
    """An odd number of fences leaves the tail of the answer inside a code
    block — the reader loses everything after it.

    Counted per line: `_FENCE` is anchored with ^ and is applied with match()
    elsewhere, so findall() over the whole string would only ever see a fence
    at position 0."""
    fences = sum(1 for line in text.split("\n") if _FENCE.match(line))
    if fences % 2 == 0:
        return text
    return text.rstrip() + "\n```"


def normalize_markdown(text: str) -> str:
    """Repair a reply's structure without changing its words."""
    if not text or not text.strip():
        return text
    repaired = _close_dangling_fence(text)
    repaired = _split_run_on_lists(repaired)
    repaired = _separate_blocks(repaired)
    # collapse 3+ blank lines and trailing spaces the model leaves behind
    repaired = re.sub(r"[ \t]+$", "", repaired, flags=re.MULTILINE)
    repaired = re.sub(r"\n{3,}", "\n\n", repaired)
    return repaired.strip()
