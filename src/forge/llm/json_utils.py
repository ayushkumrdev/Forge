"""Lenient JSON extraction for structured agent outputs.

Local models often wrap JSON in prose or markdown fences; this recovers the
first valid JSON object rather than failing on decoration around it. It also
recovers tool calls that models emit as JSON text instead of using the
provider's native tool-calling field (common with local model templates)."""

from __future__ import annotations

import json
import re
from typing import Any

from forge.llm.base import ToolCall


class JSONExtractionError(ValueError):
    pass


_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Extract the first JSON object/array from text.

    Tries, in order: the whole string, fenced code blocks, then the first
    balanced {...} or [...] span.
    """
    candidates = [text.strip()]
    candidates.extend(m.group(1).strip() for m in _FENCE_RE.finditer(text))
    balanced = _first_balanced(text)
    if balanced:
        candidates.append(balanced)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise JSONExtractionError(f"No valid JSON found in model output: {text[:200]!r}")


def extract_tool_call(text: str) -> ToolCall | None:
    """Recover a tool call a model emitted as JSON text in its content.

    Accepts {"name": ..., "arguments": {...}} directly, inside markdown
    fences, or inside Qwen-style <tool_call> tags. Returns None when the text
    is not a tool invocation (callers must also check the name is a known
    tool before acting on it)."""
    try:
        data = extract_json(text)
    except JSONExtractionError:
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    arguments = data.get("arguments", data.get("parameters", {}))
    if isinstance(name, str) and name and isinstance(arguments, dict):
        return ToolCall(name=name, arguments=arguments)
    return None


def _first_balanced(text: str) -> str | None:
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None
