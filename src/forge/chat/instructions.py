"""Project instructions: a FORGE.md (or
CLAUDE.md) at the repository root is loaded into every agent's system prompt."""

from __future__ import annotations

from pathlib import Path

_INSTRUCTION_FILES = ("FORGE.md", "CLAUDE.md")
_MAX_CHARS = 6_000


def load_project_instructions(workspace: Path) -> str:
    for name in _INSTRUCTION_FILES:
        candidate = workspace / name
        if candidate.exists():
            try:
                text = candidate.read_text(encoding="utf-8-sig", errors="replace").strip()
            except OSError:
                continue
            if text:
                if len(text) > _MAX_CHARS:
                    text = text[:_MAX_CHARS] + "\n... [instructions truncated]"
                return f"## Project instructions (from {name})\n{text}"
    return ""
