"""Permission policy: read-only tools always run; mutating
tools (file writes, shell commands, git) either run automatically ('auto', the
default for autonomous runs) or require interactive approval ('ask', the
default for chat). Denial is not an error — the model receives it as a tool
result and can adjust."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

PermissionMode = Literal["auto", "ask"]

# Every denial message starts with this. It lets callers tell "the user said
# no" apart from "the tool failed" — a distinction that matters, because
# Forge must never push the model to retry something the user refused.
DENIAL_PREFIX = "Permission denied by the user"

# approver(tool_name, human-readable detail) -> True to allow
Approver = Callable[[str, str], bool]


class PermissionPolicy:
    def __init__(self, mode: PermissionMode = "auto", approver: Approver | None = None) -> None:
        if mode == "ask" and approver is None:
            raise ValueError("ask mode requires an approver callback")
        self.mode = mode
        self._approver = approver
        self._always_allowed: set[str] = set()

    def allow_always(self, tool_name: str) -> None:
        """User chose 'always allow' for this tool in this session."""
        self._always_allowed.add(tool_name)

    def check(self, tool_name: str, mutating: bool, arguments: dict[str, Any]) -> str | None:
        """Return None to allow, or a denial message the model will see."""
        if not mutating or self.mode == "auto" or tool_name in self._always_allowed:
            return None
        detail = json.dumps(arguments, default=str)
        if len(detail) > 300:
            detail = detail[:300] + "…"
        if self._approver(tool_name, detail):
            return None
        return (
            f"{DENIAL_PREFIX} for {tool_name}. Do not retry the "
            "same call; ask the user how to proceed or try a different approach."
        )
