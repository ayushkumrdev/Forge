"""Tool abstraction. Each tool declares a JSON-schema signature (sent to the
LLM) and implements run(). The registry executes calls defensively: a tool
crash becomes an error result the agent can react to, never an unhandled
exception that kills the run."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from forge.llm.base import ToolSpec
from forge.safety.guard import SafetyViolation

_TRUNCATION_NOTICE = "\n... [output truncated by Forge] ...\n"


class ToolResult(BaseModel):
    ok: bool
    output: str = ""
    error: str | None = None

    def render(self, max_chars: int = 8_000) -> str:
        """The text the model sees as the tool response."""
        text = self.output if self.ok else f"ERROR: {self.error}"
        if len(text) <= max_chars:
            return text
        half = (max_chars - len(_TRUNCATION_NOTICE)) // 2
        return text[:half] + _TRUNCATION_NOTICE + text[-half:]


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]
    mutating: bool = False  # True when the tool changes files or system state

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult: ...


def _argument_error(
    tool: Tool, name: str, arguments: dict[str, Any], exc: TypeError
) -> str:
    """Say which argument is wrong, in the tool's own vocabulary.

    The raw TypeError leaks a Python signature — "run() missing 1 required
    positional argument: 'path'" — which tells the model nothing it can act
    on. Every other failure in Forge grounds the model in what to do next;
    this one should too."""
    schema = tool.parameters or {}
    required = list(schema.get("required") or [])
    known = list((schema.get("properties") or {}).keys())
    missing = [p for p in required if p not in arguments]
    unexpected = [p for p in arguments if known and p not in known]

    parts = []
    if missing:
        parts.append(f"missing required argument(s): {', '.join(missing)}")
    if unexpected:
        parts.append(f"unexpected argument(s): {', '.join(unexpected)}")
    if not parts:
        parts.append(str(exc))
    return (
        f"{name} was called incorrectly — {'; '.join(parts)}. "
        f"It takes: {', '.join(known) if known else 'no arguments'}"
        + (f" (required: {', '.join(required)})" if required else "")
        + f". You sent: {', '.join(arguments) or 'nothing'}. Call it again with "
        "every required argument."
    )


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None, policy=None) -> None:
        self._tools: dict[str, Tool] = {}
        self.policy = policy  # forge.safety.permissions.PermissionPolicy | None
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def is_mutating(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool is not None and tool.mutating

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                error=f"Unknown tool {name!r}. Available tools: {', '.join(self._tools)}",
            )
        if self.policy is not None:
            denial = self.policy.check(name, tool.mutating, arguments)
            if denial:
                return ToolResult(ok=False, error=denial)
        try:
            return tool.run(**arguments)
        except SafetyViolation as exc:
            return ToolResult(ok=False, error=str(exc))
        except TypeError as exc:
            return ToolResult(ok=False, error=_argument_error(tool, name, arguments, exc))
        except Exception as exc:  # noqa: BLE001 — tool crashes must not kill the run
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
