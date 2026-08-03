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
            return ToolResult(ok=False, error=f"Invalid arguments for {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 — tool crashes must not kill the run
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
