"""Terminal tools: run shell / PowerShell commands inside the workspace with a
timeout, after the safety guard has vetoed destructive patterns."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from forge.safety.guard import SafetyGuard
from forge.tools.base import Tool, ToolResult


def _run_process(
    args: list[str] | str, shell: bool, cwd: Path, timeout: float
) -> ToolResult:
    try:
        completed = subprocess.run(
            args,
            shell=shell,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, error=f"Command timed out after {timeout:.0f}s.")
    except FileNotFoundError as exc:
        return ToolResult(ok=False, error=f"Interpreter not found: {exc}")

    parts = [f"exit code: {completed.returncode}"]
    if completed.stdout:
        parts.append(f"stdout:\n{completed.stdout.rstrip()}")
    if completed.stderr:
        parts.append(f"stderr:\n{completed.stderr.rstrip()}")
    # A non-zero exit is still useful information for the agent, so ok=True;
    # the exit code is in the output for it to reason about.
    return ToolResult(ok=True, output="\n".join(parts))


class RunCommandTool(Tool):
    name = "run_command"
    mutating = True
    description = (
        "Run a shell command in the repository root (e.g. tests, linters, "
        "build steps). Returns exit code, stdout and stderr. Destructive "
        "commands are blocked."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run."},
            "timeout_s": {
                "type": "number",
                "description": "Seconds before the command is killed (default 300).",
            },
        },
        "required": ["command"],
    }

    def __init__(
        self, guard: SafetyGuard, workspace: Path, default_timeout_s: float = 300.0
    ) -> None:
        self._guard = guard
        self._workspace = workspace
        self._default_timeout_s = default_timeout_s

    def run(self, command: str, timeout_s: float | None = None) -> ToolResult:
        self._guard.check_command(command)
        timeout = min(timeout_s or self._default_timeout_s, 600.0)
        return _run_process(command, shell=True, cwd=self._workspace, timeout=timeout)


class PowerShellTool(Tool):
    name = "run_powershell"
    mutating = True
    description = (
        "Run a PowerShell command on the user's Windows machine (repository "
        "root as working directory). Full PowerShell: file operations, "
        "Get-ChildItem, environment, processes, package managers. Returns "
        "exit code, stdout and stderr. Destructive commands are blocked."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The PowerShell command to run."},
            "timeout_s": {
                "type": "number",
                "description": "Seconds before the command is killed (default 300).",
            },
        },
        "required": ["command"],
    }

    def __init__(
        self, guard: SafetyGuard, workspace: Path, default_timeout_s: float = 300.0
    ) -> None:
        self._guard = guard
        self._workspace = workspace
        self._default_timeout_s = default_timeout_s

    def run(self, command: str, timeout_s: float | None = None) -> ToolResult:
        self._guard.check_command(command)
        timeout = min(timeout_s or self._default_timeout_s, 600.0)
        return _run_process(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            shell=False,
            cwd=self._workspace,
            timeout=timeout,
        )
