"""Coding agent: a tool-loop agent that reads before it writes, makes minimal
targeted changes, and verifies its own work with the terminal when useful."""

from __future__ import annotations

from forge.agents.base import AgentOutcome, ToolLoopAgent
from forge.agents.planner import Plan, PlanTask
from forge.llm.base import LLMClient
from forge.telemetry import Recorder
from forge.tools.base import ToolRegistry

CODER_SYSTEM = """You are the Coding agent of Forge, an autonomous AI software engineer
working directly inside a real repository through tools.

Working rules:
- ALWAYS read a file (read_file) before editing it. Never overwrite blindly.
- Prefer edit_file for changes to existing files; write_file only for new files
  or full rewrites you have just read.
- edit_file old_string must be copied EXACTLY from the read_file output. If an
  edit_file call fails twice on the same file, stop retrying it: read the file
  and call write_file with the complete corrected content instead.
- Use find_symbol to jump to a definition, who_imports to see what depends on
  a file before changing it, and grep/find_files for everything else. Never
  guess paths.
- Make the smallest change that satisfies the task. Follow the existing code
  style of the repository.
- Validate your work: if the repository has tests or the task includes
  acceptance criteria, use run_command to check (e.g. run the test file,
  or `python -m py_compile <file>` for syntax).
- Never invent files, APIs or imports. If something you need is missing,
  check with grep first.
- When your new code uses a symbol (function, class, module), make sure that
  file imports or defines it. Adding a test that calls X requires importing X.
- Never give your final answer while the checks or tests are failing; keep
  fixing until they pass or you genuinely cannot proceed.

When the task is complete, reply in plain text (no tool call) with a short
summary of what you changed and how you verified it."""


class Coder:
    name = "coder"

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        recorder: Recorder,
        max_steps: int = 25,
        max_tool_output_chars: int = 8_000,
    ) -> None:
        self._agent = ToolLoopAgent(
            name=self.name,
            llm=llm,
            registry=registry,
            recorder=recorder,
            max_steps=max_steps,
            max_tool_output_chars=max_tool_output_chars,
        )

    def execute(
        self,
        task: PlanTask,
        plan: Plan,
        repo_summary: str,
        feedback: str | None = None,
        context: str | None = None,
        temperature: float | None = None,
    ) -> AgentOutcome:
        sections = [
            f"## Overall goal\n{plan.summary}",
            f"## Your current task (task {task.id}: {task.title})\n{task.description}",
        ]
        if task.target_files:
            sections.append("## Files likely involved\n" + "\n".join(task.target_files))
        if context:
            sections.append(context)
        sections.append(f"## Repository overview\n{repo_summary}")
        if feedback:
            sections.append(
                "## Reviewer feedback on your previous attempt (fix these issues)\n" + feedback
            )
        return self._agent.run(CODER_SYSTEM, "\n\n".join(sections), temperature=temperature)
