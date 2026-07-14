"""Planner agent: turns a user request plus the repository snapshot into a
small, ordered list of concrete engineering tasks with complexity estimates."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from forge.agents.structured import structured_call
from forge.llm.base import LLMClient, Usage
from forge.telemetry import Recorder

PLANNER_SYSTEM = """You are the Planner agent of Forge, an autonomous AI software engineer.

Given a user request and a repository overview, produce a minimal, ordered
execution plan. Rules:
- Prefer ONE task unless the request genuinely needs independent steps.
- Each task must be concrete and independently verifiable.
- Only include work the user asked for. No speculative refactors.
- target_files lists files you expect to create or modify (best guess).

Respond with ONLY a JSON object in this exact shape:
{
  "summary": "<one sentence describing the overall approach>",
  "tasks": [
    {
      "id": 1,
      "title": "<short imperative title>",
      "description": "<precise instructions for the coder: file paths, acceptance criteria>",
      "target_files": ["path/one.py"],
      "complexity": "low" | "medium" | "high"
    }
  ]
}"""


class PlanTask(BaseModel):
    id: int
    title: str
    description: str
    target_files: list[str] = Field(default_factory=list)
    complexity: Literal["low", "medium", "high"] = "medium"


class Plan(BaseModel):
    summary: str
    tasks: list[PlanTask]


class Planner:
    name = "planner"

    def __init__(self, llm: LLMClient, recorder: Recorder, max_tasks: int = 8) -> None:
        self._llm = llm
        self._recorder = recorder
        self._max_tasks = max_tasks
        self.usage = Usage()

    def plan(self, request: str, repo_summary: str) -> Plan:
        self._recorder.event(self.name, "planning_started", message=request)
        user_message = (
            f"## User request\n{request}\n\n## Repository overview\n{repo_summary}"
        )
        plan = structured_call(
            self._llm, PLANNER_SYSTEM, user_message, Plan, usage=self.usage
        )
        if len(plan.tasks) > self._max_tasks:
            plan.tasks = plan.tasks[: self._max_tasks]
        self._recorder.event(
            self.name,
            "plan_ready",
            summary=plan.summary,
            tasks=[task.title for task in plan.tasks],
        )
        return plan
