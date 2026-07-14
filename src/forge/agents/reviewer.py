"""Reviewer agent: independent from the coder. Judges the actual diff plus
check-command results against the task, and either approves or returns
actionable issues that are fed back to the coder."""

from __future__ import annotations

from pydantic import BaseModel, Field

from forge.agents.planner import PlanTask
from forge.agents.structured import structured_call
from forge.llm.base import LLMClient, Usage
from forge.telemetry import Recorder

REVIEWER_SYSTEM = """You are the Reviewer agent of Forge, an autonomous AI software engineer.
You are independent from the coding agent and must judge its work strictly on
the evidence: the task, the unified diff of actual file changes, and the
results of automated checks.

Reject (approved=false) when:
- The diff does not implement what the task asked for, or is empty when
  changes were required.
- Automated checks failed.
- You can see a concrete bug, syntax error, unresolved import, or the change
  breaks obviously related code.

Approve (approved=true) when the diff plausibly satisfies the task and checks
pass. Do not reject for style preferences or hypothetical improvements.

Respond with ONLY a JSON object:
{
  "approved": true | false,
  "summary": "<one or two sentences on the quality of the change>",
  "issues": ["<specific, actionable issue the coder must fix>", ...]
}
issues must be empty when approved is true."""


class Review(BaseModel):
    approved: bool
    summary: str = ""
    issues: list[str] = Field(default_factory=list)


class Reviewer:
    name = "reviewer"

    def __init__(self, llm: LLMClient, recorder: Recorder) -> None:
        self._llm = llm
        self._recorder = recorder
        self.usage = Usage()

    def review(self, task: PlanTask, diff: str, check_results: str, coder_summary: str) -> Review:
        user_message = "\n\n".join(
            [
                f"## Task under review (task {task.id}: {task.title})\n{task.description}",
                f"## Coder's own summary\n{coder_summary or '(none provided)'}",
                f"## Unified diff of actual changes\n{diff or '(no files were changed)'}",
                f"## Automated check results\n{check_results or '(no checks configured)'}",
            ]
        )
        review = structured_call(
            self._llm, REVIEWER_SYSTEM, user_message, Review, usage=self.usage
        )
        self._recorder.event(
            self.name,
            "review_done",
            approved=review.approved,
            issues=review.issues,
            summary=review.summary,
        )
        return review
