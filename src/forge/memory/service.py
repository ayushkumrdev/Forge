"""Execution memory: turns past task outcomes into planning context.

Every finished task is recorded as a lesson (request, task, verdict, reviewer
issues). Before planning a new request, the most similar past lessons are
retrieved with BM25 and rendered into the planner prompt — so a mistake the
reviewer flagged last week is in front of the planner this week."""

from __future__ import annotations

from typing import Any

from forge.memory.store import MemoryStore
from forge.retrieval.bm25 import BM25Index

_MAX_ISSUES_SHOWN = 3


class ExecutionMemory:
    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def record_task(
        self, run_id: str, request: str, task_title: str, status: str, issues: list[str]
    ) -> None:
        self._store.add_lesson(run_id, request, task_title, status, issues)

    def lessons_for(self, request: str, k: int = 3) -> list[dict[str, Any]]:
        lessons = self._store.lessons()
        if not lessons:
            return []
        index = BM25Index()
        index.add_documents(
            [
                f"{lesson['request']} {lesson['task_title']} {' '.join(lesson['issues'])}"
                for lesson in lessons
            ]
        )
        return [lessons[i] for i, _score in index.top(request, k)]

    @staticmethod
    def render(lessons: list[dict[str, Any]]) -> str:
        """Prompt-ready digest of past outcomes; empty string when nothing useful."""
        if not lessons:
            return ""
        blocks: list[str] = []
        for lesson in lessons:
            lines = [
                f"- Past task: {lesson['task_title']} "
                f"(request: {lesson['request'][:120]}) -> {lesson['status']}"
            ]
            lines.extend(
                f"  issue seen: {issue}" for issue in lesson["issues"][:_MAX_ISSUES_SHOWN]
            )
            blocks.append("\n".join(lines))
        return "\n".join(blocks)
