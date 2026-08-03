"""The execution loop:

    scan repo -> plan -> for each task:
        code -> run checks -> review -> (feedback -> code again) -> done

Stopping criteria: reviewer approval, or max_review_cycles exhausted.
Every run produces a RunReport persisted to .forge/runs/<run_id>.json."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, Field

from forge.agents.coder import Coder
from forge.agents.planner import Plan, Planner, PlanTask
from forge.agents.reviewer import Review, Reviewer
from forge.config import ForgeSettings
from forge.llm.base import LLMClient, Usage
from forge.memory.service import ExecutionMemory
from forge.memory.store import MemoryStore
from forge.repo.scanner import RepoScanner
from forge.retrieval.embeddings import OllamaEmbedder
from forge.retrieval.engine import RetrievalEngine
from forge.safety.guard import SafetyGuard
from forge.telemetry import Recorder
from forge.tools.base import ToolRegistry
from forge.tools.changes import ChangeLedger
from forge.tools.code_intel import FindSymbolTool, WhoImportsTool
from forge.tools.filesystem import (
    DeleteFileTool,
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from forge.tools.git_tool import GitTool
from forge.tools.github import GitHubFileTool, GitHubRepoTool
from forge.tools.retrieval_tool import SearchCodeTool
from forge.tools.search import GlobTool, GrepTool
from forge.tools.terminal import PowerShellTool, RunCommandTool
from forge.tools.web import FetchUrlTool


def ground_target_files(plan: Plan, known_files: set[str]) -> None:
    """Grounding: the planner's target_files are best guesses — annotate the
    ones that do not exist so the coder knows it is CREATING a file, not
    editing something the planner hallucinated."""
    for task in plan.tasks:
        task.target_files = [
            path
            if path.replace("\\", "/") in known_files
            else f"{path} (new file — does not exist yet)"
            for path in task.target_files
        ]


def retry_temperature(base: float, attempt: int) -> float | None:
    """Verifier-guided sampling diversity: attempt 1 runs at the configured
    temperature; each retry after a rejection samples hotter, so the coder
    explores instead of deterministically repeating the same failure."""
    if attempt <= 1:
        return None
    return round(min(base + 0.3 * (attempt - 1), 0.9), 2)


class TaskResult(BaseModel):
    task_id: int
    title: str
    status: str  # "approved" | "rejected" | "failed"
    attempts: int
    review: Review | None = None
    coder_summary: str = ""


class RunReport(BaseModel):
    run_id: str
    request: str
    status: str  # "success" | "partial" | "failed"
    plan_summary: str = ""
    task_results: list[TaskResult] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    diff: str = ""
    usage: Usage = Field(default_factory=Usage)
    duration_s: float = 0.0
    error: str | None = None


class ExecutionLoop:
    def __init__(
        self,
        workspace: Path,
        llm: LLMClient,
        settings: ForgeSettings | None = None,
        recorder: Recorder | None = None,
        store: MemoryStore | None = None,
        run_id: str | None = None,
        check_commands: list[str] | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.settings = settings or ForgeSettings()
        self.store = store
        self.run_id = run_id or "adhoc"
        self.recorder = recorder or Recorder(self.run_id, self.workspace, store=store)
        self.check_commands = check_commands or []

        self._llm = llm
        self._memory = ExecutionMemory(store) if store is not None else None
        self._engine: RetrievalEngine | None = None
        self._guard = SafetyGuard(self.workspace)
        self.ledger = ChangeLedger(self.workspace, self.run_id)
        self._registry = self._build_registry()

        self.planner = Planner(llm, self.recorder, max_tasks=self.settings.max_plan_tasks)
        self.coder = Coder(
            llm,
            self._registry,
            self.recorder,
            max_steps=self.settings.max_agent_steps,
            max_tool_output_chars=self.settings.max_tool_output_chars,
        )
        self.reviewer = Reviewer(llm, self.recorder)

    def _build_registry(self) -> ToolRegistry:
        tools = [
            ReadFileTool(self._guard),
            WriteFileTool(self._guard, self.ledger, self.settings.syntax_gate),
            EditFileTool(self._guard, self.ledger, self.settings.syntax_gate),
            DeleteFileTool(self._guard, self.ledger),
            ListDirTool(self._guard),
            RunCommandTool(
                self._guard, self.workspace, self.settings.command_timeout_s
            ),
            GrepTool(self.workspace),
            GlobTool(self.workspace),
            GitTool(self.workspace),
            FetchUrlTool(),
            GitHubRepoTool(self.settings.github_token),
            GitHubFileTool(self.settings.github_token),
        ]
        if os.name == "nt":
            tools.append(
                PowerShellTool(self._guard, self.workspace, self.settings.command_timeout_s)
            )
        return ToolRegistry(tools)

    def run(self, request: str) -> RunReport:
        started = time.monotonic()
        report = RunReport(run_id=self.run_id, request=request, status="failed")
        try:
            snapshot = RepoScanner(
                self.workspace,
                max_tree_entries=self.settings.max_tree_entries,
                cache_path=self.workspace / ".forge" / "repo_index.json",
            ).scan()
            repo_summary = snapshot.summary(self.settings.max_summary_chars)
            self.recorder.event(
                "orchestrator", "repo_scanned", files=len(snapshot.files)
            )
            # code-intelligence and retrieval tools need the fresh snapshot
            self._registry.register(FindSymbolTool(snapshot))
            self._registry.register(WhoImportsTool(snapshot))
            engine = RetrievalEngine(self.workspace, embedder=self._embedder())
            chunk_count = engine.build(snapshot)
            self._engine = engine
            self._registry.register(SearchCodeTool(engine))
            self.recorder.event("orchestrator", "retrieval_ready", chunks=chunk_count)

            lessons = ""
            if self._memory is not None:
                lessons = ExecutionMemory.render(self._memory.lessons_for(request))
                if lessons:
                    self.recorder.event("orchestrator", "lessons_recalled")

            plan = self.planner.plan(request, repo_summary, lessons)
            ground_target_files(
                plan, {f.path.replace("\\", "/") for f in snapshot.files}
            )
            report.plan_summary = plan.summary
            report.usage.add(self.planner.usage)

            for task in plan.tasks:
                result = self._execute_task(task, plan, repo_summary, report.usage)
                report.task_results.append(result)
                if self._memory is not None:
                    self._memory.record_task(
                        self.run_id,
                        request,
                        task.title,
                        result.status,
                        result.review.issues if result.review else [],
                    )
                if result.status != "approved":
                    # later tasks likely depend on this one; don't build on a
                    # change the reviewer refused to approve
                    break

            report.changed_files = self.ledger.changed_files
            report.diff = self.ledger.unified_diff()
            statuses = {result.status for result in report.task_results}
            if statuses == {"approved"} and report.task_results:
                report.status = "success"
            elif "approved" in statuses:
                report.status = "partial"
        except Exception as exc:  # noqa: BLE001 — a run failure must produce a report
            report.error = f"{type(exc).__name__}: {exc}"
            self.recorder.event("orchestrator", "run_error", message=report.error)

        report.usage.add(self.reviewer.usage)
        report.duration_s = round(time.monotonic() - started, 2)
        self._persist(report)
        return report

    def _execute_task(
        self, task: PlanTask, plan: Plan, repo_summary: str, usage: Usage
    ) -> TaskResult:
        self.recorder.event("orchestrator", "task_started", task=task.title)
        feedback: str | None = None
        review: Review | None = None
        outcome = None

        # pre-flight: put the real relevant code in front of the coder before
        # it generates, so it cannot hallucinate the repository's APIs
        context = (
            self._engine.preflight(f"{task.title}\n{task.description}") or None
            if self._engine is not None
            else None
        )
        for attempt in range(1, self.settings.max_review_cycles + 1):
            outcome = self.coder.execute(
                task,
                plan,
                repo_summary,
                feedback,
                context,
                temperature=retry_temperature(self.settings.temperature, attempt),
            )
            usage.add(outcome.usage)

            check_results = self._run_checks()
            review = self.reviewer.review(
                task, self.ledger.unified_diff(), check_results, outcome.final_text
            )
            if review.approved:
                self.recorder.event(
                    "orchestrator", "task_approved", task=task.title, attempts=attempt
                )
                return TaskResult(
                    task_id=task.id,
                    title=task.title,
                    status="approved",
                    attempts=attempt,
                    review=review,
                    coder_summary=outcome.final_text,
                )
            feedback = "\n".join(f"- {issue}" for issue in review.issues) or review.summary
            if check_results:
                # give the coder the real check output, not just the reviewer's
                # distillation — the concrete error is the strongest signal
                feedback += (
                    "\n\nOutput of the automated checks that must pass:\n"
                    + check_results[-2500:]
                )
            self.recorder.event(
                "orchestrator", "task_rejected", task=task.title, attempt=attempt
            )

        return TaskResult(
            task_id=task.id,
            title=task.title,
            status="rejected",
            attempts=self.settings.max_review_cycles,
            review=review,
            coder_summary=outcome.final_text if outcome else "",
        )

    def _embedder(self) -> OllamaEmbedder | None:
        """Dense retrieval is optional: only when a model is configured AND
        actually pulled in Ollama; otherwise BM25-only, silently."""
        if not self.settings.embedding_model:
            return None
        embedder = OllamaEmbedder(
            model=self.settings.embedding_model, host=self.settings.ollama_host
        )
        return embedder if embedder.available() else None

    def _run_checks(self) -> str:
        results: list[str] = []
        for command in self.check_commands:
            try:
                self._guard.check_command(command)
                completed = subprocess.run(
                    command,
                    shell=True,
                    cwd=self.workspace,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.settings.command_timeout_s,
                )
                output = (completed.stdout + completed.stderr).strip()
                results.append(
                    f"$ {command}\nexit code: {completed.returncode}\n{output[-3000:]}"
                )
            except Exception as exc:  # noqa: BLE001 — report check failure to reviewer
                results.append(f"$ {command}\nfailed to run: {exc}")
            self.recorder.event("orchestrator", "check_ran", command=command)
        return "\n\n".join(results)

    def _persist(self, report: RunReport) -> None:
        runs_dir = self.workspace / ".forge" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / f"{self.run_id}.json").write_text(
            json.dumps(report.model_dump(), indent=2, default=str), encoding="utf-8"
        )
        self.recorder.event(
            "orchestrator",
            "run_finished",
            status=report.status,
            changed_files=report.changed_files,
            duration_s=report.duration_s,
        )
