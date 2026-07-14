"""Background run execution for the API server.

Runs execute on a single worker thread (one autonomous run at a time — they
mutate the repository). Every thread opens its own MemoryStore because sqlite
connections are not shareable across threads."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from forge.config import ForgeSettings
from forge.llm.base import LLMClient
from forge.memory.store import MemoryStore
from forge.orchestrator.loop import ExecutionLoop
from forge.telemetry import Recorder

LLMFactory = Callable[[], LLMClient]


class RunManager:
    def __init__(
        self,
        workspace: Path,
        settings: ForgeSettings,
        llm_factory: LLMFactory,
    ) -> None:
        self.workspace = workspace
        self.settings = settings
        self._llm_factory = llm_factory
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forge-run")

    def submit(self, request: str, check_commands: list[str]) -> str:
        store = MemoryStore(self.workspace)
        run_id = store.start_run(request)
        store.close()
        self._executor.submit(self._execute, run_id, request, check_commands)
        return run_id

    def _execute(self, run_id: str, request: str, check_commands: list[str]) -> None:
        store = MemoryStore(self.workspace)
        try:
            recorder = Recorder(run_id, self.workspace, store=store, console=None)
            loop = ExecutionLoop(
                workspace=self.workspace,
                llm=self._llm_factory(),
                settings=self.settings,
                recorder=recorder,
                store=store,
                run_id=run_id,
                check_commands=check_commands,
            )
            report = loop.run(request)
            store.finish_run(run_id, report.status, report.model_dump())
        except Exception as exc:  # noqa: BLE001 — server must survive any run failure
            store.finish_run(run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            store.close()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
