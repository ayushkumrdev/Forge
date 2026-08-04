"""FastAPI application: REST endpoints over Forge's stores and execution loop,
plus the bundled single-page dashboard at /."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from forge import __version__
from forge.config import ForgeSettings
from forge.llm.factory import make_client
from forge.memory.store import MemoryStore
from forge.repo.scanner import RepoScanner
from forge.server.chat_api import ChatLLMFactory, ChatManager, build_chat_router
from forge.server.runs import LLMFactory, RunManager

_STATIC_DIR = Path(__file__).parent / "static"


class RunRequest(BaseModel):
    request: str = Field(min_length=3, description="What Forge should do.")
    check_commands: list[str] = Field(default_factory=list)


def create_app(
    workspace: Path,
    settings: ForgeSettings | None = None,
    llm_factory: LLMFactory | None = None,
    chat_llm_factory: ChatLLMFactory | None = None,
    app_state_path: Path | None = None,
) -> FastAPI:
    workspace = workspace.resolve()
    settings = settings or ForgeSettings()
    if app_state_path is None:
        app_state_path = Path.home() / ".forge" / "app_state.json"

    def default_llm():
        return make_client(settings)

    def default_chat_llm(model: str | None):
        return make_client(settings, model)

    manager = RunManager(workspace, settings, llm_factory or default_llm)
    chat_manager = ChatManager(
        workspace, settings, chat_llm_factory or default_chat_llm, state_path=app_state_path
    )
    app = FastAPI(title="Forge", version=__version__)
    app.state.manager = manager
    app.state.chat_manager = chat_manager
    app.include_router(build_chat_router(chat_manager))

    # -- meta -------------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict:
        client = make_client(settings)
        backend_ok = client.ping()
        model_ok = False
        if backend_ok:
            try:
                models = client.list_models()
                model_ok = settings.model in models
            except Exception:  # noqa: BLE001 — health must never raise
                pass
        return {
            "version": __version__,
            "workspace": str(workspace),
            "model": settings.model,
            "provider": settings.provider,
            "ollama_reachable": backend_ok,  # legacy key the dashboard reads
            "model_available": model_ok,
        }

    @app.get("/api/models")
    def models() -> list[str]:
        """Installed models — the app's model-switcher dropdown."""
        try:
            return make_client(settings).list_models()
        except Exception:  # noqa: BLE001 — no backend, empty dropdown
            return []

    # -- repository -------------------------------------------------------------

    @app.get("/api/repo")
    def repo() -> dict:
        snapshot = RepoScanner(
            workspace,
            max_tree_entries=settings.max_tree_entries,
            cache_path=workspace / ".forge" / "repo_index.json",
        ).scan()
        return {
            "root": snapshot.root,
            "files": len(snapshot.files),
            "tree": snapshot.tree,
            "language_stats": snapshot.language_stats,
        }

    # -- runs ---------------------------------------------------------------------

    @app.post("/api/runs", status_code=202)
    def submit_run(body: RunRequest) -> dict:
        run_id = manager.submit(body.request, body.check_commands)
        return {"run_id": run_id}

    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        store = MemoryStore(workspace)
        try:
            return store.recent_runs(limit=50)
        finally:
            store.close()

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict:
        store = MemoryStore(workspace)
        try:
            runs = {entry["id"]: entry for entry in store.recent_runs(limit=500)}
        finally:
            store.close()
        entry = runs.get(run_id)
        if entry is None:
            raise HTTPException(404, f"run {run_id} not found")
        report_path = workspace / ".forge" / "runs" / f"{run_id}.json"
        report = None
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = None
        return {**entry, "report": report}

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str, offset: int = 0) -> list[dict]:
        store = MemoryStore(workspace)
        try:
            return store.events_for_run(run_id)[offset:]
        finally:
            store.close()

    # -- memory -------------------------------------------------------------------

    @app.get("/api/memory")
    def memory() -> list[dict]:
        store = MemoryStore(workspace)
        try:
            return store.lessons(limit=100)
        finally:
            store.close()

    # -- dashboard ------------------------------------------------------------------

    # The UI is served from disk and changes whenever Forge is updated, so it
    # must never be cached: a stale copy silently hides new features and
    # re-introduces fixed bugs, with nothing on screen to explain why.
    _NO_CACHE = {"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"}

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html", headers=_NO_CACHE)

    @app.get("/app", include_in_schema=False)
    def desktop_app() -> FileResponse:
        return FileResponse(_STATIC_DIR / "app.html", headers=_NO_CACHE)

    return app
