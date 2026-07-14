"""Chat API for the desktop app: one active session, worked on a background
thread, polled by the UI. Permission requests block the worker until the user
answers through the approval endpoint — the GUI equivalent of Claude Code's
'Allow?' prompt."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from forge.chat import commands as chat_commands
from forge.chat.session import ChatSession
from forge.config import ForgeSettings
from forge.llm.base import LLMClient
from forge.safety.permissions import PermissionPolicy
from forge.telemetry import Recorder

# factory(model_name_or_None) -> client; injectable for tests
ChatLLMFactory = Callable[[str | None], LLMClient]

_APPROVAL_TIMEOUT_S = 600.0


class UIApprover:
    """Blocks the agent's worker thread until the UI answers (or times out)."""

    def __init__(self) -> None:
        self.pending: dict[str, str] | None = None
        self._decision = threading.Event()
        self._approved = False

    def __call__(self, tool_name: str, detail: str) -> bool:
        self._decision.clear()
        self._approved = False
        self.pending = {"tool": tool_name, "detail": detail}
        try:
            answered = self._decision.wait(timeout=_APPROVAL_TIMEOUT_S)
            return answered and self._approved
        finally:
            self.pending = None

    def resolve(self, approved: bool) -> None:
        self._approved = approved
        self._decision.set()


class ManagedChat:
    def __init__(
        self, chat_id: str, session: ChatSession, approver: UIApprover,
        policy: PermissionPolicy, workspace: Path, model: str, mode: str,
        events: list[dict[str, Any]],
    ) -> None:
        self.id = chat_id
        self.session = session
        self.approver = approver
        self.policy = policy
        self.workspace = workspace
        self.model = model
        self.mode = mode
        self.status = "idle"  # idle | working
        self.messages: list[dict[str, Any]] = []
        self.events = events


class ChatManager:
    def __init__(
        self,
        default_workspace: Path,
        settings: ForgeSettings,
        llm_factory: ChatLLMFactory,
    ) -> None:
        self.default_workspace = default_workspace
        self.settings = settings
        self._llm_factory = llm_factory
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forge-chat")
        self.current: ManagedChat | None = None

    def start(self, workspace: Path | None, model: str | None, mode: str) -> ManagedChat:
        workspace = (workspace or self.default_workspace).resolve()
        if not workspace.is_dir():
            raise ValueError(f"Not a directory: {workspace}")
        # a previous session blocked on an approval would jam the single
        # worker thread — deny it so the new session can start cleanly
        if self.current is not None and self.current.approver.pending:
            self.current.approver.resolve(False)
        approver = UIApprover()
        policy = (
            PermissionPolicy("auto") if mode == "auto" else PermissionPolicy("ask", approver)
        )
        llm = self._llm_factory(model)
        resolved_model = getattr(llm, "model", model or self.settings.model)
        chat_id = uuid.uuid4().hex[:12]
        events: list[dict[str, Any]] = []
        recorder = Recorder(
            f"app-{chat_id}", workspace, console=None, sink=events.append
        )
        session = ChatSession(
            workspace,
            llm,
            self.settings,
            policy=policy,
            recorder=recorder,
            session_id=f"app-{chat_id}",
        )
        self.current = ManagedChat(
            chat_id, session, approver, policy, workspace, resolved_model, mode, events
        )
        return self.current

    # -- turn execution -----------------------------------------------------------

    def submit(self, text: str) -> None:
        managed = self._require()
        if managed.status == "working":
            raise RuntimeError("A turn is already in progress.")
        managed.messages.append({"role": "user", "text": text})
        managed.status = "working"
        self._executor.submit(self._work, managed, text)

    def _work(self, managed: ManagedChat, text: str) -> None:
        try:
            if chat_commands.is_command(text):
                result = chat_commands.execute(managed.session, text)
                reply, kind = result.text, "command"
            else:
                reply, kind = managed.session.send(text), "chat"
            managed.messages.append({"role": "assistant", "text": reply, "kind": kind})
        except Exception as exc:  # noqa: BLE001 — surface, never crash the app
            managed.messages.append(
                {"role": "assistant", "kind": "error", "text": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            managed.status = "idle"

    def _require(self) -> ManagedChat:
        if self.current is None:
            raise RuntimeError("No active session. Select a folder first.")
        return self.current

    def shutdown(self) -> None:
        if self.current and self.current.approver.pending:
            self.current.approver.resolve(False)
        self._executor.shutdown(wait=False, cancel_futures=True)


# -- API models -----------------------------------------------------------------


class StartRequest(BaseModel):
    workspace: str | None = None
    model: str | None = None
    mode: str = Field(default="ask", pattern="^(ask|auto)$")


class MessageRequest(BaseModel):
    text: str = Field(min_length=1)


class ApprovalRequest(BaseModel):
    approved: bool
    always: bool = False


class ModelRequest(BaseModel):
    model: str = Field(min_length=1)


def build_chat_router(manager: ChatManager) -> APIRouter:
    router = APIRouter(prefix="/api/chat")

    @router.post("/start")
    def start(body: StartRequest) -> dict:
        try:
            managed = manager.start(
                Path(body.workspace) if body.workspace else None, body.model, body.mode
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _state(managed)

    @router.get("/updates")
    def updates(events_after: int = 0, messages_after: int = 0) -> dict:
        managed = _current()
        state = _state(managed)
        state["events"] = managed.events[events_after:]
        state["messages"] = managed.messages[messages_after:]
        return state

    @router.post("/message", status_code=202)
    def message(body: MessageRequest) -> dict:
        managed = _current()
        try:
            manager.submit(body.text)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return _state(managed)

    @router.post("/approval")
    def approval(body: ApprovalRequest) -> dict:
        managed = _current()
        pending = managed.approver.pending
        if pending is None:
            raise HTTPException(409, "Nothing is waiting for approval.")
        if body.approved and body.always:
            managed.policy.allow_always(pending["tool"])
        managed.approver.resolve(body.approved)
        return {"ok": True}

    @router.post("/model")
    def switch_model(body: ModelRequest) -> dict:
        managed = _current()
        if not hasattr(managed.session.llm, "model"):
            raise HTTPException(400, "This LLM client does not support model switching.")
        managed.session.llm.model = body.model
        managed.model = body.model
        return _state(managed)

    @router.post("/undo")
    def undo() -> dict:
        restored = _current().session.undo()
        return {"restored": restored}

    @router.get("/diff")
    def diff() -> dict:
        return {"diff": _current().session.ledger.unified_diff()}

    def _current() -> ManagedChat:
        if manager.current is None:
            raise HTTPException(409, "No active session. Select a folder first.")
        return manager.current

    def _state(managed: ManagedChat) -> dict:
        return {
            "session_id": managed.id,
            "workspace": str(managed.workspace),
            "model": managed.model,
            "mode": managed.mode,
            "status": managed.status,
            "pending_approval": managed.approver.pending,
            "message_count": len(managed.messages),
            "event_count": len(managed.events),
            "changed_files": managed.session.ledger.changed_files,
        }

    return router
