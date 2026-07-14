"""Chat API for the desktop app: multiple conversations, background turn
execution with token streaming, message queueing, soft cancel, and blocking
permission approvals — the GUI equivalent of Claude Code's 'Allow?' prompt."""

from __future__ import annotations

import json
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
_MAX_RECENT_FOLDERS = 8


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
        self.queue: list[str] = []
        self.cancel_requested = False
        self._partial: list[str] = []
        session.on_stream = self._partial.append
        session.on_step = self._partial.clear
        session.should_stop = lambda: self.cancel_requested

    @property
    def title(self) -> str:
        for message in self.messages:
            if message["role"] == "user":
                text = message["text"].strip().splitlines()[0]
                return text[:60] + ("…" if len(text) > 60 else "")
        return "New conversation"

    @property
    def partial_text(self) -> str:
        """Live streamed text for the UI. Hidden while the model is emitting a
        JSON tool call (that is machinery, not an answer)."""
        text = "".join(self._partial)
        stripped = text.lstrip()
        if stripped.startswith(("{", "<", "```json")):
            return ""
        return text

    def request_cancel(self) -> None:
        self.cancel_requested = True
        self.queue.clear()
        if self.approver.pending:
            self.approver.resolve(False)


class ChatManager:
    def __init__(
        self,
        default_workspace: Path,
        settings: ForgeSettings,
        llm_factory: ChatLLMFactory,
        state_path: Path | None = None,
    ) -> None:
        self.default_workspace = default_workspace
        self.settings = settings
        self._llm_factory = llm_factory
        self._state_path = state_path
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forge-chat")
        self.sessions: dict[str, ManagedChat] = {}
        self.current: ManagedChat | None = None

    def start(self, workspace: Path | None, model: str | None, mode: str) -> ManagedChat:
        workspace = (workspace or self.default_workspace).resolve()
        if not workspace.is_dir():
            raise ValueError(f"Not a directory: {workspace}")
        # a session blocked on an approval would jam the single worker
        # thread — deny it so the new session can start cleanly
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
        managed = ManagedChat(
            chat_id, session, approver, policy, workspace, resolved_model, mode, events
        )
        self.sessions[chat_id] = managed
        self.current = managed
        self.remember_folder(workspace)
        return managed

    def select(self, chat_id: str) -> ManagedChat:
        managed = self.sessions.get(chat_id)
        if managed is None:
            raise KeyError(chat_id)
        self.current = managed
        return managed

    # -- turn execution -----------------------------------------------------------

    def submit(self, text: str) -> str:
        """Run the turn, or queue it if one is in flight. Returns the disposition."""
        managed = self._require()
        managed.messages.append({"role": "user", "text": text})
        if managed.status == "working":
            managed.queue.append(text)
            return "queued"
        managed.status = "working"
        managed.cancel_requested = False
        self._executor.submit(self._work, managed, text)
        return "started"

    def _work(self, managed: ManagedChat, text: str) -> None:
        while True:
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
            managed._partial.clear()
            if managed.cancel_requested or not managed.queue:
                break
            text = managed.queue.pop(0)
        managed.status = "idle"
        managed.cancel_requested = False

    def _require(self) -> ManagedChat:
        if self.current is None:
            raise RuntimeError("No active session. Select a folder first.")
        return self.current

    # -- recent folders ------------------------------------------------------------

    def recent_folders(self) -> list[str]:
        if self._state_path is None or not self._state_path.exists():
            return []
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            return [p for p in data.get("recent", []) if Path(p).is_dir()]
        except (OSError, json.JSONDecodeError):
            return []

    def remember_folder(self, workspace: Path) -> None:
        if self._state_path is None:
            return
        recent = [str(workspace)] + [p for p in self.recent_folders() if p != str(workspace)]
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps({"recent": recent[:_MAX_RECENT_FOLDERS]}), encoding="utf-8"
            )
        except OSError:
            pass

    def shutdown(self) -> None:
        for managed in self.sessions.values():
            if managed.approver.pending:
                managed.approver.resolve(False)
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


class SelectRequest(BaseModel):
    session_id: str


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

    @router.get("/sessions")
    def sessions() -> list[dict]:
        return [
            {
                "session_id": managed.id,
                "title": managed.title,
                "workspace": str(managed.workspace),
                "model": managed.model,
                "status": managed.status,
                "current": manager.current is managed,
                "message_count": len(managed.messages),
            }
            for managed in reversed(list(manager.sessions.values()))
        ]

    @router.post("/select")
    def select(body: SelectRequest) -> dict:
        try:
            return _state(manager.select(body.session_id))
        except KeyError as exc:
            raise HTTPException(404, f"Unknown session {body.session_id}") from exc

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
        disposition = manager.submit(body.text)
        state = _state(managed)
        state["disposition"] = disposition
        return state

    @router.post("/stop")
    def stop() -> dict:
        managed = _current()
        managed.request_cancel()
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

    @router.get("/recent")
    def recent() -> list[str]:
        return manager.recent_folders()

    def _current() -> ManagedChat:
        if manager.current is None:
            raise HTTPException(409, "No active session. Select a folder first.")
        return manager.current

    def _state(managed: ManagedChat) -> dict:
        return {
            "session_id": managed.id,
            "title": managed.title,
            "workspace": str(managed.workspace),
            "model": managed.model,
            "mode": managed.mode,
            "status": managed.status,
            "partial": managed.partial_text if managed.status == "working" else "",
            "queued": len(managed.queue),
            "pending_approval": managed.approver.pending,
            "message_count": len(managed.messages),
            "event_count": len(managed.events),
            "changed_files": managed.session.ledger.changed_files,
            "usage": {
                "prompt_tokens": managed.session.usage.prompt_tokens,
                "completion_tokens": managed.session.usage.completion_tokens,
            },
        }

    return router
