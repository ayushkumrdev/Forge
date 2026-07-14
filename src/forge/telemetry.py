"""Telemetry: every agent step, tool call and LLM exchange is recorded to the
console (live progress), a JSONL trace file, and the SQLite memory store."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from forge.memory.store import MemoryStore


class Recorder:
    def __init__(
        self,
        run_id: str,
        workspace: Path,
        store: MemoryStore | None = None,
        console: Console | None = None,
        verbose: bool = True,
        sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.run_id = run_id
        self._store = store
        self._console = console
        self._verbose = verbose
        self._sink = sink
        log_dir = workspace / ".forge" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = log_dir / f"{run_id}.jsonl"

    def event(self, agent: str, kind: str, **payload: Any) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "agent": agent,
            "kind": kind,
            **payload,
        }
        with self._jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        if self._store is not None:
            self._store.add_event(self.run_id, agent, kind, payload)
        if self._sink is not None:
            self._sink(record)
        if self._console is not None and self._verbose:
            self._console.print(self._format(agent, kind, payload))

    @staticmethod
    def _format(agent: str, kind: str, payload: dict[str, Any]) -> str:
        detail = ""
        if kind == "tool_call":
            args = json.dumps(payload.get("arguments", {}), default=str)
            if len(args) > 120:
                args = args[:120] + "…"
            detail = f"{payload.get('tool')} {args}"
        elif kind == "tool_result":
            detail = "ok" if payload.get("ok") else f"error: {str(payload.get('error'))[:120]}"
        elif kind == "llm_response":
            detail = (
                f"{payload.get('completion_tokens', 0)} tokens "
                f"in {payload.get('duration_ms', 0):.0f} ms"
            )
        elif "message" in payload:
            detail = str(payload["message"])[:160]
        return f"[dim]{agent}[/dim] [bold cyan]{kind}[/bold cyan] {detail}"
