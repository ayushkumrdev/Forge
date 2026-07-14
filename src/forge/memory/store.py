"""Persistent execution memory, milestone 1: an SQLite store per workspace
(.forge/forge.db) recording runs and every agent/tool event. The interface is
storage-agnostic so PostgreSQL can replace SQLite in a later milestone."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    task        TEXT NOT NULL,
    status      TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    report_json TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   TEXT NOT NULL REFERENCES runs(id),
    ts       TEXT NOT NULL,
    agent    TEXT NOT NULL,
    kind     TEXT NOT NULL,
    payload  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE TABLE IF NOT EXISTS lessons (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    ts         TEXT NOT NULL,
    request    TEXT NOT NULL,
    task_title TEXT NOT NULL,
    status     TEXT NOT NULL,
    issues     TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class MemoryStore:
    def __init__(self, workspace: Path) -> None:
        forge_dir = workspace / ".forge"
        forge_dir.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(forge_dir / "forge.db")
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def start_run(self, task: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        self._connection.execute(
            "INSERT INTO runs (id, task, status, started_at) VALUES (?, ?, 'running', ?)",
            (run_id, task, _now()),
        )
        self._connection.commit()
        return run_id

    def finish_run(self, run_id: str, status: str, report: dict[str, Any] | None = None) -> None:
        self._connection.execute(
            "UPDATE runs SET status = ?, finished_at = ?, report_json = ? WHERE id = ?",
            (status, _now(), json.dumps(report) if report else None, run_id),
        )
        self._connection.commit()

    def add_event(self, run_id: str, agent: str, kind: str, payload: dict[str, Any]) -> None:
        self._connection.execute(
            "INSERT INTO events (run_id, ts, agent, kind, payload) VALUES (?, ?, ?, ?, ?)",
            (run_id, _now(), agent, kind, json.dumps(payload, default=str)),
        )
        self._connection.commit()

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT id, task, status, started_at, finished_at FROM runs "
            "ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": row[0],
                "task": row[1],
                "status": row[2],
                "started_at": row[3],
                "finished_at": row[4],
            }
            for row in rows
        ]

    def events_for_run(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT ts, agent, kind, payload FROM events WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
        return [
            {"ts": row[0], "agent": row[1], "kind": row[2], "payload": json.loads(row[3])}
            for row in rows
        ]

    def add_lesson(
        self,
        run_id: str,
        request: str,
        task_title: str,
        status: str,
        issues: list[str],
    ) -> None:
        self._connection.execute(
            "INSERT INTO lessons (run_id, ts, request, task_title, status, issues) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, _now(), request, task_title, status, json.dumps(issues)),
        )
        self._connection.commit()

    def lessons(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT ts, run_id, request, task_title, status, issues "
            "FROM lessons ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "ts": row[0],
                "run_id": row[1],
                "request": row[2],
                "task_title": row[3],
                "status": row[4],
                "issues": json.loads(row[5]),
            }
            for row in rows
        ]

    def close(self) -> None:
        self._connection.close()
