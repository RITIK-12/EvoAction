"""SQLite state + append-only JSONL event log.

Every model call, budget decision, operator run, investigation task, experiment
proposal, workflow version and per-episode evaluation is persisted here so the
report (and a restarted process) can be rebuilt from real records only.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, kind TEXT, status TEXT, started_at REAL, finished_at REAL, meta_json TEXT
);
CREATE TABLE IF NOT EXISTS model_calls (
    id TEXT PRIMARY KEY, run_id TEXT, role TEXT, purpose TEXT, model TEXT, cache_key TEXT,
    cache_hit INTEGER, status TEXT, error TEXT, prompt_tokens INTEGER, completion_tokens INTEGER,
    reasoning_tokens INTEGER, images INTEGER, cost_usd REAL, cost_is_estimate INTEGER,
    latency_s REAL, version_id TEXT, episode_id TEXT, task_id TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS budget_decisions (
    id TEXT PRIMARY KEY, run_id TEXT, role TEXT, model TEXT, allowed INTEGER, reason TEXT,
    projected_cost_usd REAL, created_at REAL
);
CREATE TABLE IF NOT EXISTS inference_cache (
    key TEXT PRIMARY KEY, model TEXT, response_json TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, kind TEXT, payload_json TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS investigation_tasks (
    id TEXT PRIMARY KEY, run_id TEXT, parent_task_id TEXT, depth INTEGER, question TEXT,
    evidence_ids TEXT, tools_allowed TEXT, budget TEXT, status TEXT, result TEXT,
    new_questions TEXT, model_used TEXT, created_at REAL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS experiment_proposals (
    id TEXT PRIMARY KEY, run_id TEXT, parent_version TEXT, status TEXT, proposal_json TEXT,
    decision_json TEXT, created_at REAL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS workflow_versions (
    id TEXT PRIMARY KEY, parent_id TEXT, status TEXT, workflow_hash TEXT, prompt_hash TEXT,
    path TEXT, manifest_json TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS eval_by_episode (
    id TEXT PRIMARY KEY, run_id TEXT, version_id TEXT, split TEXT, episode_id TEXT,
    metrics_json TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS operator_runs (
    id TEXT PRIMARY KEY, run_id TEXT, version_id TEXT, episode_id TEXT, node_id TEXT, operator TEXT,
    status TEXT, input_hash TEXT, output_hash TEXT, duration_s REAL, detail_json TEXT, created_at REAL
);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


class Store:
    def __init__(self, db_path: Path, events_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.events_path = events_path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- generic helpers -------------------------------------------------
    def insert(self, table: str, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._lock:
            self.conn.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({marks})", [_enc(v) for v in row.values()])
            self.conn.commit()

    def update(self, table: str, key: str, fields: dict[str, Any]) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self.conn.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [_enc(v) for v in fields.values()] + [key])
            self.conn.commit()

    def query(self, sql: str, params: tuple | list = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    # -- events ----------------------------------------------------------
    def event(self, run_id: str | None, kind: str, payload: dict[str, Any]) -> None:
        now = time.time()
        with self._lock:
            self.conn.execute(
                "INSERT INTO events (run_id, kind, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (run_id, kind, json.dumps(payload, default=str), now),
            )
            self.conn.commit()
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a") as f:
                f.write(json.dumps({"ts": now, "run_id": run_id, "kind": kind, **payload}, default=str) + "\n")

    # -- runs ------------------------------------------------------------
    def start_run(self, kind: str, meta: dict[str, Any] | None = None) -> str:
        run_id = new_id("RUN")
        self.insert("runs", {"id": run_id, "kind": kind, "status": "running", "started_at": time.time(), "finished_at": None, "meta_json": meta or {}})
        self.event(run_id, "run_started", {"kind": kind, **(meta or {})})
        return run_id

    def finish_run(self, run_id: str, status: str, meta: dict[str, Any] | None = None) -> None:
        fields: dict[str, Any] = {"status": status, "finished_at": time.time()}
        if meta is not None:
            fields["meta_json"] = meta
        self.update("runs", run_id, fields)
        self.event(run_id, "run_finished", {"status": status})

    # -- inference cache -------------------------------------------------
    def cache_get(self, key: str) -> dict[str, Any] | None:
        rows = self.query("SELECT response_json FROM inference_cache WHERE key = ?", (key,))
        return json.loads(rows[0]["response_json"]) if rows else None

    def cache_put(self, key: str, model: str, response: dict[str, Any]) -> None:
        self.insert("inference_cache", {"key": key, "model": model, "response_json": response, "created_at": time.time()})


def _enc(v: Any) -> Any:
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, default=str)
    if isinstance(v, bool):
        return int(v)
    return v


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        from evoaction.config import get_settings

        s = get_settings()
        _store = Store(s.db_path, s.events_path)
    return _store
