"""
store.py — the Sentinel's own receipts store (SQLite, stdlib only).

Why this exists (W07D4b spikes, 2026-10-15/17): Langfuse is the trace *viewer*, not
the metrics backend. Its read API is built for its UI: the v1 trace endpoints are
removed on Langfuse Cloud on 2026-11-16, the v2 observations endpoint truncates
metadata at 200 chars (our sentinel.* attributes come back cut mid-string), and
per-trace reads hit 429. So run_traced dual-writes: receipts go here at emit time,
spans go to Langfuse for debugging. Regression signals (sentinel_mcp/) read here.

Design rules:
  - Minimal columns: only what the four signals need. No prompt/answer text, no
    user id. What is not stored cannot leak or carry an injection into an MCP client.
  - Idempotent: task_id is the primary key; re-recording a run is a no-op.
  - Never fatal: callers wrap record_run so monitoring cannot break the agent.

Location: $SENTINEL_DB, else <repo>/.sentinel/receipts.db (gitignored).
Python 3.9-compatible.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Union

from .receipts import RunReceipt, ToolReceipt

_REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    task_id        TEXT PRIMARY KEY,
    git_sha        TEXT NOT NULL,
    run_status     TEXT NOT NULL CHECK (run_status IN ('success', 'failed')),
    run_type       TEXT NOT NULL,
    total_tokens   INTEGER NOT NULL,
    total_cost_usd REAL NOT NULL,
    latency_ms     REAL NOT NULL,
    created_at     TEXT NOT NULL,
    source         TEXT NOT NULL DEFAULT 'live'
);
CREATE INDEX IF NOT EXISTS idx_runs_sha ON runs (git_sha, run_type);

CREATE TABLE IF NOT EXISTS tool_calls (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL REFERENCES runs (task_id),
    tool_name  TEXT NOT NULL,
    tool_error INTEGER NOT NULL CHECK (tool_error IN (0, 1)),
    latency_ms REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_task ON tool_calls (task_id);
"""


def default_db_path() -> Path:
    env = os.environ.get("SENTINEL_DB")
    return Path(env) if env else _REPO_ROOT / ".sentinel" / "receipts.db"


BUSY_TIMEOUT_MS = 5000  # wait up to 5 s for the single SQLite write lock instead of failing


def connect(
    db_path: Optional[Union[str, Path]] = None,
    init_schema: bool = True,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """Open a read-write connection.

    init_schema=False: the caller ran init_schema() once at startup (the FastAPI service).
    check_same_thread=False: only for per-request connections that are never shared
    (FastAPI may run a sync dependency's setup, the handler and teardown on different
    threadpool threads).
    """
    path = Path(db_path) if db_path else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    if init_schema:
        _apply_schema(conn)
    return conn


def _apply_schema(conn: sqlite3.Connection) -> None:
    # WAL: readers (MCP server, mode=ro) don't block the writer and vice versa.
    # Persistent in the db file. Not supported on network filesystems.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)


def init_schema(db_path: Optional[Union[str, Path]] = None) -> Path:
    """Create tables + enable WAL once (service startup). Returns the resolved path."""
    path = Path(db_path) if db_path else default_db_path()
    conn = connect(path, init_schema=True)
    conn.close()
    return path


def insert_run(
    conn: sqlite3.Connection,
    run_row: dict,
    tool_rows: Iterable[dict],
    source: str = "live",
    created_at: Optional[str] = None,
) -> bool:
    """Insert one run + its tool calls atomically on the CALLER's connection.

    Used by the FastAPI service (per-request connection from Depends) and by record_run.
    Returns False if task_id already exists (idempotent: existing rows untouched).
    """
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    with conn:  # one transaction: run + its tool calls, or nothing
        cur = conn.execute(
            "INSERT INTO runs (task_id, git_sha, run_status, run_type, total_tokens,"
            " total_cost_usd, latency_ms, created_at, source) VALUES (?,?,?,?,?,?,?,?,?)"
            # ON CONFLICT skips ONLY duplicate keys; OR IGNORE would also swallow CHECK violations.
            " ON CONFLICT(task_id) DO NOTHING",
            (run_row["task_id"], run_row["git_sha"], run_row["run_status"], run_row["run_type"],
             int(run_row["total_tokens"]), float(run_row["total_cost_usd"]), float(run_row["latency_ms"]),
             created_at, source),
        )
        if cur.rowcount == 0:
            return False  # duplicate
        conn.executemany(
            "INSERT INTO tool_calls (task_id, tool_name, tool_error, latency_ms) VALUES (?,?,?,?)",
            [(run_row["task_id"], t["tool_name"], int(bool(t["tool_error"])), float(t["latency_ms"]))
             for t in tool_rows],
        )
    return True


def record_run(
    run: RunReceipt,
    tools: Iterable[ToolReceipt],
    db_path: Optional[Union[str, Path]] = None,
    source: str = "live",
    created_at: Optional[str] = None,
) -> bool:
    """Persist one run and its tool calls atomically (opens + closes its own connection).

    Returns False if task_id already exists.
    """
    run_row = {
        "task_id": run.task_id, "git_sha": run.git_sha, "run_status": run.run_status,
        "run_type": run.run_type, "total_tokens": run.total_tokens,
        "total_cost_usd": run.total_cost_usd, "latency_ms": run.latency_ms,
    }
    tool_rows = [{"tool_name": t.tool_name, "tool_error": t.tool_error, "latency_ms": t.latency_ms} for t in tools]
    conn = connect(db_path)
    try:
        return insert_run(conn, run_row, tool_rows, source=source, created_at=created_at)
    finally:
        conn.close()
