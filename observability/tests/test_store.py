"""
Offline tests for the Sentinel receipts store (temporary SQLite file, no network).

Run:
    pytest observability/tests/test_store.py
"""
from __future__ import annotations

import sqlite3

import pytest

from observability import store
from observability.receipts import RunReceipt, ToolReceipt


def _run(task_id="run-1", sha="abc1234", status="success", run_type="production", cost=0.002, latency=1500.0):
    return RunReceipt(task_id=task_id, user_id="riadh", git_sha=sha, run_status=status,
                      total_tokens=900, total_cost_usd=cost, latency_ms=latency, run_type=run_type)


def _tool(task_id="run-1", error=False, latency=400.0):
    return ToolReceipt(task_id=task_id, user_id="riadh", plan_step=1,
                       tool_name="tavily_search_results_json", tool_error=error, latency_ms=latency)


def _rows(db, sql):
    conn = store.connect(db)
    try:
        return [dict(r) for r in conn.execute(sql)]
    finally:
        conn.close()


def test_record_run_persists_run_and_tools(tmp_path):
    db = tmp_path / "r.db"
    assert store.record_run(_run(), [_tool(), _tool(error=True)], db_path=db) is True
    runs = _rows(db, "SELECT * FROM runs")
    assert len(runs) == 1
    assert runs[0]["git_sha"] == "abc1234" and runs[0]["run_status"] == "success"
    assert runs[0]["source"] == "live" and runs[0]["run_type"] == "production"
    tools = _rows(db, "SELECT tool_error FROM tool_calls ORDER BY id")
    assert [t["tool_error"] for t in tools] == [0, 1]


def test_record_run_is_idempotent(tmp_path):
    db = tmp_path / "r.db"
    assert store.record_run(_run(), [_tool()], db_path=db) is True
    assert store.record_run(_run(cost=9.99), [_tool(), _tool()], db_path=db) is False
    assert _rows(db, "SELECT total_cost_usd FROM runs")[0]["total_cost_usd"] == 0.002  # unchanged
    assert len(_rows(db, "SELECT id FROM tool_calls")) == 1  # no duplicate tool rows


def test_store_keeps_no_user_or_text_columns(tmp_path):
    db = tmp_path / "r.db"
    store.record_run(_run(), [_tool()], db_path=db)
    conn = store.connect(db)
    try:
        cols = {r[1] for t in ("runs", "tool_calls") for r in conn.execute(f"PRAGMA table_info({t})")}
    finally:
        conn.close()
    assert "user_id" not in cols
    assert not any(c in cols for c in ("question", "input", "output", "answer"))


def test_invalid_status_is_rejected(tmp_path):
    with pytest.raises(sqlite3.IntegrityError):
        store.record_run(_run(status="weird"), [], db_path=tmp_path / "r.db")


def test_db_path_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DB", str(tmp_path / "env.db"))
    assert store.default_db_path() == tmp_path / "env.db"
