"""Offline tests for POST /receipts (temp SQLite via SENTINEL_DB; no network)."""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from sentinel_api.app import app

GOOD = {
    "task_id": "run-1a2b3c4d", "git_sha": "ae2ddf6", "run_status": "success",
    "run_type": "production", "total_tokens": 900, "total_cost_usd": 0.002, "latency_ms": 1500,
    "tool_calls": [{"tool_name": "canned_search", "tool_error": False, "latency_ms": 400.5}],
}


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "receipts.db"
    # The service resolves its store through the same factory as every other caller
    # (store.default_db_path reads SENTINEL_DB), so tests redirect ALL paths at once.
    monkeypatch.setenv("SENTINEL_DB", str(path))
    return path


@pytest.fixture
def client(db):
    with TestClient(app) as c:  # `with` runs lifespan -> init_schema
        yield c


def count(db, table):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def test_post_creates_then_duplicate_is_200_noop(client, db):
    r1 = client.post("/receipts", json=GOOD)
    assert r1.status_code == 201 and r1.json() == {"task_id": GOOD["task_id"], "created": True}
    r2 = client.post("/receipts", json=GOOD)
    assert r2.status_code == 200 and r2.json()["created"] is False
    assert count(db, "runs") == 1 and count(db, "tool_calls") == 1
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT source FROM runs").fetchone()[0] == "api"
    conn.close()


@pytest.mark.parametrize("patch", [
    {"latency_ms": "1500ms"},   # wrong type
    {"prompt": "<user text>"},  # extra field (PII/text) -> forbidden
    {"git_sha": "not-a-sha"},   # domain rule
])
def test_bad_payloads_are_422_and_write_nothing(client, db, patch):
    r = client.post("/receipts", json={**GOOD, **patch})
    assert r.status_code == 422
    assert count(db, "runs") == 0


def test_failure_after_run_insert_rolls_back_everything(db):
    """Run row inserted, then the tool_calls insert fails -> nothing persisted, 500."""
    with TestClient(app, raise_server_exceptions=False) as c:
        conn = sqlite3.connect(db)
        conn.execute("CREATE TRIGGER boom BEFORE INSERT ON tool_calls BEGIN SELECT RAISE(ABORT, 'boom'); END;")
        conn.commit()
        conn.close()
        r = c.post("/receipts", json=GOOD)
    assert r.status_code == 500
    assert count(db, "runs") == 0


def test_health_and_openapi(client):
    assert client.get("/health").json() == {"status": "ok"}
    spec = client.get("/openapi.json").json()
    assert "/receipts" in spec["paths"]
    assert "RunIn" in spec["components"]["schemas"]
