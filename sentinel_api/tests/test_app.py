"""Offline tests for POST /receipts (temp SQLite via SENTINEL_DB; no network)."""
import hashlib
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from sentinel_api.app import app

OLD_KEY, NEW_KEY = "k-old-test", "k-new-test"  # test values, not secrets
AUTH = {"Authorization": f"Bearer {NEW_KEY}"}

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
    monkeypatch.setenv("SENTINEL_API_KEYS", f"{OLD_KEY}, {NEW_KEY}")  # rotation window: both valid
    return path


@pytest.fixture
def client(db):
    with TestClient(app, headers=AUTH) as c:  # `with` runs lifespan -> keys + init_schema
        yield c


@pytest.fixture
def anon(db):
    with TestClient(app) as c:  # no Authorization header
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
    with TestClient(app, raise_server_exceptions=False, headers=AUTH) as c:
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


# --- W3.5A D3 step 2: API-key auth ---

def test_missing_key_is_401_with_challenge_and_writes_nothing(anon, db):
    r = anon.post("/receipts", json=GOOD)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    assert count(db, "runs") == 0


@pytest.mark.parametrize("headers", [
    {"Authorization": "Bearer wrong-key"},
    {"Authorization": f"Basic {NEW_KEY}"},   # right secret, wrong scheme
    {"Authorization": f"Bearer {NEW_KEY}x"},  # prefix of nothing, longer than a real key
])
def test_bad_credentials_are_401(anon, db, headers):
    assert anon.post("/receipts", json=GOOD, headers=headers).status_code == 401
    assert count(db, "runs") == 0


@pytest.mark.parametrize("key", [OLD_KEY, NEW_KEY])
def test_every_key_in_the_rotation_set_is_accepted(anon, key):
    r = anon.post("/receipts", json=GOOD, headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 201


def test_auth_runs_before_validation(anon):
    """Unauthenticated + invalid body -> 401, not a 422 that describes the schema."""
    assert anon.post("/receipts", json={"junk": 1}).status_code == 401


def test_health_needs_no_key(anon):
    assert anon.get("/health").status_code == 200


@pytest.mark.parametrize("value", ["", " , ,"])
def test_empty_key_set_refuses_to_start(db, monkeypatch, value):
    monkeypatch.setenv("SENTINEL_API_KEYS", value)
    with pytest.raises(RuntimeError, match="SENTINEL_API_KEYS is empty"):
        with TestClient(app):
            pass


def test_openapi_declares_bearer_scheme(client):
    spec = client.get("/openapi.json").json()
    schemes = spec["components"]["securitySchemes"]
    assert any(s.get("scheme", "").lower() == "bearer" for s in schemes.values())
    assert "401" in spec["paths"]["/receipts"]["post"]["responses"]


# --- W3.5A D3 step 3: canonical JSON log line, allowlisted fields ---

def log_lines(capsys):
    return [json.loads(ln) for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]


def test_one_json_line_per_request_with_correlation(client, capsys):
    r = client.post("/receipts", json=GOOD, headers={"X-Request-ID": GOOD["task_id"]})
    assert r.status_code == 201 and r.headers["x-request-id"] == GOOD["task_id"]
    (line,) = [ln for ln in log_lines(capsys) if ln["event"] == "request"]
    assert line["request_id"] == GOOD["task_id"] and line["task_id"] == GOOD["task_id"]
    assert (line["method"], line["path"], line["status"], line["inserted"]) == ("POST", "/receipts", 201, True)
    assert line["key_id"] == hashlib.sha256(NEW_KEY.encode()).hexdigest()[:8]
    assert line["level"] == "INFO" and isinstance(line["latency_ms"], float)


@pytest.mark.parametrize("sent", [None, "bad id with spaces", "x" * 129])
def test_missing_or_malformed_request_id_is_replaced(client, capsys, sent):
    headers = {"X-Request-ID": sent} if sent else {}
    r = client.get("/health", headers=headers)
    rid = r.headers["x-request-id"]
    assert rid != sent and len(rid) == 32
    assert log_lines(capsys)[-1]["request_id"] == rid


def test_401_line_has_no_key_id_and_no_secret(anon, capsys):
    anon.post("/receipts", json=GOOD, headers={"Authorization": "Bearer wrong-key-value"})
    out = capsys.readouterr().out
    line = json.loads([ln for ln in out.splitlines() if ln.startswith("{")][-1])
    assert line["status"] == 401 and line["level"] == "WARNING"
    assert "key_id" not in line and "task_id" not in line
    assert "wrong-key-value" not in out and "Bearer" not in out


def test_valid_key_never_reaches_the_logs(client, capsys):
    client.post("/receipts", json=GOOD)
    client.post("/receipts", json={"junk": 1})
    out = capsys.readouterr().out
    assert NEW_KEY not in out and OLD_KEY not in out and "Authorization" not in out


def test_500_is_logged_as_error_with_request_id(db, capsys):
    with TestClient(app, raise_server_exceptions=False, headers=AUTH) as c:
        conn = sqlite3.connect(db)
        conn.execute("CREATE TRIGGER boom BEFORE INSERT ON tool_calls BEGIN SELECT RAISE(ABORT, 'boom'); END;")
        conn.commit()
        conn.close()
        c.post("/receipts", json=GOOD, headers={"X-Request-ID": "req-500"})
    line = [ln for ln in log_lines(capsys) if ln["event"] == "request"][-1]
    assert (line["status"], line["level"], line["request_id"]) == (500, "ERROR", "req-500")
