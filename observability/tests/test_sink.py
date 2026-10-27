"""Offline tests for the receipts sink + spool (W3.5A D2 step 4). Fake HTTP server on localhost."""
from __future__ import annotations

import json
import socket
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from observability import sink
from observability.receipts import RunReceipt, ToolReceipt


def _run(task_id="run-1", sha="abc1234"):
    return RunReceipt(task_id=task_id, user_id="riadh", git_sha=sha, run_status="success",
                      total_tokens=10, total_cost_usd=0.001, latency_ms=100.0)


def _tools(task_id="run-1"):
    return [ToolReceipt(task_id=task_id, user_id="riadh", plan_step=1, tool_name="canned_search",
                        tool_error=False, latency_ms=5.0)]


class FakeAPI:
    """Records POST bodies; replies with `self.status` (201 by default)."""

    def __init__(self):
        self.status, self.bodies, self.headers = 201, [], []
        api = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                api.bodies.append(body)
                api.headers.append(dict(self.headers))
                out = json.dumps({"task_id": body.get("task_id"), "created": api.status == 201}).encode()
                self.send_response(api.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _dead_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here -> connection refused
    return f"http://127.0.0.1:{port}"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DB", str(tmp_path / "receipts.db"))
    monkeypatch.setenv("SENTINEL_SINK", "http")
    return tmp_path


@pytest.fixture
def api():
    a = FakeAPI()
    yield a
    a.close()


def _spool_lines():
    p = sink.spool_path()
    return [ln for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def test_direct_sink_writes_store(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DB", str(tmp_path / "r.db"))
    monkeypatch.delenv("SENTINEL_SINK", raising=False)
    assert sink.record(_run(), _tools()) == "stored"
    assert sink.record(_run(), _tools()) == "duplicate"
    conn = sqlite3.connect(tmp_path / "r.db")
    assert conn.execute("SELECT source FROM runs").fetchone()[0] == "live"
    conn.close()


def test_invalid_receipt_is_not_recorded_and_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DB", str(tmp_path / "r.db"))
    monkeypatch.delenv("SENTINEL_SINK", raising=False)
    assert sink.record(_run(sha="NOT A SHA"), _tools()) == "invalid"
    assert not (tmp_path / "r.db").exists()


def test_service_down_spools_then_flush_delivers_once(env, api, monkeypatch):
    monkeypatch.setenv("SENTINEL_API_URL", _dead_url())
    assert sink.record(_run("run-1"), _tools("run-1")) == "spooled"
    assert sink.record(_run("run-2"), _tools("run-2")) == "spooled"
    assert len(_spool_lines()) == 2
    body = json.loads(_spool_lines()[0])
    assert "user_id" not in body and body["task_id"] == "run-1"

    monkeypatch.setenv("SENTINEL_API_URL", api.url)  # service is back
    assert sink.record(_run("run-3"), _tools("run-3")) == "stored"
    assert [b["task_id"] for b in api.bodies] == ["run-1", "run-2", "run-3"]  # spool first, in order
    assert _spool_lines() == []


def test_5xx_is_transient_and_spooled(env, api, monkeypatch):
    monkeypatch.setenv("SENTINEL_API_URL", api.url)
    api.status = 503
    assert sink.record(_run(), _tools()) == "spooled"
    assert len(_spool_lines()) == 1


def test_422_is_rejected_not_spooled(env, api, monkeypatch):
    monkeypatch.setenv("SENTINEL_API_URL", api.url)
    api.status = 422
    assert sink.record(_run(), _tools()) == "rejected"
    assert _spool_lines() == []


def test_duplicate_200_is_success(env, api, monkeypatch):
    monkeypatch.setenv("SENTINEL_API_URL", api.url)
    api.status = 200
    assert sink.record(_run(), _tools()) == "duplicate"
    assert _spool_lines() == []


def test_spool_cap_drops_loudly(env, monkeypatch, capsys):
    monkeypatch.setenv("SENTINEL_API_URL", _dead_url())
    monkeypatch.setattr(sink, "SPOOL_MAX_LINES", 2)
    outcomes = [sink.record(_run(f"run-{i}"), _tools(f"run-{i}")) for i in range(3)]
    assert outcomes == ["spooled", "spooled", "dropped"]
    assert "DROPPED" in capsys.readouterr().err


def test_flush_skips_when_another_process_holds_the_lock(env, monkeypatch):
    monkeypatch.setenv("SENTINEL_API_URL", _dead_url())
    sink.record(_run(), _tools())
    with sink._spool_lock():  # simulate a concurrent flusher
        assert sink.flush_spool()["skipped"] is True
    assert len(_spool_lines()) == 1


# --- W3.5A D3 step 1: agent auth (bearer key from env); 401/403 are spooled, not dropped ---

KEY = "test-key-not-a-secret"


def test_bearer_header_sent_when_key_set(env, api, monkeypatch):
    monkeypatch.setenv("SENTINEL_API_URL", api.url)
    monkeypatch.setenv("SENTINEL_API_KEY", KEY)
    assert sink.record(_run(), _tools()) == "stored"
    assert api.headers[-1].get("Authorization") == f"Bearer {KEY}"


def test_missing_key_warns_loudly_but_never_raises(env, api, monkeypatch, capsys):
    monkeypatch.setenv("SENTINEL_API_URL", api.url)
    monkeypatch.delenv("SENTINEL_API_KEY", raising=False)
    assert sink.record(_run(), _tools()) == "stored"  # today's server has no auth yet
    assert "Authorization" not in api.headers[-1]
    assert "SENTINEL_API_KEY is not set" in capsys.readouterr().err


@pytest.mark.parametrize("code", [401, 403])
def test_auth_failure_is_spooled_and_names_the_variable_not_the_value(env, api, monkeypatch, capsys, code):
    monkeypatch.setenv("SENTINEL_API_URL", api.url)
    monkeypatch.setenv("SENTINEL_API_KEY", KEY)
    api.status = code
    assert sink.record(_run(), _tools()) == "spooled"
    assert len(_spool_lines()) == 1
    err = capsys.readouterr().err
    assert f"auth failed (HTTP {code})" in err and "SENTINEL_API_KEY" in err
    assert KEY not in err
    assert KEY not in sink.spool_path().read_text()  # the spool holds payloads, never credentials


def test_flush_hitting_401_keeps_the_spool_in_order(env, api, monkeypatch):
    monkeypatch.setenv("SENTINEL_API_URL", _dead_url())
    sink.record(_run("run-1"), _tools("run-1"))
    sink.record(_run("run-2"), _tools("run-2"))
    monkeypatch.setenv("SENTINEL_API_URL", api.url)
    api.status = 401
    stats = sink.flush_spool()
    assert (stats["sent"], stats["rejected"], stats["kept"]) == (0, 0, 2)
    assert [json.loads(ln)["task_id"] for ln in _spool_lines()] == ["run-1", "run-2"]


def test_wrong_key_then_fixed_key_flushes_spool_in_order(env, api, monkeypatch):
    monkeypatch.setenv("SENTINEL_API_URL", api.url)
    monkeypatch.setenv("SENTINEL_API_KEY", "wrong-key")
    api.status = 401
    assert sink.record(_run("run-1"), _tools("run-1")) == "spooled"
    assert sink.record(_run("run-2"), _tools("run-2")) == "spooled"
    assert len(_spool_lines()) == 2

    monkeypatch.setenv("SENTINEL_API_KEY", KEY)  # key fixed
    api.status = 201
    n = len(api.bodies)
    assert sink.record(_run("run-3"), _tools("run-3")) == "stored"
    assert [b["task_id"] for b in api.bodies[n:]] == ["run-1", "run-2", "run-3"]
    assert all(h.get("Authorization") == f"Bearer {KEY}" for h in api.headers[n:])
    assert _spool_lines() == []
