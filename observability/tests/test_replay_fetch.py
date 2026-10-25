"""Offline tests for replay.fetch_trace on the Langfuse v2 observations API (fake urlopen, no network)."""
from __future__ import annotations

import io
import json
import urllib.parse

import pytest

from observability import replay


def _obs(name, start, attrs, parent="p"):
    # `start` kept for readability only: live v2 rows gave no usable startTime, so it is NOT sent.
    return {"name": name, "parentObservationId": parent, "metadata": {"attributes": attrs}}


RUN_ATTRS = {"sentinel.task_id": "run-1", "sentinel.user_id": "u", "sentinel.git_sha": "ae2ddf6",
             "sentinel.run_status": "success", "sentinel.total_tokens": "100",
             "sentinel.cost_usd": "0.001", "sentinel.latency_ms": "1000.0"}
LLM_ATTRS = {"gen_ai.request.model": "gpt-4o-mini", "gen_ai.usage.input_tokens": "10",
             "gen_ai.usage.output_tokens": "5", "sentinel.task_id": "run-1", "sentinel.user_id": "u",
             "sentinel.cost_usd": "0.0001", "sentinel.latency_ms": "300.0"}

PAGES = [
    {"data": [_obs("llm.call", "2026-10-25T10:00:03Z", {**LLM_ATTRS, "sentinel.plan_step": "3"}),
              _obs("agent_run", "2026-10-25T10:00:00Z", RUN_ATTRS, parent=None)],
     "meta": {"cursor": "page2"}},
    {"data": [_obs("llm.call", "2026-10-25T10:00:01Z", {**LLM_ATTRS, "sentinel.plan_step": "1"}),
              _obs("agent_run", "2026-10-25T10:00:00Z", RUN_ATTRS),          # child duplicate (seen live)
              _obs("llm.call", "2026-10-25T10:00:05Z", "{'gen_ai.request.model': 'gpt-4o-mi")],  # truncated
     "meta": {}},
]


@pytest.fixture
def fake_api(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://lf.example")
    calls = []

    def fake_urlopen(req, timeout=30):
        calls.append(req.full_url)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(req.full_url).query)
        page = PAGES[1] if q.get("cursor") == ["page2"] else PAGES[0]
        return io.BytesIO(json.dumps(page).encode())

    monkeypatch.setattr(replay.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_fetch_uses_v2_with_expand_and_follows_cursor(fake_api):
    trace = replay.fetch_trace("t1")
    assert len(fake_api) == 2
    first = urllib.parse.urlparse(fake_api[0])
    q = urllib.parse.parse_qs(first.query)
    assert first.path == "/api/public/v2/observations"
    assert q["traceId"] == ["t1"] and q["expandMetadata"] == ["attributes"]
    assert q["fields"] == ["core,basic,metadata"]
    assert "/api/public/traces" not in " ".join(fake_api)
    assert len(trace["observations"]) == 5


def test_rows_sorted_and_reconstruct_unchanged(fake_api, capsys):
    run, models, tools = replay.reconstruct(replay.fetch_trace("t1"))
    assert run is not None and run.task_id == "run-1" and run.total_tokens == 100   # duplicate agent_run ok
    assert [m.plan_step for m in models] == [1, 3]           # plan_step order (API gave 3 first), truncated row skipped
    assert "truncated attributes" in capsys.readouterr().err  # ... but reported loudly
