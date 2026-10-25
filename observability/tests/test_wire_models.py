"""Offline tests for the receipts wire contract (W3.5A D2 step 2)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from observability import store
from observability.receipts import RunReceipt, ToolReceipt
from observability.wire_models import RunIn

GOOD = {
    "task_id": "run-1a2b3c4d", "git_sha": "ae2ddf6", "run_status": "success",
    "run_type": "production", "total_tokens": 900, "total_cost_usd": 0.002, "latency_ms": 1500,
    "tool_calls": [{"tool_name": "tavily_search_results_json", "tool_error": False, "latency_ms": 400.5}],
}


def test_good_json_payload_validates():
    run = RunIn.model_validate_json(__import__("json").dumps(GOOD))  # JSON int 1500 ok for float
    assert run.latency_ms == 1500.0 and len(run.tool_calls) == 1


@pytest.mark.parametrize("patch", [
    {"latency_ms": "1500"},          # string number: strict, no coercion
    {"latency_ms": -1},              # negative
    {"git_sha": "AE2DDF6!"},         # not hex
    {"run_status": "ok"},            # not in Literal
    {"user_id": "riadh"},            # PII field: forbidden, not silently dropped
    {"prompt": "hello"},             # text field: forbidden
    {"tool_calls": [{"tool_name": "x", "tool_error": 0, "latency_ms": 1}]},  # 0 is not a bool
])
def test_bad_payloads_rejected(patch):
    with pytest.raises(ValidationError):
        RunIn.model_validate({**GOOD, **patch})


def test_unknown_sha_allowed():
    assert RunIn.model_validate({**GOOD, "git_sha": "unknown"}).git_sha == "unknown"


def test_from_receipts_drops_user_id_and_round_trips_to_store(tmp_path):
    run = RunReceipt(task_id="run-9", user_id="riadh", git_sha="abc1234", run_status="failed",
                     total_tokens=10, total_cost_usd=0.0, latency_ms=12.0, run_type="injection_test")
    tools = [ToolReceipt(task_id="run-9", user_id="riadh", plan_step=1, tool_name="canned_search",
                         tool_error=True, latency_ms=3.0)]
    wire = RunIn.from_receipts(run, tools)
    assert "user_id" not in wire.model_dump()
    run_row, tool_rows = wire.to_rows()
    conn = store.connect(tmp_path / "r.db")
    try:
        assert store.insert_run(conn, run_row, tool_rows) is True
        assert store.insert_run(conn, run_row, tool_rows) is False  # idempotent on task_id
        assert conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0] == 1
    finally:
        conn.close()
