"""Fixture receipts store: a clean baseline commit and a regressed candidate commit (synthetic data)."""
from __future__ import annotations

from observability import store
from observability.receipts import RunReceipt, ToolReceipt

BASE, CAND = "aaaa111", "bbbb222"


def build(db, base_runs=5, cand_runs=5, cand_failures=2, cand_latency=40000.0, cand_tool_errors=2):
    for i in range(base_runs):
        tid = f"b{i}"
        store.record_run(
            RunReceipt(tid, "u", BASE, "success", 5000, 0.0015, 30000.0 + i * 100),
            [ToolReceipt(tid, "u", 1, "canned_search", False, 5.0) for _ in range(2)], db_path=db)
    for i in range(cand_runs):
        tid = f"c{i}"
        status = "failed" if i < cand_failures else "success"
        tools = [ToolReceipt(tid, "u", 1, "canned_search", i < cand_tool_errors, 5.0), 
                 ToolReceipt(tid, "u", 1, "canned_search", False, 5.0)]
        store.record_run(RunReceipt(tid, "u", CAND, status, 5000, 0.0015, cand_latency + i * 100), tools, db_path=db)
    # injection traffic on the candidate must NOT affect production signals
    store.record_run(RunReceipt("inj0", "u", CAND, "failed", 1, 9.0, 999999.0, run_type="injection_test"), [], db_path=db)
