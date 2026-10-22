"""
Offline tests: canned_search is a real LangChain tool, so the Sentinel callback records it,
and the fault-injection drill produces tool_error receipts. No LLM, no network.
"""
from __future__ import annotations

import pytest

from observability import demo_backends as D
from observability.tracing_callback import SentinelCallbackHandler


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    D._calls["n"] = 0
    monkeypatch.delenv("SENTINEL_FAULT_EVERY", raising=False)
    monkeypatch.delenv("SEARCH_DELAY_S", raising=False)


def _handler():
    return SentinelCallbackHandler(task_id="t1", user_id="u", git_sha="abc")


def test_canned_search_emits_tool_receipt():
    h = _handler()
    out = D.canned_search.invoke("what is mcp", config={"callbacks": [h]})
    assert isinstance(out, list) and len(out) == 2
    assert len(h.tool_receipts) == 1
    r = h.tool_receipts[0]
    assert r.tool_name == "canned_search" and r.tool_error is False


def test_fault_drill_records_tool_error(monkeypatch):
    monkeypatch.setenv("SENTINEL_FAULT_EVERY", "2")
    h = _handler()
    D.canned_search.invoke("q1", config={"callbacks": [h]})  # call 1: ok
    with pytest.raises(D.InjectedFault):
        D.canned_search.invoke("q2", config={"callbacks": [h]})  # call 2: injected fault
    assert [r.tool_error for r in h.tool_receipts] == [False, True]


def test_no_faults_by_default():
    h = _handler()
    for i in range(5):
        D.canned_search.invoke(f"q{i}", config={"callbacks": [h]})
    assert not any(r.tool_error for r in h.tool_receipts)
