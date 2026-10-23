"""Offline tests for the regression-signal math (fixture SQLite, no network)."""
from __future__ import annotations

import pytest

from sentinel_mcp import signals as S
from sentinel_mcp.tests.fixtures import BASE, CAND, build


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "r.db"
    build(db)
    c = S.connect_ro(db)
    yield c
    c.close()


def _by_name(cmp):
    return {s.name: s for s in cmp.signals}


def test_p95_nearest_rank():
    assert S.p95([]) is None
    assert S.p95([5.0]) == 5.0
    assert S.p95([float(i) for i in range(1, 21)]) == 19.0  # ceil(0.95*20)=19th value
    assert S.p95([1.0, 2.0, 3.0, 4.0, 100.0]) == 100.0      # n<20 -> max


def test_regressed_candidate_flags_latency_tools_success_not_cost(conn):
    cmp = S.compare(conn, BASE, CAND)
    sig = _by_name(cmp)
    assert cmp.verdict == "regression"
    assert sig["cost_delta"].status == "ok" and sig["cost_delta"].delta == 0.0
    assert sig["p95_latency"].status == "regression" and sig["p95_latency"].delta > 25
    assert sig["tool_errors"].status == "regression"
    assert sig["tool_errors"].baseline == 0.0 and sig["tool_errors"].candidate == 20.0
    assert "canned_search" in sig["tool_errors"].detail
    assert sig["success_rate"].baseline == 100.0 and sig["success_rate"].candidate == 60.0
    assert sig["success_rate"].status == "regression"


def test_same_commit_is_ok(conn):
    cmp = S.compare(conn, BASE, BASE)
    assert cmp.verdict == "ok"
    assert all(s.status == "ok" for s in cmp.signals)


def test_injection_runs_excluded_from_production(conn):
    cmp = S.compare(conn, BASE, CAND)
    assert cmp.candidate_runs == 5  # the injection_test run is not counted
    assert _by_name(cmp)["cost_delta"].candidate == 0.0015


def test_insufficient_data_gives_no_verdict(tmp_path):
    db = tmp_path / "small.db"
    build(db, cand_runs=3, cand_failures=3)
    c = S.connect_ro(db)
    try:
        cmp = S.compare(c, BASE, CAND)
    finally:
        c.close()
    assert cmp.verdict == "insufficient_data"
    assert all(s.status == "insufficient_data" for s in cmp.signals)
    assert any("need >= 5 runs" in n for n in cmp.notes)


@pytest.mark.parametrize("bad", ["", "xyz", "abc", "ABCDEF1'; DROP TABLE runs;--", "a" * 41])
def test_bad_sha_rejected(conn, bad):
    with pytest.raises(ValueError):
        S.compare(conn, bad, CAND)


def test_bad_run_type_rejected(conn):
    with pytest.raises(ValueError):
        S.compare(conn, BASE, CAND, run_type="everything")


def test_store_opened_read_only(tmp_path):
    import sqlite3
    db = tmp_path / "r.db"
    build(db)
    c = S.connect_ro(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            c.execute("DELETE FROM runs")
    finally:
        c.close()


def test_list_commits_counts(conn):
    lst = S.list_commits(conn)
    by = {c.git_sha: c for c in lst.commits}
    assert by[BASE].runs == 5 and by[BASE].failed_runs == 0
    assert by[CAND].runs == 5 and by[CAND].failed_runs == 2
