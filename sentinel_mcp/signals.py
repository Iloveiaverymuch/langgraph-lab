"""
signals.py — the four Sentinel regression signals, computed in code (no I/O beyond one SQLite read).

Compares a baseline commit against a candidate commit on runs of one run_type
(default "production", so W07D2 injection traffic never pollutes the signal):

  cost_delta    mean cost per run            regression if +cost_pct %   or more
  p95_latency   nearest-rank p95 of run ms   regression if +p95_pct %    or more
  tool_errors   tool error rate, new tools   regression if any tool newly errors, or the rate rises
  success_rate  success / total (percent)    regression if it drops success_drop_pts points or more

Why computed here, not by the MCP client's model: deterministic, unit-tested arithmetic,
a few dozen bytes of output instead of thousands of span tokens, and no untrusted trace
text ever reaches the client's context (indirect prompt injection; see W07D2).

Sample-size guard: fewer than MIN_RUNS runs on either side -> "insufficient_data", no verdict.
"""
from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field

SHA_RE = re.compile(r"^[0-9a-f]{4,40}$")
RUN_TYPES = ("production", "injection_test")
MIN_RUNS = 5

Status = Literal["ok", "regression", "insufficient_data"]


@dataclass(frozen=True)
class Thresholds:
    cost_pct: float = 20.0
    p95_pct: float = 25.0
    success_drop_pts: float = 10.0


class SignalResult(BaseModel):
    name: Literal["cost_delta", "p95_latency", "tool_errors", "success_rate"]
    status: Status
    baseline: Optional[float] = Field(None, description="baseline value, in `unit`")
    candidate: Optional[float] = Field(None, description="candidate value, in `unit`")
    delta: Optional[float] = Field(None, description="candidate - baseline, in `delta_unit`")
    unit: str
    delta_unit: str
    detail: str


class Comparison(BaseModel):
    baseline_sha: str
    candidate_sha: str
    run_type: str
    baseline_runs: int
    candidate_runs: int
    verdict: Status
    signals: List[SignalResult]
    notes: List[str] = Field(default_factory=list)


class CommitSummary(BaseModel):
    git_sha: str
    runs: int
    failed_runs: int
    first_run_at: str
    last_run_at: str


class CommitList(BaseModel):
    run_type: str
    commits: List[CommitSummary] = Field(description="most recent first")


# ---------------------------------------------------------------- validation
def validate_sha(sha: str) -> str:
    sha = (sha or "").strip().lower()
    if not SHA_RE.match(sha):
        raise ValueError("git sha must be 4-40 lowercase hex characters")
    return sha


def validate_run_type(run_type: str) -> str:
    if run_type not in RUN_TYPES:
        raise ValueError(f"run_type must be one of {RUN_TYPES}")
    return run_type


# ---------------------------------------------------------------- db access
def connect_ro(db_path: Union[str, Path]) -> sqlite3.Connection:
    """Open the receipts store READ-ONLY (the MCP server can never write to it)."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError("receipts store not found; record a run first (observability.run_traced)")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def list_commits(conn: sqlite3.Connection, run_type: str = "production") -> CommitList:
    rows = conn.execute(
        "SELECT git_sha, COUNT(*) AS runs, SUM(run_status = 'failed') AS failed_runs,"
        " MIN(created_at) AS first_run_at, MAX(created_at) AS last_run_at"
        " FROM runs WHERE run_type = ? GROUP BY git_sha ORDER BY MAX(created_at) DESC",
        (run_type,),
    ).fetchall()
    return CommitList(run_type=run_type, commits=[CommitSummary(**dict(r)) for r in rows])


def _runs(conn: sqlite3.Connection, sha: str, run_type: str) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT run_status, total_cost_usd, latency_ms FROM runs WHERE git_sha = ? AND run_type = ?",
        (sha, run_type),
    ).fetchall()


def _tool_stats(conn: sqlite3.Connection, sha: str, run_type: str) -> dict:
    rows = conn.execute(
        "SELECT t.tool_name, COUNT(*) AS calls, SUM(t.tool_error) AS errors"
        " FROM tool_calls t JOIN runs r ON r.task_id = t.task_id"
        " WHERE r.git_sha = ? AND r.run_type = ? GROUP BY t.tool_name",
        (sha, run_type),
    ).fetchall()
    return {r["tool_name"]: (int(r["calls"]), int(r["errors"] or 0)) for r in rows}


# ---------------------------------------------------------------- math
def p95(values: Sequence[float]) -> Optional[float]:
    """Nearest-rank 95th percentile. With n < 20 this is the maximum."""
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


def pct_change(base: Optional[float], cand: Optional[float]) -> Optional[float]:
    if base is None or cand is None or base == 0:
        return None
    return round((cand - base) / base * 100.0, 2)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _r(x: Optional[float], nd: int) -> Optional[float]:
    return None if x is None else round(x, nd)


# ---------------------------------------------------------------- signals
def compare(
    conn: sqlite3.Connection,
    baseline_sha: str,
    candidate_sha: str,
    run_type: str = "production",
    thresholds: Thresholds = Thresholds(),
) -> Comparison:
    baseline_sha, candidate_sha = validate_sha(baseline_sha), validate_sha(candidate_sha)
    run_type = validate_run_type(run_type)
    b_runs, c_runs = _runs(conn, baseline_sha, run_type), _runs(conn, candidate_sha, run_type)
    nb, nc = len(b_runs), len(c_runs)
    enough = nb >= MIN_RUNS and nc >= MIN_RUNS
    gate: Optional[Status] = None if enough else "insufficient_data"
    notes: List[str] = []
    if not enough:
        notes.append(f"need >= {MIN_RUNS} runs per commit (baseline {nb}, candidate {nc}); no verdict")
    if min(nb, nc) < 20:
        notes.append("fewer than 20 runs on a side: p95 equals the slowest run")

    # 1) cost delta (mean USD per run)
    b_cost = _mean([r["total_cost_usd"] for r in b_runs])
    c_cost = _mean([r["total_cost_usd"] for r in c_runs])
    cost_pct = pct_change(b_cost, c_cost)
    cost = SignalResult(
        name="cost_delta",
        status=gate or ("regression" if cost_pct is not None and cost_pct >= thresholds.cost_pct else "ok"),
        baseline=_r(b_cost, 6), candidate=_r(c_cost, 6), delta=cost_pct,
        unit="usd_per_run", delta_unit="percent",
        detail=f"threshold +{thresholds.cost_pct:g}%",
    )

    # 2) p95 latency
    b_p95 = p95([r["latency_ms"] for r in b_runs])
    c_p95 = p95([r["latency_ms"] for r in c_runs])
    p95_pct = pct_change(b_p95, c_p95)
    latency = SignalResult(
        name="p95_latency",
        status=gate or ("regression" if p95_pct is not None and p95_pct >= thresholds.p95_pct else "ok"),
        baseline=_r(b_p95, 1), candidate=_r(c_p95, 1), delta=p95_pct,
        unit="ms", delta_unit="percent",
        detail=f"threshold +{thresholds.p95_pct:g}%",
    )

    # 3) tool errors (overall rate + tools that newly error)
    bt, ct = _tool_stats(conn, baseline_sha, run_type), _tool_stats(conn, candidate_sha, run_type)

    def rate(stats: dict) -> Optional[float]:
        calls = sum(c for c, _ in stats.values())
        return round(100.0 * sum(e for _, e in stats.values()) / calls, 2) if calls else None

    b_rate, c_rate = rate(bt), rate(ct)
    newly_failing = sorted(n for n, (_, e) in ct.items() if e > 0 and bt.get(n, (0, 0))[1] == 0)
    rate_up = b_rate is not None and c_rate is not None and c_rate > b_rate
    tools = SignalResult(
        name="tool_errors",
        status=gate or ("regression" if newly_failing or rate_up else "ok"),
        baseline=b_rate, candidate=c_rate,
        delta=None if b_rate is None or c_rate is None else round(c_rate - b_rate, 2),
        unit="percent_of_tool_calls", delta_unit="points",
        detail=("newly failing tools: " + ", ".join(newly_failing)) if newly_failing else "no newly failing tools",
    )

    # 4) success rate
    def succ(rows: List[sqlite3.Row]) -> Optional[float]:
        return round(100.0 * sum(r["run_status"] == "success" for r in rows) / len(rows), 2) if rows else None

    b_s, c_s = succ(b_runs), succ(c_runs)
    drop = None if b_s is None or c_s is None else round(b_s - c_s, 2)
    success = SignalResult(
        name="success_rate",
        status=gate or ("regression" if drop is not None and drop >= thresholds.success_drop_pts else "ok"),
        baseline=b_s, candidate=c_s, delta=None if drop is None else -drop,
        unit="percent_of_runs", delta_unit="points",
        detail=f"threshold -{thresholds.success_drop_pts:g} points",
    )

    signals = [cost, latency, tools, success]
    verdict: Status = gate or ("regression" if any(s.status == "regression" for s in signals) else "ok")
    return Comparison(
        baseline_sha=baseline_sha, candidate_sha=candidate_sha, run_type=run_type,
        baseline_runs=nb, candidate_runs=nc, verdict=verdict, signals=signals, notes=notes,
    )
