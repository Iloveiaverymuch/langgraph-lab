"""
server.py — Agent Regression Sentinel exposed over MCP (stdio). Read-only.

Tools:
  list_commits(run_type)                              -> which commits have recorded runs
  compare_commits(baseline_sha, candidate_sha, run_type) -> 5 regression signals + verdict

Security posture:
  - read-only: the receipts store is opened with SQLite mode=ro
  - inputs validated (hex sha, run_type allow-list); SQL uses bound parameters only
  - outputs are numbers, shas and fixed strings: no prompt/answer/trace text reaches the
    client model (no indirect-injection path from production traffic into the dev agent)
  - nothing is printed to stdout (stdout IS the MCP transport on stdio)

Run (an MCP client normally launches this):
    .venv/bin/python -m sentinel_mcp.server
Store location: $SENTINEL_DB, else <repo>/.sentinel/receipts.db.
SDK: official MCP Python SDK v2 (`MCPServer`, formerly FastMCP).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # allow launch from any cwd
    sys.path.insert(0, str(_REPO_ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from observability.store import default_db_path  # noqa: E402
from sentinel_mcp import signals  # noqa: E402

mcp = MCPServer(
    "agent-regression-sentinel",
    instructions=(
        "Read-only regression signals for the LangGraph supervisor agent. "
        "Call list_commits first to find commit shas with recorded runs, then "
        "compare_commits(baseline_sha, candidate_sha) to check cost, run p95 latency, "
        "tool errors, success rate and per-tool p95 latency. A verdict of insufficient_data means record more runs."
    ),
)


def _open():
    try:
        return signals.connect_ro(default_db_path())
    except FileNotFoundError as e:
        raise ToolError(str(e)) from None


@mcp.tool()
def list_commits(run_type: str = "production") -> signals.CommitList:
    """List commits that have recorded agent runs (most recent first), with run and failure counts."""
    try:
        run_type = signals.validate_run_type(run_type)
    except ValueError as e:
        raise ToolError(str(e)) from None
    conn = _open()
    try:
        return signals.list_commits(conn, run_type)
    finally:
        conn.close()


@mcp.tool()
def compare_commits(baseline_sha: str, candidate_sha: str, run_type: str = "production") -> signals.Comparison:
    """Compare a candidate commit against a baseline commit on five regression signals:
    cost per run, run p95 latency, tool-call errors, success rate, and per-tool p95 latency.
    Returns per-signal
    status (ok | regression | insufficient_data) and an overall verdict."""
    conn = _open()
    try:
        return signals.compare(conn, baseline_sha, candidate_sha, run_type)
    except ValueError as e:
        raise ToolError(str(e)) from None
    finally:
        conn.close()


if __name__ == "__main__":
    mcp.run(transport="stdio")
