"""
run_traced.py — wrap ONE real `supervisor` run with Sentinel tracing.

This is the integration layer. It does NOT edit supervisor/ — it wraps around
`app.invoke(...)`:
  - picks a task_id, uses your user_id, and reads the current git commit,
  - attaches the recorder via the `callbacks` slot (same one the eval harness uses),
  - times the run and marks it success / failed (success == it finished without
    throwing — the definition we agreed on for now),
  - builds the run receipt and ships the whole trace to Langfuse.

Run on your Mac (needs OPENAI_API_KEY + TAVILY_API_KEY + Langfuse keys, and spends
a few cents of real model calls):
    set -a; . ./.env.local; set +a
    python -m observability.run_traced
    python -m observability.run_traced "your own question here"
"""
from __future__ import annotations

import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from .receipts import RunReceipt
from .tracing_callback import SentinelCallbackHandler
from . import langfuse_export

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_sha() -> str:
    """Short git commit of the code running this task. 'unknown' if not a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() or "unknown"
    except Exception:
        return "unknown"


def run_traced(question: str, user_id: str = "riadh", task_id: Optional[str] = None) -> dict:
    # Ensure the repo root is importable (so `supervisor` resolves), like the eval harness does.
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from langchain_core.messages import HumanMessage
    from supervisor import app  # compiled graph, untouched

    task_id = task_id or f"run-{uuid.uuid4().hex[:8]}"
    git_sha = _git_sha()
    handler = SentinelCallbackHandler(task_id=task_id, user_id=user_id, git_sha=git_sha)

    t0 = time.perf_counter()
    status = "success"
    error: Optional[Exception] = None
    final_state: dict = {}
    try:
        final_state = app.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "next": "",
                "final_answer": "",
                "search_iterations": 0,
            },
            config={"recursion_limit": 20, "callbacks": [handler]},
        )
    except Exception as e:  # noqa: BLE001 — record the failure, then re-raise below
        status = "failed"
        error = e

    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    # Totals come straight from the recorded receipts — single source of truth.
    total_tokens = sum(r.tokens_in + r.tokens_out for r in handler.model_receipts)
    total_cost = round(sum(r.cost_usd for r in handler.model_receipts), 8)

    run = RunReceipt(
        task_id=task_id,
        user_id=user_id,
        git_sha=git_sha,
        run_status=status,
        total_tokens=total_tokens,
        total_cost_usd=total_cost,
        latency_ms=latency_ms,
    )
    trace_id = langfuse_export.export_run(run, handler.timeline)
    langfuse_export.flush()

    print(
        f"\n[sentinel] task_id={task_id} | status={status} | git={git_sha} | "
        f"llm_calls={len(handler.model_receipts)} | tool_calls={len(handler.tool_receipts)} | "
        f"tokens={total_tokens} | cost=${total_cost} | {latency_ms} ms"
    )
    print(f"[sentinel] trace_id={trace_id}")
    print(f"[sentinel] replay it:  python -m observability.replay {trace_id}")

    if error is not None:
        raise error
    return final_state


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or (
        "What are the key tradeoffs between LangGraph and building raw agent loops from scratch?"
    )
    run_traced(question)
