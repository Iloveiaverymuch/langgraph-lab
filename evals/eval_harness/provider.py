"""
Promptfoo Python provider for the LangGraph supervisor (in-repo).

Promptfoo calls `call_api(prompt, options, context)` and expects:
    { "output": <str>, "metadata": {...}, "tokenUsage": {...} }

We return the final report as `output` and the trajectory under `metadata`, so the
deterministic assertions in promptfooconfig.yaml can check:
  - tools called + order   -> context.metadata.worker_sequence
  - termination / no loops -> context.metadata.terminated / step_count
  - token budget           -> context.metadata.total_tokens (tokenUsage.total)
  - final output contains  -> output

Execution mode:
  - RUN_REAL_SUPERVISOR=1 AND OPENAI_API_KEY present -> run the real supervisor graph.
  - otherwise -> deterministic fake (CI-safe, no network, no spend).

The supervisor is the SAME repo's package — imported from the repo root, no edits to it.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

from trajectory import worker_sequence, step_count, terminated, final_answer

# Repo root = two levels up from evals/eval_harness/ . The `supervisor` package lives there.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_real(question: str) -> tuple[dict, int]:
    """Run the actual compiled supervisor graph. Returns (final_state, total_tokens)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from langchain_core.messages import HumanMessage
    from langchain_core.callbacks import UsageMetadataCallbackHandler
    from supervisor import app  # compiled graph, untouched

    cb = UsageMetadataCallbackHandler()
    final_state = app.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "next": "",
            "final_answer": "",
            "search_iterations": 0,
        },
        config={"recursion_limit": 20, "callbacks": [cb]},
    )
    total = 0
    for usage in getattr(cb, "usage_metadata", {}).values():
        total += usage.get("total_tokens", 0)
    return final_state, total


def _run_fake(question: str) -> tuple[dict, int]:
    from fake_supervisor import run_fake
    final_state = run_fake(question)
    total = step_count(final_state) * 400  # deterministic estimate for the budget assert
    return final_state, total


def _use_real() -> bool:
    return os.environ.get("RUN_REAL_SUPERVISOR") == "1" and bool(
        os.environ.get("OPENAI_API_KEY")
    )


def call_api(prompt: str, options=None, context=None):
    """Promptfoo entry point."""
    question = prompt
    if context and isinstance(context, dict):
        question = context.get("vars", {}).get("question", prompt)

    try:
        if _use_real():
            final_state, total_tokens = _run_real(question)
            mode = "real"
        else:
            final_state, total_tokens = _run_fake(question)
            mode = "fake"
    except Exception as e:
        return {"error": f"supervisor invocation failed: {type(e).__name__}: {e}"}

    return {
        "output": final_answer(final_state),
        "metadata": {
            "mode": mode,
            "worker_sequence": worker_sequence(final_state),
            "step_count": step_count(final_state),
            "terminated": terminated(final_state),
            "search_iterations": final_state.get("search_iterations", 0),
            "total_tokens": total_tokens,
        },
        "tokenUsage": {"total": total_tokens},
    }
