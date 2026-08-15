"""
Smoke test — prove the Langfuse pipe works, WITHOUT running the real agent.

This builds one synthetic run (a couple of fake receipts) and ships it to
Langfuse. If it lands in the dashboard, the OTLP endpoint, auth, and attribute
mapping are all good — so when we wire the real supervisor next, any problem is
in the agent glue, not the pipe.

Run on your Mac (needs network + deps):
    pip install -r observability/requirements.txt
    set -a; . ./.env.local; set +a
    python -m observability.smoke_test_export

Then open Langfuse -> Traces. You should see a trace named 'agent_run' with two
children: 'llm.call' and 'tool.tavily_search_results_json', carrying cost,
tokens, latency, task_id, git_sha, and run_status attributes.
"""
from __future__ import annotations

from .receipts import ModelReceipt, RunReceipt, ToolReceipt
from . import langfuse_export


def main() -> None:
    task_id = "smoke-0001"
    user_id = "riadh"
    git_sha = "smoke-test"

    model_receipt = ModelReceipt(
        task_id=task_id, user_id=user_id, plan_step=1,
        model="gpt-4o-mini", tokens_in=800, tokens_out=120,
        cost_usd=0.000192, latency_ms=910.0,
    )
    tool_receipt = ToolReceipt(
        task_id=task_id, user_id=user_id, plan_step=1,
        tool_name="tavily_search_results_json", tool_error=False, latency_ms=1340.0,
    )
    run = RunReceipt(
        task_id=task_id, user_id=user_id, git_sha=git_sha, run_status="success",
        total_tokens=920, total_cost_usd=0.000192, latency_ms=2250.0,
    )

    langfuse_export.export_run(run, timeline=[model_receipt, tool_receipt])
    langfuse_export.flush()
    print("Exported smoke trace 'agent_run' (task_id=smoke-0001). Check Langfuse -> Traces.")


if __name__ == "__main__":
    main()
