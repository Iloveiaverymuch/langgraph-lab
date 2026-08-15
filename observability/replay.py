"""
replay.py — reconstruct a past run's receipts from its Langfuse trace.

Given a trace ID, this fetches the trace over Langfuse's REST API (same keys as
the exporter, no `langfuse` SDK, stdlib only) and rebuilds the internal receipts
— casting Langfuse's stringified attributes back to real numbers via the adapter.

Design:
  - fetch_trace(trace_id)   -> raw Langfuse JSON        (network)
  - reconstruct(trace_json) -> (RunReceipt, [Model], [Tool])   (pure, testable)
  - replay(trace_id)        -> fetch + reconstruct + pretty-print

Usage on your Mac:
    set -a; . ./.env.local; set +a
    python -m observability.replay <trace_id>
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from . import adapter
from .receipts import ModelReceipt, RunReceipt, ToolReceipt


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------
def _auth_header() -> str:
    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    if not (public and secret):
        raise RuntimeError(
            "Missing LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY. "
            "Run:  set -a; . ./.env.local; set +a"
        )
    token = base64.b64encode(f"{public}:{secret}".encode()).decode()
    return f"Basic {token}"


def fetch_trace(trace_id: str) -> Dict[str, Any]:
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
    url = f"{host}/api/public/traces/{trace_id}"
    request = urllib.request.Request(url, headers={"Authorization": _auth_header()})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 (trusted host)
        return json.loads(response.read().decode())


# ---------------------------------------------------------------------------
# pure reconstruction
# ---------------------------------------------------------------------------
def _attrs_of(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the OTel attribute bag out of a Langfuse observation.

    Langfuse nests it under metadata.attributes; fall back to any flattened
    sentinel.* / gen_ai.* keys just in case the shape differs.
    """
    metadata = observation.get("metadata") or {}
    nested = metadata.get("attributes")
    if isinstance(nested, dict):
        return nested
    return {
        k: v
        for k, v in metadata.items()
        if isinstance(k, str) and (k.startswith("sentinel.") or k.startswith("gen_ai."))
    }


def reconstruct(
    trace_json: Dict[str, Any],
) -> Tuple[Optional[RunReceipt], List[ModelReceipt], List[ToolReceipt]]:
    observations = trace_json.get("observations") or []
    run: Optional[RunReceipt] = None
    models: List[ModelReceipt] = []
    tools: List[ToolReceipt] = []

    for observation in observations:
        name = observation.get("name", "")
        attrs = _attrs_of(observation)
        if name == "llm.call" and adapter.GEN_AI_REQUEST_MODEL in attrs:
            models.append(adapter.attributes_to_model_receipt(attrs))
        elif name.startswith("tool.") and adapter.S_TOOL_NAME in attrs:
            tools.append(adapter.attributes_to_tool_receipt(attrs))
        elif name == "agent_run" and adapter.S_RUN_STATUS in attrs:
            # our real root span (Langfuse's trace-wrapper node has no sentinel attrs)
            run = adapter.attributes_to_run_receipt(attrs)

    return run, models, tools


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print(run: Optional[RunReceipt], models: List[ModelReceipt], tools: List[ToolReceipt]) -> None:
    if run is not None:
        print(
            f"\nRUN  task_id={run.task_id} | status={run.run_status} | git={run.git_sha} | "
            f"tokens={run.total_tokens} | cost=${run.total_cost_usd} | {run.latency_ms} ms"
        )
    else:
        print("\nRUN  (no root agent_run receipt found in this trace)")

    print(f"\nLLM calls ({len(models)}):")
    for i, m in enumerate(models, 1):
        print(
            f"  {i:>2}. step={m.plan_step} {m.model} | in={m.tokens_in} out={m.tokens_out} "
            f"| ${m.cost_usd} | {m.latency_ms} ms"
        )

    print(f"\nTool calls ({len(tools)}):")
    for i, t in enumerate(tools, 1):
        flag = "ERROR" if t.tool_error else "ok"
        print(f"  {i:>2}. step={t.plan_step} {t.tool_name} | {flag} | {t.latency_ms} ms")


def replay(trace_id: str) -> None:
    run, models, tools = reconstruct(fetch_trace(trace_id))
    _print(run, models, tools)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m observability.replay <trace_id>")
    replay(sys.argv[1])
