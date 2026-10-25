"""
replay.py — reconstruct a past run's receipts from its Langfuse trace.

Given a trace ID, this fetches the trace's observations over Langfuse's REST API
(same keys as the exporter, no `langfuse` SDK, stdlib only) and rebuilds the
internal receipts — casting Langfuse's stringified attributes back to real numbers
via the adapter.

API (migrated 2026-10-25, before Langfuse Cloud removes GET /api/public/traces on
2026-11-16): GET /api/public/v2/observations?traceId=...
  - v2 has no "one trace" getter: rows are flat, paginated with a cursor.
  - metadata values are cut at 200 chars unless the TOP-LEVEL key is listed in
    expandMetadata. All our sentinel.*/gen_ai.* attributes live under ONE key,
    `attributes` (>200 chars for llm.call / agent_run), so we ask for
    expandMetadata=attributes. Verified live on trace 608b780c: without it,
    llm.call/agent_run attrs come back as truncated strings; with it, full dicts.
  - rows are not time-ordered: reconstruct() orders receipts by sentinel.plan_step.
  - a still-truncated bag (str instead of dict) is reported loudly, never dropped silently.

Design:
  - fetch_trace(trace_id)   -> {"id", "observations": [...]}   (network, v2 API)
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
import urllib.parse
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


V2_FIELDS = "core,basic,metadata"
EXPAND_METADATA = "attributes"   # the one metadata key holding every sentinel.*/gen_ai.* attribute
PAGE_LIMIT = 1000                # v2 max; one agent run is ~10-20 observations
MAX_PAGES = 50                   # guard against a cursor that never ends


def _get_json(url: str) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers={"Authorization": _auth_header()})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 (trusted host)
        return json.loads(response.read().decode())


def fetch_trace(trace_id: str) -> Dict[str, Any]:
    """All observations of one trace via the v2 API, time-ordered, wrapped as {"observations": [...]}.

    The wrapper keeps reconstruct() unchanged (it was written against the v1 trace shape).
    """
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
    rows: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    for _ in range(MAX_PAGES):
        params = {"traceId": trace_id, "fields": V2_FIELDS, "expandMetadata": EXPAND_METADATA,
                  "limit": PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor
        page = _get_json(f"{host}/api/public/v2/observations?{urllib.parse.urlencode(params)}")
        rows.extend(page.get("data") or [])
        meta = page.get("meta") or {}
        cursor = meta.get("cursor") or meta.get("nextCursor")
        if not cursor:
            break
    else:
        print(f"[replay] WARNING: stopped after {MAX_PAGES} pages; trace may be incomplete", file=sys.stderr)

    truncated = [r.get("name") for r in rows if isinstance((r.get("metadata") or {}).get("attributes"), str)]
    if truncated:
        print(f"[replay] WARNING: {len(truncated)} observation(s) still have truncated attributes "
              f"(not reconstructed): {sorted(set(truncated))}", file=sys.stderr)

    rows.sort(key=lambda r: r.get("startTime") or "")
    return {"id": trace_id, "observations": rows}


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

    # Order by OUR plan_step, not by API row order: v2 rows arrive unordered and the
    # requested field groups did not give a usable startTime (live check, trace 608b780c).
    models.sort(key=lambda m: m.plan_step)
    tools.sort(key=lambda t: t.plan_step)
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
