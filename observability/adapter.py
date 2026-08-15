"""
Adapter — the ONLY module that knows OpenTelemetry-GenAI attribute names.

Responsibilities:
  1. emit  — turn an internal receipt into the attribute dict attached to a span
  2. read  — turn span attributes back into an internal receipt (used by replay)

Naming policy:
  - gen_ai.*   -> standard OTel-GenAI semantic conventions (experimental spec)
  - sentinel.* -> OUR product attributes, for things the spec does not cover
The split is deliberate: use the standard where one exists, a clearly-private
namespace where it does not. When the OTel-GenAI spec moves, only the constants
below change — and `tests/test_adapter_pinning.py` fails loudly so we notice
instead of the Sentinel's numbers silently going blank.
"""
from __future__ import annotations

from typing import Any, Dict

from .receipts import ModelReceipt, ToolReceipt, RunReceipt

# ---------------------------------------------------------------------------
# Pinned attribute names
# ---------------------------------------------------------------------------
# OTel-GenAI semantic conventions (spec status: experimental as of 2026-05).
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_USAGE_IN = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUT = "gen_ai.usage.output_tokens"

# Sentinel-private attributes (no OTel standard exists for these yet).
S_TASK_ID = "sentinel.task_id"
S_USER_ID = "sentinel.user_id"
S_PLAN_STEP = "sentinel.plan_step"
S_GIT_SHA = "sentinel.git_sha"
S_RUN_STATUS = "sentinel.run_status"
S_COST_USD = "sentinel.cost_usd"
S_LATENCY_MS = "sentinel.latency_ms"
S_TOOL_NAME = "sentinel.tool_name"
S_TOOL_ERROR = "sentinel.tool_error"
S_TOTAL_TOKENS = "sentinel.total_tokens"

# Value used for gen_ai.system for OpenAI-backed calls.
_OPENAI_SYSTEM = "openai"
_CHAT_OPERATION = "chat"


# ---------------------------------------------------------------------------
# emit: internal receipt -> span attributes
# ---------------------------------------------------------------------------
def model_receipt_to_attributes(r: ModelReceipt) -> Dict[str, Any]:
    return {
        GEN_AI_SYSTEM: _OPENAI_SYSTEM,
        GEN_AI_OPERATION: _CHAT_OPERATION,
        GEN_AI_REQUEST_MODEL: r.model,
        GEN_AI_USAGE_IN: r.tokens_in,
        GEN_AI_USAGE_OUT: r.tokens_out,
        S_TASK_ID: r.task_id,
        S_USER_ID: r.user_id,
        S_PLAN_STEP: r.plan_step,
        S_COST_USD: r.cost_usd,
        S_LATENCY_MS: r.latency_ms,
    }


def tool_receipt_to_attributes(r: ToolReceipt) -> Dict[str, Any]:
    return {
        S_TASK_ID: r.task_id,
        S_USER_ID: r.user_id,
        S_PLAN_STEP: r.plan_step,
        S_TOOL_NAME: r.tool_name,
        S_TOOL_ERROR: r.tool_error,
        S_LATENCY_MS: r.latency_ms,
    }


def run_receipt_to_attributes(r: RunReceipt) -> Dict[str, Any]:
    return {
        S_TASK_ID: r.task_id,
        S_USER_ID: r.user_id,
        S_GIT_SHA: r.git_sha,
        S_RUN_STATUS: r.run_status,
        S_TOTAL_TOKENS: r.total_tokens,
        S_COST_USD: r.total_cost_usd,
        S_LATENCY_MS: r.latency_ms,
    }


# ---------------------------------------------------------------------------
# read: span attributes -> internal receipt  (used by the replay tool)
# ---------------------------------------------------------------------------
# Langfuse returns OTel attributes as STRINGS (e.g. "1861", "0.0005", "true").
# These coercers make the read side robust whether it gets native types (our own
# tests) or the stringified values that come back over the API.
def _as_int(v: Any) -> int:
    return int(float(v)) if isinstance(v, str) else int(v)


def _as_float(v: Any) -> float:
    return float(v)


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def attributes_to_model_receipt(a: Dict[str, Any]) -> ModelReceipt:
    return ModelReceipt(
        task_id=str(a[S_TASK_ID]),
        user_id=str(a[S_USER_ID]),
        plan_step=_as_int(a[S_PLAN_STEP]),
        model=str(a[GEN_AI_REQUEST_MODEL]),
        tokens_in=_as_int(a[GEN_AI_USAGE_IN]),
        tokens_out=_as_int(a[GEN_AI_USAGE_OUT]),
        cost_usd=_as_float(a[S_COST_USD]),
        latency_ms=_as_float(a[S_LATENCY_MS]),
    )


def attributes_to_tool_receipt(a: Dict[str, Any]) -> ToolReceipt:
    return ToolReceipt(
        task_id=str(a[S_TASK_ID]),
        user_id=str(a[S_USER_ID]),
        plan_step=_as_int(a[S_PLAN_STEP]),
        tool_name=str(a[S_TOOL_NAME]),
        tool_error=_as_bool(a[S_TOOL_ERROR]),
        latency_ms=_as_float(a[S_LATENCY_MS]),
    )


def attributes_to_run_receipt(a: Dict[str, Any]) -> RunReceipt:
    return RunReceipt(
        task_id=str(a[S_TASK_ID]),
        user_id=str(a[S_USER_ID]),
        git_sha=str(a[S_GIT_SHA]),
        run_status=str(a[S_RUN_STATUS]),
        total_tokens=_as_int(a[S_TOTAL_TOKENS]),
        total_cost_usd=_as_float(a[S_COST_USD]),
        latency_ms=_as_float(a[S_LATENCY_MS]),
    )
