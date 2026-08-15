"""
Pure receipt-building logic — NO LangChain, NO network imports.

Kept separate from `tracing_callback.py` on purpose: the LangChain callback is
just thin glue that extracts raw data and calls the functions here. All the real
logic (token extraction, cost, receipt construction) lives in this module so it
can be unit-tested offline, without LangChain or an API key.

Everything here is duck-typed: it reads dicts and objects defensively rather than
importing LangChain result classes, so a version bump in LangChain can't break it.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from .receipts import ModelReceipt, ToolReceipt

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
# USD per 1,000,000 tokens as (input, output).
# As of 2025-05. VERIFY against current OpenAI pricing before quoting absolute
# numbers in the W07D5 article — Langfuse computes its own cost too, and the two
# should agree. Relative cost *deltas* (what the Sentinel alerts on) hold even if
# these drift slightly.
PRICES_PER_1M_USD = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def _match_price_key(model: Optional[str]) -> Optional[str]:
    if not model:
        return None
    for key in PRICES_PER_1M_USD:
        if model.startswith(key):
            return key
    return None


def cost_usd(model: Optional[str], tokens_in: int, tokens_out: int) -> float:
    """Compute cost from tokens. Unknown model -> 0.0 (Langfuse still prices it)."""
    key = _match_price_key(model)
    if key is None:
        return 0.0
    p_in, p_out = PRICES_PER_1M_USD[key]
    return round(tokens_in / 1_000_000 * p_in + tokens_out / 1_000_000 * p_out, 8)


# ---------------------------------------------------------------------------
# Extraction from LangChain's LLM result (duck-typed)
# ---------------------------------------------------------------------------
def extract_token_usage(llm_output: Any, generations: Any) -> Tuple[int, int]:
    """
    Return (tokens_in, tokens_out).

    Preferred source: llm_output['token_usage'] (what ChatOpenAI reports).
    Fallback: sum usage_metadata across generation messages (newer LangChain).
    """
    if isinstance(llm_output, dict):
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if isinstance(usage, dict):
            t_in = usage.get("prompt_tokens", usage.get("input_tokens"))
            t_out = usage.get("completion_tokens", usage.get("output_tokens"))
            if t_in is not None or t_out is not None:
                return int(t_in or 0), int(t_out or 0)

    t_in = t_out = 0
    for gen_list in (generations or []):
        for gen in (gen_list or []):
            message = getattr(gen, "message", None)
            usage_md = getattr(message, "usage_metadata", None) if message is not None else None
            if isinstance(usage_md, dict):
                t_in += int(usage_md.get("input_tokens", 0) or 0)
                t_out += int(usage_md.get("output_tokens", 0) or 0)
    return t_in, t_out


def extract_model_name(llm_output: Any, fallback: Optional[str]) -> str:
    if isinstance(llm_output, dict):
        model = llm_output.get("model_name") or llm_output.get("model")
        if model:
            return model
    return fallback or "unknown"


# ---------------------------------------------------------------------------
# Receipt construction — the one place receipts are built
# ---------------------------------------------------------------------------
def build_model_receipt(
    task_id: str,
    user_id: str,
    plan_step: int,
    model: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: float,
) -> ModelReceipt:
    return ModelReceipt(
        task_id=task_id,
        user_id=user_id,
        plan_step=plan_step,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd(model, tokens_in, tokens_out),
        latency_ms=latency_ms,
    )


def build_tool_receipt(
    task_id: str,
    user_id: str,
    plan_step: int,
    tool_name: str,
    tool_error: bool,
    latency_ms: float,
) -> ToolReceipt:
    return ToolReceipt(
        task_id=task_id,
        user_id=user_id,
        plan_step=plan_step,
        tool_name=tool_name,
        tool_error=tool_error,
        latency_ms=latency_ms,
    )
