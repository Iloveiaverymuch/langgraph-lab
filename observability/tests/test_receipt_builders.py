"""
Offline tests for the pure receipt-building logic (no LangChain, no network).

Covers: cost math, token extraction from both shapes ChatOpenAI can emit, model
name resolution, and receipt construction.

Run:
    python -m observability.tests.test_receipt_builders
    pytest observability/tests/test_receipt_builders.py
"""
from __future__ import annotations

from types import SimpleNamespace

from observability._receipt_builders import (
    build_model_receipt,
    cost_usd,
    extract_model_name,
    extract_token_usage,
)


def test_cost_known_model():
    # 1M input tokens at $0.15/1M = $0.15; 1M output at $0.60/1M = $0.60
    assert cost_usd("gpt-4o-mini", 1_000_000, 0) == 0.15
    assert cost_usd("gpt-4o-mini", 0, 1_000_000) == 0.60


def test_cost_model_prefix_match():
    # dated variants like "gpt-4o-mini-2024-07-18" still price correctly
    assert cost_usd("gpt-4o-mini-2024-07-18", 1_000_000, 0) == 0.15


def test_cost_unknown_model_is_zero():
    assert cost_usd("some-future-model", 1000, 1000) == 0.0


def test_extract_tokens_from_llm_output():
    llm_output = {"token_usage": {"prompt_tokens": 120, "completion_tokens": 30}}
    assert extract_token_usage(llm_output, None) == (120, 30)


def test_extract_tokens_from_generations_fallback():
    # No token_usage in llm_output -> fall back to usage_metadata on messages
    gen = SimpleNamespace(
        message=SimpleNamespace(usage_metadata={"input_tokens": 200, "output_tokens": 45})
    )
    assert extract_token_usage({}, [[gen]]) == (200, 45)


def test_extract_model_name_prefers_llm_output():
    assert extract_model_name({"model_name": "gpt-4o-mini"}, "fallback") == "gpt-4o-mini"
    assert extract_model_name({}, "gpt-4o-mini") == "gpt-4o-mini"
    assert extract_model_name(None, None) == "unknown"


def test_build_model_receipt_computes_cost():
    r = build_model_receipt("t1", "u1", 2, "gpt-4o-mini", 1_000_000, 0, 900.0)
    assert r.task_id == "t1"
    assert r.plan_step == 2
    assert r.cost_usd == 0.15
    assert r.tokens_in == 1_000_000


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} builder checks passed.")
