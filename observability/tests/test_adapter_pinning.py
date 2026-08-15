"""
Pinning test: freezes the attribute-name contract between our internal receipts
and the OpenTelemetry-GenAI span attributes.

If a future OTel-GenAI spec bump — or a careless edit — renames an attribute,
these assertions fail loudly, instead of the Sentinel's cost/token numbers
silently going blank in production. Update the literals here ONLY as a
deliberate, reviewed change (that is the whole point).

Runs two ways:
    pytest observability/tests/test_adapter_pinning.py
    python -m observability.tests.test_adapter_pinning
"""
from __future__ import annotations

from observability import adapter as A
from observability.receipts import ModelReceipt, ToolReceipt, RunReceipt


def test_genai_attribute_names_are_pinned():
    # These strings are the contract with the OTel-GenAI spec. Do not drift silently.
    assert A.GEN_AI_SYSTEM == "gen_ai.system"
    assert A.GEN_AI_OPERATION == "gen_ai.operation.name"
    assert A.GEN_AI_REQUEST_MODEL == "gen_ai.request.model"
    assert A.GEN_AI_USAGE_IN == "gen_ai.usage.input_tokens"
    assert A.GEN_AI_USAGE_OUT == "gen_ai.usage.output_tokens"


def test_sentinel_attribute_names_are_pinned():
    assert A.S_TASK_ID == "sentinel.task_id"
    assert A.S_USER_ID == "sentinel.user_id"
    assert A.S_PLAN_STEP == "sentinel.plan_step"
    assert A.S_GIT_SHA == "sentinel.git_sha"
    assert A.S_RUN_STATUS == "sentinel.run_status"
    assert A.S_COST_USD == "sentinel.cost_usd"
    assert A.S_LATENCY_MS == "sentinel.latency_ms"
    assert A.S_TOOL_NAME == "sentinel.tool_name"
    assert A.S_TOOL_ERROR == "sentinel.tool_error"
    assert A.S_TOTAL_TOKENS == "sentinel.total_tokens"


def test_model_receipt_emits_exact_key_set():
    r = ModelReceipt("t1", "u1", 1, "gpt-4o-mini", 100, 50, 0.0001, 850.0)
    attrs = A.model_receipt_to_attributes(r)
    assert set(attrs) == {
        "gen_ai.system",
        "gen_ai.operation.name",
        "gen_ai.request.model",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "sentinel.task_id",
        "sentinel.user_id",
        "sentinel.plan_step",
        "sentinel.cost_usd",
        "sentinel.latency_ms",
    }


def test_model_receipt_roundtrip():
    r = ModelReceipt("t1", "u1", 1, "gpt-4o-mini", 100, 50, 0.0001, 850.0)
    assert A.attributes_to_model_receipt(A.model_receipt_to_attributes(r)) == r


def test_tool_receipt_roundtrip():
    r = ToolReceipt("t1", "u1", 2, "tavily_search_results_json", False, 1200.0)
    assert A.attributes_to_tool_receipt(A.tool_receipt_to_attributes(r)) == r


def test_run_receipt_roundtrip():
    r = RunReceipt("t1", "u1", "abc1234", "success", 1500, 0.0025, 9300.0)
    assert A.attributes_to_run_receipt(A.run_receipt_to_attributes(r)) == r


if __name__ == "__main__":
    # Plain-python runner so the test works without pytest installed.
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} pinning checks passed.")
