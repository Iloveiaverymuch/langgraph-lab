"""
Offline test for replay.reconstruct — proves we rebuild correctly-typed receipts
from a Langfuse-shaped payload where every attribute value is a STRING (which is
how Langfuse returns them over the API).

Run:
    python -m observability.tests.test_replay_reconstruct
    pytest observability/tests/test_replay_reconstruct.py
"""
from __future__ import annotations

from observability.replay import reconstruct


def _langfuse_shaped_trace() -> dict:
    """Mimics the real Langfuse trace JSON: observations with metadata.attributes
    as strings, plus Langfuse's own trace-wrapper node (no sentinel attrs)."""
    def obs(name, attrs):
        return {"name": name, "metadata": {"attributes": attrs}}

    return {
        "observations": [
            # Langfuse's trace-wrapper node — named agent_run but NO sentinel attrs; must be ignored.
            {"name": "agent_run", "metadata": {"attributes": {}}},
            # our real root
            obs("agent_run", {
                "sentinel.task_id": "run-0af8cda0",
                "sentinel.user_id": "riadh",
                "sentinel.git_sha": "7f1c8c9",
                "sentinel.run_status": "success",
                "sentinel.total_tokens": "11276",
                "sentinel.cost_usd": "0.0027039",
                "sentinel.latency_ms": "43125.839",
            }),
            obs("llm.call", {
                "gen_ai.request.model": "gpt-4o-mini-2024-07-18",
                "gen_ai.usage.input_tokens": "1861",
                "gen_ai.usage.output_tokens": "479",
                "sentinel.task_id": "run-0af8cda0",
                "sentinel.user_id": "riadh",
                "sentinel.plan_step": "7",
                "sentinel.cost_usd": "0.00056655",
                "sentinel.latency_ms": "5342.208",
            }),
            obs("tool.tavily_search_results_json", {
                "sentinel.task_id": "run-0af8cda0",
                "sentinel.user_id": "riadh",
                "sentinel.plan_step": "2",
                "sentinel.tool_name": "tavily_search_results_json",
                "sentinel.tool_error": "false",
                "sentinel.latency_ms": "1340.0",
            }),
        ]
    }


def test_reconstruct_counts_and_ignores_wrapper():
    run, models, tools = reconstruct(_langfuse_shaped_trace())
    assert run is not None            # the wrapper node was correctly ignored
    assert len(models) == 1
    assert len(tools) == 1


def test_reconstruct_casts_types():
    run, models, tools = reconstruct(_langfuse_shaped_trace())
    # run
    assert run.run_status == "success"
    assert isinstance(run.total_tokens, int) and run.total_tokens == 11276
    assert isinstance(run.total_cost_usd, float) and run.total_cost_usd == 0.0027039
    # model
    m = models[0]
    assert isinstance(m.plan_step, int) and m.plan_step == 7
    assert isinstance(m.tokens_in, int) and m.tokens_in == 1861
    assert isinstance(m.tokens_out, int) and m.tokens_out == 479
    assert isinstance(m.cost_usd, float) and m.cost_usd == 0.00056655
    # tool
    t = tools[0]
    assert isinstance(t.tool_error, bool) and t.tool_error is False
    assert isinstance(t.latency_ms, float) and t.latency_ms == 1340.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} replay checks passed.")
