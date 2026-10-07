"""
Offline tests for the durable layer (no network, no Inngest server): step_stubs replace step execution.

Run (repo root):  python durable/tests/test_workflow.py     (needs Python >= 3.10 and `pip install -r durable/requirements.txt`)
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
os.environ.setdefault("TAVILY_API_KEY", "tvly-dummy")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import inngest  # noqa: E402
from inngest.experimental import mocked  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from durable import codec, routing, workflow  # noqa: E402
from supervisor import nodes  # noqa: E402

Q = {"role": "human", "name": None, "content": "What is CAP?"}
S = {"role": "ai", "name": "search_worker", "content": "findings"}
A_OK = {"role": "ai", "name": "analyst_worker", "content": "Fine. SUFFICIENT: coverage is adequate"}
A_MORE = {"role": "ai", "name": "analyst_worker", "content": "Gap. NEEDS_MORE: partition behaviour"}
W = {"role": "ai", "name": "writer_worker", "content": "## Executive Summary ..."}
G = {"report": "## Executive Summary ...", "redacted": False}
CLIENT = mocked.Inngest(app_id="test")


def test_codec_roundtrip_keeps_names_and_roles():
    msgs = [HumanMessage(content="q"), AIMessage(content="a", name="search_worker")]
    back = codec.to_messages([codec.to_dict(m) for m in msgs])
    assert [type(m) for m in back] == [HumanMessage, AIMessage] and back[1].name == "search_worker"
    assert back[0].name is None


def test_next_step_agrees_with_the_graphs_supervisor_for_every_reachable_state():
    """Parity: the durable loop must route exactly like supervisor_node (LLM proposal, guard, cap)."""
    histories = [
        [Q], [Q, S], [Q, S, A_OK], [Q, S, A_OK, W],
        [Q, S, A_MORE], [Q, S, A_MORE, S], [Q, S, A_MORE, S, A_OK], [Q, S, A_MORE, S, A_MORE],
        [Q, S, A_MORE, S, A_MORE, W],            # cap reached, writer ran
        [Q, S, A_OK, S, S],                       # cap reached via two searches
    ]
    real = nodes.llm
    try:
        for proposal in ("search_worker", "FINISH", "garbage", "writer_worker"):  # the LLM's answer must not matter
            nodes.llm = NS(invoke=lambda msgs, p=proposal: NS(content=p))
            for h in histories:
                state = {"messages": codec.to_messages(h), "next": "", "final_answer": "",
                         "search_iterations": sum(1 for m in h if m["name"] == "search_worker")}
                import io, contextlib
                with contextlib.redirect_stdout(io.StringIO()):
                    graph_choice = nodes.supervisor_node(state)["next"]
                assert routing.next_step(h) == graph_choice, (proposal, [m["name"] for m in h])
    finally:
        nodes.llm = real


def _run(stubs, **data):
    event = inngest.Event(name=workflow.REQUESTED, data={"question": "What is CAP?", "run_id": "r1", **data})
    return mocked.trigger(workflow.run_research, event, CLIENT, step_stubs=stubs)


HAPPY = {"search_worker-1": S, "analyst_worker-1": A_OK, "writer_worker-1": W, "output-guard": G}


def test_released_without_approval_gate():
    res = _run(HAPPY, require_approval=False)
    assert res.status == mocked.Status.COMPLETED, res
    assert res.output == {"status": "released", "run_id": "r1", "report": G["report"], "redacted": False}


def test_second_search_round_uses_stable_step_ids():
    stubs = {"search_worker-1": S, "analyst_worker-1": A_MORE, "search_worker-2": S,
             "writer_worker-1": W, "output-guard": G}   # cap (2 searches) -> writer, analyst not asked again
    res = _run(stubs, require_approval=False)
    assert res.status == mocked.Status.COMPLETED and res.output["status"] == "released", res


def test_approval_timeout_releases_nothing():
    res = _run({**HAPPY, "await-approval": None})
    assert res.status == mocked.Status.COMPLETED and res.output == {"status": "timed_out", "run_id": "r1"}, res
    assert "report" not in res.output


def test_rejection_releases_nothing():
    ev = inngest.Event(name=workflow.DECISION, data={"run_id": "r1", "approved": False})
    res = _run({**HAPPY, "await-approval": ev})
    assert res.output == {"status": "rejected", "run_id": "r1"}, res


def test_approval_releases_the_report():
    ev = inngest.Event(name=workflow.DECISION, data={"run_id": "r1", "approved": True})
    res = _run({**HAPPY, "await-approval": ev})
    assert res.output["status"] == "released" and res.output["report"] == G["report"], res


def test_event_without_run_id_is_rejected_not_retried():
    event = inngest.Event(name=workflow.REQUESTED, data={"question": "x"})
    res = mocked.trigger(workflow.run_research, event, CLIENT, step_stubs={})
    assert res.status == mocked.Status.FAILED, res


def test_function_is_configured_for_idempotency_and_step_retries():
    cfg = workflow.run_research._opts
    assert cfg.idempotency == "event.data.run_id" and cfg.retries == 3


def test_every_step_handler_runs_exactly_once_across_replays():
    """The durability property in miniature: the function body is re-invoked from the top after each step
    (that is what resuming after a crash does), yet each worker executes ONCE; saved steps are replayed."""
    calls = []

    def fake(name):
        def worker(state):
            calls.append(name)
            text = "SUFFICIENT: ok" if name == "analyst_worker" else f"{name} output"
            return {"messages": [AIMessage(content=text, name=name)]}
        return worker

    saved = {n: getattr(nodes, n) for n in ("search_worker", "analyst_worker", "writer_worker", "output_guard_node")}
    try:
        for n in ("search_worker", "analyst_worker", "writer_worker"):
            setattr(nodes, n, fake(n))
        nodes.output_guard_node = lambda state: (calls.append("output_guard"), {"final_answer": "final"})[1]
        res = _run({}, require_approval=False)
    finally:
        for n, f in saved.items():
            setattr(nodes, n, f)
    assert res.status == mocked.Status.COMPLETED and res.output["report"] == "final", res
    assert calls == ["search_worker", "analyst_worker", "writer_worker", "output_guard"], calls


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} checks passed.")
