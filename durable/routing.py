"""
Deterministic routing for the durable loop.

In the graph, supervisor_node asks an LLM for the next worker and then OVERRIDES it with the legal
step (supervisor/guard.py, L1), after a hard search cap. The LLM's answer therefore never changes the
outcome, so the durable version does not call it at all: the next step is a pure function of the
message history. That is also what makes replay safe: the same recorded steps always lead to the same
next step.

test_workflow.py asserts this function agrees with supervisor_node for every reachable state.
"""
from __future__ import annotations

from supervisor.guard import expected_next
from supervisor.state import MAX_SEARCH_ITERATIONS


def next_step(messages: list) -> str:
    """messages: list of {"role", "name", "content"} dicts. Returns a worker name or "FINISH"."""
    names = [m.get("name") for m in messages]
    searches = names.count("search_worker")
    if searches >= MAX_SEARCH_ITERATIONS:                     # hard cap: bypass the analyst, write now
        return "FINISH" if "writer_worker" in names else "writer_worker"
    return expected_next(_Shim(m) for m in messages)


class _Shim:
    """expected_next reads .name and .content; dicts need a thin adapter."""
    def __init__(self, d: dict):
        self.name = d.get("name")
        self.content = d.get("content", "")
