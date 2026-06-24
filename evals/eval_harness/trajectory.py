"""
Trajectory reconstruction for the LangGraph supervisor.

The supervisor tags every worker AIMessage with `name=<worker>`. That tag IS the
observable trajectory: reading the ordered list of names from final_state["messages"]
gives the exact sequence of workers that ran. Used for path/order + termination asserts.

Pure functions over a final-state dict — import-safe with no API keys.
"""

from __future__ import annotations

WORKERS = ("search_worker", "analyst_worker", "writer_worker")


def worker_sequence(final_state: dict) -> list[str]:
    """Ordered list of worker names that produced messages — the trajectory."""
    seq = []
    for m in final_state.get("messages", []):
        name = getattr(m, "name", None)
        if name in WORKERS:
            seq.append(name)
    return seq


def step_count(final_state: dict) -> int:
    """Total worker invocations (trajectory length). For the no-loop assertion."""
    return len(worker_sequence(final_state))


def terminated(final_state: dict) -> bool:
    """Clean termination iff supervisor's last route was FINISH AND a report exists."""
    next_val = final_state.get("next", "")
    wrote_report = "writer_worker" in worker_sequence(final_state)
    return next_val == "FINISH" and wrote_report


def final_answer(final_state: dict) -> str:
    """The final deliverable string (last message content)."""
    msgs = final_state.get("messages", [])
    if not msgs:
        return ""
    return getattr(msgs[-1], "content", "") or ""
