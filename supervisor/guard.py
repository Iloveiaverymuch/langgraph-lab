"""
Deterministic transition guard for the supervisor (W07D3, layer L1).

Why this exists: the supervisor's five routing rules are a state machine — the legal
next step follows from WHICH WORKER SPOKE LAST, not from judgement. Letting an LLM
re-derive that from a context that includes attacker-controlled text made routing
hijackable (ignore-previous 7, system-prompt-leak 3, tool-confusion 1/4/8: 14 of the 43
baseline failures). Now the LLM only PROPOSES a next step; this module DECIDES.

Trust model: only the `name` metadata on worker messages is trusted. Worker messages are
created by our own nodes (AIMessage(name=...)); a user's question arrives as a
HumanMessage with no name, so text inside it can never forge "search_worker already ran".
Message CONTENT is read in exactly one place: the analyst's own SUFFICIENT / NEEDS_MORE
verdict, which our analyst node produced.

Pure python, no langchain/openai imports, so it is unit-testable offline.
"""
from __future__ import annotations

from typing import Iterable, Tuple

_WORKERS = ("search_worker", "analyst_worker", "writer_worker")


def _analyst_wants_more(text: str) -> bool:
    """The analyst is told to END with SUFFICIENT or NEEDS_MORE. Whichever verdict
    appears LAST wins. Neither present -> not 'wants more' (prefer progress over a loop)."""
    needs = text.rfind("NEEDS_MORE")
    enough = text.rfind("SUFFICIENT")
    return needs != -1 and needs > enough


def expected_next(messages: Iterable) -> str:
    """The one legal next step given who spoke last."""
    last = None
    last_analyst_text = ""
    for m in messages:
        name = getattr(m, "name", None)
        if name in _WORKERS:
            last = name
            if name == "analyst_worker":
                last_analyst_text = str(getattr(m, "content", "") or "")
    if last is None:
        return "search_worker"
    if last == "search_worker":
        return "analyst_worker"
    if last == "analyst_worker":
        return "search_worker" if _analyst_wants_more(last_analyst_text) else "writer_worker"
    return "FINISH"  # writer_worker spoke last: the report exists


def apply_guard(llm_choice: str, messages: Iterable) -> Tuple[str, bool]:
    """Return (next_node, overridden). The LLM's choice survives only if it IS the legal step."""
    legal = expected_next(messages)
    return legal, llm_choice != legal
