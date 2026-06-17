"""
Shared state schema — the contract between all nodes in the supervisor graph.
Every node reads from this and writes partial updates back.
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


def _increment(current: int, update: int) -> int:
    """Reducer: adds update to current. Used for search_iterations counter."""
    return current + update


MAX_SEARCH_ITERATIONS = 2


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # append-only reducer
    next: str                                              # supervisor's routing decision
    final_answer: str                                      # populated by writer_worker
    search_iterations: Annotated[int, _increment]         # incremented by search_worker, capped by supervisor
