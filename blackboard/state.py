"""
Blackboard state schema — shared typed workspace for all agents.

Design contract:
- Workers write to named typed fields (findings, code, critique) — the blackboard
- Supervisor reads those fields directly instead of parsing message history
- messages is demoted to audit log / human-facing trace only

Reducer decisions:
- findings: list[str] with operator.add — accumulates across researcher iterations
- code:     str, no reducer — last-write-wins (revisions replace, not append)
- critique: str, no reducer — latest critique is what matters
- messages: operator.add — append-only audit log
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


MAX_ITERATIONS = 6  # hard cap across all worker calls


class AgentState(TypedDict):
    # --- input ---
    task:       str                                         # original user input, set once at invocation
    messages:   Annotated[list[BaseMessage], operator.add] # audit log — append-only

    # --- supervisor control ---
    next:       str                                         # routing target, written by supervisor
    task_type:  str                                         # "research" | "code" | "review", set by classifier
    iteration:  Annotated[int, operator.add]               # incremented by workers, read by supervisor

    # --- blackboard ---
    findings:   Annotated[list[str], operator.add]          # researcher appends one entry per run
    code:       str                                         # coder overwrites — latest revision wins
    critique:   str                                         # critic overwrites — latest assessment wins
