"""
Internal receipt model — the single source of truth for what the Sentinel records.

A "receipt" is one recorded step of an agent run. Everything downstream
(regression signals, the MCP server in W07D4b, replay) reads THESE field names,
never the raw OpenTelemetry / Langfuse attribute names.

Why the indirection: the OTel-GenAI semantic conventions are still experimental
(spec status "Development" as of 2026-05) and their attribute names change
between releases. Isolating them behind this model + `adapter.py` means a spec
change is a one-file edit, not a rename scattered across the codebase.

Python 3.9-compatible (matches the project venv): no PEP 604 unions, no slots=.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelReceipt:
    """One LLM call (e.g. a supervisor routing decision or a worker generation)."""
    task_id: str
    user_id: str
    plan_step: int      # which step of the run this call belongs to (0, 1, 2, ...)
    model: str          # e.g. "gpt-4o-mini"
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float


@dataclass
class ToolReceipt:
    """One tool call (e.g. the Tavily search inside search_worker)."""
    task_id: str
    user_id: str
    plan_step: int
    tool_name: str      # e.g. "tavily_search_results_json"
    tool_error: bool    # did the tool raise / fail?
    latency_ms: float


@dataclass
class RunReceipt:
    """The whole agent run (the root of the trace)."""
    task_id: str
    user_id: str
    git_sha: str        # code version this run executed — powers "which change regressed"
    run_status: str     # "success" | "failed"  (failed == an error was thrown)
    total_tokens: int
    total_cost_usd: float
    latency_ms: float
