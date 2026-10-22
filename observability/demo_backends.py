"""
demo_backends.py — canned search as a real LangChain tool, for Sentinel demos without Tavily credits.

Why not durable/backends.FakeTavily: that object is not a LangChain tool, so the Sentinel
callback never sees a tool call (no ToolReceipt, no tool-error signal). This one is a @tool,
so on_tool_start / on_tool_end / on_tool_error fire exactly like the real Tavily tool.

Env (read by run_traced when SEARCH_BACKEND=fake):
  SENTINEL_FAULT_EVERY=<n>   fault-injection DRILL: every n-th search call raises (0/unset = never).
                             Used to demonstrate the regression signals; label it as a drill.
  SEARCH_DELAY_S=<seconds>   sleep inside every search call (latency drill).
The LLM stays real.
"""
from __future__ import annotations

import os
import time

from langchain_core.tools import tool

_calls = {"n": 0}


class InjectedFault(RuntimeError):
    """Raised by the fault-injection drill (never in normal runs)."""


@tool("canned_search")
def canned_search(query: str) -> list:
    """Return two canned search results for the query (demo backend, no network)."""
    _calls["n"] += 1
    delay = float(os.getenv("SEARCH_DELAY_S", "0") or 0)
    if delay:
        time.sleep(delay)
    every = int(os.getenv("SENTINEL_FAULT_EVERY", "0") or 0)
    if every and _calls["n"] % every == 0:
        raise InjectedFault("fault-injection drill: canned_search failed on purpose")
    return [
        {"url": f"https://example.test/{_calls['n']}/a",
         "content": f"Canned source A for '{query[:80]}': core definition, main trade-offs, and one worked example."},
        {"url": f"https://example.test/{_calls['n']}/b",
         "content": f"Canned source B for '{query[:80]}': common failure modes, measurements, and operational advice."},
    ]


def configure_search() -> str:
    """Swap supervisor.nodes.tavily for canned_search when SEARCH_BACKEND=fake. Returns the backend name."""
    if os.getenv("SEARCH_BACKEND", "tavily").lower() != "fake":
        return "tavily"
    from supervisor import nodes  # heavy import, only when asked
    nodes.tavily = canned_search
    return "fake"
