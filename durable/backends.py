"""
Demo backends. Durability is orthogonal to retrieval, so the crash demo can run without Tavily credits.

SEARCH_BACKEND=fake        replace the Tavily client in supervisor.nodes with canned results
SEARCH_DELAY_S=<seconds>   sleep inside every search call (see also DEMO_STEP_DELAY_S in workflow.py)
Default (unset): real Tavily, untouched. The LLM is always real.
"""
from __future__ import annotations

import os
import time


class FakeTavily:
    """Same surface as the LangChain Tavily tool that search_worker uses: invoke(query) -> list of dicts."""

    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.calls = 0

    def invoke(self, query: str) -> list:
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        return [
            {"url": f"https://example.test/{self.calls}/a",
             "content": f"Canned source A for '{query[:80]}': core definition, main trade-offs, and one worked example."},
            {"url": f"https://example.test/{self.calls}/b",
             "content": f"Canned source B for '{query[:80]}': common failure modes, measurements, and operational advice."},
        ]


def configure_backends() -> str:
    """Call once at app start. Returns the active search backend name."""
    from supervisor import nodes  # heavy import: kept out of module import time
    if os.getenv("SEARCH_BACKEND", "tavily").lower() == "fake":
        nodes.tavily = FakeTavily(delay_s=float(os.getenv("SEARCH_DELAY_S", "0")))
        return "fake"
    return "tavily"
