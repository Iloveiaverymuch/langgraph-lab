"""
Deterministic fake supervisor — no LLM, no network.

Why it exists:
- Lets the gate's OWN logic be tested without API keys or spend, and is the seam we
  use to inject deliberate regressions (FAKE_BROKEN) to prove the gate goes red.
- The real supervisor runs when RUN_REAL_SUPERVISOR=1 + OPENAI_API_KEY (see provider.py).

It reproduces the real supervisor's final_state shape: tagged messages + `next` + counters.
"""

from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass
class _Msg:
    """Mimics a LangChain AIMessage just enough for trajectory reconstruction."""
    content: str
    name: str | None = None


_CANNED_REPORT = """## Executive Summary
Deterministic fixture report produced by the fake supervisor.

## Key Findings
- Finding A
- Finding B

## Analysis
Trade-offs considered.

## Conclusion
Done."""


def run_fake(question: str) -> dict:
    """Return a final_state dict matching the real supervisor's schema.

    Regression injection via FAKE_BROKEN:
      skip_analyst | no_finish | loop | empty_report
    Unset / "" = healthy canonical run.
    """
    broken = os.environ.get("FAKE_BROKEN", "")
    messages = [_Msg(content=question)]  # HumanMessage stand-in (no name)

    if broken == "skip_analyst":
        messages += [
            _Msg("search findings", "search_worker"),
            _Msg(_CANNED_REPORT, "writer_worker"),
        ]
        return {"messages": messages, "next": "FINISH", "search_iterations": 1}

    if broken == "no_finish":
        messages += [
            _Msg("search findings", "search_worker"),
            _Msg("analysis SUFFICIENT", "analyst_worker"),
            _Msg(_CANNED_REPORT, "writer_worker"),
        ]
        return {"messages": messages, "next": "writer_worker", "search_iterations": 1}

    if broken == "loop":
        messages += [
            _Msg("s", "search_worker"),
            _Msg("a NEEDS_MORE", "analyst_worker"),
            _Msg("s", "search_worker"),
            _Msg("a NEEDS_MORE", "analyst_worker"),
            _Msg("s", "search_worker"),
            _Msg("a NEEDS_MORE", "analyst_worker"),
            _Msg(_CANNED_REPORT, "writer_worker"),
        ]
        return {"messages": messages, "next": "FINISH", "search_iterations": 3}

    if broken == "empty_report":
        messages += [
            _Msg("search findings", "search_worker"),
            _Msg("analysis SUFFICIENT", "analyst_worker"),
            _Msg("(report generation failed)", "writer_worker"),
        ]
        return {"messages": messages, "next": "FINISH", "search_iterations": 1}

    # healthy canonical run: search -> analyst -> writer -> FINISH
    messages += [
        _Msg("search findings on the topic", "search_worker"),
        _Msg("coverage is adequate SUFFICIENT", "analyst_worker"),
        _Msg(_CANNED_REPORT, "writer_worker"),
    ]
    return {"messages": messages, "next": "FINISH", "search_iterations": 1}
