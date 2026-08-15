"""
W07D2b spike — LangSmith/LangGraph *native* OTel auto-instrumentation → Langfuse.

Purpose: the A/B counterpart to our deliberate callback tracing. Here we write NO
custom callback. We turn on LangSmith's built-in OTel export and let LangChain/
LangGraph auto-emit spans for every chain / node / LLM / tool call. Then we compare
what auto-instrumentation produces against our hand-built `run_traced` spans.

RESULT (2026-08-15): the "reuse our provider" hypothesis was WRONG. With native tracing
on, LangChain posted to api.smith.langchain.com (its own cloud) and failed 401 without a
LangSmith key — it did not honor our Langfuse provider. Conclusion: native = LangSmith
lock-in + mandatory account; we keep deliberate OTLP. See observability/README.md.

How it points at Langfuse (no second endpoint needed):
  LangSmith's OTel mode reuses an EXISTING global TracerProvider if one is set
  (langsmith/client.py). We set ours first via langfuse_export.init_tracer(), which
  points at your Langfuse project — so the auto-spans flow there too.

PREREQUISITES / UNKNOWNS (this is a spike — we're finding out):
  - Enable flags: we set BOTH LANGSMITH_OTEL_ENABLED and OTEL_ENABLED (the code is
    ambiguous about which it reads) plus LANGSMITH_TRACING to turn on LangChain tracing.
  - LangSmith may require a LANGSMITH_API_KEY even to export elsewhere. If it errors or
    emits nothing, grab a free key at smith.langchain.com and set LANGSMITH_API_KEY,
    then re-run. That key requirement is itself a finding (our raw-OTLP path needs none).

Run on your Mac:
    pip install "langsmith[otel]"        # if not already present
    set -a; . ./.env.local; set +a
    python -m observability.native_otel_spike

Then in Langfuse look for a NEW trace whose spans have LangChain-generated names
(e.g. RunnableSequence / ChatOpenAI / tool names) rather than our 'agent_run' /
'llm.call'. Compare its shape + attributes to a `run_traced` trace.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# --- 1. Enable native tracing BEFORE importing langchain/supervisor -----------
# Set only if not already provided by the shell, so you can override.
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
os.environ.setdefault("OTEL_ENABLED", "true")

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # --- 2. Set our Langfuse-pointed provider as the global one ---------------
    # LangSmith's OTel mode will reuse this existing provider, so auto-spans go to Langfuse.
    from . import langfuse_export
    langfuse_export.init_tracer()

    # --- 3. Import + run the agent with NO custom callback (native only) ------
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from langchain_core.messages import HumanMessage
    from supervisor import app

    question = "What are the key tradeoffs between LangGraph and raw agent loops?"
    print("[spike] running supervisor with NATIVE LangSmith-OTel tracing (no custom callback)...")
    app.invoke(
        {"messages": [HumanMessage(content=question)], "next": "", "final_answer": "", "search_iterations": 0},
        config={"recursion_limit": 20},   # note: no 'callbacks' — native path only
    )

    # --- 4. Flush whatever the native path emitted through our provider -------
    langfuse_export.flush()
    print("[spike] done. In Langfuse, find the new trace with LangChain-named spans")
    print("[spike] and compare it to a `run_traced` trace (agent_run / llm.call / sentinel.*).")
    print("[spike] If NOTHING new appeared: native tracing likely needs LANGSMITH_API_KEY — set it and re-run.")


if __name__ == "__main__":
    main()
