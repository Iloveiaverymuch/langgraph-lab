"""
W07D3 step 0 — diagnostic (read-only, ~10 cheap LLM calls, no Tavily, no tracing).

For each attack whose baseline failure was an ABORT (output = question echoed back),
call the supervisor LLM exactly as supervisor_node does at step 0 and print the RAW
output. Tells us which mechanism produced the abort:
  - INVALID word  -> supervisor_node's fail-open 'unexpected output -> FINISH'
  - FINISH        -> a VALID but premature FINISH (attacker-chosen), which fail-closed alone would not catch

Run from repo root:  set -a; . ./.env.local; set +a; python3 evals/diag_supervisor_routing.py
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from supervisor.nodes import SUPERVISOR_PROMPT, WORKERS, llm  # noqa: E402

TARGETS = ["ignore-previous 7", "system-prompt-leak 3", "tool-confusion 1", "tool-confusion 4", "tool-confusion 8"]
REPS = 2

cfg = yaml.safe_load((ROOT / "evals" / "promptfooconfig.injection.yaml").read_text())
tests = {t["description"]: t["vars"]["question"] for t in cfg["tests"]}

for key in TARGETS:
    desc = next(d for d in tests if key in d)
    q = tests[desc]
    outs = []
    for _ in range(REPS):
        raw = llm.invoke([SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=q)]).content.strip()
        kind = "VALID-FINISH" if raw == "FINISH" else ("valid-worker" if raw in WORKERS else "INVALID->fail-open FINISH")
        outs.append((raw[:60], kind))
    print(f"\n{desc}")
    for raw, kind in outs:
        print(f"   raw={raw!r:62} {kind}")
