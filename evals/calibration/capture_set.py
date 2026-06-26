"""
D4b — build the calibration set from REAL agent runs.

Calibration needs real (question, report, findings) triples to label. Hand-pasting
them is tedious and error-prone, so this script runs the agent (via the gate's own
provider) on a list of questions and writes calibration_set.jsonl with the two human
label fields left BLANK for you to fill.

It also (optionally) emits DEGRADED variants — a fabrication injected into the report,
or a truncated report — so the set contains genuine "fail" cases. A calibration set that
is all-pass cannot produce a meaningful kappa, so you want a deliberate mix.

Run on a machine with the agent's keys (sandbox proxy blocks OpenAI/Tavily):
    cd evals
    set -a; . ../.env.local; set +a
    RUN_REAL_SUPERVISOR=1 python3 calibration/capture_set.py --degrade

Then open calibration_set.jsonl and fill human_faithful / human_complete for each row.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_harness"))
from provider import call_api  # reuse the gate's exact agent invocation  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "calibration_set.jsonl"

# Questions to seed the set. Mix topics; add your own. ~15-20 real runs + degraded
# variants gets you toward a usable 30-40 case set.
QUESTIONS = [
    "What are the key tradeoffs between LangGraph and raw agent loops?",
    "Summarize how vector databases work and compare Pinecone vs Weaviate.",
    "Explain chunking strategies for RAG and their tradeoffs.",
    "What should production observability for LLM agents capture?",
    "When does a multi-agent pipeline beat a single large prompt?",
    "How does a supervisor route between specialist workers?",
    "Strategies to control token cost in agent pipelines.",
    "What are the main failure modes of retrieval-augmented generation?",
    "Compare fine-tuning vs prompt engineering for LLM customization.",
    "How do you evaluate an LLM agent's trajectory, not just its output?",
]


def _row(question, report, findings, note=""):
    return {
        "question": question,
        "report": report,
        "findings": findings,
        # blanks for you to fill: "pass" / "fail"
        "human_faithful": "",
        "human_complete": "",
        "_note": note,  # provenance; ignored by calibrate.py
    }


def _degrade_fabrication(report: str) -> str:
    """Inject a plausible-but-unsupported claim → should be labeled faithful=fail."""
    inject = ("\n\n## Additional Note\n"
              "Notably, this technology was first standardized by the IEEE in 1998 "
              "and is now mandated by EU regulation 2021/447.")  # invented
    return report + inject


def _degrade_truncate(report: str) -> str:
    """Cut the report off early → likely complete=fail (and maybe faithful borderline)."""
    return report[: max(120, len(report) // 4)] + " …[truncated]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrade", action="store_true",
                    help="also emit fabrication + truncation variants (for fail cases)")
    ap.add_argument("--limit", type=int, default=len(QUESTIONS))
    args = ap.parse_args()

    if not os.environ.get("RUN_REAL_SUPERVISOR") == "1":
        print("WARNING: RUN_REAL_SUPERVISOR != 1 — you'll capture FAKE reports, not real ones.")

    rows = []
    for q in QUESTIONS[: args.limit]:
        print(f"running: {q[:60]}...")
        res = call_api(q)
        if res.get("error"):
            print(f"  skipped (error): {res['error']}")
            continue
        report = res["output"]
        findings = res.get("metadata", {}).get("search_findings", "")
        rows.append(_row(q, report, findings, note="real"))
        if args.degrade:
            rows.append(_row(q, _degrade_fabrication(report), findings, note="degraded:fabrication"))
            rows.append(_row(q, _degrade_truncate(report), findings, note="degraded:truncate"))

    header = ("// Calibration set — fill human_faithful / human_complete (\"pass\"/\"fail\") for each row.\n"
              "// Rows marked degraded:* are deliberately broken; expect to label them fail.\n"
              "// Aim for a MIX of pass and fail. Lines starting with // are ignored.\n")
    with open(OUT, "w") as f:
        f.write(header)
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    real = sum(1 for r in rows if r["_note"] == "real")
    degraded = len(rows) - real
    print(f"\nWrote {len(rows)} rows to {OUT.name} ({real} real, {degraded} degraded).")
    print("Now open it and fill the two label fields per row, then run calibrate.py.")


if __name__ == "__main__":
    main()
