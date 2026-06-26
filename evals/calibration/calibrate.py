"""
D4b — Judge calibration loop.

Measures agreement between the LLM-as-judge and human labels using Cohen's kappa.
This is the senior-differentiator: it proves the judge can be trusted to gate, rather
than assuming it.

Workflow:
  1. Hand-label cases in calibration_set.jsonl (human "pass"/"fail" per criterion).
  2. Run this script: it runs the SAME judge functions used by the gate (eval_harness/judge.py)
     on each case, then computes Cohen's kappa (judge vs human) per criterion.
  3. Read kappa:
        kappa < 0.6  -> judge unreliable; revise the rubric / add few-shot, re-measure.
        0.6-0.85     -> usable; tighten if it gates high-stakes merges.
        > 0.85       -> strong; judge can carry weight in the CI gate.
  4. Track kappa over time (results appended to calibration_history.jsonl).

Run (on a machine with ANTHROPIC_API_KEY, since the Haiku judge calls the model):
    cd evals && python3 calibration/calibrate.py

No sklearn dependency — Cohen's kappa is implemented directly and unit-checked.
"""

from __future__ import annotations
import os
import sys
import json
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_harness"))
from judge import judge_faithfulness, judge_completion  # noqa: E402

HERE = Path(__file__).resolve().parent
SET_PATH = HERE / "calibration_set.jsonl"
HISTORY_PATH = HERE / "calibration_history.jsonl"


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two raters over binary labels (1=pass, 0=fail).
    kappa = (po - pe) / (1 - pe), where po=observed agreement, pe=chance agreement."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    # marginal probabilities
    pa1, pb1 = sum(a) / n, sum(b) / n
    pa0, pb0 = 1 - pa1, 1 - pb1
    pe = pa1 * pb1 + pa0 * pb0
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0  # degenerate: all one class
    return (po - pe) / (1 - pe)


def _to_bin(label) -> int:
    """Normalize a human/judge label to 1 (pass) / 0 (fail)."""
    if isinstance(label, bool):
        return 1 if label else 0
    s = str(label).strip().lower()
    return 1 if s in ("pass", "1", "true", "yes", "y") else 0


def load_set() -> list[dict]:
    if not SET_PATH.exists():
        print(f"No calibration set at {SET_PATH}. See calibration_set.jsonl template.")
        return []
    rows = []
    for line in SET_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            rows.append(json.loads(line))
    return rows


def run():
    rows = load_set()
    if not rows:
        return

    faith_human, faith_judge = [], []
    comp_human, comp_judge = [], []

    for i, row in enumerate(rows, 1):
        report = row.get("report", "")
        findings = row.get("findings", "")
        question = row.get("question", "")

        # Only score rows that have a NON-BLANK human label (blank = not yet labeled).
        if str(row.get("human_faithful", "")).strip():
            jr = judge_faithfulness(report, findings)
            faith_human.append(_to_bin(row["human_faithful"]))
            faith_judge.append(1 if jr["pass"] else 0)
            print(f"[{i}] faithful  human={row['human_faithful']:<5} judge={'pass' if jr['pass'] else 'fail':<5} {jr['reason'][:60]}")

        if str(row.get("human_complete", "")).strip():
            jr = judge_completion(report, question)
            comp_human.append(_to_bin(row["human_complete"]))
            comp_judge.append(1 if jr["pass"] else 0)
            print(f"[{i}] complete  human={row['human_complete']:<5} judge={'pass' if jr['pass'] else 'fail':<5} {jr['reason'][:60]}")

    print("\n=== Cohen's kappa (judge vs human) ===")
    results = {"timestamp": datetime.datetime.utcnow().isoformat() + "Z", "n": len(rows)}
    if faith_human:
        k = cohens_kappa(faith_human, faith_judge)
        results["faithfulness_kappa"] = round(k, 3)
        print(f"faithfulness: kappa={k:.3f}  (n={len(faith_human)})  {_verdict(k)}")
    if comp_human:
        k = cohens_kappa(comp_human, comp_judge)
        results["completion_kappa"] = round(k, 3)
        print(f"completion:   kappa={k:.3f}  (n={len(comp_human)})  {_verdict(k)}")

    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(results) + "\n")
    print(f"\nAppended to {HISTORY_PATH.name} (tracking agreement over time).")


def _verdict(k: float) -> str:
    if k != k:  # nan
        return "n/a"
    if k < 0.6:
        return "UNRELIABLE — revise rubric / add few-shot"
    if k < 0.85:
        return "usable"
    return "STRONG — judge can gate"


if __name__ == "__main__":
    run()
