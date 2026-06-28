"""
Print a per-assertion pass/fail summary from a promptfoo JSON result file.

The default promptfoo console table shows only [PASS]/[FAIL] + output text — it does
NOT say WHICH assertion failed. This script makes the CI log explicit, so you can tell
at a glance whether a PR was blocked by a deterministic check (order / termination /
budget / sections) or by an LLM-as-judge (faithfulness / completion).

Usage:  python3 report_failures.py output.json
"""

from __future__ import annotations
import sys
import json
from collections import Counter


def label_of(reason: str) -> str:
    """Bucket an assertion by its reason text so we can tally what's blocking."""
    r = reason.lower()
    if "trajectory" in r or "order ok" in r:
        return "trajectory/order"
    if "terminate" in r:
        return "termination"
    if "budget" in r or ("steps=" in r and "tokens=" in r):
        return "budget"
    if "faithful" in r:
        return "judge:faithfulness"
    if "completion" in r or "complete" in r:
        return "judge:completion"
    if "contain" in r or "expected output to contain" in r:
        return "output-contains"
    return "other"


def main(path: str) -> int:
    data = json.load(open(path))
    results = data.get("results", {}).get("results", [])

    blocking = Counter()   # which assertion types caused failures
    any_fail = False

    print("\n" + "=" * 70)
    print("PER-ASSERTION RESULT  (what passed / what blocked each case)")
    print("=" * 70)

    for i, r in enumerate(results, 1):
        q = (r.get("vars", {}) or {}).get("question", "")[:50]
        comps = r.get("gradingResult", {}).get("componentResults", []) or []
        print(f"\n[{i}] {q}")
        for c in comps:
            ok = c.get("pass")
            reason = (c.get("reason", "") or "").replace("\n", " ")[:90]
            tag = label_of(reason)
            mark = "PASS" if ok else "FAIL"
            print(f"    {mark:4} | {tag:20} | {reason}")
            if not ok:
                any_fail = True
                blocking[tag] += 1

    print("\n" + "=" * 70)
    if any_fail:
        print("GATE RESULT: BLOCKED. Failures by assertion type:")
        for tag, n in blocking.most_common():
            print(f"   {n:3} × {tag}")
        print("\n(If 'judge:*' appears, the LLM-as-judge blocked it. If only "
              "trajectory/termination/budget/output-contains, the deterministic gate did.)")
    else:
        print("GATE RESULT: PASS — all assertions green.")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "output.json"))
