"""
D4b — push the calibration set to Langfuse as a dataset for human annotation.

Why: Langfuse gives a UI to review each case and (via annotation queues) attach human
labels with multiple reviewers — better than editing JSONL by hand once the set grows.
This script uploads calibration_set.jsonl as a Langfuse dataset; you then review/label
in the Langfuse UI, and (optionally) export labels back to the JSONL for calibrate.py.

Run on a machine with network access to Langfuse (the sandbox proxy blocks it):
    cd evals
    set -a; . ../.env.local; set +a       # LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST
    python3 calibration/langfuse_push.py

Verified against langfuse SDK 4.x API: create_dataset / create_dataset_item / create_score.
If your SDK differs, the method names are the only thing to adjust.
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET_PATH = HERE / "calibration_set.jsonl"
DATASET_NAME = os.environ.get("LANGFUSE_DATASET", "judge-calibration")


def _host() -> str:
    # tolerate either LANGFUSE_HOST (SDK default) or LANGFUSE_BASE_URL
    return os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"


def load_rows() -> list[dict]:
    rows = []
    for line in SET_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            rows.append(json.loads(line))
    return rows


def main():
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        sys.exit("Missing LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY. Source ../.env.local first.")

    from langfuse import Langfuse
    lf = Langfuse(host=_host())

    rows = load_rows()
    if not rows:
        sys.exit("calibration_set.jsonl is empty.")

    # 1. dataset (idempotent — create_dataset is safe to call repeatedly)
    lf.create_dataset(
        name=DATASET_NAME,
        description="Judge calibration cases (faithfulness + completion). Annotate human labels here.",
    )

    # 2. one item per case. input = what the judge sees; expected_output = human labels (if present)
    for i, row in enumerate(rows, 1):
        lf.create_dataset_item(
            dataset_name=DATASET_NAME,
            input={
                "question": row.get("question", ""),
                "report": row.get("report", ""),
                "findings": row.get("findings", ""),
            },
            expected_output={
                "human_faithful": row.get("human_faithful"),
                "human_complete": row.get("human_complete"),
            },
            metadata={"source": "calibration_set.jsonl", "row": i},
        )

    lf.flush()
    print(f"Pushed {len(rows)} items to Langfuse dataset '{DATASET_NAME}' at {_host()}.")
    print("Open Langfuse → Datasets → annotate, or use Annotation Queues for multi-reviewer labeling.")


if __name__ == "__main__":
    main()
