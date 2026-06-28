# Judge Calibration (D4b)

Proving the LLM-as-judge can be **trusted to gate** — by measuring its agreement with
human labels (Cohen's κ), not assuming it. This is the senior differentiator on top of
the judge itself.

## Calibration results (Claude Haiku judge, n=30: 10 real + 20 degraded)

The loop was run 5 times, tuning the rubrics between runs. κ vs. human labels:

| Round | Faithfulness κ | Completeness κ | Change made |
|-------|---------------|----------------|-------------|
| 1 | 0.41 | −0.08 | baseline rubrics |
| 2 | 0.39 | 0.41 | completeness: explicit truncation = fail |
| 3 | 0.29 | 0.34 | faithfulness: aggressive "any soft reference = fail" — **overshot** (started failing human-pass rows) |
| 4 | 0.52 | 0.34 | reverted to balanced strictness |
| 5 | **0.52** | **0.53** | completeness: judge ONLY answering, ignore correctness (separate the criteria) |

**Where the judge is reliable (safe to gate on):**
- **Fabricated facts/statistics** — every injected-fabrication case caught (faithfulness).
- **Truncated / cut-off reports** — every truncation case caught (completeness).
These are unambiguous; the judge agrees with the human essentially 1:1 on them.

**Where it is advisory only (κ capped ~0.5, do NOT hard-gate):**
- The subjective edge of faithfulness: when search drift puts off-topic junk in the
  *findings* and the report *gestures* at it ("...gaps in X, Y, Z") without making hard
  claims. Human and judge genuinely disagree here — and the human labels themselves were
  not perfectly self-consistent on this boundary. **κ is capped by label consistency, not
  judge competence.** A handful of `real` rows (7, 16, 19, 25) are this case.

**Conclusion:** gate on the crisp failures (fabrication, truncation); treat the subjective
faithfulness edge as an advisory signal, not a merge blocker. κ≈0.5 reflects an honest
moderate-agreement judge on a genuinely ambiguous criterion — raising it further requires
sharper human label definitions, not more rubric text.

> Note on the verdict line printed by `calibrate.py`: it flags κ<0.6 as "UNRELIABLE" as a
> generic guideline. Here that's the correct *cautious* reading — these judges are used to
> gate only their reliable sub-cases (fabrication/truncation) and advise on the rest.

## The pieces

| File | Role |
|------|------|
| `../eval_harness/judge.py` | The two judges (faithfulness, completion). **Same module the gate uses** — so we calibrate the judge that actually gates. |
| `calibration_set.jsonl` | Hand-labeled cases: question + report + findings + your `human_faithful` / `human_complete` labels. |
| `calibrate.py` | Runs the judges on the set, computes Cohen's κ (judge vs human) per criterion. |
| `langfuse_push.py` | Pushes the set to Langfuse as a dataset for UI annotation (optional, for scale). |
| `calibration_history.jsonl` | Appended each run — track κ over time. |

## Workflow

1. **Build the set (automated capture).** Don't hand-paste — run the agent and let it
   write the rows for you:

   ```bash
   cd evals
   set -a; . ../.env.local; set +a
   RUN_REAL_SUPERVISOR=1 python3 calibration/capture_set.py --degrade
   ```

   This runs the agent on the seed questions and writes `calibration_set.jsonl` with
   `{question, report, findings}` filled and the two label fields **blank**. `--degrade`
   also emits deliberately-broken variants (injected fabrication, truncation) so you have
   genuine **fail** cases. Then open the file and fill `human_faithful` / `human_complete`
   (`pass`/`fail`) for each row. **A mix of pass and fail is required** — an all-pass set
   can't calibrate (κ is undefined/0 when one rater never varies). Blank rows are skipped.

2. **Run calibration** (needs `ANTHROPIC_API_KEY` — the Haiku judge calls the model):

   ```bash
   cd evals
   set -a; . ../.env.local; set +a
   python3 calibration/calibrate.py
   ```

   Reads κ per criterion:
   - `κ < 0.6` → **unreliable**: revise the rubric or add few-shot examples, re-measure.
   - `0.6–0.85` → usable.
   - `> 0.85` → strong; the judge can carry weight in the gate.

3. **Iterate.** If κ is low, tune the rubric in `judge.py` (or fill `FAITHFULNESS_FEWSHOT` /
   `COMPLETION_FEWSHOT` with calibrated examples drawn from disagreements), then re-run.
   κ is appended to `calibration_history.jsonl` so you can see it improve.

4. **(Optional) Langfuse UI labeling** — for multi-reviewer or larger sets:

   ```bash
   cd evals
   set -a; . ../.env.local; set +a    # LANGFUSE_* keys
   python3 calibration/langfuse_push.py
   ```

   Then annotate in Langfuse → Datasets / Annotation Queues, and reflect labels back
   into `calibration_set.jsonl` for `calibrate.py`.

## How the judge connects to the gate

`promptfooconfig.yaml` calls `judge_faithfulness` / `judge_completion` from
`eval_harness/judge.py` as `python` asserts (scoped to run when `ANTHROPIC_API_KEY` is set;
they no-op on keyless/fake CI runs). Because calibration runs the **same functions**, a
κ you measure here is the κ of the judge that gates your PRs.

## Note on environment

The judge needs network access to the Anthropic API (Claude Haiku), `capture_set.py`
needs OpenAI + Tavily (it runs the agent), and `langfuse_push.py` needs Langfuse Cloud.
Run them on a machine with that access (not a restricted CI sandbox). The κ math in
`calibrate.py` is pure-Python and unit-checked (matches the textbook 0.400 example), so
the calibration logic itself is verifiable offline.
