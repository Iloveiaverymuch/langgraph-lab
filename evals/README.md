# evals/ — Agent Regression Sentinel (CI eval gate)

A CI gate that **blocks a PR** when the supervisor regresses on its *trajectory* or
*output* — not just its final answer.

> **Status: live and proven on GitHub.** A deliberately-broken PR (supervisor skipping
> the analyst worker) was caught and **blocked** — red check, exit 100, merge disabled —
> with the failure reason `bad trajectory: ["search_worker","search_worker","writer_worker"]`.
> See [Verified end-to-end](#verified-end-to-end).

## What it checks (per frozen case)

Two layers: **deterministic** checks (cheap, exact) and **LLM-as-judge** checks (content quality).

| # | Criterion | Type | Source of truth | Catches |
|---|-----------|------|-----------------|---------|
| 1 | Tools called + order | deterministic | `metadata.worker_sequence` (ordered subsequence) | wrong routing, skipped specialist |
| 2 | Termination / no loops | deterministic | `metadata.terminated` + `step_count ≤ 5` | non-termination, routing loops |
| 3 | Token budget | deterministic | `metadata.total_tokens ≤ 20000` | "right answer, expensive path" |
| 4 | Output contains | deterministic | required report sections in `output` | empty/malformed deliverable |
| 5 | **Faithfulness** | LLM-as-judge | report vs. `metadata.search_findings` | **fabrication / unsupported claims** |
| 6 | **Task completion** | LLM-as-judge | report vs. question | **truncated / off-topic / unanswered** |

The token ceiling (20k) is calibrated to real gpt-4o-mini runs (~12k tokens/case observed).
The two judges run on **Claude Haiku** (a different model family than the agent, to reduce
shared blind spots) and were **calibrated** against human labels — see `calibration/`.

## How it works

```
.github/workflows/eval-gate.yml         # runs on every PR to master
        │
        ▼
evals/promptfooconfig.yaml              # 8 frozen cases + 6 assertion rules
        │ calls file://eval_harness/provider.py
        ▼
provider.py ─► output (str) + context.metadata {worker_sequence, step_count, terminated, total_tokens, search_findings}
   │  ├── trajectory.py        # reconstruct path + extract search_findings from message tags
   │  ├── fake_supervisor.py   # deterministic fixture (+ FAKE_BROKEN switch)
   │  └── judge.py             # faithfulness + completion judges (Claude Haiku)
   │        (real agent: RUN_REAL_SUPERVISOR=1 + OPENAI_API_KEY/TAVILY_API_KEY)
   │        (judges:      ANTHROPIC_API_KEY — else they no-op/skip)
   ▼
supervisor/  (this repo, UNCHANGED)     # the real compiled LangGraph graph
        │ any assertion fails → exit non-zero → job red → merge blocked
        ▼
report_failures.py             # prints per-assertion PASS/FAIL + which type blocked
```

The provider imports the repo's own `supervisor` package — **no changes to agent code**.
Tokens are captured with a LangChain `UsageMetadataCallbackHandler`. The judges call the
same `judge.py` used by `calibration/` — so the gated judge is the calibrated judge.

After every run, **`report_failures.py`** parses the JSON result and prints a per-assertion
breakdown to the CI log, so you can see whether a PR was blocked by a deterministic check
or by the LLM-as-judge (and which one).

## Run locally

From the **repo root**. Activate the venv first — the provider runs the supervisor in
a Python subprocess, so its deps (langgraph, langchain-community, …) must be importable:

```bash
source .venv/bin/activate                 # REQUIRED — else: ModuleNotFoundError: langgraph
pip install -r requirements.txt -r evals/requirements.txt
cd evals

# fake (free, deterministic) — proves the gate logic
npx promptfoo@latest eval -c promptfooconfig.yaml ; echo "exit: $?"      # 8 passed, exit 0

# simulate regressions → red, exit 100
FAKE_BROKEN=skip_analyst  npx promptfoo@latest eval -c promptfooconfig.yaml ; echo "exit: $?"
FAKE_BROKEN=no_finish     npx promptfoo@latest eval -c promptfooconfig.yaml ; echo "exit: $?"
FAKE_BROKEN=loop          npx promptfoo@latest eval -c promptfooconfig.yaml ; echo "exit: $?"
FAKE_BROKEN=empty_report  npx promptfoo@latest eval -c promptfooconfig.yaml ; echo "exit: $?"

# real agent + judges (costs API calls; ~6 min for 8 cases)
set -a; . ../.env.local; set +a          # OPENAI_API_KEY + TAVILY_API_KEY (agent) + ANTHROPIC_API_KEY (judges)
RUN_REAL_SUPERVISOR=1 npx promptfoo@latest eval -c promptfooconfig.yaml -o result.json

# print which assertion blocked each case (deterministic vs. judge)
python3 report_failures.py result.json
```

> `provider.py` only runs the **real** agent when `RUN_REAL_SUPERVISOR=1` **and**
> `OPENAI_API_KEY` is present; otherwise it falls back to the fake. The **judges** only
> run when `ANTHROPIC_API_KEY` is present; otherwise they no-op (skip = pass). So a
> "passing" CI run with empty secrets is the fake + skipped judges — not a real eval.

## CI: make it actually block PRs

1. The workflow runs on every PR to `master`. By default it runs the **real** agent +
   judges — add **three** repo secrets (Settings → Secrets → Actions):
   `OPENAI_API_KEY` + `TAVILY_API_KEY` (agent) and `ANTHROPIC_API_KEY` (Haiku judges).
   Missing the Anthropic key ⇒ judges silently skip (gate still runs deterministic checks).
   To run the free fake instead, set Actions variable `RUN_REAL_SUPERVISOR=0`.
2. A red ✗ is advisory until you add a **branch protection rule**: Settings → Branches →
   Add rule on `master` → require the `eval-gate` status check. *Then* a failing gate
   disables the merge button.

## Verified end-to-end

Proven on real PRs against this repo:

| PR | What broke | Blocked by | Result |
|----|-----------|-----------|--------|
| Healthy supervisor | nothing | — | ✓ pass, exit 0 — **merge allowed** |
| Skip-analyst | routing (analyst dropped) | **deterministic: trajectory** | ✗ exit 100 — **blocked** (`bad trajectory: [...]`) |
| Fabrication-in-writer | injected fake statistics | **LLM-as-judge: faithfulness** | ✗ exit 100 — **blocked** (8× faithfulness `[2/5] unsupported claim`) |

The fabrication PR is the key result: the reports had **valid structure, trajectory,
termination, and budget** (all deterministic checks passed) — yet the gate still blocked
the merge because the **faithfulness judge** caught the invented statistics. A plain
output check could never catch that. The CI log's per-assertion summary made the cause
explicit:

```
GATE RESULT: BLOCKED. Failures by assertion type:
     8 × judge:faithfulness
     3 × judge:completion
     2 × output-contains
     1 × trajectory/order
```

## Known limitation: non-determinism on real PRs

The real supervisor routes via an LLM following a prompt, so the trajectory varies
run-to-run. In the broken-PR test, CI saw 2/8 fail while a local re-run saw 1/8 fail —
same code, different counts. Implications:

- A genuinely-broken PR is still reliably blocked (one failure → exit 100).
- But a *healthy* PR can occasionally flip red by chance, which is noisy for a required check.

**Recommended hardening:** run the deterministic fake on PRs (`RUN_REAL_SUPERVISOR=0`)
and run the real agent on a nightly schedule. The fake exercises the full gate logic
deterministically; the nightly real run catches genuine behavioral drift without gating
every PR on a flaky signal.

## Calibration

The two judges are calibrated against human labels (Cohen's κ) — see **`calibration/`**.
Current agreement: faithfulness κ≈0.52, completion κ≈0.53 (moderate). The judges are
reliable on crisp failures (fabrication, truncation) and advisory on subjective edges.
Run `python3 calibration/calibrate.py` to re-measure after any rubric change.

## Possible next steps

- Raise judge κ toward 0.7 with sharper human-label definitions on the subjective edge.
- Mitigate real-agent non-determinism: fake-on-PR + real-agent nightly (see below).
