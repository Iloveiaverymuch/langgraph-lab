# evals/ — Agent Regression Sentinel (CI eval gate)

A CI gate that **blocks a PR** when the supervisor regresses on its *trajectory* or
*output* — not just its final answer.

## What it checks (per frozen case)

| # | Criterion | Source of truth | Catches |
|---|-----------|-----------------|---------|
| 1 | Tools called + order | `metadata.worker_sequence` (ordered subsequence) | wrong routing, skipped specialist |
| 2 | Termination / no loops | `metadata.terminated` + `step_count ≤ 5` | non-termination, routing loops |
| 3 | Token budget | `metadata.total_tokens ≤ 8000` | "right answer, expensive path" |
| 4 | Output contains | required report sections in `output` | empty/malformed deliverable |

## How it works

```
.github/workflows/eval-gate.yml         # runs on every PR to main
        │
        ▼
evals/promptfooconfig.yaml              # 8 frozen cases + 4 assertion rules
        │ calls file://eval_harness/provider.py
        ▼
provider.py ─► output (str) + context.metadata {worker_sequence, step_count, terminated, total_tokens}
   │  ├── trajectory.py        # reconstruct path from message name-tags
   │  └── fake_supervisor.py   # deterministic fixture (+ FAKE_BROKEN switch)
   │        (or RUN_REAL_SUPERVISOR=1 + OPENAI_API_KEY)
   ▼
supervisor/  (this repo, UNCHANGED)     # the real compiled LangGraph graph
        │ any assertion fails → exit non-zero → job red → merge blocked
```

The provider imports the repo's own `supervisor` package — **no changes to agent code**.
Tokens are captured with a LangChain `UsageMetadataCallbackHandler`.

## Run locally

From the **repo root**:

```bash
pip install -r requirements.txt -r evals/requirements.txt
cd evals

# fake (free, deterministic) — proves the gate logic
npx promptfoo@latest eval -c promptfooconfig.yaml ; echo "exit: $?"      # 8 passed, exit 0

# simulate regressions → red, exit 100
FAKE_BROKEN=skip_analyst  npx promptfoo@latest eval -c promptfooconfig.yaml ; echo "exit: $?"
FAKE_BROKEN=no_finish     npx promptfoo@latest eval -c promptfooconfig.yaml ; echo "exit: $?"
FAKE_BROKEN=loop          npx promptfoo@latest eval -c promptfooconfig.yaml ; echo "exit: $?"
FAKE_BROKEN=empty_report  npx promptfoo@latest eval -c promptfooconfig.yaml ; echo "exit: $?"

# real agent (costs API calls)
export OPENAI_API_KEY=... TAVILY_API_KEY=... RUN_REAL_SUPERVISOR=1
npx promptfoo@latest eval -c promptfooconfig.yaml
```

## CI: make it actually block PRs

1. The workflow runs on every PR to `master`. By default it runs the **real** agent —
   add repo secrets `OPENAI_API_KEY` and `TAVILY_API_KEY` (Settings → Secrets → Actions).
   To run the free fake instead, set Actions variable `RUN_REAL_SUPERVISOR=0`.
2. A red ✗ is advisory until you add a **branch protection rule**: Settings → Branches →
   Add rule on `master` → require the `eval-gate` status check. *Then* a failing gate
   disables the merge button.

## Note on real-agent-on-every-PR

The real supervisor is **non-deterministic** (routing + writer output vary run-to-run),
so a passing PR can occasionally flip red on re-run. If that flakiness becomes noise,
switch to `RUN_REAL_SUPERVISOR=0` on PRs and run the real agent on a nightly schedule
instead — the fake still exercises the full gate logic deterministically.

## Next

Add one **Haiku LLM-as-judge** assertion (faithfulness / task-completion), scoped to
merge-to-main, then **calibrate** it against human labels in Langfuse (Cohen's κ ≥ 0.7)
before it's allowed to gate.
