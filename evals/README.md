# evals/ — Agent Regression Sentinel (CI eval gate)

A CI gate that **blocks a PR** when the supervisor regresses on its *trajectory* or
*output* — not just its final answer.

> **Status: live and proven on GitHub.** A deliberately-broken PR (supervisor skipping
> the analyst worker) was caught and **blocked** — red check, exit 100, merge disabled —
> with the failure reason `bad trajectory: ["search_worker","search_worker","writer_worker"]`.
> See [Verified end-to-end](#verified-end-to-end).

## What it checks (per frozen case)

| # | Criterion | Source of truth | Catches |
|---|-----------|-----------------|---------|
| 1 | Tools called + order | `metadata.worker_sequence` (ordered subsequence) | wrong routing, skipped specialist |
| 2 | Termination / no loops | `metadata.terminated` + `step_count ≤ 5` | non-termination, routing loops |
| 3 | Token budget | `metadata.total_tokens ≤ 20000` | "right answer, expensive path" |
| 4 | Output contains | required report sections in `output` | empty/malformed deliverable |

The token ceiling (20k) is calibrated to real gpt-4o-mini runs (~12k tokens/case observed).

## How it works

```
.github/workflows/eval-gate.yml         # runs on every PR to master
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

# real agent (costs API calls; ~6 min for 8 cases)
set -a; . ../.env.local; set +a          # loads OPENAI_API_KEY + TAVILY_API_KEY
RUN_REAL_SUPERVISOR=1 npx promptfoo@latest eval -c promptfooconfig.yaml -o result.json

# read failure reasons from the saved run
python3 -c "import json; d=json.load(open('result.json')); [print('—', c.get('reason','')) for r in d['results']['results'] for c in r.get('gradingResult',{}).get('componentResults',[]) if not c.get('pass')]"
```

> `provider.py` only runs the **real** agent when `RUN_REAL_SUPERVISOR=1` **and** an API
> key is present. With no key it silently falls back to the fake — so a "passing" CI run
> with empty secrets is really the fake passing, not the real agent.

## CI: make it actually block PRs

1. The workflow runs on every PR to `master`. By default it runs the **real** agent —
   add repo secrets `OPENAI_API_KEY` and `TAVILY_API_KEY` (Settings → Secrets → Actions).
   To run the free fake instead, set Actions variable `RUN_REAL_SUPERVISOR=0`.
2. A red ✗ is advisory until you add a **branch protection rule**: Settings → Branches →
   Add rule on `master` → require the `eval-gate` status check. *Then* a failing gate
   disables the merge button.

## Verified end-to-end

Proven on real PRs against this repo:

| PR | Agent run | Result | Reason |
|----|-----------|--------|--------|
| Healthy supervisor | real (8 cases) | ✓ 8/8 pass, exit 0 — **merge allowed** | — |
| Broken (skip analyst) | real | ✗ failed, exit 100 — **merge blocked** | `bad trajectory: ["search_worker","search_worker","writer_worker"]` |

The broken PR was caught by the **trajectory** assertion (analyst worker missing from the
path) — a behavioral regression a plain output check would have missed. This is the whole
point: the gate guards *how the agent works*, not just what it returns.

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

## Next

Add one **Haiku LLM-as-judge** assertion (faithfulness / task-completion), scoped to
merge-to-main, then **calibrate** it against human labels in Langfuse (Cohen's κ ≥ 0.7)
before it's allowed to gate.
