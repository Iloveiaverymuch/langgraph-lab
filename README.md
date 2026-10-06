# langgraph-lab — multi-agent patterns, a CI eval gate, runtime tracing, and prompt-injection hardening

A LangGraph lab that grew into an **Agent Regression Sentinel**: two multi-agent implementations
(message-passing and blackboard) plus the tooling to keep an agent honest in production.

| Part | What it is | Where |
|---|---|---|
| Agents | Supervisor/worker research agent (message-passing) and a blackboard variant | [`supervisor/`](supervisor/), [`blackboard/`](blackboard/) |
| **Pre-merge gate** | Promptfoo CI gate: deterministic trajectory/budget checks + calibrated LLM judges | [`evals/`](evals/README.md) |
| **Post-deploy watch** | Every LLM/tool call → typed receipt → OpenTelemetry-GenAI → Langfuse | [`observability/`](observability/README.md) |
| **Hardening** | 30-attack prompt-injection suite, frozen baseline, and 3 defensive layers | [`supervisor/README.md`](supervisor/README.md) |

## Headline results

- **Eval gate, proven on real PRs:** a fabrication injected into the writer (valid structure, valid
  trajectory, in budget) was blocked by the faithfulness judge; a skipped-analyst PR was blocked by the
  deterministic trajectory check.
- **Prompt injection (30 attacks × 3 runs, pass = agent resisted):** baseline **45/90 (50%)** →
  with the L1 transition guard **66/90 (73%)**, live. Control-flow attacks are fully fixed
  (tool-confusion 10/24 → 24/24). The remaining failures are content-level and are handled by an output
  guard (offline replay: exfiltration leaks 21 → 0 of 24 recorded runs) and an input guard (writer-only
  probe: 10/10 → 0/10 on the remaining hijack). The end-to-end rate of all three layers together is
  **not yet re-measured live** (search-API credits exhausted); numbers and caveats are in
  [`supervisor/README.md`](supervisor/README.md).

## Implementations

### `supervisor/` — message-passing

Workers share information only through `state["messages"]`.

```
START → supervisor ⇢ search_worker  → supervisor
                   ⇢ analyst_worker → supervisor
                   ⇢ writer_worker  → supervisor
                   ⇢ output_guard → END
```

- **search_worker** — queries Tavily, synthesizes real web results into structured findings
- **analyst_worker** — assesses coverage and gaps, ends with `SUFFICIENT` or `NEEDS_MORE: [gap]`
- **writer_worker** — synthesizes everything into a structured report
- **supervisor** — an LLM *proposes* the next worker; a deterministic transition guard *decides* (see
  [`supervisor/README.md`](supervisor/README.md)). The loop ends when the analyst approves coverage or
  the search cap (`MAX_SEARCH_ITERATIONS = 2`) is hit.
- **output_guard** — deterministic last line of defence on the final report

### `blackboard/` — blackboard memory

Workers write to named typed fields (`findings`, `code`, `critique`); the supervisor routes by reading
those fields directly — no LLM call, no message parsing.

**Pipeline:** `researcher → critic` (research/review) or `researcher → coder → critic` (code).
It ends at `critic` on purpose: the point is to explore the blackboard pattern and dynamic routing across
task types, not to reproduce the message-passing output shape. The memory mechanism is the variable.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.9+ (developed on 3.9; CI uses 3.11)
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...  TAVILY_API_KEY=tvly-...     # or keep them in .env.local (git-ignored)

python main.py                                            # blackboard 5-task suite (default)
python main.py --mode message_passing                     # supervisor agent
python main.py --mode blackboard --task "Write a retry decorator in Python"
python main.py --mode compare                             # both on one question (outputs differ by design)

# offline tests (no keys, seconds)
python supervisor/tests/test_guard.py && python supervisor/tests/test_input_guard.py && python supervisor/tests/test_output_guard.py
```

Tune the caps: `MAX_SEARCH_ITERATIONS` in `supervisor/state.py` (default 2), `MAX_ITERATIONS` in
`blackboard/state.py` (default 6).

## Eval gate (CI)

A Promptfoo gate that blocks a PR when the supervisor regresses, on two layers:

- **Deterministic checks** — worker order, termination/no-loops, token budget, report sections.
- **LLM-as-judge checks** (Claude Haiku, calibrated against human labels, Cohen's κ) — **faithfulness**
  (no fabricated/unsupported claims) and **task completion**.

The CI log prints a per-assertion summary showing exactly which check blocked. Offline unit tests run on
every PR in `.github/workflows/unit-tests.yml`. Full design:
**[`evals/README.md`](evals/README.md)**; judge calibration:
**[`evals/calibration/README.md`](evals/calibration/README.md)**.

## Observability (runtime tracing → Langfuse)

The eval gate is the *pre-merge* half; `observability/` is the *post-deploy* half. It watches the running
agent and surfaces drift the gate can't see (a provider silently updating a model, latency creeping up,
cost doubling) — none of which open a PR.

```bash
pip install -r observability/requirements.txt
set -a; . ./.env.local; set +a                          # LANGFUSE_* + OPENAI/TAVILY keys
python -m observability.smoke_test_export               # prove the Langfuse pipe (fake data)
python -m observability.run_traced "your question"      # trace a real supervisor run
python -m observability.replay <trace_id>               # rebuild a past run's receipts
```

Each run is pinned to its git commit (`sentinel.git_sha`) so a regression can be attributed to the change
that caused it. Details: **[`observability/README.md`](observability/README.md)**.

## Project structure

```
langgraph-lab/
├── main.py                    entry point (--mode selector)
├── supervisor/                message-passing agent + hardening layers  → supervisor/README.md
│   ├── state.py  nodes.py  graph.py
│   ├── guard.py               L1 transition guard      (pure python)
│   ├── input_guard.py         L2 input defences        (pure python)
│   ├── output_guard.py        L3 output guard          (pure python)
│   └── tests/                 offline tests for L1/L2/L3
├── blackboard/                blackboard-memory agent
├── evals/                     Agent Regression Sentinel                  → evals/README.md
│   ├── promptfooconfig.yaml            quality gate (8 frozen cases)
│   ├── promptfooconfig.injection.yaml  prompt-injection suite (30 attacks)
│   ├── eval_harness/                   providers, judges, trajectory, detectors
│   ├── calibration/                    judge-vs-human calibration (κ)
│   ├── baselines/                      FROZEN injection baseline (+ exfil rerun): diff against, never overwrite
│   ├── results/                        committed D3 run (evidence for the L1 numbers)
│   └── replay_l3.py  probe_writer_l2.py  report_*.py
├── observability/             runtime tracing → Langfuse                 → observability/README.md
└── .github/workflows/         eval-gate.yml (paid, real agent) · unit-tests.yml (offline)
```

## Key design decisions

1. **Supervisor owns routing, workers own content.** Workers never touch `state["next"]`; the supervisor
   does no substantive work. Nodes write facts, edges make decisions (`route_supervisor` is separate from
   `supervisor_node`, so topology stays declarative and inspectable).
2. **Infrastructure enforces constraints, not prompts.** The search cap, the transition guard and the
   output guard are code. Prompts only propose.
3. **Gap-targeted second search.** On `NEEDS_MORE: [gap]` the gap text becomes the next Tavily query.
4. **Worker factory.** `make_worker(prompt, name)` removes the repeated LLM-call structure; `search_worker`
   is hand-written because it calls Tavily first.
5. **Measure before keeping a layer.** The input-side "fence the question as data" defence was dropped
   because it showed no effect and caused over-blocking; only measured fixes shipped.

State is a `TypedDict` with reducers (append-only `messages`, last-write-wins `next`/`final_answer`,
additive `search_iterations`) — the typed contract between nodes.

## Problems encountered (supervisor, early lab)

1. **Supervisor stopped after one worker.** `gpt-4o-mini` ignored vague routing rules. Fix: a numbered
   decision tree with explicit `→ output:` lines. *Lesson:* small models need unambiguous routing prompts —
   later made moot by moving the decision into code (L1).
2. **Infinite analyst loop after adding the cap.** The cap waited for an LLM approval the analyst's prompt
   biased against. Fix: once capped, bypass the analyst and force the writer. *Lesson:* don't combine a hard
   cap with an LLM approval gate; one control mechanism per decision point.
3. **Hallucinated results.** `search_worker` called the LLM with no tools. Fix: wire in Tavily (query →
   search → LLM synthesis with source URLs). *Lesson:* an agent loop without grounded retrieval is an
   expensive hallucination engine.
4. **Off-topic second query** *(open).* The extracted gap bullet is used verbatim as the Tavily query and can
   drift off-topic. Planned fix: a small LLM call that rewrites the gap into a focused query.
