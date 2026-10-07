# durable/ — the supervisor agent as an Inngest durable function (W07D4)

The same agent (`supervisor/`, untouched), run as **one Inngest function in which every worker is a durable
step**. Kill the process mid-run and the run resumes where it stopped instead of starting over; a human
approval gate sits before the report is released; a re-sent trigger does not start a second run.

```
research/requested ──► run-research (Inngest function)
                         loop: next_step(messages)           pure function of saved messages (L1 guard + search cap)
                               step.run("search_worker-1")    each step = one LLM / search call, result saved as JSON
                               step.run("analyst_worker-1")
                               ...
                         step.run("output-guard")            L3 output guard
                         step.wait_for_event("await-approval")   pauses (timeout), resumes on research/report.decision
                         ──► {"status": "released" | "rejected" | "timed_out", "report": ...}
```

## Mental model

A durable function is **re-run from the top after any interruption, but every finished `step.run` returns its saved
result instead of executing again.** Therefore: side effects (LLM, search) live inside step handlers only; the code
between steps must be deterministic; step ids must be stable (`search_worker-1`, `-2`, ...). Steps are
**at-least-once**: a step that dies after its side effect but before its result is saved runs again.

Why this fits the agent: after the L1 work, routing is a pure function of "who spoke last" (`routing.next_step`).
The graph's supervisor LLM call never changes the outcome (L1 overrides it), so the durable loop skips it.
`tests/test_workflow.py` asserts `next_step` agrees with the graph's `supervisor_node` for every reachable state,
whatever the LLM proposes.

## Files

| file | role |
|---|---|
| `workflow.py` | the Inngest client + `run_research` function (steps, approval gate, idempotency) |
| `routing.py` | deterministic next-step logic (L1 guard + search cap) |
| `codec.py` | LangChain messages <-> JSON dicts (step results must be JSON) |
| `backends.py` | demo-only canned search (`SEARCH_BACKEND=fake`) |
| `app.py` | FastAPI app serving the function at `/api/inngest` |
| `send.py` | CLI to send the trigger / approval events |
| `tests/test_workflow.py` | offline tests (no network, no server) |

## Run it (Python 3.10+ required by the Inngest SDK; the repo venv is 3.11)

```bash
pip install -r requirements.txt -r durable/requirements.txt
set -a; . ./.env.local; set +a                                   # OPENAI_API_KEY

# terminal 1: the app (canned search, so no Tavily credits are needed; every step held 10 s after its work)
SEARCH_BACKEND=fake DEMO_STEP_DELAY_S=10 uvicorn durable.app:app --port 8000
# terminal 2: Inngest dev server + UI on http://localhost:8288
npx --yes inngest-cli@latest dev -u http://localhost:8000/api/inngest
# terminal 3: trigger a run
python -m durable.send request "What is the CAP theorem?" --run-id demo-1
```

### Drill 1: crash recovery (the demo video)

1. Watch terminal 1. When `[run:demo-1][step:analyst_worker-1] EXECUTING` appears, `kill -9` the app process
   (`kill -9 $(pgrep -f "uvicorn durable.app")`). Search is already saved; analysis is in flight.
2. Restart terminal 1's command. The dev server retries the failed step after its backoff (about 30 s in dev).
3. Terminal 1 now logs `analyst_worker-1`, `writer_worker-1`, `output-guard` as EXECUTING, and **never
   `search_worker-1` again**: it was saved. In the UI the run shows the errored step followed by the retry.
4. Killing earlier (during `search_worker-1`) re-runs that search: the step was not saved yet. That is at-least-once.

### Drill 2: human approval

The run parks at `await-approval` (UI: "Waiting for event"). Release it with
`python -m durable.send decide demo-1 approve` (or `reject`). With no decision it returns `timed_out` after
`--approval-timeout-s` (default 300) and releases nothing. `--no-approval` skips the gate.

### Drill 3: idempotency

`python -m durable.send request "x" --run-id idem-1 --no-approval --repeat 3` sends the same event three times:
three events, **one run** (`idempotency="event.data.run_id"`).

## Rehearsal evidence (2026-10-07)

Run against the real Inngest dev server (v1.46) and this app with a fake LLM and search (the sandbox cannot reach
OpenAI), single run `kill -9` during `analyst_worker-1`:

```
process 1 (killed):    [step:search_worker-1] EXECUTING   [step:analyst_worker-1] EXECUTING   <- killed here
process 2 (restarted): [step:analyst_worker-1] EXECUTING   [step:writer_worker-1] EXECUTING   [step:output-guard] EXECUTING
run history:           search_worker-1 completed once -> step errored (kill) -> analyst retried -> writer -> guard -> COMPLETED
```

Also verified there: approve -> `released` with the report; 3 events with one `run_id` -> 1 run; a second in-flight run
caught by the same kill resumed independently. Your own recording with the real LLM is still to do.

## Tests

```bash
python durable/tests/test_workflow.py     # 10 checks, offline
```

They cover: JSON round-trip, routing parity with the graph, stable step ids, released / rejected / timed-out /
approved paths, rejection of events without `run_id`, idempotency + retry configuration, and that every worker runs
**exactly once** while the function body is re-invoked after each step (the durability property in miniature).

## Limits (read before relying on it)

- Dev server only. Production needs Inngest Cloud or self-hosting with event/signing keys (`INNGEST_PROD=1`).
- The approval gate is a workflow pause, **not authentication**: anyone who can send a `research/report.decision`
  event for a `run_id` can approve it. Gate that endpoint in production.
- The unreleased report is stored in Inngest's run state (every step result is). Treat that store as sensitive.
- Steps are at-least-once. Today's steps (read-only LLM and search calls) are safe to repeat; a step with an external
  side effect (send an email, charge a card) needs its own idempotency key.
- Worker nodes are synchronous (`llm.invoke`), so each step runs them in a thread; the function is async.
- `DEMO_STEP_DELAY_S` exists only to make the crash window reliable; leave it at 0 outside demos.
