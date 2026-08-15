# observability/ — Agent Regression Sentinel (runtime half)

Deliberate tracing for the `supervisor` graph: every LLM and tool call is recorded,
shipped to **Langfuse** over OpenTelemetry (OTLP), and can be replayed from a trace ID.
This is the **post-deploy** half of the Sentinel — the eval gate (`evals/`) blocks bad
*changes* before merge; this watches the *running* agent for drift after merge.

## Why not the auto-instrumentation?

Langfuse ships an SDK/callback that auto-traces LangChain. We don't use it as a black box.
The value of the Sentinel is the **signal schema** (cost, latency, tool errors, success),
so we instrument deliberately and keep every span explainable. Spans follow the
OpenTelemetry **GenAI semantic conventions** (`gen_ai.*`), which are still experimental —
so all knowledge of those attribute names is isolated in one adapter with a pinning test.
Mechanism: LangChain `callbacks` → OTLP/HTTP → Langfuse (langgraph 0.6.11 exposes no
native-OTel module; this is the installed-stack path, and the design is identical).

## Native OTel vs. deliberate — the W07D2b decision

We spiked the "native" path (`native_otel_spike.py`): enable LangSmith's OTel export
(`LANGSMITH_OTEL_ENABLED`) and let LangChain/LangGraph auto-instrument the graph.
Empirical result:

- Native tracing is **LangSmith-cloud-first** — it posted to `api.smith.langchain.com`
  and failed `401 Unauthorized` without a LangSmith account/key. It did **not** reuse our
  Langfuse OTel provider; redirecting it to Langfuse needs explicit `OTEL_EXPORTER_OTLP_*`
  config and, in this version, a LangSmith key just to switch the tracer on.
- Its spans are framework-shaped (`RunnableSequence`, `ChatOpenAI`, chain nodes) with no
  `sentinel.*` signals, no `git_sha`, no `run_status` — not a product signal schema.
- Tracing failures were non-fatal; the agent completed normally.

**Decision: keep deliberate OTLP.** No vendor lock-in (raw OTLP → any OTel backend), no
mandatory LangSmith account, exact signal schema, git-pinned, explainable. Native's
zero-code auto-capture is real, but the wrong tradeoff for a product whose value is a
curated signal schema.

## Data flow

```
supervisor run
   │  (attached via config={"callbacks":[SentinelCallbackHandler]})
   ▼
tracing_callback.py     records each LLM/tool call
   ▼
receipts.py             typed receipt (ModelReceipt / ToolReceipt / RunReceipt)
   ▼
adapter.py              receipt  ⇄  OTel-GenAI span attributes   (the ONLY file that
                                                                   knows the wire names)
   ▼
langfuse_export.py      spans → Langfuse OTLP endpoint (Basic auth from .env.local)
   ▼
Langfuse dashboard      trace tree, cost, tokens, latency, sentinel.* labels
   ▲
replay.py               fetch a trace by ID → rebuild receipts (casts strings → numbers)
```

## Files

| File | Role | Network? | Tested |
|---|---|---|---|
| `receipts.py` | Typed receipt model — the internal vocabulary | no | via others |
| `adapter.py` | receipt ⇄ OTel-GenAI attribute mapping (+ read coercion) | no | `test_adapter_pinning` |
| `_receipt_builders.py` | Pure token/cost/receipt logic (no LangChain) | no | `test_receipt_builders` |
| `tracing_callback.py` | LangChain callback → receipts (thin glue) | no | (glue) |
| `langfuse_export.py` | receipts → OTLP → Langfuse; returns trace ID | **yes** | endpoint/auth offline |
| `run_traced.py` | Wrap one real supervisor run + export | **yes** | on Mac |
| `replay.py` | Fetch trace by ID → rebuild receipts | **yes** (fetch) | `test_replay_reconstruct` |
| `smoke_test_export.py` | Ship a fake trace — prove the pipe | **yes** | on Mac |
| `debug_export.py` | One synchronous span + verbose logs (diagnostics) | **yes** | on Mac |

## Attribute namespaces

- `gen_ai.*` — standard OTel-GenAI: `gen_ai.request.model`, `gen_ai.usage.input_tokens`, …
- `sentinel.*` — our product signals with no standard: `task_id`, `user_id`, `plan_step`,
  `git_sha`, `run_status`, `cost_usd`, `latency_ms`, `tool_name`, `tool_error`.

`test_adapter_pinning.py` freezes these strings, so a spec rename fails loudly instead of
silently blanking the numbers.

## Usage

```bash
pip install -r observability/requirements.txt
set -a; . ./.env.local; set +a

python -m observability.smoke_test_export         # fake trace — proves the pipe
python -m observability.run_traced "your question"    # trace a real run; prints trace_id
python -m observability.replay <trace_id>         # rebuild that run's receipts

# offline tests (no network)
python -m observability.tests.test_adapter_pinning
python -m observability.tests.test_receipt_builders
python -m observability.tests.test_replay_reconstruct
```

## Known limitations (deliberate, documented)

- **Waterfall timing is approximate.** Child spans use current wall-clock time, so Langfuse
  shows ~0 duration per span. The *real* per-call latency is on `sentinel.latency_ms`, which
  is what the Sentinel reads. Exact timestamps are a future polish.
- **Success = finished without throwing** (the current definition). Upgrade target: success ==
  the W05 eval grader passing.
- **Pricing table** in `_receipt_builders.py` is dated (2025-05) — verify before quoting
  absolute dollars. Langfuse computes its own cost as a cross-check (they currently agree).

## Definition of "regression" vs. the eval gate

See the eval-vs-Sentinel distinction: the eval gate answers *"is this change good enough to
ship?"* (quality, on a fixed set, before merge). The Sentinel answers *"is the live system
still behaving like yesterday?"* (cost / latency / errors / success, on real traffic, after
merge). They compose; neither replaces the other.
