"""
The supervisor agent as ONE Inngest function: every worker is a durable step.

How replay works: when a run is resumed (after a crash, a deploy, a pause) Inngest calls this function
again from the top. Every `step.run` that already finished returns its saved result WITHOUT executing the
handler; execution continues at the first unfinished step. So:
  - side effects (LLM, search) live inside step handlers only;
  - the code between steps must be deterministic (routing.next_step is a pure function of the saved messages);
  - step ids must be stable and unique ("search_worker-1", "analyst_worker-1", ...).

Steps are at-least-once: a step that dies after its side effect but before its result is saved runs again.
Whole-run duplicates are prevented by the function-level idempotency key (event.data.run_id).

Human-in-the-loop: after the output guard, the function pauses (step.wait_for_event) until an approval
decision for the same run_id arrives, or the timeout expires. Nothing is released without approval.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import os

import inngest

from supervisor import nodes
from supervisor.state import MAX_SEARCH_ITERATIONS

from .codec import to_dict, to_messages
from .routing import next_step

logger = logging.getLogger("uvicorn")

client = inngest.Inngest(
    app_id="langgraph-lab-durable",
    logger=logger,
    is_production=os.getenv("INNGEST_PROD") == "1",
)

REQUESTED = "research/requested"
DECISION = "research/report.decision"
MAX_LOOP = 2 * MAX_SEARCH_ITERATIONS + 3  # search+analyst per round, plus writer, with headroom


def _state(messages: list) -> dict:
    searches = sum(1 for m in messages if m.get("name") == "search_worker")
    return {"messages": to_messages(messages), "next": "", "final_answer": "", "search_iterations": searches}


async def run_worker(run_id: str, step_id: str, name: str, messages: list) -> dict:
    """Step handler. Executes ONLY when the step is not already saved: the log line is the replay proof."""
    logger.info("[run:%s][step:%s] EXECUTING (not a replay)", run_id, step_id)
    fn = getattr(nodes, name)  # looked up at call time so tests can substitute workers
    update = await asyncio.to_thread(fn, _state(messages))  # workers are synchronous and block on the network
    # Demo only: hold the finished result for N seconds BEFORE it is returned (saved), to give `kill -9` a window
    # in exactly the spot where a step has done its side effect but is not yet durable. 0 in normal use.
    await asyncio.sleep(float(os.getenv("DEMO_STEP_DELAY_S", "0")))
    return to_dict(update["messages"][0])


async def run_output_guard(run_id: str, messages: list) -> dict:
    logger.info("[run:%s][step:output-guard] EXECUTING (not a replay)", run_id)
    update = await asyncio.to_thread(nodes.output_guard_node, _state(messages))
    return {"report": update.get("final_answer", ""), "redacted": bool(update.get("messages"))}


@client.create_function(
    fn_id="run-research",
    trigger=inngest.TriggerEvent(event=REQUESTED),
    idempotency="event.data.run_id",  # CEL: one run per run_id; a re-sent event is dropped
    retries=3,                        # per step: a transient OpenAI/network error retries that step only
)
async def run_research(ctx: inngest.Context) -> dict:
    data = ctx.event.data
    question, run_id = data.get("question"), data.get("run_id")
    if not question or not run_id:
        raise inngest.NonRetriableError("event data needs 'question' and 'run_id'")

    messages = [{"role": "human", "name": None, "content": question}]
    counts: dict = {}
    for _ in range(MAX_LOOP):
        nxt = next_step(messages)
        if nxt == "FINISH":
            break
        counts[nxt] = counts.get(nxt, 0) + 1
        step_id = f"{nxt}-{counts[nxt]}"
        messages.append(await ctx.step.run(step_id, run_worker, run_id, step_id, nxt, messages))
    else:
        raise inngest.NonRetriableError("routing did not finish within the loop bound")

    guarded = await ctx.step.run("output-guard", run_output_guard, run_id, messages)

    if data.get("require_approval", True):
        decision = await ctx.step.wait_for_event(
            "await-approval",
            event=DECISION,
            if_exp="event.data.run_id == async.data.run_id",
            timeout=datetime.timedelta(seconds=int(data.get("approval_timeout_s", 300))),
        )
        if decision is None:
            return {"status": "timed_out", "run_id": run_id}
        if not decision.data.get("approved"):
            return {"status": "rejected", "run_id": run_id}

    return {"status": "released", "run_id": run_id, "report": guarded["report"], "redacted": guarded["redacted"]}
