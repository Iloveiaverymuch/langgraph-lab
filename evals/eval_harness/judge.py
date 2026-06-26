"""
LLM-as-judge for the eval gate — two rubrics, Claude Haiku.

Judge model = Claude Haiku (Anthropic), deliberately a DIFFERENT model family from the
agent under test (gpt-4o-mini). Judging with an independent model reduces shared blind
spots — a fabrication the generator is prone to making is less likely to also be missed
by a judge from a different lab.

Design intent: this is the SINGLE source of judge logic, used by BOTH
  - the promptfoo gate (asserts in promptfooconfig.yaml call judge_*),
  - the D4b calibration harness (calibrate.py runs the same functions on labeled data).
That shared path is what makes calibration meaningful — we calibrate the exact judge
that gates.

Each judge returns a dict: { "pass": bool, "score": float (0..1), "reason": str, "raw": int 1..5 }.
A 1-5 rubric score is mapped to pass/fail at PASS_THRESHOLD (tunable during calibration).

Offline/deterministic fallback: if no ANTHROPIC_API_KEY (e.g. fake CI runs), judges return
a neutral PASS so the gate's deterministic checks still run without paid calls. The real
judge only engages when a key is present — same pattern as the supervisor provider.
"""

from __future__ import annotations
import os
import re
import json

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5-20251001")
PASS_THRESHOLD = float(os.environ.get("JUDGE_PASS_THRESHOLD", "4"))  # 1-5 scale; >=4 passes

# Calibration few-shot examples (D4b). Drawn from real judge↔human disagreements where
# the judge was too LENIENT — they teach the strict bar: any claim not traceable to the
# findings is a fail, even if it is topic-relevant or plausibly true.
FAITHFULNESS_FEWSHOT = """
CALIBRATED EXAMPLES (match this strictness):

Example A — FINDINGS describe vector DBs and list Pinecone (managed) and Weaviate
(open-source). REPORT adds "Pinecone processes over 2 billion queries daily and is
SOC2-certified." → score 2. Reason: the query volume and certification are NOT in the
findings; topic-relevant but unsupported additions are a fail.

Example B — FINDINGS cover RAG chunking strategies. REPORT is accurate on chunking but
adds a paragraph on "telemedicine UX best practices" pulled from an off-topic source.
→ score 2. Reason: off-topic content not grounded in the target findings is unfaithful,
even though the rest is fine.

Example C — FINDINGS list four RAG failure modes. REPORT covers them and adds "studies
show a 40% accuracy drop" with no such figure in the findings. → score 2. Reason: an
invented statistic is an unsupported claim → fail.
"""
COMPLETION_FEWSHOT = """
CALIBRATED EXAMPLES (match this strictness):

Example A — REPORT ends mid-sentence: "...the main strategies include …[truncated]".
→ score 1. Reason: a truncated/cut-off report is incomplete regardless of how much
topic it covered before being cut.

Example B — QUESTION asks to "compare Pinecone vs Weaviate"; REPORT only describes
Pinecone and never covers Weaviate. → score 2. Reason: a multi-part question with a
part missing is incomplete.
"""


# ---------------------------------------------------------------------------
# Rubric prompts
# ---------------------------------------------------------------------------

FAITHFULNESS_RUBRIC = """You are a STRICT evaluator of factual faithfulness (source grounding).

You are given SEARCH FINDINGS (the ONLY source material the writer was allowed to use)
and a REPORT produced from them. Judge ONLY whether the report's claims trace back to
the findings — NOT whether they are true in the real world. A claim can be factually
true and still FAIL if it is not in the findings.

Score 1-5:
5 = every claim is traceable to the findings; no additions.
4 = traceable, with only trivial rewording (no new facts at all).
3 = at least one claim is not in the findings.
2 = several unsupported claims, OR any off-topic content not from the findings,
    OR any invented fact/statistic/name — EVEN IF topic-relevant or plausibly true.
1 = largely fabricated / ignores the findings.

Be strict: ANY claim, statistic, certification, name, or section that cannot be located
in the findings caps the score at 2 (fail). "Topic-relevant" is NOT the same as "supported".
Note: the findings may contain off-topic junk from a drifting search — a report that
correctly ignores it is fine; a report that pulls that junk in as content fails.
(A brief, passing mention of "gaps/further work" is NOT by itself a failure — judge the
report's actual claims, not its hedging.)
{fewshot}
SEARCH FINDINGS:
{findings}

REPORT:
{report}

Respond ONLY with JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""


COMPLETION_RUBRIC = """You are a STRICT evaluator of task completion.

You are given a QUESTION and a REPORT. Score how fully and directly the report answers it.

Score 1-5:
5 = fully and directly answers the question, on-topic and complete.
4 = answers it well, minor gaps.
3 = partially answers; notable gaps or drift.
2 = mostly off-topic, misses the core, OR a multi-part question with a part missing.
1 = does not answer the question, OR is truncated / cut off / ends mid-thought.

IMPORTANT: a report that is truncated, cut off, or ends mid-sentence is INCOMPLETE
(score 1) even if the visible part covered the topic well. Check whether the report
actually finishes.

SCOPE: judge ONLY whether the question was answered. Do NOT lower the score because a
claim is unsupported, inaccurate, fabricated, or off-topic — factual faithfulness is
judged by a SEPARATE evaluator. A report can be fully complete (high score here) while
also containing false claims. Score completeness independently of correctness.
{fewshot}
QUESTION:
{question}

REPORT:
{report}

Respond ONLY with JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _has_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _call_judge(prompt: str) -> dict:
    """Call the judge model (Claude Haiku), parse {score, reason}.
    Robust to stray prose around the JSON."""
    from anthropic import Anthropic
    client = Anthropic()
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        temperature=0,  # deterministic judging
        messages=[{"role": "user", "content": prompt}],
    )
    # Anthropic returns a list of content blocks; concatenate any text blocks.
    text = "".join(getattr(b, "text", "") for b in resp.content) or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"score": 1, "reason": f"unparseable judge output: {text[:120]}"}
    try:
        data = json.loads(m.group(0))
        return {"score": int(data.get("score", 1)), "reason": str(data.get("reason", ""))[:200]}
    except Exception as e:
        return {"score": 1, "reason": f"judge parse error: {type(e).__name__}"}


def _result(raw_score: int, reason: str) -> dict:
    passed = raw_score >= PASS_THRESHOLD
    return {
        "pass": passed,
        "score": (raw_score - 1) / 4.0,   # normalize 1-5 -> 0..1
        "raw": raw_score,
        "reason": f"[{raw_score}/5] {reason}",
    }


def judge_faithfulness(report: str, findings: str) -> dict:
    if not _has_key():
        return {"pass": True, "score": 1.0, "raw": 5, "reason": "[skipped: no ANTHROPIC_API_KEY] faithfulness"}
    if not (findings or "").strip():
        return {"pass": True, "score": 1.0, "raw": 5, "reason": "[no findings to check against] faithfulness skipped"}
    prompt = FAITHFULNESS_RUBRIC.format(fewshot=FAITHFULNESS_FEWSHOT, findings=findings[:6000], report=report[:6000])
    out = _call_judge(prompt)
    return _result(out["score"], out["reason"])


def judge_completion(report: str, question: str) -> dict:
    if not _has_key():
        return {"pass": True, "score": 1.0, "raw": 5, "reason": "[skipped: no ANTHROPIC_API_KEY] completion"}
    prompt = COMPLETION_RUBRIC.format(fewshot=COMPLETION_FEWSHOT, question=question, report=report[:6000])
    out = _call_judge(prompt)
    return _result(out["score"], out["reason"])
