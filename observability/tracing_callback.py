"""
SentinelCallbackHandler — the "recorder".

A LangChain callback handler that listens to the supervisor graph and produces
one internal receipt per LLM call and per tool call. It touches NO node logic:
you attach it via `config={"callbacks": [handler]}` on `app.invoke(...)`, exactly
like the eval harness already does with UsageMetadataCallbackHandler.

This module is deliberately thin — all real logic lives in `_receipt_builders.py`
(pure, offline-testable). Here we only:
  - time each call (start -> end),
  - track which model/tool a run_id belongs to,
  - hand the raw data to the pure builders.

plan_step: a per-run counter incremented on each model call. Tool calls carry the
step of the model call that triggered them. Simple and explainable; good enough
for grouping cost/latency by step.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from .receipts import ModelReceipt, ToolReceipt
from ._receipt_builders import (
    build_model_receipt,
    build_tool_receipt,
    extract_model_name,
    extract_token_usage,
)


def _model_from_serialized(serialized: Any) -> Optional[str]:
    """Best-effort model name from the callback's `serialized` payload."""
    if isinstance(serialized, dict):
        kwargs = serialized.get("kwargs")
        if isinstance(kwargs, dict):
            return kwargs.get("model_name") or kwargs.get("model")
    return None


class SentinelCallbackHandler(BaseCallbackHandler):
    """Collects ModelReceipt / ToolReceipt for a single agent run.

    One handler instance per run — created by the integration layer (task #4),
    which passes the run's task_id / user_id / git_sha.
    """

    def __init__(self, task_id: str, user_id: str, git_sha: str) -> None:
        self.task_id = task_id
        self.user_id = user_id
        self.git_sha = git_sha
        self.model_receipts: List[ModelReceipt] = []
        self.tool_receipts: List[ToolReceipt] = []
        # Combined, in call order (LLM + tool interleaved) — used to build the
        # trace tree so the Langfuse waterfall reflects what actually happened.
        self.timeline: List[Any] = []
        # run_id -> perf_counter at start, and run_id -> model/tool label
        self._start: Dict[Any, float] = {}
        self._model_at_start: Dict[Any, Optional[str]] = {}
        self._tool_at_start: Dict[Any, str] = {}
        self._step = 0

    # -- LLM ---------------------------------------------------------------
    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        # ChatOpenAI (chat models) fire THIS, not on_llm_start.
        run_id = kwargs.get("run_id")
        self._step += 1
        self._start[run_id] = time.perf_counter()
        self._model_at_start[run_id] = _model_from_serialized(serialized)

    def on_llm_start(self, serialized: Any, prompts: Any, **kwargs: Any) -> None:
        # Kept for completion-model providers that use this instead.
        run_id = kwargs.get("run_id")
        self._step += 1
        self._start[run_id] = time.perf_counter()
        self._model_at_start[run_id] = _model_from_serialized(serialized)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        latency_ms = self._latency(run_id)
        llm_output = getattr(response, "llm_output", None)
        generations = getattr(response, "generations", None)
        tokens_in, tokens_out = extract_token_usage(llm_output, generations)
        model = extract_model_name(llm_output, self._model_at_start.pop(run_id, None))
        receipt = build_model_receipt(
            self.task_id, self.user_id, self._step,
            model, tokens_in, tokens_out, latency_ms,
        )
        self.model_receipts.append(receipt)
        self.timeline.append(receipt)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        self._start.pop(run_id, None)
        self._model_at_start.pop(run_id, None)

    # -- Tools -------------------------------------------------------------
    def on_tool_start(self, serialized: Any, input_str: Any, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        self._start[run_id] = time.perf_counter()
        name = (serialized or {}).get("name", "unknown_tool") if isinstance(serialized, dict) else "unknown_tool"
        self._tool_at_start[run_id] = name

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        receipt = build_tool_receipt(
            self.task_id, self.user_id, self._step,
            self._tool_at_start.pop(run_id, "unknown_tool"), False, self._latency(run_id),
        )
        self.tool_receipts.append(receipt)
        self.timeline.append(receipt)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        receipt = build_tool_receipt(
            self.task_id, self.user_id, self._step,
            self._tool_at_start.pop(run_id, "unknown_tool"), True, self._latency(run_id),
        )
        self.tool_receipts.append(receipt)
        self.timeline.append(receipt)

    # -- helpers -----------------------------------------------------------
    def _latency(self, run_id: Any) -> float:
        t0 = self._start.pop(run_id, None)
        return round((time.perf_counter() - t0) * 1000, 3) if t0 is not None else 0.0
