"""
wire_models.py — the receipts contract at the API boundary (W3.5A D2).

One definition of a valid row, used by BOTH sinks:
  - http sink:   run_traced -> RunIn -> POST /receipts -> service validates again -> store
  - direct sink: run_traced -> RunIn -> store.insert_run
so a row the API would reject can never reach the store through the side door.

Rules (mirror store.py's "minimal columns" design):
  - strict types: "1200" is NOT coerced to 1200 (client bugs fail loudly as 422)
  - extra="forbid": unknown fields (e.g. user_id, prompt) are rejected, not silently dropped
  - no user id, no prompt/answer text: what is not accepted cannot be stored or leak
  - schema_version: lets server and client deploy in any order later (expand-then-contract)
"""
from __future__ import annotations

from typing import Iterable, List, Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .receipts import RunReceipt, ToolReceipt

_ID = r"^[A-Za-z0-9_.:-]+$"
_MAX_MS = 24 * 3600 * 1000.0  # one day: anything longer is a unit bug, not a slow run


class _Wire(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class ToolCallIn(_Wire):
    tool_name: str = Field(min_length=1, max_length=128, pattern=_ID)
    tool_error: bool
    latency_ms: float = Field(ge=0, le=_MAX_MS)


class RunIn(_Wire):
    schema_version: Literal[1] = 1
    task_id: str = Field(min_length=1, max_length=128, pattern=_ID)
    git_sha: str = Field(pattern=r"^([0-9a-f]{7,40}|unknown)$")
    run_status: Literal["success", "failed"]
    run_type: str = Field(pattern=r"^[a-z_]{1,32}$")
    total_tokens: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0)
    latency_ms: float = Field(ge=0, le=_MAX_MS)
    tool_calls: List[ToolCallIn] = Field(default_factory=list, max_length=1000)

    @classmethod
    def from_receipts(cls, run: RunReceipt, tools: Iterable[ToolReceipt]) -> "RunIn":
        """Internal dataclasses -> wire model. Drops user_id by construction."""
        return cls(
            task_id=run.task_id,
            git_sha=run.git_sha,
            run_status=run.run_status,
            run_type=run.run_type,
            total_tokens=int(run.total_tokens),
            total_cost_usd=float(run.total_cost_usd),
            latency_ms=float(run.latency_ms),
            tool_calls=[
                ToolCallIn(tool_name=t.tool_name, tool_error=bool(t.tool_error), latency_ms=float(t.latency_ms))
                for t in tools
            ],
        )

    def to_rows(self) -> Tuple[dict, List[dict]]:
        """Wire model -> (runs row, tool_calls rows) for store.insert_run."""
        run_row = self.model_dump(exclude={"tool_calls", "schema_version"})
        return run_row, [t.model_dump() for t in self.tool_calls]
