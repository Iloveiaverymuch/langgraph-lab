"""
sentinel_api/app.py — Sentinel receipts ingestion service (W3.5A D2).

    SENTINEL_DB=.sentinel/receipts.db uvicorn sentinel_api.app:app --port 8100

The service OWNS the receipts store: agents POST one run (+ its tool calls) per request.
Design (see claude/sprint/w07.5a-d1-fastapi-notes.md):
  - POST /receipts is a plain `def`: sqlite3 blocks, so it runs in the threadpool and
    never stalls the event loop. aiosqlite would only wrap a thread: no gain on SQLite.
  - The insert happens INSIDE the request: commit, then 201. Answering before the data is
    durable would make a lost receipt look like a success (client never retries).
  - Idempotent on task_id: duplicate -> 200 + created=false (not 409), so a client
    re-sending its spool treats "already stored" as success.
  - Validation = observability.wire_models.RunIn (strict, extra=forbid), the SAME model
    the direct sink uses: one definition of a valid row.
  - Auth / rate limiting / logging: W3.5A D3.
"""
import sqlite3
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import Depends, FastAPI, Request, Response
from pydantic import BaseModel

from observability import store
from observability.wire_models import RunIn


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schema + WAL once at startup; per-request connections skip it.
    app.state.db_path = store.init_schema(store.default_db_path())
    yield


app = FastAPI(title="Sentinel receipts API", version="0.1.0", lifespan=lifespan)


def get_conn(request: Request) -> Iterator[sqlite3.Connection]:
    """One connection per request (never shared, hence check_same_thread=False)."""
    conn = store.connect(request.app.state.db_path, init_schema=False, check_same_thread=False)
    try:
        yield conn
    except Exception:
        conn.rollback()  # belt and braces: insert_run's `with conn:` already rolls back
        raise            # re-raise, or FastAPI never sees the error
    finally:
        conn.close()


class ReceiptAck(BaseModel):
    task_id: str
    created: bool


@app.post(
    "/receipts",
    response_model=ReceiptAck,
    status_code=201,
    responses={200: {"model": ReceiptAck, "description": "Duplicate task_id: already stored, nothing changed"}},
)
def post_receipt(run: RunIn, response: Response, conn: sqlite3.Connection = Depends(get_conn)) -> ReceiptAck:
    run_row, tool_rows = run.to_rows()
    created = store.insert_run(conn, run_row, tool_rows, source="api")
    if not created:
        response.status_code = 200
    return ReceiptAck(task_id=run.task_id, created=created)


@app.get("/health")
async def health() -> dict:
    # async + no I/O: answers even when every threadpool thread is busy writing.
    return {"status": "ok"}
