"""
sentinel_api/app.py — Sentinel receipts ingestion service (W3.5A D2).

    SENTINEL_API_KEYS=<k1>[,<k2>] SENTINEL_DB=.sentinel/receipts.db \
        uvicorn sentinel_api.app:app --port 8100 --no-access-log

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
  - Auth (W3.5A D3, decided 2026-10-27): static API keys, `Authorization: Bearer <key>`.
    SENTINEL_API_KEYS is a comma-separated SET so a key can rotate with no downtime
    (add new -> move the agent -> remove old). Empty set = refuse to start (fail closed,
    never silently open). Compared with hmac.compare_digest (constant time). Auth runs
    before body validation and before the DB connection: an unauthenticated caller gets
    401, never a 422 that describes the schema. /health stays open (liveness probe).
    JWT deferred until there is a real issuer (W08).
  - Logging (W3.5A D3 step 3): one JSON canonical line per request (sentinel_api/logs.py),
    allowlisted fields only; X-Request-ID accepted/generated and echoed (the agent sends
    its task_id). --no-access-log avoids double lines.
  - Rate limiting (W3.5A D3 step 4): per-key token bucket (sentinel_api/ratelimit.py),
    SENTINEL_RATE_PER_S=10 / SENTINEL_RATE_BURST=50 by default; 429 + Retry-After, which
    the sink spools. Order on POST /receipts: auth -> rate limit -> validation -> DB.
    Per-process state: N workers/replicas = N x the limit (shared limit = Redis, W3.5B).
"""
import hmac
import math
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from typing import Iterator, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from observability import store
from observability.wire_models import RunIn

from . import logs, ratelimit


API_KEYS_ENV = "SENTINEL_API_KEYS"


def load_api_keys() -> frozenset:
    """Accepted keys from SENTINEL_API_KEYS. Empty -> RuntimeError (fail closed)."""
    keys = frozenset(k.strip() for k in os.environ.get(API_KEYS_ENV, "").split(",") if k.strip())
    if not keys:
        raise RuntimeError(f"{API_KEYS_ENV} is empty: refusing to start without auth (fail closed)")
    return keys


@asynccontextmanager
async def lifespan(app: FastAPI):
    logs.configure(os.environ.get("SENTINEL_LOG_LEVEL", "INFO"))
    app.state.api_keys = load_api_keys()
    app.state.limiter = ratelimit.from_env()  # before anything else: no keys, no service
    # Schema + WAL once at startup; per-request connections skip it.
    app.state.db_path = store.init_schema(store.default_db_path())
    yield


app = FastAPI(title="Sentinel receipts API", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def canonical_log_line(request: Request, call_next):
    """One JSON line per request. request.state carries task_id/key_id back from the
    endpoint (the state dict lives in the ASGI scope, shared with the endpoint)."""
    rid = logs.request_id_from(request.headers.get("x-request-id"))
    token = logs.request_id_var.set(rid)
    start = time.perf_counter()
    status = 500  # if the endpoint raises, call_next re-raises: logged as 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        extra = {
            "method": request.method, "path": request.url.path, "status": status,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }
        for field in ("task_id", "inserted", "key_id"):
            if hasattr(request.state, field):
                extra[field] = getattr(request.state, field)
        level = logs.logging.ERROR if status >= 500 else logs.logging.WARNING if status >= 400 else logs.logging.INFO
        logs.log.log(level, "request", extra=extra)
        logs.request_id_var.reset(token)


# auto_error=False: we raise our own 401 (+ WWW-Authenticate) instead of FastAPI's default,
# whose status code has changed across versions. HTTPBearer also adds the OpenAPI scheme.
_bearer = HTTPBearer(auto_error=False, description="Sentinel API key (SENTINEL_API_KEY on the agent)")


def require_api_key(
    request: Request, creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)
) -> str:
    """Returns the caller's key_id (sha256[:8]); 401 otherwise."""
    token = creds.credentials.encode() if creds else b""
    matched = None
    for key in request.app.state.api_keys:  # check every key, no early exit
        if hmac.compare_digest(token, key.encode()):
            matched = key
    if not (creds and matched):
        raise HTTPException(
            status_code=401, detail="invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},  # required on 401 (RFC 9110)
        )
    request.state.key_id = logs.key_id(matched)  # fingerprint for the log, never the key
    return request.state.key_id


async def rate_limit(request: Request, key_id: str = Depends(require_api_key)) -> None:
    """async on purpose: runs on the event loop thread, so the bucket dict needs no lock.
    Depends on require_api_key explicitly: auth always runs first, and 401s spend no token."""
    wait = request.app.state.limiter.acquire(key_id)
    if wait > 0:
        raise HTTPException(
            status_code=429, detail="rate limit exceeded",
            headers={"Retry-After": str(max(1, math.ceil(wait)))},
        )


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
    dependencies=[Depends(rate_limit)],  # -> require_api_key; both resolved before the body and the DB
    responses={
        200: {"model": ReceiptAck, "description": "Duplicate task_id: already stored, nothing changed"},
        401: {"description": "Missing or invalid API key (the agent's sink spools and retries)"},
        429: {"description": "Rate limit exceeded for this key; see Retry-After (the sink spools)"},
    },
)
def post_receipt(
    run: RunIn, request: Request, response: Response, conn: sqlite3.Connection = Depends(get_conn)
) -> ReceiptAck:
    request.state.task_id = run.task_id
    run_row, tool_rows = run.to_rows()
    created = store.insert_run(conn, run_row, tool_rows, source="api")
    request.state.inserted = created
    if not created:
        response.status_code = 200
    return ReceiptAck(task_id=run.task_id, created=created)


@app.get("/health")
async def health() -> dict:
    # async + no I/O: answers even when every threadpool thread is busy writing.
    return {"status": "ok"}
