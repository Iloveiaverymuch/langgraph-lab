"""
sentinel_api/logs.py — structured JSON logging for the receipts API (W3.5A D3 step 3).

One "canonical log line" per request: a single JSON object with everything needed to
answer "what happened to receipt X" (request_id = the agent's task_id when it sends one).

Secrets policy = ALLOWLIST, not redaction: the formatter emits a fixed set of fields and
never headers, query strings or bodies. Redacting `Authorization` would be a denylist
that misses the next secret header someone adds. The caller's key appears only as
key_id = sha256(key)[:8] hex: enough to see which key is still in use during a rotation,
useless to recover the key.

stdlib logging only (decided 2026-10-27): no dependency; uvicorn already logs via stdlib.
Run uvicorn with --no-access-log, or every request is logged twice.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import re
import sys
import uuid
from datetime import datetime, timezone

LOGGER_NAME = "sentinel_api"
log = logging.getLogger(LOGGER_NAME)

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# Only these `extra=` fields ever reach the output. Names must not collide with LogRecord
# attributes (e.g. `created` is the record timestamp: logging refuses to overwrite it).
ALLOWED_FIELDS = ("method", "path", "status", "latency_ms", "task_id", "inserted", "key_id")

# Same charset as wire_models._ID: a task_id is always a valid request id.
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def request_id_from(header_value: str | None) -> str:
    """Caller's X-Request-ID if well-formed, else a fresh one (no log/header injection)."""
    if header_value and _REQUEST_ID.match(header_value):
        return header_value
    return uuid.uuid4().hex


def key_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:8]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for field in ALLOWED_FIELDS:
            if field in record.__dict__:
                out[field] = record.__dict__[field]
        if record.exc_info and record.exc_info[0] is not None:
            out["exc_type"] = record.exc_info[0].__name__  # type only: messages can carry data
        return json.dumps(out, default=str)


class _StdoutHandler(logging.StreamHandler):
    """Writes to whatever sys.stdout is at emit time (not at startup): survives stdout
    being swapped later (pytest capture, reloaders)."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value) -> None:
        pass


def configure(level: str = "INFO") -> None:
    """Idempotent. Called at app startup."""
    handler = _StdoutHandler()
    handler.setFormatter(JsonFormatter())
    log.handlers[:] = [handler]
    log.setLevel(level)
    log.propagate = False  # don't duplicate into uvicorn's/root's handlers
