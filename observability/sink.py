"""
sink.py — where run receipts go (W3.5A D2).

    SENTINEL_SINK=direct (default)  write to the local SQLite store in-process
    SENTINEL_SINK=http              POST to the receipts API (sentinel_api), spool on failure
    SENTINEL_API_URL                default http://127.0.0.1:8100
    SENTINEL_API_KEY                bearer token for the receipts API (W3.5A D3); never printed

Both sinks validate through observability.wire_models.RunIn first: one definition of a
valid row, whichever path writes it.

http failure policy (decided 2026-10-25, "spool"):
Rule: spool whenever resending the SAME payload could succeed later.
  - 201/200          stored / duplicate (idempotent on task_id) -> done
  - 401/403          auth failure (W3.5A D3, decided 2026-10-27): the payload is fine, the
                     credential is wrong and gets fixed separately -> logged loudly (names
                     SENTINEL_API_KEY, never its value) and spooled; the fix flushes the spool
  - other 4xx (422..) rejected: the payload is bad, a resend fails identically forever ->
                     logged, NOT spooled
  - network / 5xx / 408 / 429  transient -> appended to .sentinel/spool.jsonl
  - the spool is flushed (in order) at the start of the next http run; resends are safe
    because the store ignores duplicate task_ids. Capped at SPOOL_MAX_LINES; overflow is
    dropped LOUDLY (counted on stderr), never silently.
  - concurrency: promptfoo runs several agent processes at once, so spool appends and
    rewrites hold an fcntl lock (POSIX: macOS/Linux). A flush that cannot get the lock
    skips: another process is already flushing.
Never fatal: record() never raises; monitoring must not break the agent.
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from pydantic import ValidationError

from . import store
from .receipts import RunReceipt, ToolReceipt
from .wire_models import RunIn

DEFAULT_API_URL = "http://127.0.0.1:8100"
TIMEOUT_S = 2.0
SPOOL_MAX_LINES = 10_000
_TRANSIENT_4XX = {408, 429}
_AUTH_4XX = {401, 403}
API_KEY_ENV = "SENTINEL_API_KEY"


class Rejected(Exception):
    """The API refused the payload (4xx): resending cannot succeed."""


def _warn(msg: str) -> None:
    print(f"[sentinel] WARNING: {msg}", file=sys.stderr)


def api_url() -> str:
    return os.environ.get("SENTINEL_API_URL", DEFAULT_API_URL).rstrip("/")


def api_key() -> str | None:
    return os.environ.get(API_KEY_ENV) or None


def _is_auth_failure(e: BaseException) -> bool:
    return isinstance(e, urllib.error.HTTPError) and e.code in _AUTH_4XX


def spool_path() -> Path:
    return store.default_db_path().parent / "spool.jsonl"


@contextmanager
def _spool_lock(blocking: bool = True) -> Iterator[bool]:
    path = spool_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def post(payload: str, timeout: float = TIMEOUT_S) -> bool:
    """POST one RunIn JSON. Returns created (True) / duplicate (False).

    Raises Rejected on a permanent 4xx; OSError (URLError, timeout, 5xx / 401 / 403 HTTPError)
    when a resend could succeed later.
    """
    headers = {"Content-Type": "application/json"}
    key = api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        f"{api_url()}/receipts", data=payload.encode("utf-8"), headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return bool(json.loads(resp.read() or b"{}").get("created", resp.status == 201))
    except urllib.error.HTTPError as e:
        if 400 <= e.code < 500 and e.code not in _TRANSIENT_4XX | _AUTH_4XX:
            raise Rejected(f"HTTP {e.code}: {e.read()[:300]!r}") from None
        raise  # 5xx / 408 / 429 / 401 / 403: resend may succeed (HTTPError is an OSError)


def _spool(payload: str) -> bool:
    with _spool_lock():
        path = spool_path()
        n = sum(1 for _ in open(path)) if path.exists() else 0
        if n >= SPOOL_MAX_LINES:
            _warn(f"spool full ({n} lines): receipt DROPPED")
            return False
        with open(path, "a") as f:
            f.write(payload + "\n")
        return True


def flush_spool() -> dict:
    """Resend spooled receipts in order; stop at the first transient failure."""
    stats = {"sent": 0, "rejected": 0, "kept": 0, "skipped": False}
    with _spool_lock(blocking=False) as locked:
        if not locked:
            stats["skipped"] = True
            return stats
        path = spool_path()
        if not path.exists():
            return stats
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        keep: list = []
        for i, line in enumerate(lines):
            try:
                post(line)
                stats["sent"] += 1
            except Rejected as e:
                stats["rejected"] += 1
                _warn(f"spooled receipt rejected, discarded: {e}")
            except OSError:
                keep = lines[i:]  # service still down: keep the rest, in order
                break
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(ln + "\n" for ln in keep))
        os.replace(tmp, path)  # atomic rewrite
        stats["kept"] = len(keep)
    return stats


def record(run: RunReceipt, tools: Iterable[ToolReceipt]) -> str:
    """Send one run to the configured sink. Never raises.

    Returns: stored | duplicate | spooled | rejected | dropped | invalid | error
    """
    try:
        try:
            wire = RunIn.from_receipts(run, tools)
        except ValidationError as e:
            _warn(f"receipt failed the wire contract, not recorded: {e.errors()[:3]}")
            return "invalid"

        sink = os.environ.get("SENTINEL_SINK", "direct")
        if sink == "direct":
            run_row, tool_rows = wire.to_rows()
            conn = store.connect()
            try:
                return "stored" if store.insert_run(conn, run_row, tool_rows) else "duplicate"
            finally:
                conn.close()

        if sink == "http":
            if not api_key():
                _warn(f"SENTINEL_SINK=http but {API_KEY_ENV} is not set: an authenticated "
                      "receipts API will answer 401 and receipts will be spooled")
            stats = flush_spool()
            if stats["sent"] or stats["rejected"] or stats["kept"]:
                print(f"[sentinel] spool flush: {stats}", file=sys.stderr)
            payload = wire.model_dump_json()
            try:
                return "stored" if post(payload) else "duplicate"
            except Rejected as e:
                _warn(f"receipts API rejected the run, not spooled: {e}")
                return "rejected"
            except OSError as e:
                if _is_auth_failure(e):
                    _warn(f"auth failed (HTTP {e.code}): check {API_KEY_ENV}; spooling")
                else:
                    _warn(f"receipts API unreachable ({type(e).__name__}), spooling")
                return "spooled" if _spool(payload) else "dropped"

        _warn(f"unknown SENTINEL_SINK={sink!r}; expected direct|http")
        return "invalid"
    except Exception as e:  # noqa: BLE001 — never fatal
        _warn(f"receipts sink failed: {type(e).__name__}: {e}")
        return "error"
