"""
sentinel_api/ratelimit.py — per-API-key token bucket (W3.5A D3 step 4, decided 2026-10-27).

Each key_id gets a bucket of `burst` tokens refilled at `rate` tokens/s; a request spends
one token; an empty bucket -> 429 + Retry-After. Bursts are allowed on purpose: the
agent's spool flush sends its backlog in one go, and a flush that hits 429 stops and
keeps the rest in order (sink.py), so the client throttles itself.

Keyed by key_id AFTER auth, not by IP: every agent shares localhost/proxy IPs, and only
valid keys create buckets, so memory is bounded by the key set.

Not thread-safe by design: acquire() is called only from an `async def` dependency, i.e.
on the event loop's single thread; no lock needed. Calling it from a `def` path (thread
pool) would need a lock.

Limit: state is per process. With `--workers N` or N replicas the effective limit is N x.
A shared limit needs Redis (W3.5B).
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, Tuple

RATE_ENV, BURST_ENV = "SENTINEL_RATE_PER_S", "SENTINEL_RATE_BURST"
DEFAULT_RATE, DEFAULT_BURST = 10.0, 50


class TokenBucketLimiter:
    def __init__(self, rate: float, burst: int, clock: Callable[[], float] = time.monotonic):
        if rate <= 0 or burst < 1:
            raise ValueError(f"rate must be > 0 and burst >= 1 (got rate={rate}, burst={burst})")
        self.rate, self.burst, self.clock = float(rate), int(burst), clock
        self._buckets: Dict[str, Tuple[float, float]] = {}  # key -> (tokens, last refill time)

    def acquire(self, key: str) -> float:
        """Spend one token. Returns 0.0 if allowed, else seconds until a token is available."""
        now = self.clock()
        tokens, last = self._buckets.get(key, (float(self.burst), now))
        tokens = min(float(self.burst), tokens + (now - last) * self.rate)
        if tokens >= 1.0:
            self._buckets[key] = (tokens - 1.0, now)
            return 0.0
        self._buckets[key] = (tokens, now)
        return (1.0 - tokens) / self.rate


def from_env() -> TokenBucketLimiter:
    """Invalid config fails at startup (loud), never silently unlimited."""
    try:
        rate = float(os.environ.get(RATE_ENV, DEFAULT_RATE))
        burst = int(os.environ.get(BURST_ENV, DEFAULT_BURST))
        return TokenBucketLimiter(rate, burst)
    except ValueError as e:
        raise RuntimeError(f"invalid rate limit config ({RATE_ENV}/{BURST_ENV}): {e}") from None
