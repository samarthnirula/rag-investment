"""Server-side per-user rate limiter — Redis sliding-window implementation.

Uses Redis sorted sets for accurate, multi-worker, restart-safe rate limiting.
Falls back to in-memory counters automatically when Redis is unavailable so the
app never crashes due to a Redis outage (fail-open behaviour with a warning log).

Public API (unchanged from the original in-memory version):
    check_query_rate_limit(uid, *, queries_this_month, monthly_limit)
    check_upload_rate_limit(uid, *, uploads_this_month, monthly_upload_limit,
                            file_size_bytes, max_upload_bytes)
    check_demo_rate_limit(ip)
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from fastapi import HTTPException

_log = logging.getLogger(__name__)

# ── Limits ────────────────────────────────────────────────────────────────────
_HOURLY_LIMIT   = 30
_MINUTE_LIMIT   = 5
_HOUR_SECONDS   = 3600
_MINUTE_SECONDS = 60

_DEMO_HOURLY_LIMIT = 10
_DEMO_MINUTE_LIMIT = 3

# ── Redis client (lazy singleton) ─────────────────────────────────────────────
_redis_lock = threading.Lock()
_redis_client: Optional["redis.Redis"] = None  # type: ignore[name-defined]
_redis_unavailable = False  # set True after first failed connect; retried every 60 s
_redis_retry_at: float = 0.0


def _get_redis():
    """Return a connected Redis client or None if Redis is unreachable."""
    global _redis_client, _redis_unavailable, _redis_retry_at

    if _redis_client is not None:
        return _redis_client

    now = time.monotonic()
    if _redis_unavailable and now < _redis_retry_at:
        return None  # back-off: don't hammer an unavailable Redis

    with _redis_lock:
        if _redis_client is not None:
            return _redis_client

        try:
            import redis as _redis_mod
            url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            client = _redis_mod.Redis.from_url(
                url,
                socket_connect_timeout=1,
                socket_timeout=1,
                decode_responses=True,
            )
            client.ping()
            _redis_client = client
            _redis_unavailable = False
            _log.info("Rate limiter connected to Redis at %s", url)
        except Exception as exc:
            _redis_unavailable = True
            _redis_retry_at = time.monotonic() + 60  # retry in 60 s
            _log.warning(
                "Redis unavailable for rate limiter (falling back to in-memory): %s", exc
            )
        return _redis_client


# ── Sliding-window check via Redis sorted set ─────────────────────────────────

def _redis_check(key: str, limit: int, window_seconds: int) -> bool:
    """Sliding-window check.  Returns True if the request is ALLOWED.

    Uses a Redis sorted set keyed by ``key``.  Each member is a unique string
    (timestamp + random suffix); the score is ``time.time()``.  Old members
    are evicted before counting so the window truly slides.

    Performs 4 pipelined commands (no round-trip overhead):
      ZREMRANGEBYSCORE, ZCARD, ZADD, EXPIRE.

    The count returned by ZCARD reflects the number of requests in the window
    *before* the current one is added.  If count < limit the request is allowed.
    """
    rc = _get_redis()
    if rc is None:
        return True  # fail open

    now = time.time()
    # Unique member prevents collisions under high concurrency.
    member = f"{now:.6f}:{random.getrandbits(32)}"

    try:
        pipe = rc.pipeline()
        pipe.zremrangebyscore(key, 0, now - window_seconds)
        pipe.zcard(key)
        pipe.zadd(key, {member: now})
        pipe.expire(key, window_seconds + 1)
        _, count, _, _ = pipe.execute()
        return int(count) < limit
    except Exception as exc:
        _log.warning("Redis rate-limit check failed (fail-open): %s", exc)
        return True  # fail open


# ── In-memory fallback (single-process, resets on restart) ────────────────────
# Used only when Redis is unavailable.

_mem_lock = threading.Lock()


@dataclass
class _MemBucket:
    hour_window:   int = 0
    hour_count:    int = 0
    minute_window: int = 0
    minute_count:  int = 0


_mem_buckets:      dict[str, _MemBucket] = {}
_mem_demo_buckets: dict[str, _MemBucket] = {}


def _mem_check(buckets: dict, key: str, hourly_limit: int, minute_limit: int) -> tuple[bool, bool]:
    """Return (hour_ok, minute_ok) and advance the bucket counters."""
    now = int(time.time())
    with _mem_lock:
        if key not in buckets:
            buckets[key] = _MemBucket(
                hour_window=now // _HOUR_SECONDS,
                minute_window=now // _MINUTE_SECONDS,
            )
        b = buckets[key]
        if now // _HOUR_SECONDS != b.hour_window:
            b.hour_window = now // _HOUR_SECONDS
            b.hour_count = 0
        if now // _MINUTE_SECONDS != b.minute_window:
            b.minute_window = now // _MINUTE_SECONDS
            b.minute_count = 0

        hour_ok   = b.hour_count   < hourly_limit
        minute_ok = b.minute_count < minute_limit
        if hour_ok and minute_ok:
            b.hour_count   += 1
            b.minute_count += 1
        return hour_ok, minute_ok


# ── Public API ────────────────────────────────────────────────────────────────

def check_query_rate_limit(
    uid: str,
    *,
    queries_this_month: int,
    monthly_limit: int,
) -> None:
    """Enforce per-user query rate limits.  Raises HTTPException 429 if exceeded."""
    # Monthly cap (DB-backed — survives restarts regardless of Redis)
    if monthly_limit and queries_this_month >= monthly_limit:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Monthly query limit reached ({monthly_limit} queries/month). "
                "This protects plan costs and keeps the service sustainable."
            ),
        )

    if _get_redis() is not None:
        # Redis sliding-window path
        if not _redis_check(f"rl:q:hour:{uid}", _HOURLY_LIMIT, _HOUR_SECONDS):
            now = int(time.time())
            mins = (_HOUR_SECONDS - now % _HOUR_SECONDS) // 60
            raise HTTPException(
                status_code=429,
                detail=f"Hourly query limit reached ({_HOURLY_LIMIT}/hr). Resets in {mins} min.",
            )
        if not _redis_check(f"rl:q:min:{uid}", _MINUTE_LIMIT, _MINUTE_SECONDS):
            now = int(time.time())
            secs = _MINUTE_SECONDS - now % _MINUTE_SECONDS
            raise HTTPException(
                status_code=429,
                detail=f"Slow down — max {_MINUTE_LIMIT} queries/min. Try again in {secs}s.",
            )
    else:
        # In-memory fallback
        hour_ok, minute_ok = _mem_check(_mem_buckets, uid, _HOURLY_LIMIT, _MINUTE_LIMIT)
        if not hour_ok:
            now = int(time.time())
            mins = (_HOUR_SECONDS - now % _HOUR_SECONDS) // 60
            raise HTTPException(
                status_code=429,
                detail=f"Hourly query limit reached ({_HOURLY_LIMIT}/hr). Resets in {mins} min.",
            )
        if not minute_ok:
            now = int(time.time())
            secs = _MINUTE_SECONDS - now % _MINUTE_SECONDS
            raise HTTPException(
                status_code=429,
                detail=f"Slow down — max {_MINUTE_LIMIT} queries/min. Try again in {secs}s.",
            )


def check_upload_rate_limit(
    uid: str,
    *,
    uploads_this_month: int,
    monthly_upload_limit: int,
    file_size_bytes: int,
    max_upload_bytes: int,
) -> None:
    """Enforce upload-specific limits.  Raises HTTPException 429/413 if exceeded."""
    if file_size_bytes > max_upload_bytes:
        max_mb = max_upload_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=(
                f"File exceeds the {max_mb} MB plan limit. "
                "Compress or split the PDF before uploading."
            ),
        )
    if monthly_upload_limit and uploads_this_month >= monthly_upload_limit:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Monthly upload limit reached ({monthly_upload_limit} uploads/month). "
                "Upgrade your plan or wait until next month."
            ),
        )


def check_demo_rate_limit(ip: str) -> None:
    """Enforce per-IP rate limits for the unauthenticated demo endpoint."""
    if _get_redis() is not None:
        if not _redis_check(f"rl:demo:hour:{ip}", _DEMO_HOURLY_LIMIT, _HOUR_SECONDS):
            now = int(time.time())
            mins = (_HOUR_SECONDS - now % _HOUR_SECONDS) // 60
            raise HTTPException(
                status_code=429,
                detail=f"Demo limit reached. Try again in {mins} min or create a free account.",
            )
        if not _redis_check(f"rl:demo:min:{ip}", _DEMO_MINUTE_LIMIT, _MINUTE_SECONDS):
            now = int(time.time())
            secs = _MINUTE_SECONDS - now % _MINUTE_SECONDS
            raise HTTPException(
                status_code=429,
                detail=f"Slow down — try again in {secs}s.",
            )
    else:
        hour_ok, minute_ok = _mem_check(_mem_demo_buckets, ip, _DEMO_HOURLY_LIMIT, _DEMO_MINUTE_LIMIT)
        if not hour_ok:
            now = int(time.time())
            mins = (_HOUR_SECONDS - now % _HOUR_SECONDS) // 60
            raise HTTPException(
                status_code=429,
                detail=f"Demo limit reached. Try again in {mins} min or create a free account.",
            )
        if not minute_ok:
            now = int(time.time())
            secs = _MINUTE_SECONDS - now % _MINUTE_SECONDS
            raise HTTPException(
                status_code=429,
                detail=f"Slow down — try again in {secs}s.",
            )
