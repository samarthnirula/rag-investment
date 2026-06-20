"""Token-bucket rate limiter stored in Streamlit session state.

Each user session gets a bucket of tokens that refills over time.
When the bucket is empty, queries are blocked with a friendly message.

Limits:
  - monthly plan cap for margin protection
  - hourly cap for sustained abuse protection
  - minute cap for burst protection
"""
from __future__ import annotations

import time

import streamlit as st

from insightlens.billing import default_plan

# ── Configuration ──────────────────────────────────────────────────────────────
_HOURLY_LIMIT  = 30      # max queries per 1-hour window
_MINUTE_LIMIT  = 5       # max queries per 1-minute window (burst guard)
_HOUR_SECONDS  = 3600
_MINUTE_SECONDS = 60


def _init() -> None:
    if "rl_hour_window" not in st.session_state:
        st.session_state.rl_hour_window   = int(time.time()) // _HOUR_SECONDS
        st.session_state.rl_hour_count    = 0
        st.session_state.rl_minute_window = int(time.time()) // _MINUTE_SECONDS
        st.session_state.rl_minute_count  = 0
        st.session_state.rl_month_count   = 0


def check_rate_limit() -> tuple[bool, str]:
    """Return (allowed, reason_if_blocked).

    Call before processing any user query.
    Returns (True, "") when the query is allowed.
    Returns (False, message) when the limit is hit.
    """
    _init()
    now = int(time.time())

    # Roll windows if time has passed
    current_hour   = now // _HOUR_SECONDS
    current_minute = now // _MINUTE_SECONDS

    if current_hour != st.session_state.rl_hour_window:
        st.session_state.rl_hour_window = current_hour
        st.session_state.rl_hour_count  = 0

    if current_minute != st.session_state.rl_minute_window:
        st.session_state.rl_minute_window = current_minute
        st.session_state.rl_minute_count  = 0

    # Check limits
    monthly_limit = default_plan().monthly_query_limit
    if monthly_limit and st.session_state.rl_month_count >= monthly_limit:
        return (
            False,
            f"Monthly query limit reached ({monthly_limit} queries/month). "
            "This protects plan costs and keeps the service sustainable.",
        )

    if st.session_state.rl_hour_count >= _HOURLY_LIMIT:
        remaining = _HOUR_SECONDS - (now % _HOUR_SECONDS)
        mins = remaining // 60
        return False, f"Hourly query limit reached ({_HOURLY_LIMIT} queries/hour). Resets in {mins} min."

    if st.session_state.rl_minute_count >= _MINUTE_LIMIT:
        remaining = _MINUTE_SECONDS - (now % _MINUTE_SECONDS)
        return False, f"Slow down — max {_MINUTE_LIMIT} queries per minute. Try again in {remaining}s."

    # Consume tokens
    st.session_state.rl_hour_count   += 1
    st.session_state.rl_minute_count += 1
    st.session_state.rl_month_count  += 1
    return True, ""


def queries_remaining() -> tuple[int, int, int]:
    """Return (monthly_remaining, hourly_remaining, minute_remaining)."""
    _init()
    monthly_limit = default_plan().monthly_query_limit
    return (
        max(0, monthly_limit - st.session_state.rl_month_count),
        max(0, _HOURLY_LIMIT  - st.session_state.rl_hour_count),
        max(0, _MINUTE_LIMIT  - st.session_state.rl_minute_count),
    )


def seed_from_db(queries_today: int, queries_this_month: int = 0) -> None:
    """Seed counters from the DB so limits survive reloads and new tabs.

    Called once per session on startup when a user is authenticated.
    Only seeds if the session counter is lower than the DB count.
    """
    _init()
    if queries_today > st.session_state.rl_hour_count:
        st.session_state.rl_hour_count = min(queries_today, _HOURLY_LIMIT)
    if queries_this_month > st.session_state.rl_month_count:
        st.session_state.rl_month_count = min(
            queries_this_month,
            default_plan().monthly_query_limit,
        )
