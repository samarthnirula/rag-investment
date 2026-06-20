"""Data page — usage analytics dashboard (Power BI-style)."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from insightlens.billing import TARGET_GROSS_MARGIN, default_plan
from insightlens.config import AppConfig
from insightlens.storage.audit_repository import AuditRepository
from insightlens.storage.billing_repository import BillingRepository
from insightlens.storage.jobs_repository import JobsRepository
from insightlens.storage.snowflake_client import open_connection
from insightlens.storage.usage_repository import UsageRepository

_NAVY  = "#0f172a"
_BLUE  = "#1e40af"
_MUTED = "#64748b"
_BORDER = "#e2e8f0"
_LIGHT  = "#f8fafc"

_RANGE_OPTIONS = {"Last 7 days": 7, "Last 30 days": 30, "Last 60 days": 60, "Last 90 days": 90}
_PAGE_OPTIONS  = ["All", "insightlens", "epstein"]
_PAGE_LABELS   = {"All": "All chats", "insightlens": "Investment Chat", "epstein": "Epstein Chat"}


def _kpi(label: str, value: str, delta: str = "", color: str = _NAVY) -> str:
    delta_html = (
        f"<div style='font-size:0.72rem;color:#16a34a;margin-top:3px'>{delta}</div>"
        if delta else ""
    )
    return (
        f"<div style='background:#fff;border:1px solid {_BORDER};"
        f"border-radius:10px;padding:18px 20px'>"
        f"<div style='font-size:0.72rem;font-weight:600;color:{_MUTED};"
        f"letter-spacing:0.04em;margin-bottom:6px'>{label}</div>"
        f"<div style='font-size:1.9rem;font-weight:800;color:{color}'>{value}</div>"
        f"{delta_html}"
        f"</div>"
    )


def _section(title: str) -> None:
    st.html(
        f"<div style='font-size:0.72rem;font-weight:700;color:{_MUTED};"
        f"letter-spacing:0.06em;margin:28px 0 10px'>{title}</div>"
    )


def render_data_page(cfg: AppConfig, user: dict | None) -> None:
    st.html(
        "<div style='padding:16px 0 4px'>"
        "<span style='font-size:1.4rem;font-weight:700;color:#1e293b'>Data Usage</span>"
        "</div>"
        "<div style='font-size:0.85rem;color:#475569;margin-bottom:16px'>"
        "Query analytics and usage metrics for your account."
        "</div>"
    )

    if not user:
        st.info("Sign in to view your usage data.")
        return

    uid = user["uid"]
    user_email = (user.get("email") or "").lower()
    admin_emails = {
        item.strip().lower()
        for item in os.getenv("ATTICUS_ADMIN_EMAILS", "").split(",")
        if item.strip()
    }
    is_admin = bool(user_email and user_email in admin_emails)

    # ── Filter bar ────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        range_label = st.selectbox(
            "Date range", options=list(_RANGE_OPTIONS.keys()),
            index=1, label_visibility="collapsed",
        )
    with fc2:
        page_filter = st.selectbox(
            "Chat type", options=_PAGE_OPTIONS,
            format_func=lambda x: _PAGE_LABELS[x],
            label_visibility="collapsed",
        )
    with fc3:
        refresh = st.button("↺ Refresh", use_container_width=True)

    days = _RANGE_OPTIONS[range_label]
    page_arg = None if page_filter == "All" else page_filter

    # ── Load all data ─────────────────────────────────────────────────────────
    with st.spinner("Loading analytics…"):
        try:
            with open_connection(cfg.db) as conn:
                repo = AuditRepository(conn)
                stats       = repo.get_user_stats(uid)
                daily       = repo.get_daily_counts(uid, days=days)
                by_page     = repo.get_page_breakdown(uid, days=days)
                by_hour     = repo.get_hourly_distribution(uid, days=days)
                avg_sources = repo.get_chunks_over_time(uid, days=days)
                recent      = repo.get_recent_queries(uid, limit=25, page_filter=page_filter)
                usage_repo = UsageRepository(conn)
                uploads_this_month = usage_repo.count_uploads_this_month(uid)
                upload_cost_this_month = usage_repo.estimated_upload_cost_this_month(uid)
        except Exception as exc:
            st.error(f"Could not load analytics: {exc}")
            return

    # ── KPI cards ─────────────────────────────────────────────────────────────
    _section("KEY METRICS")
    k1, k2, k3, k4 = st.columns(4, gap="medium")

    # Calculate period queries from daily data
    period_queries = sum(d["Queries"] for d in daily)
    avg_src = (
        round(sum(r["Avg Sources"] for r in avg_sources) / len(avg_sources), 1)
        if avg_sources else 0.0
    )
    plan = default_plan()
    upload_cost_this_month = locals().get("upload_cost_this_month", 0.0)
    uploads_this_month = locals().get("uploads_this_month", 0)
    est_cost = float(stats.get("estimated_cost_usd", 0.0)) + float(upload_cost_this_month)
    cost_cap = plan.monthly_variable_cost_cap_usd
    margin_color = "#16a34a" if est_cost <= cost_cap else "#dc2626"

    with k1:
        st.html(_kpi("Total queries (all time)", f"{stats['total_queries']:,}", color=_BLUE))
    with k2:
        st.html(_kpi(f"Queries ({range_label.lower()})", f"{period_queries:,}"))
    with k3:
        st.html(_kpi("Queries today", f"{stats['queries_today']:,}"))
    with k4:
        st.html(_kpi("Avg sources per query", f"{avg_src}", color="#16a34a"))

    k5, k6 = st.columns(2, gap="medium")
    with k5:
        st.html(_kpi("Estimated AI cost", f"${est_cost:.2f}", color=margin_color))
    with k6:
        st.html(
            _kpi(
                f"{int(TARGET_GROSS_MARGIN * 100)}% margin cost cap",
                f"${cost_cap:.2f}",
                color="#0f172a",
            )
        )
    st.caption(
        f"Uploads this month: {uploads_this_month} · "
        f"estimated upload cost: ${float(upload_cost_this_month):.2f}. "
        "These are planning estimates, not provider invoices."
    )

    # ── Queries over time ─────────────────────────────────────────────────────
    _section(f"QUERIES OVER TIME — {range_label.upper()}")
    if daily:
        df_daily = pd.DataFrame(daily).set_index("date")
        st.area_chart(df_daily, height=220, color="#1e40af")
    else:
        st.html(
            "<div style='text-align:center;padding:40px;color:#94a3b8;"
            "background:#f8fafc;border-radius:10px;font-size:0.88rem'>"
            "No queries in this period yet.</div>"
        )

    # ── Side-by-side breakdowns ───────────────────────────────────────────────
    _section("BREAKDOWNS")
    left, right = st.columns(2, gap="large")

    with left:
        st.html(
            f"<div style='font-size:0.75rem;font-weight:600;color:{_NAVY};"
            f"margin-bottom:8px'>Queries by chat type</div>"
        )
        if by_page:
            df_page = pd.DataFrame(by_page).set_index("Page")
            st.bar_chart(df_page, height=200, color="#1e40af")
        else:
            st.html("<div style='color:#94a3b8;font-size:0.82rem'>No data.</div>")

    with right:
        st.html(
            f"<div style='font-size:0.75rem;font-weight:600;color:{_NAVY};"
            f"margin-bottom:8px'>Peak usage hours (UTC)</div>"
        )
        peak_data = [h for h in by_hour if h["Queries"] > 0]
        if peak_data:
            df_hour = pd.DataFrame(by_hour).set_index("Hour")
            st.bar_chart(df_hour, height=200, color="#f59e0b")
        else:
            st.html("<div style='color:#94a3b8;font-size:0.82rem'>No data.</div>")

    # ── Average sources trend ─────────────────────────────────────────────────
    if avg_sources:
        _section("AVERAGE SOURCES RETRIEVED PER DAY")
        df_src = pd.DataFrame(avg_sources).set_index("date")
        st.line_chart(df_src, height=160, color="#16a34a")

    # ── Recent queries table ───────────────────────────────────────────────────
    _section("RECENT QUERIES")
    if recent:
        df_recent = pd.DataFrame(recent)
        st.dataframe(
            df_recent,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Time":     st.column_config.TextColumn("Time",       width="medium"),
                "Chat":     st.column_config.TextColumn("Chat",       width="small"),
                "Query":    st.column_config.TextColumn("Query",      width="large"),
                "Sources":  st.column_config.NumberColumn("Sources",  width="small", format="%d"),
                "Resp len": st.column_config.NumberColumn("Resp len", width="small", format="%d chars"),
            },
            height=min(40 * len(df_recent) + 40, 500),
        )
        st.html(
            f"<div style='font-size:0.72rem;color:{_MUTED};margin-top:6px'>"
            f"Showing {len(recent)} most recent queries"
            + (f" · filtered to {_PAGE_LABELS[page_filter]}" if page_filter != "All" else "")
            + "</div>"
        )
    else:
        st.html(
            "<div style='text-align:center;padding:40px;color:#94a3b8;"
            "background:#f8fafc;border-radius:10px;font-size:0.88rem'>"
            "No queries yet. Start chatting to see your usage data here.</div>"
        )

    # ── Usage limit indicator ─────────────────────────────────────────────────
    st.html("<div style='height:20px'></div>")
    from insightlens.ui.rate_limiter import queries_remaining
    monthly_rem, hourly_rem, minute_rem = queries_remaining()
    st.html(
        f"<div style='background:{_LIGHT};border:1px solid {_BORDER};"
        f"border-radius:8px;padding:12px 16px;"
        f"display:flex;justify-content:space-between;align-items:center'>"
        f"<div style='font-size:0.78rem;color:{_NAVY};font-weight:600'>Current session limits</div>"
        f"<div style='font-size:0.78rem;color:{_MUTED}'>"
        f"{monthly_rem} left this month &nbsp;·&nbsp; "
        f"{hourly_rem} left this hour &nbsp;·&nbsp; "
        f"{minute_rem} left this minute"
        f"</div></div>"
    )

    if is_admin:
        _section("ADMIN MARGIN WATCH")
        try:
            with open_connection(cfg.db) as conn:
                rows = BillingRepository(conn).admin_margin_rows(limit=50)
                job_stats = JobsRepository(conn).stats()
        except Exception:
            rows = []
            job_stats = {}

        if rows:
            admin_df = pd.DataFrame(
                [
                    {
                        "User": row.user_id,
                        "Queries": row.queries_this_month,
                        "Uploads": row.uploads_this_month,
                        "Estimated cost": round(row.estimated_cost_usd, 4),
                        "Over starter cap": row.estimated_cost_usd > plan.monthly_variable_cost_cap_usd,
                    }
                    for row in rows
                ]
            )
            st.dataframe(admin_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No admin usage data yet.")

        if job_stats:
            st.caption(
                "Background jobs: "
                + ", ".join(f"{status}={count}" for status, count in sorted(job_stats.items()))
            )
