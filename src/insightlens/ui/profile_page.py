"""Profile page — user stats, account info, GDPR data deletion."""
from __future__ import annotations

import html

import streamlit as st

from insightlens.config import AppConfig
from insightlens.storage.audit_repository import AuditRepository
from insightlens.storage.cases_repository import CasesRepository
from insightlens.storage.chat_repository_persistent import PersistentChatRepository
from insightlens.storage.consent_repository import ConsentRepository
from insightlens.storage.snowflake_client import open_connection
from insightlens.ui.auth import sign_out


def render_profile_page(cfg: AppConfig, user: dict | None) -> None:
    st.html(
        "<div style='padding:16px 0 4px'>"
        "<span style='font-size:1.4rem;font-weight:700;color:#1e293b'>Profile</span>"
        "</div>"
    )

    if not user:
        st.info("Sign in to view your profile.")
        return

    uid = user.get("uid", "")
    email = user.get("email", "")
    display_name = user.get("display_name", email)

    # ── User info card ────────────────────────────────────────────────────────
    st.html(
        "<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;"
        "padding:20px 24px;margin-bottom:20px'>"
        "<div style='display:flex;align-items:center;gap:16px'>"
        "<div style='width:52px;height:52px;border-radius:50%;background:#e0e7ff;"
        "display:flex;align-items:center;justify-content:center;"
        "font-size:1.4rem;font-weight:700;color:#4338ca'>"
        + html.escape((display_name or "?")[0].upper())
        + "</div>"
        "<div>"
        f"<div style='font-size:1.05rem;font-weight:600;color:#1e293b'>{html.escape(display_name)}</div>"
        f"<div style='font-size:0.82rem;color:#64748b'>{html.escape(email)}</div>"
        f"<div style='font-size:0.72rem;color:#94a3b8;margin-top:2px;font-family:monospace'>{html.escape(uid[:16])}…</div>"
        "</div>"
        "</div>"
        "</div>"
    )

    # ── Usage statistics ──────────────────────────────────────────────────────
    st.html(
        "<div style='font-size:0.78rem;font-weight:600;color:#64748b;"
        "letter-spacing:0.05em;margin-bottom:8px'>USAGE STATISTICS</div>"
    )

    stats = {"total_queries": 0, "queries_this_month": 0, "queries_today": 0, "chunks_retrieved": 0}
    with st.spinner("Loading usage stats…"):
        try:
            with open_connection(cfg.db) as conn:
                stats = AuditRepository(conn).get_user_stats(uid)
        except Exception:
            pass

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [
        (c1, "Total queries", stats["total_queries"]),
        (c2, "This month", stats["queries_this_month"]),
        (c3, "Today", stats["queries_today"]),
        (c4, "Chunks retrieved", stats["chunks_retrieved"]),
    ]:
        with col:
            st.html(
                "<div style='background:#fff;border:1px solid #e2e8f0;border-radius:8px;"
                "padding:14px 16px;text-align:center'>"
                f"<div style='font-size:1.6rem;font-weight:700;color:#1e293b'>{value:,}</div>"
                f"<div style='font-size:0.75rem;color:#64748b;margin-top:2px'>{label}</div>"
                "</div>"
            )

    st.html("<div style='height:24px'></div>")

    # ── Cases & chats summary ──────────────────────────────────────────────────
    st.html(
        "<div style='font-size:0.78rem;font-weight:600;color:#64748b;"
        "letter-spacing:0.05em;margin-bottom:8px'>WORKSPACE</div>"
    )

    cases_count = 0
    chats_count = 0
    consent_dates: dict = {}
    with st.spinner("Loading workspace…"):
        try:
            with open_connection(cfg.db) as conn:
                cases_count   = len(CasesRepository(conn).list_cases(uid))
                chats_count   = len(PersistentChatRepository(conn).list_chats(uid))
                consent_dates = ConsentRepository(conn).get_consent_dates(uid)
        except Exception:
            pass

    c5, c6 = st.columns(2)
    with c5:
        st.html(
            "<div style='background:#fff;border:1px solid #e2e8f0;border-radius:8px;"
            "padding:14px 16px;text-align:center'>"
            f"<div style='font-size:1.6rem;font-weight:700;color:#1e293b'>{cases_count}</div>"
            "<div style='font-size:0.75rem;color:#64748b;margin-top:2px'>Cases</div>"
            "</div>"
        )
    with c6:
        st.html(
            "<div style='background:#fff;border:1px solid #e2e8f0;border-radius:8px;"
            "padding:14px 16px;text-align:center'>"
            f"<div style='font-size:1.6rem;font-weight:700;color:#1e293b'>{chats_count}</div>"
            "<div style='font-size:0.75rem;color:#64748b;margin-top:2px'>Saved chats</div>"
            "</div>"
        )

    st.html("<div style='height:20px'></div>")

    # ── Legal consent record ──────────────────────────────────────────────────
    st.html(
        "<div style='font-size:0.78rem;font-weight:600;color:#64748b;"
        "letter-spacing:0.05em;margin-bottom:8px'>LEGAL CONSENT</div>"
    )
    _terms_dt   = consent_dates.get("terms_v1")
    _privacy_dt = consent_dates.get("privacy_v1")
    _fmt = lambda dt: dt.strftime("%b %d, %Y at %H:%M UTC") if dt else "Not recorded"
    st.html(
        "<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;"
        "padding:12px 16px;font-size:0.78rem;color:#166534;line-height:1.8'>"
        f"<strong>Terms of Service v1</strong> accepted: {_fmt(_terms_dt)}<br>"
        f"<strong>Privacy Policy v1</strong> accepted: {_fmt(_privacy_dt)}"
        "</div>"
    )

    st.html("<div style='height:20px'></div>")

    # ── Danger zone ───────────────────────────────────────────────────────────
    with st.expander("⚠ Danger zone — delete account data"):
        st.warning(
            "This will permanently delete all your query logs, saved chats, cases, "
            "and uploaded documents. This action cannot be undone."
        )
        confirm = st.text_input(
            "Type your email to confirm deletion:",
            key="delete_confirm_email",
            placeholder=email,
        )
        if st.button("Delete all my data", type="primary", key="btn_delete_data"):
            if confirm.strip().lower() != email.lower():
                st.error("Email does not match. No data was deleted.")
            else:
                try:
                    with open_connection(cfg.db) as conn:
                        AuditRepository(conn).delete_user_logs(uid)
                        PersistentChatRepository(conn).delete_user_chats(uid)
                        CasesRepository(conn).delete_user_cases(uid)
                        ConsentRepository(conn).delete_user_consents(uid)
                    st.success("All your data has been deleted.")
                    sign_out()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Deletion failed: {exc}")
