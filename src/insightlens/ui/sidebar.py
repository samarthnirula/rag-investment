"""Sidebar — single source of truth. Called once from streamlit_app.py."""
from __future__ import annotations

import html as _html
from typing import Callable

import streamlit as st

from insightlens.ui.auth import current_user, sign_out

_CHAT_PAGES    = {"insightlens", "epstein"}
_DEFAULT_TOP_K = 8

_CSS = """
<style>
/* ── Premium Light Sidebar ───────────────────────────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background: #FFFFFF !important;
    border-right: 1px solid rgba(0,0,0,0.06) !important;
    box-shadow: 4px 0 20px rgba(0,0,0,0.03) !important;
}

/* ── Text styling — dark and elegant ─────────────────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #4A4A5A !important;
}

/* ── All buttons — premium light style ──────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    background: #FAFAF8 !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    border-radius: 12px !important;
    color: #5A5A6A !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 12px 14px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #F5F4F0 !important;
    border-color: rgba(102, 126, 234, 0.3) !important;
    color: #667EEA !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.06) !important;
}
/* Active/primary button — gradient accent */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(118, 75, 162, 0.3) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 16px rgba(118, 75, 162, 0.4) !important;
    transform: translateY(-1px);
}

/* ── Selectbox ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stSelectbox label {
    color: #667EEA !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background: #FAFAF8 !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    color: #4A4A5A !important;
    font-size: 0.82rem !important;
    border-radius: 10px !important;
}

/* ── Slider ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stSlider label {
    color: #667EEA !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stSliderThumb"]    { background: #667EEA !important; }
[data-testid="stSidebar"] .stSlider [data-testid="stSliderTrackFill"] { background: linear-gradient(90deg, #667EEA, #764BA2) !important; }
[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"]         { display: none; }

/* ── Divider ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] hr { border-color: rgba(0,0,0,0.06) !important; margin: 8px 0 !important; }

/* ── Profile icon (popover trigger) — gradient circle ────────────────────────── */
[data-testid="stSidebar"] [data-testid="stPopover"] button {
    width: 32px !important;
    height: 32px !important;
    min-width: 0 !important;
    min-height: 0 !important;
    max-width: 32px !important;
    max-height: 32px !important;
    padding: 0 !important;
    background: linear-gradient(135deg, #667EEA, #764BA2) !important;
    border: none !important;
    border-radius: 50% !important;
    aspect-ratio: 1 !important;
    overflow: hidden !important;
    box-shadow: 0 4px 12px rgba(118, 75, 162, 0.35) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stSidebar"] [data-testid="stPopover"] button *,
[data-testid="stSidebar"] [data-testid="stPopover"] button::before,
[data-testid="stSidebar"] [data-testid="stPopover"] button::after {
    display: none !important;
}

/* ── Tooltips off ──────────────────────────────────────────────────────────── */
[data-testid="stTooltipIcon"],
[role="tooltip"],
div[data-baseweb="tooltip"],
.stTooltipContent {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
button[title],
button[aria-label] {
    pointer-events: auto !important;
}
button[title]:hover::after,
button[aria-label]:hover::after {
    display: none !important;
}

/* ── Header column ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:first-of-type {
    align-items: center !important;
    padding: 16px 12px 12px !important;
    gap: 10px !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"] {
    padding: 0 !important;
    min-width: 0 !important;
}

/* ── Chat section label ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] .chat-section-label {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    color: #8A8A9A !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── Chat history hover state ─────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
    background: #F5F4F0 !important;
    color: #667EEA !important;
}
[data-testid="stSidebar"] .stButton > button:focus {
    box-shadow: none !important;
    outline: none !important;
}

/* ── Sidebar content padding ─────────────────────────────────────────────── */
[data-testid="stSidebarUserContent"] { padding-bottom: 24px !important; }
</style>
"""


def render_sidebar(
    companies: list[str],
    new_chat_fn: Callable,
    switch_page_fn: Callable,
    rename_chat_fn: Callable | None = None,
    delete_chat_fn: Callable | None = None,
) -> tuple[str, int]:
    """Render the sidebar. Returns (company_filter, top_k)."""
    st.html(_CSS)

    with st.sidebar:
        _user       = current_user()
        _user_label = (_user.get("display_name") or _user.get("email", "")) if _user else ""
        _ap         = st.session_state.get("active_page", "insightlens")

        # ── Header: gradient icon (dropdown menu) + app name + username ─────────
        _icon_col, _title_col = st.columns([1, 6], gap="small")

        with _icon_col:
            with st.popover(""):
                if _user_label:
                    st.html(
                        f"<div style='font-size:0.82rem;font-weight:600;color:#2D2D3A;"
                        f"padding:2px 0 10px;border-bottom:1px solid rgba(0,0,0,0.06);"
                        f"margin-bottom:8px'>{_html.escape(_user_label)}</div>"
                    )
                for _btn_key, _btn_page, _btn_label in [
                    ("menu_profile",    "profile",    "👤  Profile"),
                    ("menu_data",       "data",       "📊  Data"),
                    ("menu_cases",      "cases",      "⚖  Cases"),
                    ("menu_team",       "team",       "🏢  Team"),
                    ("menu_discussion", "discussion", "💬  Discussion"),
                ]:
                    if st.button(
                        _btn_label, key=_btn_key, use_container_width=True,
                        type="primary" if _ap == _btn_page else "secondary",
                    ):
                        st.session_state.active_page = _btn_page
                        st.rerun()
                st.divider()
                if _user:
                    if st.button("Sign out", key="menu_sign_out", use_container_width=True):
                        sign_out()
                        st.rerun()
                else:
                    if st.button("Sign in", key="menu_sign_in", use_container_width=True):
                        st.session_state.pre_auth_page = "landing"
                        st.rerun()

        with _title_col:
            st.html(
                "<div style='display:flex;align-items:center;justify-content:space-between;"
                "padding-right:6px'>"
                "<span style='font-size:0.95rem;font-weight:700;color:#2D2D3A;"
                "letter-spacing:-0.3px'>Atticus</span>"
                f"<span style='font-size:0.7rem;font-weight:500;color:#667EEA;"
                f"max-width:90px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>"
                f"{_html.escape(_user_label)}</span>"
                "</div>"
            )

        # ── Primary actions ───────────────────────────────────────────────────
        st.html("<div style='padding:0 10px 4px'>")
        if st.button(
            "＋  New chat", key="nav_new_chat", use_container_width=True,
            type="primary" if _ap == "insightlens" else "secondary",
        ):
            new_chat_fn("insightlens")
            st.session_state.active_page = "insightlens"
            st.rerun()

        if st.button(
            "⚖  Epstein's Case", key="nav_epstein", use_container_width=True,
            type="primary" if _ap == "epstein" else "secondary",
        ):
            switch_page_fn("epstein")
            st.rerun()
        st.html("</div>")

        # ── Page nav ──────────────────────────────────────────────────────────
        st.html("<div style='padding:0 10px 4px'>")
        _page_nav = [
            ("nav_profile", "profile", "👤  Profile"),
            ("nav_data",    "data",    "📊  Data"),
            ("nav_cases",   "cases",   "⚖  Cases"),
        ]
        _nav_cols = st.columns(3, gap="small")
        for _col, (_key, _pg, _lbl) in zip(_nav_cols, _page_nav):
            with _col:
                if st.button(
                    _lbl, key=_key, use_container_width=True,
                    type="primary" if _ap == _pg else "secondary",
                ):
                    st.session_state.active_page = _pg
                    st.rerun()
        st.html("</div>")

        st.divider()

        # ── Filters ───────────────────────────────────────────────────────────
        st.html("<div style='padding:0 10px'>")
        company_choice = st.selectbox(
            "Company filter",
            options=["All companies"] + companies,
        )
        top_k = st.slider("Sources per answer", min_value=3, max_value=15, value=_DEFAULT_TOP_K)
        st.html("</div>")

        st.divider()

        # ── Chat history ──────────────────────────────────────────────────────
        st.html(
            "<div style='padding:4px 16px 6px'>"
            "<span style='font-size:0.65rem;font-weight:600;color:#667EEA;"
            "letter-spacing:0.1em;text-transform:uppercase'>Chats</span>"
            "</div>"
        )
        st.html("<div style='min-height:160px;padding:0 10px'>")

        _chats_loaded = st.session_state.get("_chats_loaded", False)

        if _ap in _CHAT_PAGES and not _chats_loaded:
            # Show skeleton buttons while chat history loads from DB
            st.html(
                "<div style='padding:2px 0'>"
                "<span style='height:36px;margin-bottom:6px;border-radius:10px;"
                "background:linear-gradient(90deg,#f0f0f5,#e8e8f2,#f0f0f5);"
                "background-size:900px 100%;animation:sk-shimmer 1.05s linear infinite;"
                "display:block'></span>"
                "<span style='height:36px;margin-bottom:6px;border-radius:10px;width:85%;"
                "background:linear-gradient(90deg,#f0f0f5,#e8e8f2,#f0f0f5);"
                "background-size:900px 100%;animation:sk-shimmer 1.05s linear infinite;"
                "display:block'></span>"
                "<span style='height:36px;border-radius:10px;width:65%;"
                "background:linear-gradient(90deg,#f0f0f5,#e8e8f2,#f0f0f5);"
                "background-size:900px 100%;animation:sk-shimmer 1.05s linear infinite;"
                "display:block'></span>"
                "</div>"
            )
        elif _ap in _CHAT_PAGES:
            page_chats = [
                (cid, c)
                for cid, c in st.session_state.get("chats", {}).items()
                if c["page"] == _ap
            ]
            for cid, chat in reversed(page_chats):
                is_active = cid == st.session_state.get("current_chat_id")
                label = (chat["name"] or "Untitled chat")[:38]
                if st.button(
                    label, key=f"chat_{cid}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.current_chat_id = cid
                    st.rerun()

        st.html("</div>")

    return company_choice, top_k