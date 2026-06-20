"""Discussion board — shared team forum for legal research notes."""
from __future__ import annotations

import html

import streamlit as st

from insightlens.config import AppConfig
from insightlens.storage.discussion_repository import DiscussionRepository
from insightlens.storage.snowflake_client import open_connection

_POST_TYPES = ["Finding", "Question", "Note", "Risk", "Timeline", "Other"]

_TYPE_STYLE: dict[str, tuple[str, str]] = {
    "Finding":  ("#dcfce7", "#166534"),
    "Question": ("#dbeafe", "#1e40af"),
    "Note":     ("#f3f4f6", "#374151"),
    "Risk":     ("#fee2e2", "#991b1b"),
    "Timeline": ("#fef9c3", "#854d0e"),
    "Other":    ("#f3e8ff", "#6b21a8"),
}

_TYPE_ICON = {
    "Finding":  "✅",
    "Question": "❓",
    "Note":     "📝",
    "Risk":     "🚩",
    "Timeline": "📅",
    "Other":    "💬",
}


def _post_type_badge(post_type: str) -> str:
    bg, color = _TYPE_STYLE.get(post_type, ("#f3f4f6", "#374151"))
    icon = _TYPE_ICON.get(post_type, "💬")
    return (
        f"<span style='background:{bg};color:{color};font-size:0.65rem;font-weight:700;"
        f"padding:2px 8px;border-radius:4px;margin-right:6px'>{icon} {html.escape(post_type)}</span>"
    )


def render_discussion_page(cfg: AppConfig, user: dict | None) -> None:
    st.html(
        "<div style='padding:16px 0 4px'>"
        "<span style='font-size:1.4rem;font-weight:700;color:#1e293b'>Discussion</span>"
        "</div>"
        "<div style='font-size:0.85rem;color:#475569;margin-bottom:20px'>"
        "Share findings, flag risks, and post questions with your team."
        "</div>"
    )

    if not user:
        st.info("Sign in to read and post in the discussion board.")
        return

    uid = user["uid"]
    display_name = user.get("display_name") or user.get("email", "User")

    # ── Filter bar ────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        filter_type = st.selectbox(
            "Filter by type",
            ["All"] + _POST_TYPES,
            label_visibility="collapsed",
        )
    with fc3:
        if st.button("↺ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── New post ──────────────────────────────────────────────────────────────
    with st.expander("＋ Post a finding or question", expanded=False):
        with st.form("discussion_post_form", clear_on_submit=True):
            post_type = st.selectbox("Type", _POST_TYPES, key="new_post_type")
            content = st.text_area(
                "Your message",
                height=100,
                max_chars=4000,
                placeholder="Share a finding, risk, date, or question from your documents…",
                key="new_post_content",
            )
            submitted = st.form_submit_button("Post", type="primary", use_container_width=True)
            if submitted:
                if not content.strip():
                    st.error("Message cannot be empty.")
                else:
                    try:
                        with open_connection(cfg.db) as conn:
                            DiscussionRepository(conn).add_post(
                                author=display_name,
                                post_type=post_type,
                                content=content.strip(),
                            )
                        st.success("Posted.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not post: {exc}")

    st.divider()

    # ── Load posts ────────────────────────────────────────────────────────────
    posts: list[dict] = []
    with st.spinner("Loading posts…"):
        try:
            with open_connection(cfg.db) as conn:
                posts = DiscussionRepository(conn).list_posts(limit=200)
        except Exception as exc:
            st.error(f"Could not load posts: {exc}")

    if filter_type != "All":
        posts = [p for p in posts if p["type"] == filter_type]

    if not posts:
        st.html(
            "<div style='padding:48px 0;text-align:center;color:#94a3b8;font-size:0.9rem'>"
            "No posts yet — be the first to share a finding."
            "</div>"
        )
        return

    st.html(
        f"<div style='font-size:0.72rem;color:#94a3b8;margin-bottom:12px'>"
        f"{len(posts)} post{'s' if len(posts) != 1 else ''}"
        + (f" · filtered to {filter_type}" if filter_type != "All" else "")
        + "</div>"
    )

    # ── Render posts (newest first) ───────────────────────────────────────────
    for post in reversed(posts):
        is_own = post["author"] == display_name
        border_color = _TYPE_STYLE.get(post["type"], ("#e2e8f0", "#64748b"))[0]

        col_post, col_del = st.columns([10, 1])
        with col_post:
            st.html(
                f"<div style='border-left:3px solid {border_color};"
                f"background:#fafafa;border-radius:0 8px 8px 0;"
                f"padding:12px 16px;margin-bottom:10px'>"
                f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:6px'>"
                f"{_post_type_badge(post['type'])}"
                f"<span style='font-size:0.78rem;font-weight:600;color:#1e293b'>"
                f"{html.escape(post['author'])}</span>"
                f"<span style='font-size:0.72rem;color:#94a3b8;margin-left:4px'>"
                f"{html.escape(post['time'])}</span>"
                f"</div>"
                f"<div style='font-size:0.85rem;color:#334155;line-height:1.65;white-space:pre-wrap'>"
                f"{html.escape(post['content'])}</div>"
                f"</div>"
            )
        with col_del:
            if is_own:
                if st.button("✕", key=f"del_post_{post['post_id']}", help="Delete your post"):
                    try:
                        with open_connection(cfg.db) as conn:
                            DiscussionRepository(conn).delete_post(post["post_id"])
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

    # ── Export ────────────────────────────────────────────────────────────────
    if posts:
        st.divider()
        export_md = "# Discussion Board Export\n\n"
        for post in posts:
            export_md += (
                f"## [{post['type']}] {post['author']} — {post['time']}\n\n"
                f"{post['content']}\n\n---\n\n"
            )
        st.download_button(
            "Export discussion as Markdown",
            data=export_md,
            file_name="atticus_discussion.md",
            mime="text/markdown",
        )
