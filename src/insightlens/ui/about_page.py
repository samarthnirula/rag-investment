"""About Us page — shown from the landing page footer."""
from __future__ import annotations

import streamlit as st

_NAVY = "#0f172a"
_BLUE = "#1e40af"
_MUTED = "#64748b"
_BORDER = "#e2e8f0"
_LIGHT = "#f8fafc"


def render_about_page() -> None:
    st.html("""<style>
    [data-testid="stSidebar"] { display: none !important; }
    footer { display: none !important; }
    #MainMenu { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
    </style>""")

    # ── Nav ────────────────────────────────────────────────────────────────────
    nc1, _, nc2 = st.columns([2, 5, 1])
    with nc1:
        st.html(
            f"<div style='padding:16px 0 12px 24px'>"
            f"<span style='font-family:\"Times New Roman\",Times,serif;"
            f"font-size:1.4rem;font-weight:800;color:{_NAVY}'>Atticus</span>"
            f"<span style='font-size:0.68rem;color:{_MUTED};margin-left:8px'>Legal Research Intelligence</span>"
            f"</div>"
        )
    with nc2:
        if st.button("← Back", key="about_back", use_container_width=True):
            st.session_state.pre_auth_page = "landing"
            st.rerun()

    st.html(f"<div style='height:1px;background:{_BORDER}'></div>")
    st.html("<div style='height:40px'></div>")

    _, col, _ = st.columns([1, 4, 1])
    with col:
        # ── Mission ───────────────────────────────────────────────────────────
        st.html(
            f"<div style='font-size:0.72rem;font-weight:700;color:{_BLUE};"
            f"letter-spacing:0.1em;margin-bottom:12px'>OUR MISSION</div>"
            f"<h1 style='font-size:2.4rem;font-weight:900;color:{_NAVY};"
            f"line-height:1.15;letter-spacing:-1px;margin:0 0 20px'>"
            f"Give every lawyer the research depth of a 10-person team.</h1>"
            f"<p style='font-size:1rem;color:{_MUTED};line-height:1.8;margin:0 0 40px'>"
            f"Legal research has not meaningfully changed since Westlaw was founded in 1975. "
            f"Lawyers still read linearly through thousands of pages, highlight by hand, "
            f"and cross-reference manually. The result: cases are won or lost on how much "
            f"time a firm can bill for research — not on who has the best argument."
            f"<br><br>"
            f"Atticus changes that. Upload a case file, ask a question in plain English, "
            f"and get a cited answer from the exact page in seconds. Every response is "
            f"grounded in your documents — not the internet, not a general-purpose chatbot."
            f"</p>"
        )

        st.html(f"<div style='height:1px;background:{_BORDER};margin-bottom:40px'></div>")

        # ── Product principles ─────────────────────────────────────────────────
        st.html(
            f"<div style='font-size:0.72rem;font-weight:700;color:{_BLUE};"
            f"letter-spacing:0.1em;margin-bottom:20px'>PRODUCT PRINCIPLES</div>"
        )
        principles = [
            ("Document-grounded answers",
             "Atticus only answers from what is in your documents. It does not guess, "
             "extrapolate from the internet, or hallucinate precedents. If the answer is "
             "not in your files, it says so."),
            ("Not legal advice — ever",
             "Every response carries a legal disclaimer. Atticus is a research tool, "
             "not a lawyer. No attorney-client relationship is created. We built the "
             "disclaimer into the system prompt, not just the UI."),
            ("Your data stays yours",
             "Your documents are never used to train AI models. Processed in isolated "
             "PostgreSQL/pgvector-backed workspaces. GDPR data deletion available from your Profile "
             "at any time, permanently."),
            ("Transparency over magic",
             "Every AI answer shows exactly which source page it came from. Lawyers "
             "should never trust a system they cannot verify — so we make verification "
             "the default, not an afterthought."),
        ]
        for title, body in principles:
            st.html(
                f"<div style='border-left:3px solid {_BLUE};padding:12px 16px;"
                f"margin-bottom:16px;background:{_LIGHT};border-radius:0 8px 8px 0'>"
                f"<div style='font-size:0.88rem;font-weight:700;color:{_NAVY};"
                f"margin-bottom:4px'>{title}</div>"
                f"<div style='font-size:0.8rem;color:{_MUTED};line-height:1.65'>{body}</div>"
                f"</div>"
            )

        st.html(f"<div style='height:1px;background:{_BORDER};margin:32px 0'></div>")

        # ── Stack ──────────────────────────────────────────────────────────────
        st.html(
            f"<div style='font-size:0.72rem;font-weight:700;color:{_BLUE};"
            f"letter-spacing:0.1em;margin-bottom:16px'>TECHNOLOGY</div>"
            f"<p style='font-size:0.88rem;color:{_MUTED};line-height:1.8;margin:0 0 16px'>"
            f"Atticus is built on a custom 8-stage hybrid retrieval system combining "
            f"BM25 keyword search, vector semantic search, Reciprocal Rank Fusion, "
            f"version-aware document scoring, and cross-encoder reranking. "
            f"The answer generation uses Claude (Anthropic) with document-grounded "
            f"system prompts designed specifically for legal research use cases."
            f"</p>"
            f"<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:32px'>"
            + "".join(
                f"<span style='background:#e0e7ff;color:#3730a3;font-size:0.72rem;"
                f"font-weight:600;border-radius:6px;padding:4px 12px'>{t}</span>"
                for t in ["Anthropic Claude", "PostgreSQL + pgvector", "Firebase Auth",
                          "SentenceTransformers", "BM25 + Vector Hybrid",
                          "Cross-encoder Rerank", "Tesseract OCR", "PyMuPDF",
                          "Python 3.12", "Streamlit", "Docker", "nginx TLS"]
            )
            + "</div>"
        )

        st.html(f"<div style='height:1px;background:{_BORDER};margin-bottom:32px'></div>")

        # ── Contact ────────────────────────────────────────────────────────────
        st.html(
            f"<div style='font-size:0.72rem;font-weight:700;color:{_BLUE};"
            f"letter-spacing:0.1em;margin-bottom:16px'>CONTACT</div>"
            f"<div style='font-size:0.88rem;color:{_MUTED};line-height:1.9'>"
            f"General enquiries: <strong>hello@atticus.ai</strong><br>"
            f"Legal & compliance: <strong>legal@atticus.ai</strong><br>"
            f"Access code requests: <strong>access@atticus.ai</strong><br>"
            f"Security disclosures: <strong>security@atticus.ai</strong>"
            f"</div>"
        )

        st.html("<div style='height:60px'></div>")

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.html(f"<div style='height:1px;background:{_BORDER}'></div>")
    fl, _, fr = st.columns([2, 4, 2])
    with fl:
        st.html(
            f"<div style='padding:16px 0;font-size:0.75rem;color:{_MUTED}'>"
            f"© 2026 Atticus</div>"
        )
    with fr:
        fc = st.columns(3)
        for col, (label, page) in zip(fc, [("Terms", "terms"), ("Privacy", "privacy"), ("About", "about")]):
            with col:
                if st.button(label, key=f"about_ft_{page}", use_container_width=True):
                    st.session_state.pre_auth_page = page
                    st.rerun()
