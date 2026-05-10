"""InsightLens — chat-style UI for investment document Q&A."""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit Cloud runs `pip install -r requirements.txt` but NOT `pip install -e .`,
# so the local `insightlens` package isn't on sys.path by default.
# Insert src/ so imports work both locally and in the cloud.
sys.path.insert(0, str(Path(__file__).parents[2]))

import os

import streamlit as st

# When deployed on Streamlit Community Cloud, secrets live in st.secrets (not .env).
# Inject them into os.environ so the rest of the app (load_config) sees them normally.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass  # running locally — .env already loaded by python-dotenv

from insightlens.config import ConfigError, load_config
from insightlens.embeddings.embedder import Embedder
from insightlens.generation.llm_client import ClaudeClient
from insightlens.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from insightlens.retrieval.vector_search import RetrievalRequest, VectorSearchService
from insightlens.storage.chunk_repository import ChunkRepository, RetrievedChunk
from insightlens.storage.snowflake_client import open_connection

st.set_page_config(page_title="InsightLens", layout="wide", page_icon="🔍")

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Startup (cached — runs once per session) ───────────────────────────────────
@st.cache_resource
def _bootstrap():
    cfg = load_config()
    embedder = Embedder(model=cfg.embedding_model)
    llm = ClaudeClient(api_key=cfg.anthropic_api_key, model=cfg.generation_model)
    return cfg, embedder, llm

@st.cache_data(ttl=300)
def _load_companies(_cfg):
    with open_connection(_cfg.snowflake) as conn:
        return ChunkRepository(conn).list_companies()

try:
    cfg, embedder, llm = _bootstrap()
except ConfigError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()

try:
    companies = _load_companies(cfg)
except Exception as exc:
    st.error(f"Could not connect to Snowflake: {exc}")
    st.info("Check that your secrets are configured in the Streamlit Cloud dashboard.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 InsightLens")
    st.caption("Investment document Q&A")
    st.divider()
    company_choice = st.selectbox(
        "Filter by company",
        options=["All companies"] + companies,
    )
    top_k = st.slider("Sources to retrieve", min_value=3, max_value=15, value=cfg.retrieval_top_k)
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Source cards ───────────────────────────────────────────────────────────────
def _render_sources(chunks: list[RetrievedChunk]) -> None:
    if not chunks:
        return
    st.markdown("---")
    st.markdown("**Sources**")
    for i, chunk in enumerate(chunks, start=1):
        company = chunk.company or "Unknown"
        version = chunk.version_label or "unversioned"
        pct = f"{chunk.similarity:.0%}"
        label = f"[{i}]  {chunk.file_name}   ·   Page {chunk.page_number}   ·   {company}   ·   {pct} match"
        with st.expander(label):
            cols = st.columns(2)
            cols[0].caption(f"Version: {version}")
            cols[1].caption(f"Similarity: {chunk.similarity:.3f}")
            st.markdown(chunk.chunk_text)

# ── Empty state — centered welcome ─────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("<div style='height:28vh'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            "<h2 style='text-align:center'>InsightLens</h2>"
            "<p style='text-align:center;color:gray'>"
            "Ask anything about your investment documents."
            "</p>",
            unsafe_allow_html=True,
        )

# ── Conversation history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("chunks"):
            _render_sources(msg["chunks"])

# ── Chat input (sticks to bottom of page automatically) ────────────────────────
question = st.chat_input("Ask about your investment documents…")

if question:
    company_filter = None if company_choice == "All companies" else company_choice

    # Show user bubble immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve chunks then stream the answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            with open_connection(cfg.snowflake) as conn:
                repo = ChunkRepository(conn)
                retrieval = VectorSearchService(embedder=embedder, repository=repo)
                chunks = retrieval.retrieve(
                    RetrievalRequest(query=question, top_k=top_k, company_filter=company_filter)
                )

        user_prompt = build_user_prompt(question, chunks)
        # st.write_stream feeds the generator token-by-token into the UI
        answer_text = st.write_stream(llm.stream(SYSTEM_PROMPT, user_prompt))
        _render_sources(chunks)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "chunks": chunks,
    })
