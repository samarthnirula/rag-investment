"""Atticus — chat-style UI for legal and public-record document Q&A."""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

# Streamlit Cloud runs `pip install -r requirements.txt` but NOT `pip install -e .`,
# so the local `insightlens` package isn't on sys.path by default.
# Insert src/ so imports work both locally and in the cloud.
sys.path.insert(0, str(Path(__file__).parents[2]))

import html
import json
import os
import re

import pandas as pd
import streamlit as st

# set_page_config must be the very first Streamlit call in the script.
st.set_page_config(page_title="Atticus", layout="wide", page_icon="⚖")

# On Streamlit Cloud there is no .env file — secrets come from the dashboard.
# Locally, python-dotenv handles everything, so skip st.secrets to avoid the
# "No secrets found" warning that appears when no secrets.toml is present.
_env_file = Path(__file__).parents[3] / ".env"
if not _env_file.exists():
    try:
        for _k, _v in st.secrets.items():
            if isinstance(_v, str):
                os.environ.setdefault(_k, _v)
    except Exception:
        pass

from insightlens.config import ConfigError, load_config
from insightlens.billing import estimate_query_cost_usd
from insightlens.embeddings.embedder import Embedder
from insightlens.generation.llm_client import ClaudeClient
from insightlens.generation.prompts import CASE_SYSTEM_PROMPT, SYSTEM_PROMPT, build_user_prompt
from insightlens.retrieval.hybrid_search import HybridSearchService
from insightlens.retrieval.reranker import Reranker
from insightlens.retrieval.vector_search import RetrievalRequest
from insightlens.storage.audit_repository import AuditRepository
from insightlens.storage.chat_repository_persistent import PersistentChatRepository
from insightlens.storage.chunk_repository import ChunkRepository, RetrievedChunk
from insightlens.storage.consent_repository import ConsentRepository
from insightlens.storage.image_repository import ImageRepository
from insightlens.storage.snowflake_client import open_connection
from insightlens.ui.about_page import render_about_page
from insightlens.ui.discussion_page import render_discussion_page
from insightlens.ui.org_page import render_org_page
from insightlens.ui.auth import (current_user, get_email_verified, is_authenticated,
                                  refresh_token, send_email_verification, sign_out)
from insightlens.ingestion.cloud_auth import (
    exchange_dropbox_code,
    exchange_google_drive_code,
    exchange_onedrive_code,
    parse_oauth_state,
)
from insightlens.ui.landing_page import render_landing_page
from insightlens.ui.cases_page import render_cases_page
from insightlens.storage.cloud_credentials_repository import CloudCredentialsRepository
from insightlens.ui.data_page import render_data_page
from insightlens.ui.input_guard import InputGuardError, validate_query, validate_text_input
from insightlens.ui.legal_page import render_legal_page
from insightlens.ui.profile_page import render_profile_page
from insightlens.ui.rate_limiter import check_rate_limit, queries_remaining, seed_from_db
from insightlens.ui.sidebar import render_sidebar

# ── Constants ──────────────────────────────────────────────────────────────────
_DEFAULT_TOP_K = 8
_CHAT_PAGES = {"insightlens", "epstein"}  # pages that have chat history
_VALID_PAGES = {"insightlens", "epstein", "profile", "data", "cases", "legal", "about", "discussion"}

# ── Session state ──────────────────────────────────────────────────────────────
if "active_page" not in st.session_state:
    st.session_state.active_page = "insightlens"
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None


def _new_chat(page: str) -> str:
    chat_id = uuid.uuid4().hex[:16]
    st.session_state.chats[chat_id] = {
        "name": "New Chat", "messages": [], "page": page, "persisted": False
    }
    st.session_state.current_chat_id = chat_id
    # Persist to DB when user is authenticated and cfg is available
    user = current_user()
    if user:
        try:
            with open_connection(cfg.db) as conn:
                PersistentChatRepository(conn).create_chat(
                    user["uid"], page, "New Chat", chat_id=chat_id
                )
            st.session_state.chats[chat_id]["persisted"] = True
        except Exception:
            pass
    return chat_id


def _db_rename_chat(chat_id: str, new_name: str) -> None:
    """Rename a chat in both session state and DB."""
    if chat_id in st.session_state.chats:
        st.session_state.chats[chat_id]["name"] = new_name
    user = current_user()
    if user and st.session_state.chats.get(chat_id, {}).get("persisted"):
        try:
            with open_connection(cfg.db) as conn:
                PersistentChatRepository(conn).update_chat_name(chat_id, new_name, user["uid"])
        except Exception:
            pass


def _db_delete_chat(chat_id: str, page: str) -> None:
    """Delete a chat from session state and DB, switch to next available."""
    del st.session_state.chats[chat_id]
    if st.session_state.get("current_chat_id") == chat_id:
        remaining = [cid for cid, c in st.session_state.chats.items() if c["page"] == page]
        if remaining:
            st.session_state.current_chat_id = remaining[-1]
        else:
            _new_chat(page)
    user = current_user()
    if user:
        try:
            with open_connection(cfg.db) as conn:
                PersistentChatRepository(conn).delete_chat(chat_id, user["uid"])
        except Exception:
            pass


def _save_message(chat_id: str, role: str, content: str) -> None:
    """Persist a single message to PostgreSQL (fire-and-forget)."""
    if not st.session_state.chats.get(chat_id, {}).get("persisted"):
        return
    try:
        with open_connection(cfg.db) as conn:
            PersistentChatRepository(conn).save_message(chat_id, role, content)
    except Exception:
        pass


def _get_current_chat() -> dict | None:
    cid = st.session_state.current_chat_id
    if cid and cid in st.session_state.chats:
        c = st.session_state.chats[cid]
        if c["page"] == st.session_state.active_page:
            return c
    return None


def _switch_page(page: str) -> None:
    st.session_state.active_page = page
    page_chats = [
        (cid, c) for cid, c in st.session_state.chats.items() if c["page"] == page
    ]
    if page_chats:
        st.session_state.current_chat_id = page_chats[-1][0]
    else:
        _new_chat(page)

# ── Startup (cached — runs once per session) ───────────────────────────────────
@st.cache_resource
def _bootstrap():
    cfg = load_config()
    embedder = Embedder(model=cfg.embedding_model)
    llm = ClaudeClient(api_key=cfg.anthropic_api_key, model=cfg.generation_model)
    reranker = Reranker()

    # Start the background job runner (daemon thread — processes queued jobs)
    from insightlens.jobs.runner import get_runner
    from insightlens.jobs.handlers import register_all_handlers
    runner = get_runner(cfg)
    register_all_handlers(runner, cfg)
    runner.start()

    return cfg, embedder, llm, reranker

@st.cache_data(ttl=300)
def _load_companies(_cfg):
    with open_connection(_cfg.db) as conn:
        return ChunkRepository(conn).list_companies()

@st.cache_data(ttl=600)
def _load_corpus(_cfg):
    """Load document chunks for BM25. Refreshes every 10 minutes."""
    with open_connection(_cfg.db) as conn:
        return ChunkRepository(conn).get_all_chunks(company_filter=None)

@st.cache_data(ttl=600)
def _load_corpus_epstein(_cfg):
    """Load Epstein-only chunks for BM25. Keeps the index focused."""
    with open_connection(_cfg.db) as conn:
        return ChunkRepository(conn).get_all_chunks(company_filter="Epstein")

@st.cache_data(ttl=600)
def _load_image_index(_cfg) -> dict[tuple[str, int], list]:
    """Return {(document_id, page_number): [ImageRecord, ...]} for inline display."""
    try:
        with open_connection(_cfg.db) as conn:
            records = ImageRepository(conn).get_all_image_metadata()
        index: dict[tuple[str, int], list] = {}
        for rec in records:
            key = (rec.document_id, rec.page_number)
            index.setdefault(key, []).append(rec)
        return index
    except Exception:
        return {}

# ── Google Sign-In callback (Firebase JS SDK passes params after redirect auth) ─
# The refresh token is NOT passed in the URL (avoids server-log exposure).
# It was stored in sessionStorage by the sign-in JS and is read back below.
import time as _time
_fb_token = st.query_params.get("firebase_token", "")
_fb_refresh_token = st.query_params.get("firebase_refresh_token", "")
_fb_uid   = st.query_params.get("firebase_uid", "")
_fb_email = st.query_params.get("firebase_email", "")
_fb_name  = st.query_params.get("firebase_name", "")
if _fb_token and _fb_uid and not is_authenticated():
    st.session_state.user_uid           = _fb_uid
    st.session_state.user_email         = _fb_email
    st.session_state.user_display_name  = _fb_name or (_fb_email.split("@")[0] if _fb_email else "User")
    st.session_state.user_id_token      = _fb_token
    st.session_state.user_refresh_token = _fb_refresh_token
    st.session_state._email_verified    = True   # Google accounts are pre-verified
    st.session_state._token_refreshed_at = _time.time()
    st.session_state.active_page = "insightlens"
    st.query_params.clear()
    st.rerun()

# ── Cloud storage OAuth callback ────────────────────────────────────────────────
_cloud_code = st.query_params.get("cloud_code", "")
_cloud_state = st.query_params.get("cloud_state", "")
if _cloud_code and _cloud_state and is_authenticated():
    try:
        provider_key, return_url = parse_oauth_state(_cloud_state)
        uid = st.session_state.get("user_uid", "")
        if not uid:
            raise ValueError("No user session")

        if provider_key == "google_drive":
            tokens = exchange_google_drive_code(_cloud_code, _get_base_url())
        elif provider_key == "dropbox":
            tokens = exchange_dropbox_code(_cloud_code, _get_base_url())
        elif provider_key == "onedrive":
            tokens = exchange_onedrive_code(_cloud_code, _get_base_url())
        else:
            raise ValueError(f"Unknown provider: {provider_key}")

        from datetime import datetime, timezone
        exp = datetime.fromtimestamp(_time.time() + tokens.expires_in, tz=timezone.utc) if tokens.expires_in else None
        with open_connection(_cfg.db) as conn:
            CloudCredentialsRepository(conn).upsert(
                user_id=uid,
                provider=provider_key,
                refresh_token=tokens.refresh_token or "",
                access_token=tokens.access_token,
                token_expires_at=exp,
            )
        st.query_params.clear()
        st.session_state.active_page = "cases"
        st.rerun()
    except Exception as exc:
        st.error(f"Cloud connection failed: {exc}")
        st.query_params.clear()


def _get_base_url() -> str:
    return "https://" + os.getenv("STREAMLIT_SERVER_BASE_URL", "app.atticus.ai")


# Second leg: pick up the refresh token from the secure sessionStorage redirect
_fb_rt = st.query_params.get("firebase_rt", "")
if _fb_rt and is_authenticated() and not st.session_state.get("user_refresh_token"):
    st.session_state.user_refresh_token = _fb_rt
    st.query_params.clear()
    st.rerun()

# Trigger second leg: if authenticated but no refresh token yet, read from sessionStorage
# (fires once after Google Sign-In redirect lands; JS is a no-op if sessionStorage is empty)
if is_authenticated() and not st.session_state.get("user_refresh_token"):
    st.components.v1.html("""<script>
    (function(){
      try {
        var rt = window.top.sessionStorage.getItem('__att_rt');
        if (rt) {
          window.top.sessionStorage.removeItem('__att_rt');
          var base = window.top.location.origin + window.top.location.pathname;
          window.top.location.href = base + '?firebase_rt=' + encodeURIComponent(rt);
        }
      } catch(_) {}
    })();
    </script>""", height=0)

# ── Public demo mode ─────────────────────────────────────────────────────────
if st.query_params.get("demo", "") in {"1", "true", "yes"}:
    st.session_state.demo_mode = True
    st.session_state.active_page = "epstein"
    st.query_params.clear()
    st.rerun()

_demo_mode = bool(st.session_state.get("demo_mode")) and not is_authenticated()

# ── Auth gate — landing page is the entry point ───────────────────────────────
_firebase_configured = bool(os.getenv("FIREBASE_API_KEY") or os.getenv("FIREBASE_WEB_API_KEY"))
if not is_authenticated() and not _demo_mode:
    _pre = st.session_state.get("pre_auth_page", "landing")
    if _pre == "about":
        render_about_page()
    elif _pre == "terms":
        render_legal_page("terms", pre_auth=True)
    elif _pre == "privacy":
        render_legal_page("privacy", pre_auth=True)
    else:
        render_landing_page()
    st.stop()

# ── Skeleton loading screen ────────────────────────────────────────────────────
# Shown while DB queries, chat loading, and rate-limiter seeding complete on
# first session load (browser refresh). Cleared just before the page renders.
_SK_CSS = """<style>
@keyframes sk-shimmer {
    0%   { background-position: -900px 0; opacity: .72; }
    50%  { opacity: 1; }
    100% { background-position:  900px 0; opacity: .72; }
}
.sk {
    background: linear-gradient(90deg,
        #f0f0f5 0%, #e8e8f2 35%, #f8f8fc 50%, #e8e8f2 65%, #f0f0f5 100%);
    background-size: 900px 100%;
    animation: sk-shimmer 1.05s linear infinite !important;
    animation-iteration-count: infinite !important;
    animation-fill-mode: none !important;
    border-radius: 6px;
    display: block;
}
.sk-d {
    background: linear-gradient(90deg,
        #1a1a22 0%, #222230 35%, #2a2a38 50%, #222230 65%, #1a1a22 100%);
    background-size: 900px 100%;
    animation: sk-shimmer 1.05s linear infinite !important;
    animation-iteration-count: infinite !important;
    animation-fill-mode: none !important;
    border-radius: 6px;
    display: block;
}
.sk-shell {
    min-height: 100vh;
    background: #f8fafc;
    display: grid;
    grid-template-columns: 260px minmax(0, 1fr);
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.sk-sidebar {
    background: #0d0e1a;
    border-right: 1px solid #1f2433;
    padding: 18px 16px;
}
.sk-main {
    padding: 22px 28px 30px;
}
.sk-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
}
.sk-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.55fr) minmax(280px, .85fr);
    gap: 18px;
}
.sk-panel {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 12px 32px rgba(15, 23, 42, .04);
}
@media (max-width: 780px) {
    .sk-shell { grid-template-columns: 1fr; }
    .sk-sidebar { display: none; }
    .sk-main { padding: 18px 14px; }
    .sk-grid { grid-template-columns: 1fr; }
}
</style>"""

_SK_CARDS_HTML = """
<div class="sk-shell">
  <aside class="sk-sidebar">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:28px">
      <div class="sk-d" style="width:28px;height:28px;border-radius:7px"></div>
      <div>
        <div class="sk-d" style="width:92px;height:17px;margin-bottom:6px"></div>
        <div class="sk-d" style="width:138px;height:9px"></div>
      </div>
    </div>
    <div class="sk-d" style="height:34px;width:100%;border-radius:8px;margin-bottom:18px"></div>
    <div class="sk-d" style="height:10px;width:58px;margin-bottom:10px"></div>
    <div class="sk-d" style="height:36px;width:100%;border-radius:8px;margin-bottom:8px"></div>
    <div class="sk-d" style="height:36px;width:92%;border-radius:8px;margin-bottom:8px"></div>
    <div class="sk-d" style="height:36px;width:96%;border-radius:8px;margin-bottom:22px"></div>
    <div class="sk-d" style="height:10px;width:82px;margin-bottom:10px"></div>
    <div class="sk-d" style="height:46px;width:100%;border-radius:9px;margin-bottom:8px"></div>
    <div class="sk-d" style="height:46px;width:100%;border-radius:9px;margin-bottom:8px"></div>
    <div class="sk-d" style="height:46px;width:86%;border-radius:9px"></div>
  </aside>
  <main class="sk-main">
    <div class="sk-topbar">
      <div>
        <div class="sk" style="height:24px;width:190px;margin-bottom:9px"></div>
        <div class="sk" style="height:12px;width:320px"></div>
      </div>
      <div style="display:flex;gap:8px">
        <div class="sk" style="width:34px;height:34px;border-radius:8px"></div>
        <div class="sk" style="width:92px;height:34px;border-radius:8px"></div>
      </div>
    </div>
    <div class="sk-grid">
      <section class="sk-panel" style="min-height:560px;display:flex;flex-direction:column">
        <div style="display:flex;justify-content:flex-end;margin-bottom:18px">
          <div class="sk" style="height:46px;width:58%;border-radius:12px"></div>
        </div>
        <div style="display:flex;gap:12px;margin-bottom:20px">
          <div class="sk" style="width:32px;height:32px;border-radius:50%"></div>
          <div style="flex:1">
            <div class="sk" style="height:14px;width:32%;margin-bottom:10px"></div>
            <div class="sk" style="height:12px;width:96%;margin-bottom:7px"></div>
            <div class="sk" style="height:12px;width:90%;margin-bottom:7px"></div>
            <div class="sk" style="height:12px;width:62%"></div>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:4px 0 22px 44px">
          <div class="sk" style="height:76px;border-radius:10px"></div>
          <div class="sk" style="height:76px;border-radius:10px"></div>
        </div>
        <div style="flex:1"></div>
        <div style="border:1px solid #e5e7eb;border-radius:12px;padding:12px">
          <div class="sk" style="height:14px;width:45%;margin-bottom:10px"></div>
          <div class="sk" style="height:38px;width:100%;border-radius:9px"></div>
        </div>
      </section>
      <aside class="sk-panel">
        <div class="sk" style="height:15px;width:110px;margin-bottom:14px"></div>
        <div class="sk" style="height:128px;width:100%;border-radius:10px;margin-bottom:14px"></div>
        <div class="sk" style="height:12px;width:85%;margin-bottom:8px"></div>
        <div class="sk" style="height:12px;width:72%;margin-bottom:22px"></div>
        <div class="sk" style="height:15px;width:96px;margin-bottom:12px"></div>
        <div class="sk" style="height:54px;width:100%;border-radius:9px;margin-bottom:8px"></div>
        <div class="sk" style="height:54px;width:100%;border-radius:9px;margin-bottom:8px"></div>
        <div class="sk" style="height:54px;width:100%;border-radius:9px"></div>
      </aside>
    </div>
  </main>
</div>"""

_sk_first_load = not st.session_state.get("_app_initialized")
_sk_placeholder = st.empty()
if _sk_first_load:
    _sk_placeholder.html(_SK_CSS + _SK_CARDS_HTML)

# ── Token refresh: re-check every 50 minutes (tokens expire after 1 hour) ────────
if is_authenticated():
    _rt = st.session_state.get("user_refresh_token")
    _last_refresh = st.session_state.get("_token_refreshed_at", 0)
    if _rt and (_time.time() - _last_refresh > 50 * 60):
        try:
            _fresh = refresh_token(_rt)
            if _fresh:
                st.session_state.user_id_token      = _fresh["id_token"]
                st.session_state.user_refresh_token = _fresh["refresh_token"]
        except Exception:
            pass
        st.session_state._token_refreshed_at = _time.time()

try:
    cfg, embedder, llm, reranker = _bootstrap()
except ConfigError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()

companies: list[str] = []
try:
    companies = _load_companies(cfg)
except Exception:
    pass

image_index: dict = {}
try:
    image_index = _load_image_index(cfg)
except Exception:
    pass

# ── Load chats from DB once per session ────────────────────────────────────────
if not st.session_state.get("_chats_loaded"):
    _user = current_user()
    if _user:
        try:
            with open_connection(cfg.db) as conn:
                _repo = PersistentChatRepository(conn)
                for _cr in _repo.list_chats(_user["uid"]):
                    if _cr.chat_id not in st.session_state.chats:
                        _msgs_raw = _repo.load_messages(_cr.chat_id, _user["uid"])
                        st.session_state.chats[_cr.chat_id] = {
                            "name":      _cr.chat_name or "Chat",
                            "messages":  [{"role": m.role, "content": m.content} for m in _msgs_raw],
                            "page":      _cr.page,
                            "persisted": True,
                        }
        except Exception:
            pass
    st.session_state._chats_loaded = True

# ── Ensure there is always an active chat for chat pages ─────────────────────────
if st.session_state.active_page in _CHAT_PAGES:
    _ap_chats = [
        cid for cid, c in st.session_state.chats.items()
        if c["page"] == st.session_state.active_page
    ]
    _cid = st.session_state.current_chat_id
    if (
        not _cid
        or _cid not in st.session_state.chats
        or st.session_state.chats.get(_cid, {}).get("page") != st.session_state.active_page
    ):
        if _ap_chats:
            st.session_state.current_chat_id = _ap_chats[-1]
        else:
            _new_chat(st.session_state.active_page)

# ── Seed rate limiter from DB (persists across page reloads) ──────────────────
if not st.session_state.get("_rl_seeded"):
    _user = current_user()
    if _user:
        try:
            with open_connection(cfg.db) as conn:
                _stats = AuditRepository(conn).get_user_stats(_user["uid"])
                seed_from_db(
                    _stats.get("queries_today", 0),
                    _stats.get("queries_this_month", 0),
                )
        except Exception:
            pass
    st.session_state._rl_seeded = True

# ── Periodic DB re-seed: closes multi-tab rate-limit bypass ──────────────────
# Session-state counters reset when a new tab opens; re-syncing from the DB
# every 5 minutes ensures all tabs share the same authoritative count.
if is_authenticated() and _time.time() - st.session_state.get("_rl_reseeded_at", 0) > 300:
    _user = current_user()
    if _user:
        try:
            with open_connection(cfg.db) as conn:
                _stats = AuditRepository(conn).get_user_stats(_user["uid"])
                seed_from_db(
                    _stats.get("queries_today", 0),
                    _stats.get("queries_this_month", 0),
                )
        except Exception:
            pass
    st.session_state._rl_reseeded_at = _time.time()

# ── Session timeout warning (token expires after 1h, we refresh at 50min) ────────
if _firebase_configured and is_authenticated():
    _age = _time.time() - st.session_state.get("_token_refreshed_at", _time.time())
    if _age > 55 * 60:
        st.warning(
            "⚠ Your session is about to expire. Save your work and refresh the page.",
            icon="⏳",
        )

# ── Email verification gate — block access until verified ─────────────────────
if _firebase_configured and is_authenticated() and not st.session_state.get("_email_verified"):
    _id_tok = st.session_state.get("user_id_token", "")
    if _id_tok:
        _verified = get_email_verified(_id_tok)
        st.session_state._email_verified = _verified
        if not _verified:
            _u_email = st.session_state.get("user_email", "your email address")
            st.html("""<style>
            [data-testid="stSidebar"] { display: none !important; }
            [data-testid="stMain"] { background: #08090e !important; }
            .main .block-container { background: transparent !important; }
            </style>""")
            _, _vc, _ = st.columns([1, 2, 1])
            with _vc:
                st.html("<div style='height:80px'></div>")
                st.html(
                    "<div style='text-align:center;padding:40px 32px;"
                    "background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);"
                    "border-radius:14px;backdrop-filter:blur(12px)'>"
                    "<div style='font-size:2.5rem;margin-bottom:16px'>📧</div>"
                    "<div style='font-size:1.2rem;font-weight:700;color:#f4f4f5;"
                    "margin-bottom:8px'>Verify your email address</div>"
                    f"<div style='font-size:0.88rem;color:#94a3b8;line-height:1.7;"
                    f"margin-bottom:24px'>We sent a verification link to "
                    f"<strong style='color:#a5b4fc'>{html.escape(_u_email)}</strong>.<br>"
                    f"Click the link in the email to unlock your account.</div>"
                    "</div>"
                )
                st.html("<div style='height:16px'></div>")
                if st.button("Resend verification email", use_container_width=True,
                             type="primary", key="gate_resend"):
                    send_email_verification(_id_tok)
                    st.toast("Verification email sent — check your inbox.")
                if st.button("I've verified — refresh", use_container_width=True,
                             key="gate_refresh"):
                    del st.session_state["_email_verified"]
                    st.rerun()
                st.html("<div style='height:12px'></div>")
                if st.button("Sign out", use_container_width=True, key="gate_signout"):
                    sign_out()
                    st.rerun()
            st.stop()
    else:
        st.session_state._email_verified = True

# ── GDPR consent logging (once per session, on first login) ──────────────────
if is_authenticated() and not st.session_state.get("_consent_logged"):
    _user = current_user()
    if _user:
        try:
            with open_connection(cfg.db) as conn:
                ConsentRepository(conn).log_acceptance(_user["uid"])
        except Exception:
            pass
    st.session_state._consent_logged = True

# ── Helpers ────────────────────────────────────────────────────────────────────

def _sanitize_error(exc: Exception) -> str:
    """Return a user-facing error message that never leaks credentials or internals."""
    msg = str(exc)
    sensitive_kw = ("password", "token", "key", "secret", "credential", "auth")
    if any(kw in msg.lower() for kw in sensitive_kw):
        return "An internal error occurred. Please try again or contact support."
    return msg[:250] if len(msg) > 250 else msg


def _log_query_async(
    cfg,
    user: dict | None,
    page: str,
    query: str,
    chunks_retrieved: int,
    model: str,
    response_length: int,
) -> None:
    """Fire-and-forget audit log — never raises or blocks the UI."""
    try:
        uid = user["uid"] if user else None
        estimated_cost_usd = estimate_query_cost_usd(
            query_text=query,
            response_text="x" * max(response_length, 0),
            chunks_retrieved=chunks_retrieved,
        )
        with open_connection(cfg.db) as conn:
            AuditRepository(conn).log_query(
                user_id=uid,
                page=page,
                query_text=query,
                chunks_retrieved=chunks_retrieved,
                model_used=model,
                response_length=response_length,
                estimated_cost_usd=estimated_cost_usd,
            )
    except Exception:
        pass  # audit failures must never crash the app


def _render_disclaimer() -> None:
    st.html(
        "<div style='margin-top:12px;padding:8px 12px;background:#fafafa;"
        "border:1px solid #f1f5f9;border-radius:6px;font-size:0.7rem;color:#94a3b8;"
        "line-height:1.5'>"
        "⚠ AI-generated response. Always verify against source documents. "
        "Does not constitute legal advice and does not create an attorney-client relationship."
        "</div>"
    )


def _render_chat_input_with_upload(placeholder: str, key: str) -> tuple[str, list]:
    """Render chat input with an upload button. Returns (question, uploaded_files)."""
    # Initialize session state for this input
    if f"{key}_files" not in st.session_state:
        st.session_state[f"{key}_files"] = []

    uploaded_files = st.session_state[f"{key}_files"]

    # Show file preview chips with remove buttons if files are uploaded
    if uploaded_files:
        cols = st.columns(len(uploaded_files) + 1)
        for i, file in enumerate(uploaded_files):
            file_name = file.name if hasattr(file, 'name') else str(file)
            with cols[i]:
                st.html(
                    f"<div style='display:inline-flex;align-items:center;gap:6px;"
                    f"background:linear-gradient(135deg,rgba(102,126,234,0.1),rgba(118,75,162,0.1));"
                    f"border:1px solid rgba(102,126,234,0.3);border-radius:20px;"
                    f"padding:6px 10px;font-size:0.78rem;color:#667EEA;'>"
                    f"📄 {html.escape(file_name[:35])}"
                    f"</div>"
                )
                if st.button("✕", key=f"rm_{key}_{i}", use_container_width=False):
                    st.session_state[f"{key}_files"].pop(i)
                    st.rerun()

    # Create columns for input and upload button
    input_col, btn_col = st.columns([8, 1])

    with input_col:
        question = st.chat_input(placeholder, key=key)

    with btn_col:
        st.html("<div style='height: 8px'></div>")  # Align with input
        # Upload button using custom HTML
        st.html(
            "<button class='chat-upload-btn' onclick='window.triggerFileUpload()' "
            "title='Upload files'>➕</button>"
        )

    # Hidden file uploader
    uploaded = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "doc", "txt", "md", "csv", "xlsx", "xls"],
        key=f"{key}_file_uploader",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Handle file additions
    if uploaded:
        existing_names = {
            f.name if hasattr(f, 'name') else str(f)
            for f in st.session_state.get(f"{key}_files", [])
        }
        for f in uploaded:
            if f.name not in existing_names:
                st.session_state[f"{key}_files"].append(f)

    return question, st.session_state.get(f"{key}_files", [])


# ── Structured legal response renderer ────────────────────────────────────────

import json as _json
import re as _re

def _parse_legal_response(raw: str) -> dict | None:
    """Extract and parse the structured JSON from a legal AI response.
    Returns None if the response is not structured JSON."""
    text = raw.strip()
    # Strip markdown code fences if present
    text = _re.sub(r"^```(?:json)?\s*", "", text)
    text = _re.sub(r"\s*```$", "", text)
    # Find the outermost JSON object
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = _json.loads(text[start:end + 1])
        if "summary" in data or "answer" in data:
            return data
    except Exception:
        pass
    return None


_SEVERITY_DOT = {
    "high":   "<span style='display:inline-block;width:9px;height:9px;border-radius:50%;"
              "background:#dc2626;flex-shrink:0;margin-top:4px'></span>",
    "medium": "<span style='display:inline-block;width:9px;height:9px;border-radius:50%;"
              "background:#d97706;flex-shrink:0;margin-top:4px'></span>",
    "low":    "<span style='display:inline-block;width:9px;height:9px;border-radius:50%;"
              "background:#16a34a;flex-shrink:0;margin-top:4px'></span>",
}

def _render_image_search_results(scored_images: list) -> None:
    """Render semantically matched images directly in the answer."""
    if not scored_images:
        return
    from pathlib import Path as _Path
    st.html(
        "<div style='font-size:0.72rem;font-weight:700;color:#64748b;"
        "letter-spacing:0.06em;margin:18px 0 10px'>📷 RELEVANT IMAGES</div>"
    )
    cols = st.columns(min(len(scored_images), 3))
    for col, img in zip(cols, scored_images):
        with col:
            img_path = _Path(img.file_path)
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
                badge_color = "#16a34a" if img.similarity >= 0.55 else "#d97706"
                desc_text = (img.ai_description[:220] + "…") if img.ai_description and len(img.ai_description) > 220 else (img.ai_description or "")
                st.html(
                    f"<div style='font-size:0.69rem;color:#64748b;line-height:1.5;"
                    f"margin-top:5px'>{html.escape(desc_text)}</div>"
                    f"<div style='font-size:0.65rem;font-weight:600;"
                    f"color:{badge_color};margin-top:3px'>"
                    f"{img.similarity:.0%} match</div>"
                )


def _render_legal_ai_message(
    raw: str,
    chunks: list | None = None,
    image_index: dict | None = None,
    scored_images: list | None = None,
    key_suffix: str = "",
) -> None:
    """Render a structured legal AI response matching the Atticus card design."""
    parsed = _parse_legal_response(raw)

    if parsed is None:
        # Fallback: plain markdown for non-JSON responses.
        st.markdown(raw)
        _render_disclaimer()
        _copy_answer_button(raw, key_suffix=key_suffix)
        _render_image_search_results(scored_images or [])
        if chunks:
            _render_sources(chunks, image_index)
        return

    summary         = parsed.get("summary", "")
    risk_flags      = parsed.get("risk_flags") or []
    answer          = parsed.get("answer", "")
    follow_ups      = parsed.get("follow_up_actions") or []

    # ── 1. Summary ────────────────────────────────────────────────────────────
    if summary:
        st.html(
            "<div style='font-size:0.93rem;font-weight:500;color:#0f172a;"
            "line-height:1.65;margin-bottom:14px'>"
            f"{html.escape(summary)}"
            "</div>"
        )

    # ── 2. Risk flags ─────────────────────────────────────────────────────────
    if risk_flags:
        st.html(
            "<div style='font-size:0.68rem;font-weight:700;color:#94a3b8;"
            "letter-spacing:0.08em;margin-bottom:8px'>RISK FLAGS</div>"
        )
        rows_html = ""
        for flag in risk_flags:
            sev         = str(flag.get("severity", "medium")).lower()
            desc        = html.escape(str(flag.get("description", "")))
            clause      = html.escape(str(flag.get("clause", "")))
            page        = flag.get("page", 0)
            dot         = _SEVERITY_DOT.get(sev, _SEVERITY_DOT["medium"])
            page_badge  = (
                f"<span style='background:#ede9fe;color:#4f46e5;font-size:0.68rem;"
                f"font-weight:600;padding:2px 7px;border-radius:99px;white-space:nowrap'>"
                f"p.{page}</span>"
                if page else ""
            )
            clause_html = (
                f"<span style='font-size:0.78rem;color:#64748b;white-space:nowrap;"
                f"margin-right:8px'>{clause}</span>"
                if clause and clause != "—" else ""
            )
            rows_html += (
                f"<div style='display:flex;align-items:flex-start;gap:10px;"
                f"padding:10px 12px;border:1px solid #e2e8f0;border-radius:8px;"
                f"margin-bottom:6px;background:#fff'>"
                f"{dot}"
                f"<span style='flex:1;font-size:0.82rem;color:#1e293b;line-height:1.55'>{desc}</span>"
                f"{clause_html}{page_badge}"
                f"</div>"
            )
        st.html(
            f"<div style='margin-bottom:16px'>{rows_html}</div>"
            f"<div style='height:1px;background:#f1f5f9;margin-bottom:16px'></div>"
        )

    # ── 3. Full answer ────────────────────────────────────────────────────────
    if answer:
        st.markdown(answer)

    # ── 4. Follow-up action buttons ───────────────────────────────────────────
    if follow_ups:
        st.html("<div style='height:4px'></div>")
        cols = st.columns(len(follow_ups))
        for i, action in enumerate(follow_ups):
            label = action.get("label", "")
            prompt = action.get("prompt", "")
            with cols[i]:
                if st.button(f"{label} ↗", key=f"followup_{key_suffix}_{i}",
                             use_container_width=True):
                    st.session_state[f"_prefill_question_{key_suffix}"] = prompt
                    st.rerun()

    # ── 5. Semantically matched images ────────────────────────────────────────
    _render_image_search_results(scored_images or [])

    # ── 6. Sources ────────────────────────────────────────────────────────────
    _render_disclaimer()
    _copy_answer_button(answer or raw, key_suffix=key_suffix)
    if chunks:
        _render_sources(chunks, image_index)


def _copy_answer_button(answer_text: str, key_suffix: str) -> None:
    """Render a small popover that exposes the answer text with a native copy button."""
    with st.popover("⎘ Copy answer", use_container_width=False):
        st.code(answer_text, language=None, wrap_lines=True)


def _render_text(text: str) -> None:
    """Render body text: gradient-fade scroll for long passages."""
    escaped = html.escape(text)
    if len(text) > 800:
        st.html(
            "<div style='position:relative'>"
            f"<div style='max-height:220px;overflow-y:auto;font-size:0.875rem;"
            f"line-height:1.65;color:inherit;white-space:pre-wrap'>{escaped}</div>"
            "<div style='position:absolute;bottom:18px;left:0;right:0;height:36px;"
            "background:linear-gradient(to bottom,transparent,rgba(255,255,255,0.95));"
            "pointer-events:none'></div>"
            "<div style='text-align:center;font-size:0.7rem;color:#94a3b8'>"
            "↕ scroll for full text</div>"
            "</div>"
        )
    else:
        st.html(
            f"<div style='font-size:0.875rem;line-height:1.65;color:inherit;"
            f"white-space:pre-wrap'>{escaped}</div>"
        )


# ── Financial-body auto-detection ─────────────────────────────────────────────
_FIN_SIGNALS = re.compile(r"\$\s*[\d,]+|\(\s*[\d,]+\s*\)")
_FIN_VALUE_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)|\(\s*([\d,]+(?:\.\d+)?)\s*\)|(—|-{1,2})")
_PERIOD_HEADER_RE = re.compile(
    r"(?:Q[1-4]\s+\d{4}|(?:December|March|June|September|October|January|April|July)"
    r"\s+\d{1,2},?\s+\d{4}|\d{4})",
    re.IGNORECASE,
)


def _try_render_financial_table_from_body(text: str) -> bool:
    """Parse financial-statement rows from raw body text and render as a table.

    Handles patterns like:
      FFO attributable to...   $311,007   $307,347   $301,769   $289,513
      Straight-line rent        (25,710)   (30,105)   (24,533)   (30,968)
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]

    # Collect data rows: label + ≥2 monetary values
    data_rows: list[list[str]] = []
    for line in lines:
        first_val = _FIN_SIGNALS.search(line)
        if not first_val:
            continue
        label = line[: first_val.start()].strip()
        if not label or len(label) < 3:
            continue
        rest = line[first_val.start():]
        vals: list[str] = []
        for m in _FIN_VALUE_RE.finditer(rest):
            if m.group(1):
                vals.append(f"${m.group(1)}")
            elif m.group(2):
                vals.append(f"({m.group(2)})")
            else:
                vals.append("—")
        if len(vals) >= 2:
            data_rows.append([label] + vals)

    if len(data_rows) < 3:
        return False

    # Normalise column count
    max_cols = max(len(r) for r in data_rows)
    data_rows = [(r + [""] * max_cols)[:max_cols] for r in data_rows]

    # Try to extract period labels from a header line near the top
    periods: list[str] = []
    for line in lines[:8]:
        if _FIN_SIGNALS.search(line):
            continue
        found = _PERIOD_HEADER_RE.findall(line)
        if found:
            periods = found
            break

    n_val = max_cols - 1
    col_headers = (periods[:n_val] if len(periods) >= n_val
                   else [f"Period {i + 1}" for i in range(n_val)])
    header = (["Metric"] + col_headers + [""] * max_cols)[:max_cols]

    # Render
    _th0 = ("border:1px solid #cbd5e1;padding:5px 10px;text-align:left;"
            "background:#f1f5f9;font-weight:600;font-size:0.75rem")
    _th  = ("border:1px solid #cbd5e1;padding:5px 10px;text-align:right;"
            "background:#f1f5f9;font-weight:600;font-size:0.75rem;white-space:nowrap")
    _td0 = "border:1px solid #e2e8f0;padding:4px 10px;text-align:left;font-size:0.78rem"
    _td  = ("border:1px solid #e2e8f0;padding:4px 10px;text-align:right;"
            "font-size:0.78rem;font-family:monospace")

    h_html = (
        f"<th style='{_th0}'>{html.escape(header[0])}</th>"
        + "".join(f"<th style='{_th}'>{html.escape(h)}</th>" for h in header[1:])
    )
    b_html = "".join(
        "<tr>"
        + f"<td style='{_td0}'>{html.escape(row[0])}</td>"
        + "".join(f"<td style='{_td}'>{html.escape(v)}</td>" for v in row[1:])
        + "</tr>"
        for row in data_rows
    )
    st.html(
        "<div style='overflow-x:auto;margin-top:4px'>"
        "<div style='font-size:0.7rem;font-weight:600;color:#64748b;margin-bottom:5px'>"
        "📋 FINANCIAL STATEMENT — auto-parsed table"
        "</div>"
        "<table style='border-collapse:collapse;width:100%'>"
        f"<thead><tr>{h_html}</tr></thead>"
        f"<tbody>{b_html}</tbody>"
        "</table></div>"
    )
    return True


def _render_body_content(text: str) -> None:
    """Render a body chunk as a structured artifact:
    title + 3-line preview + expandable full text.
    Financial reconciliation tables are auto-parsed into a real table.
    """
    # Auto-detect financial statement content (≥5 dollar/paren amounts)
    if len(_FIN_SIGNALS.findall(text)) >= 5:
        if _try_render_financial_table_from_body(text):
            return

    # Smart preview: title line + 2-3 excerpt lines + HTML <details> for the rest
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return

    title = lines[0]
    excerpt_lines = lines[1:4]
    has_more = len(lines) > 4 or sum(len(l) for l in lines[1:]) > 300

    title_html = (
        f"<div style='font-weight:600;font-size:0.875rem;color:#1e293b;"
        f"margin-bottom:5px;line-height:1.4'>{html.escape(title)}</div>"
    )
    excerpt_html = ""
    if excerpt_lines:
        excerpt = " ".join(excerpt_lines)[:300]
        if has_more:
            excerpt += "…"
        excerpt_html = (
            f"<div style='font-size:0.82rem;color:#475569;line-height:1.55;"
            f"margin-bottom:6px'>{html.escape(excerpt)}</div>"
        )

    if has_more:
        full_escaped = html.escape(text)
        expand_html = (
            "<details style='margin-top:2px'>"
            "<summary style='cursor:pointer;font-size:0.75rem;color:#4a7fcb;"
            "font-weight:500;user-select:none'>▶ Show full text</summary>"
            f"<div style='margin-top:8px;max-height:280px;overflow-y:auto;"
            f"font-size:0.8rem;line-height:1.6;white-space:pre-wrap;"
            f"color:inherit;padding-right:4px'>{full_escaped}</div>"
            "</details>"
        )
    else:
        full_escaped = html.escape(text)
        expand_html = (
            f"<div style='font-size:0.82rem;line-height:1.6;white-space:pre-wrap;"
            f"color:inherit'>{full_escaped}</div>"
        )

    st.html(title_html + excerpt_html + expand_html)


# ── Chart-line classification patterns ────────────────────────────────────────
# Require $ prefix OR unit suffix OR decimal — bare integers (page numbers
# like "14", "5") must NOT be treated as financial values.
_MONEY_RE = re.compile(
    r"(?:"
    r"[~\$]+[\d,\.]+\s*(?:bn|mn|mm|m|b|billion|million|trillion|%|×|x|k|bps)?"  # $X or $Xbn
    r"|[\d,\.]*\d\.\d[\d,\.]*\s*(?:bn|mn|mm|m|b|billion|million|trillion|%|×|x|k|bps)?"  # X.Y decimal
    r"|[\d,\.]+\s*(?:bn|mn|mm|m|b|billion|million|trillion|%|×|x|k|bps)"  # Xbn / X% (unit required)
    r")\s*$",
    re.IGNORECASE,
)
# Year-row patterns — "Q1 2024 Q2 2024" or "2021 2022 2023"
_YEAR_ROW_RE = re.compile(
    r"^(?:Q[1-4]\s+)?\d{4}(?:[\s\-–·]+(?:Q[1-4]\s+)?\d{4})+$"
)
_YEAR_SINGLE_RE = re.compile(r"^\d{4}$")
# Short lines (≤70 chars) containing an embedded dollar value
_INLINE_VALUE_RE = re.compile(
    r"^.{3,70}[~\$]\d[\d,\.]*\s*(?:bn|mn|mm|m|b|billion|million|trillion|%|bps)?.{0,20}$",
    re.IGNORECASE,
)
_FOOTNOTE_RE = re.compile(r"^\(?\d+\)?[\.\)]")


def _classify_chart_lines(
    text: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split chart text into (titles, values, year_rows, notes)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    titles, values, year_rows, notes = [], [], [], []
    seen_value = False
    for line in lines:
        if _FOOTNOTE_RE.match(line):
            continue
        if _MONEY_RE.match(line):
            values.append(line)
            seen_value = True
        elif _YEAR_ROW_RE.match(line) or _YEAR_SINGLE_RE.match(line):
            year_rows.append(line)
        elif not seen_value and _INLINE_VALUE_RE.match(line):
            # FIX 13: short line with an embedded dollar figure → treat as value
            values.append(line)
            seen_value = True
        elif not seen_value and len(titles) < 4:
            titles.append(line)
        else:
            notes.append(line)
    return titles, values, year_rows, notes


def _render_chart_text(text: str) -> None:
    """Render chart/infographic text as a structured info card."""
    titles, values, year_rows, notes = _classify_chart_lines(text)

    def e(s: str) -> str:
        return html.escape(s)

    title_html = ""
    if titles:
        title_html = (
            "<div style='font-size:0.9rem;font-weight:600;margin-bottom:8px;line-height:1.4'>"
            + "<br>".join(e(t) for t in titles)
            + "</div>"
        )

    values_html = ""
    if values:
        chips = "".join(
            f"<span style='display:inline-block;background:#dbeafe;color:#1e40af;"
            f"border-radius:4px;padding:2px 8px;margin:2px 3px;"
            f"font-weight:600;font-size:0.85rem'>{e(v)}</span>"
            for v in values
        )
        values_html = (
            "<div style='margin-bottom:6px'>"
            "<span style='font-size:0.72rem;color:#64748b;font-weight:600;"
            "letter-spacing:0.04em;display:block;margin-bottom:3px'>DATA POINTS</span>"
            + chips + "</div>"
        )

    # FIX 8 & 9: extract "Q1 2024" or plain "2024" tokens from year rows
    years_html = ""
    if year_rows:
        year_tags = ""
        for row in year_rows:
            tokens = re.findall(r"Q[1-4]\s+\d{4}|\d{4}", row)
            for tok in tokens:
                year_tags += (
                    f"<span style='display:inline-block;background:#f1f5f9;color:#475569;"
                    f"border-radius:3px;padding:1px 6px;margin:1px 2px;"
                    f"font-size:0.78rem;font-family:monospace'>{e(tok)}</span>"
                )
        years_html = (
            "<div style='margin-top:6px'>"
            "<span style='font-size:0.72rem;color:#64748b;font-weight:600;"
            "letter-spacing:0.04em;display:block;margin-bottom:3px'>PERIOD</span>"
            + year_tags + "</div>"
        )

    notes_html = ""
    if notes:
        notes_html = (
            "<div style='margin-top:8px;font-size:0.78rem;color:#64748b;"
            "line-height:1.5;white-space:pre-wrap'>"
            + "<br>".join(e(n) for n in notes)
            + "</div>"
        )

    if not title_html and not values_html and not years_html:
        body_html = (
            f"<div style='font-size:0.85rem;white-space:pre-wrap;"
            f"font-family:monospace;line-height:1.8'>{e(text)}</div>"
        )
    else:
        body_html = title_html + values_html + years_html + notes_html

    st.html(
        "<div style='background:#f0f6ff;border-left:3px solid #4a7fcb;"
        "border-radius:4px;padding:10px 14px;margin-top:4px'>"
        "<div style='font-size:0.7rem;font-weight:600;color:#4a7fcb;"
        "letter-spacing:0.05em;margin-bottom:8px'>"
        "📈 CHART / VISUAL SLIDE — data extracted from PDF; bar-to-year mapping is visual"
        "</div>"
        + body_html + "</div>"
    )


# ── Table validation ──────────────────────────────────────────────────────────

def _is_genuine_financial_table(rows: list) -> bool:
    """Return True only when structured_content looks like real financial data.

    Rejects:
    - TOC pages: cells are text strings + short page-number ints (1-3 digits)
    - Text slides: cells contain paragraph-length strings (>100 chars)
    - Mostly-empty tables: <25% of data cells contain any number
    """
    if not rows or len(rows) < 2:
        return False
    header, *data = rows
    if len(header) < 2 or not data:
        return False

    # Reject if ANY cell is a paragraph (>100 chars) — that's a text slide
    for row in rows[:6]:
        for cell in row:
            if len(str(cell).strip()) > 100:
                return False

    # Collect all non-empty data cells
    data_cells = [str(c).strip() for row in data for c in row if str(c).strip()]
    if not data_cells:
        return False

    # Reject TOC pattern: most cells are 1-3 digit integers (page numbers)
    short_ints = sum(1 for c in data_cells if c.isdigit() and len(c) <= 3)
    if short_ints / len(data_cells) > 0.40:
        return False

    # Must have meaningful numeric content ($, %, decimal, comma-separated)
    numeric = sum(1 for c in data_cells if re.search(r"[\$\%]|[\d,]{4,}|\d\.\d", c))
    if numeric / len(data_cells) < 0.20:
        return False

    return True


def _try_render_table(structured_content: str) -> bool:
    """Render structured_content as a pandas DataFrame (strict: genuine tables only)."""
    try:
        rows = json.loads(structured_content)
        if not _is_genuine_financial_table(rows):
            return False
        header, *data = rows
        safe_header = [str(h)[:120] for h in header]
        n = len(safe_header)
        normalized = [(list(row) + [""] * n)[:n] for row in data]
        df = pd.DataFrame(normalized, columns=safe_header)
        st.dataframe(df, use_container_width=True)
        return True
    except Exception:
        return False


def _render_structured_as_html_table(structured_content: str) -> bool:
    """HTML-table fallback for genuine financial tables that pandas can't handle."""
    try:
        rows = json.loads(structured_content)
        if not _is_genuine_financial_table(rows):
            return False
        header, *data = rows
        n = len(header)
        _th = ("border:1px solid #cbd5e1;padding:5px 10px;text-align:left;"
               "background:#f1f5f9;font-weight:600")
        _td = "border:1px solid #e2e8f0;padding:5px 10px;text-align:left"
        h_html = "<tr>" + "".join(
            f"<th style='{_th}'>{html.escape(str(h))}</th>" for h in header
        ) + "</tr>"
        b_html = "".join(
            "<tr>" + "".join(
                f"<td style='{_td}'>{html.escape(str(v))}</td>"
                for v in (list(row) + [""] * n)[:n]
            ) + "</tr>"
            for row in data
        )
        st.html(
            "<div style='overflow-x:auto;margin-top:4px'>"
            "<table style='border-collapse:collapse;font-size:0.82rem;width:100%'>"
            f"<thead>{h_html}</thead><tbody>{b_html}</tbody>"
            "</table></div>"
        )
        return True
    except Exception:
        return False


def _try_render_pipe_table(text: str) -> bool:
    """Render pipe-delimited text as an HTML table."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip() and "|" in ln]
    if len(lines) < 2:
        return False
    rows = [[c.strip() for c in ln.split("|")] for ln in lines]
    max_cols = max(len(r) for r in rows)
    if max_cols < 2:
        return False
    rows = [(r + [""] * max_cols)[:max_cols] for r in rows]
    header, *data = rows
    if not data:
        return False
    # Use same genuine-table validator on the reconstructed rows
    if not _is_genuine_financial_table([header] + data):
        return False
    _th = ("border:1px solid #cbd5e1;padding:5px 10px;text-align:left;"
           "background:#f1f5f9;font-weight:600")
    _td = "border:1px solid #e2e8f0;padding:5px 10px;text-align:left"
    h_html = "<tr>" + "".join(f"<th style='{_th}'>{html.escape(h)}</th>" for h in header) + "</tr>"
    b_html = "".join(
        "<tr>" + "".join(f"<td style='{_td}'>{html.escape(v)}</td>" for v in row) + "</tr>"
        for row in data
    )
    st.html(
        "<div style='overflow-x:auto;margin-top:4px'>"
        "<table style='border-collapse:collapse;font-size:0.82rem;width:100%'>"
        f"<thead>{h_html}</thead><tbody>{b_html}</tbody>"
        "</table></div>"
    )
    return True


# ── Badge + dispatch ───────────────────────────────────────────────────────────

def _resolve_content_kind(chunk: RetrievedChunk) -> str:
    """Classify what a chunk will actually render as — drives badge and dispatch.

    Returns one of: "table" | "statement" | "chart" | "text" | "reference"
    """
    if chunk.chunk_type == "financial_table" and chunk.structured_content:
        try:
            rows = json.loads(chunk.structured_content)
            if _is_genuine_financial_table(rows):
                return "table"
        except Exception:
            pass

    if chunk.chunk_type == "chart_caption":
        return "chart"

    # financial_table with no usable structured_content
    if chunk.chunk_type == "financial_table":
        if len(_FIN_SIGNALS.findall(chunk.chunk_text)) >= 4:
            return "statement"
        _, vals, years, _ = _classify_chart_lines(chunk.chunk_text)
        if vals or years:
            return "chart"

    # body — check for financial statement content
    if chunk.chunk_type == "body":
        if len(_FIN_SIGNALS.findall(chunk.chunk_text)) >= 5:
            return "statement"

    return "text"


_BADGE = {
    "table":     "📊 ",
    "statement": "📋 ",
    "chart":     "📈 ",
    "text":      "",
    "reference": "",
}

_KIND_CHIP = {
    "table":     ("<span style='font-size:0.68rem;font-weight:700;letter-spacing:0.05em;"
                  "color:#166534;background:#dcfce7;border-radius:3px;padding:1px 6px'>"
                  "FINANCIAL TABLE</span>"),
    "statement": ("<span style='font-size:0.68rem;font-weight:700;letter-spacing:0.05em;"
                  "color:#1e40af;background:#dbeafe;border-radius:3px;padding:1px 6px'>"
                  "FINANCIAL STATEMENT</span>"),
    "chart":     ("<span style='font-size:0.68rem;font-weight:700;letter-spacing:0.05em;"
                  "color:#92400e;background:#fef3c7;border-radius:3px;padding:1px 6px'>"
                  "CHART / VISUAL</span>"),
    "text":      ("<span style='font-size:0.68rem;font-weight:700;letter-spacing:0.05em;"
                  "color:#374151;background:#f3f4f6;border-radius:3px;padding:1px 6px'>"
                  "TEXT</span>"),
    "reference": ("<span style='font-size:0.68rem;font-weight:700;letter-spacing:0.05em;"
                  "color:#6b7280;background:#f9fafb;border-radius:3px;padding:1px 6px'>"
                  "REFERENCE / INDEX</span>"),
}


# ── Source card content ────────────────────────────────────────────────────────

def _render_chunk_content(chunk: RetrievedChunk, kind: str) -> None:
    """Render the content artifact for a source card."""
    # ── table ─────────────────────────────────────────────────────────────────
    if kind == "table":
        if chunk.structured_content:
            if _try_render_table(chunk.structured_content):
                return
            if _render_structured_as_html_table(chunk.structured_content):
                return
        if _try_render_pipe_table(chunk.chunk_text):
            return
        kind = "statement"

    # ── statement ─────────────────────────────────────────────────────────────
    if kind == "statement":
        if _try_render_financial_table_from_body(chunk.chunk_text):
            return
        kind = "text"

    # ── chart ─────────────────────────────────────────────────────────────────
    if kind == "chart":
        _render_chart_text(chunk.chunk_text)
        return

    # ── text / reference ──────────────────────────────────────────────────────
    _render_body_content(chunk.chunk_text)


def _format_doc_display_name(file_name: str) -> str:
    """Return a human-readable document name for source cards."""
    import urllib.parse
    name = urllib.parse.unquote(file_name)
    # EFTA-coded Epstein production documents: EFTA00005586.pdf → Case File EFTA-00005586
    m = re.match(r"EFTA(\d+)\.pdf$", name, re.IGNORECASE)
    if m:
        return f"Case File EFTA-{m.group(1)}"
    # Strip extension and clean URL-encoded / underscored names
    stem = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    stem = stem.replace("_", " ").strip()
    return stem[:72] + "…" if len(stem) > 72 else stem


def _match_label(similarity: float) -> str:
    if similarity == 0.0:
        return "keyword match"
    return f"{similarity:.0%} match"


def _render_page_images(chunk: RetrievedChunk, image_index: dict) -> None:
    """Display any images associated with a chunk's document + page number."""
    images = image_index.get((chunk.document_id, chunk.page_number), [])
    if not images:
        return
    st.html(
        "<div style='font-size:0.72rem;font-weight:600;color:#64748b;"
        "letter-spacing:0.05em;margin:10px 0 6px'>📷 PAGE IMAGES</div>"
    )
    cols = st.columns(min(len(images), 3))
    for col, img_rec in zip(cols, images):
        with col:
            img_path = Path(img_rec.file_path)
            if img_path.exists():
                st.image(
                    str(img_path),
                    use_container_width=True,
                    caption=f"Image {img_rec.image_index + 1}",
                )
                if img_rec.ai_description:
                    st.html(
                        f"<div style='font-size:0.72rem;color:#64748b;"
                        f"line-height:1.5;margin-top:4px'>"
                        f"{html.escape(img_rec.ai_description[:200])}…</div>"
                        if len(img_rec.ai_description) > 200
                        else f"<div style='font-size:0.72rem;color:#64748b;"
                        f"line-height:1.5;margin-top:4px'>"
                        f"{html.escape(img_rec.ai_description)}</div>"
                    )
            else:
                st.html(
                    f"<div style='font-size:0.72rem;color:#94a3b8'>"
                    f"Image file not found on disk.</div>"
                )


def _render_sources(chunks: list[RetrievedChunk], image_index: dict | None = None) -> None:
    if not chunks:
        return
    if image_index is None:
        image_index = {}
    st.markdown("---")
    st.markdown("#### Sources")
    for i, chunk in enumerate(chunks, start=1):
        kind      = _resolve_content_kind(chunk)
        badge     = _BADGE[kind]
        doc_name  = _format_doc_display_name(chunk.file_name)
        match_lbl = _match_label(chunk.similarity)

        has_images = bool(image_index.get((chunk.document_id, chunk.page_number)))
        img_tag = " 📷" if has_images else ""
        label = f"{badge}Source {i}  ·  {doc_name}  ·  p.{chunk.page_number}  ·  {match_lbl}{img_tag}"

        with st.expander(label):
            version = chunk.version_label or ""
            section = chunk.section_header or ""

            meta_parts = [f"📄 {html.escape(doc_name)}"]

            # Show section only when it adds information (not when it duplicates the filename)
            if section and section != chunk.file_name:
                meta_parts.append(f"Section: <em>{html.escape(section)}</em>")

            if version:
                meta_parts.append(f"Version: {html.escape(version)}")

            meta_parts.append(f"Page {chunk.page_number}")

            if chunk.similarity > 0.0:
                meta_parts.append(
                    f"<span style='color:#16a34a;font-weight:600'>"
                    f"{chunk.similarity:.0%} vector match</span>"
                )
            else:
                meta_parts.append(
                    "<span style='color:#64748b'>keyword match</span>"
                )

            st.html(
                "<div style='font-size:0.75rem;color:#64748b;line-height:1.8;"
                "border-bottom:1px solid #e2e8f0;padding-bottom:8px;margin-bottom:8px'>"
                + "  &nbsp;·&nbsp;  ".join(meta_parts)
                + "</div>"
            )

            st.html(_KIND_CHIP[kind] + "<div style='margin-bottom:8px'></div>")
            _render_chunk_content(chunk, kind)
            _render_page_images(chunk, image_index)


# ── Query helpers ──────────────────────────────────────────────────────────────

_FOLLOWUP_RE = re.compile(
    r"\b(it|its|this|that|those|these|they|them|their|he|she|his|her|"
    r"the\s+company|the\s+fund|the\s+reit|the\s+same)\b",
    re.IGNORECASE,
)


def _contextualize_query(raw: str, history: list[dict]) -> str:
    """Rewrite a follow-up question so retrieval has full context."""
    user_turns = [m for m in history if m["role"] == "user"]
    if not user_turns:
        return raw

    words = raw.split()
    is_short = len(words) <= 12
    has_reference = bool(_FOLLOWUP_RE.search(raw))

    if not (is_short or has_reference):
        return raw

    prev_question = user_turns[-1]["content"]
    return f"Context: {prev_question}\nQuestion: {raw}"


# ── Cross-company coverage guarantee ──────────────────────────────────────────
_COMPANY_ALIASES: dict[str, str] = {
    "realty income":  "Realty Income",
    "digital realty": "Digital",
    "public storage": "Psa",
    "eastgroup":      "Egp",
    "east group":     "Egp",
    "simon property": "Simon",
    "simon":          "Simon",
    "bxp":            "Bxp",
    "vici":           "Vici",
    "dlr":            "Digital",
}


def _ensure_company_coverage(
    query: str,
    chunks: list[RetrievedChunk],
    retrieval: HybridSearchService,
) -> list[RetrievedChunk]:
    """Guarantee every company mentioned in a cross-company query has ≥1 chunk."""
    if not companies:
        return chunks

    represented = {(c.company or "").lower() for c in chunks}
    query_lower = query.lower()

    company_lookup = {c.lower(): c for c in companies}

    missing: list[str] = []
    seen_missing: set[str] = set()

    for stored_lower, stored_name in company_lookup.items():
        if stored_lower in query_lower and stored_lower not in represented:
            if stored_lower not in seen_missing:
                missing.append(stored_name)
                seen_missing.add(stored_lower)

    for alias, canonical in _COMPANY_ALIASES.items():
        if alias in query_lower:
            actual = company_lookup.get(canonical.lower(), canonical)
            if actual.lower() not in represented and actual.lower() not in seen_missing:
                missing.append(actual)
                seen_missing.add(actual.lower())

    if not missing:
        return chunks

    augmented = list(chunks)
    seen_ids = {c.chunk_id for c in chunks}
    for company_name in missing:
        try:
            secondary = retrieval.retrieve(
                RetrievalRequest(query=query, top_k=2, company_filter=company_name)
            )
            for c in secondary:
                if c.chunk_id not in seen_ids:
                    augmented.append(c)
                    seen_ids.add(c.chunk_id)
        except Exception:
            pass

    return augmented


# ── Query-param navigation (bottom icon bar links fire ?nav=page) ───────────────
_nav_qp = st.query_params.get("nav")
if _nav_qp in _VALID_PAGES:
    if st.session_state.active_page != _nav_qp:
        if _nav_qp in _CHAT_PAGES:
            _switch_page(_nav_qp)
        else:
            st.session_state.active_page = _nav_qp
    st.query_params.clear()
    st.rerun()

# ── Global CSS — Premium Light Theme ─────────────────────────────────────────────
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

/* ── Base typography ──────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Light background canvas ─────────────────────────────────────────────── */
[data-testid="stMain"],
[data-testid="stAppViewContainer"] > section.main {
    background: #F5F4F0 !important;
}

/* ── Main content card ────────────────────────────────────────────────────── */
.main .block-container {
    background: transparent !important;
    max-width: 100% !important;
    padding: 0 !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    margin: 0 !important;
}

/* ── Chat container ─────────────────────────────────────────────────────── */
.stChatMessage {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    gap: 16px !important;
}

/* ── Chat message content ─────────────────────────────────────────────────── */
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
    color: #2D2D3A !important;
}

/* ── User bubble — elegant warm tone ─────────────────────────────────────── */
[data-testid="stChatMessage"][data-testid*="user"] [data-testid="stChatMessageContent"],
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%) !important;
    border-radius: 20px 20px 4px 20px !important;
    padding: 14px 18px !important;
    border: none !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(118, 75, 162, 0.25) !important;
}

/* ── Assistant bubble — premium card ─────────────────────────────────────── */
.stChatMessage:not(:has([data-testid="chatAvatarIcon-user"])) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:not([data-testid*="user"]) [data-testid="stChatMessageContent"] {
    background: #FFFFFF !important;
    border-radius: 20px 4px 20px 20px !important;
    padding: 16px 20px !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.03) !important;
}

/* ── Expanders (source cards) ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: 16px !important;
    background: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03) !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.82rem !important;
    color: #4A4A5A !important;
    font-weight: 500 !important;
    padding: 12px 16px !important;
}
[data-testid="stExpander"]:has([data-testid="stExpanderToolbar"]) {
    border-color: #667EEA !important;
}
[data-testid="stExpander"] [data-testid="stExpanderToolbar"] {
    color: #667EEA !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid rgba(0,0,0,0.06);
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #6B6B7A !important;
    padding: 10px 20px !important;
    border-radius: 10px 10px 0 0 !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #667EEA !important;
    border-bottom: 2px solid #667EEA !important;
    background: rgba(102, 126, 234, 0.06) !important;
}

/* ── Chat input — premium floating box ────────────────────────────────────── */
[data-testid="stChatInput"] > div {
    border: none !important;
    border-radius: 24px !important;
    background: #FFFFFF !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08), 0 2px 8px rgba(0,0,0,0.04) !important;
    padding: 4px !important;
    position: relative !important;
}
[data-testid="stChatInput"] textarea {
    font-size: 0.9rem !important;
    color: #2D2D3A !important;
    border-radius: 20px !important;
}
[data-testid="stBottom"] {
    background: transparent !important;
    padding: 0 20px 20px !important;
}

/* ── Upload button in chat input area ─────────────────────────────────────── */
.chat-upload-btn {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    min-height: 36px !important;
    max-width: 36px !important;
    max-height: 36px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%) !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(118, 75, 162, 0.3) !important;
    margin-left: 8px !important;
}
.chat-upload-btn:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 4px 12px rgba(118, 75, 162, 0.4) !important;
}

/* ── Drag-and-drop overlay ─────────────────────────────────────────────────── */
.drag-drop-overlay {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    background: rgba(102, 126, 234, 0.15) !important;
    backdrop-filter: blur(4px) !important;
    z-index: 9999 !important;
    display: none !important;
    align-items: center !important;
    justify-content: center !important;
    pointer-events: none !important;
}
.drag-drop-overlay.active {
    display: flex !important;
    pointer-events: all !important;
}
.drag-drop-content {
    background: #ffffff !important;
    border: 3px dashed #667EEA !important;
    border-radius: 16px !important;
    padding: 40px 60px !important;
    text-align: center !important;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25) !important;
}
.drag-drop-content .icon {
    font-size: 3rem !important;
    margin-bottom: 16px !important;
}
.drag-drop-content .text {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #667EEA !important;
    margin-bottom: 8px !important;
}
.drag-drop-content .subtext {
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
}

/* ── File preview chips ─────────────────────────────────────────────────────── */
.file-preview-chip {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    border-radius: 20px !important;
    padding: 6px 12px !important;
    font-size: 0.78rem !important;
    color: #667EEA !important;
    margin-right: 8px !important;
    margin-bottom: 8px !important;
}
.file-preview-chip .remove-btn {
    background: rgba(102, 126, 234, 0.2) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 18px !important;
    height: 18px !important;
    min-width: 18px !important;
    min-height: 18px !important;
    max-width: 18px !important;
    max-height: 18px !important;
    font-size: 0.7rem !important;
    color: #667EEA !important;
    cursor: pointer !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    transition: all 0.2s ease !important;
    background: #FFFFFF !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 4px 12px rgba(118, 75, 162, 0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #5B71E8 0%, #6A4199 100%) !important;
    box-shadow: 0 6px 16px rgba(118, 75, 162, 0.4) !important;
    transform: translateY(-1px);
}

/* ── Spinner ──────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div { color: #667EEA !important; }

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.25); }

/* ── Sidebar bottom padding ───────────────────────────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-bottom: 20px !important;
}

/* ── Mobile responsive ───────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .main .block-container {
        padding: 0 !important;
    }
}
</style>
""")

# ── Drag-and-drop JavaScript ──────────────────────────────────────────────────
st.html("""
<div id="dragDropOverlay" class="drag-drop-overlay">
    <div class="drag-drop-content">
        <div class="icon">📄</div>
        <div class="text">Drop files to upload</div>
        <div class="subtext">PDF, DOCX, TXT, and more</div>
    </div>
</div>

<script>
(function() {
    const overlay = document.getElementById('dragDropOverlay');
    let dragCounter = 0;
    let isDraggingFile = false;

    // Detect file drag vs text drag
    document.addEventListener('dragenter', function(e) {
        if (e.dataTransfer && e.dataTransfer.types &&
            (e.dataTransfer.types.includes('Files') || e.dataTransfer.types.includes('application/x-moz-file'))) {
            dragCounter++;
            isDraggingFile = true;
            overlay.classList.add('active');
            e.preventDefault();
        }
    });

    document.addEventListener('dragleave', function(e) {
        dragCounter--;
        if (dragCounter === 0) {
            overlay.classList.remove('active');
            isDraggingFile = false;
        }
    });

    document.addEventListener('dragover', function(e) {
        if (e.dataTransfer && e.dataTransfer.types &&
            (e.dataTransfer.types.includes('Files') || e.dataTransfer.types.includes('application/x-moz-file'))) {
            e.preventDefault();
            e.stopPropagation();
        }
    });

    document.addEventListener('drop', function(e) {
        dragCounter = 0;
        overlay.classList.remove('active');
        isDraggingFile = false;

        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            e.preventDefault();
            e.stopPropagation();

            const files = e.dataTransfer.files;
            const input = document.querySelector('input[type="file"]');

            if (input) {
                const dataTransfer = new DataTransfer();
                for (let i = 0; i < files.length; i++) {
                    dataTransfer.items.add(files[i]);
                }
                input.files = dataTransfer.files;

                // Trigger Streamlit file uploader
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
    });

    // Upload button click handler
    window.triggerFileUpload = function() {
        const input = document.querySelector('input[type="file"]');
        if (input) {
            input.click();
        }
    };
})();
</script>
""")

# ── Sidebar ────────────────────────────────────────────────────────────────────
company_choice, top_k = render_sidebar(
    companies=companies,
    new_chat_fn=_new_chat,
    switch_page_fn=_switch_page,
    rename_chat_fn=_db_rename_chat,
    delete_chat_fn=_db_delete_chat,
)

if _demo_mode and st.session_state.active_page != "epstein":
    st.session_state.active_page = "epstein"
    _switch_page("epstein")
    st.toast("Public demo is limited to the read-only case workspace.")
    st.rerun()


# ── Page functions ─────────────────────────────────────────────────────────────

def _render_insightlens_page() -> None:
    chat = _get_current_chat()
    if chat is None:
        return
    messages: list[dict] = chat["messages"]

    # Empty state welcome
    if not messages:
        st.markdown("<div style='height:20vh'></div>", unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.html(
                "<div style='text-align:center;padding:48px 32px;background:linear-gradient(135deg,rgba(102,126,234,0.06),rgba(118,75,162,0.04));"
                "border:1px solid rgba(102,126,234,0.15);border-radius:20px;box-shadow:0 4px 20px rgba(102,126,234,0.08)'>"
                "<div style='font-size:1.8rem;font-weight:700;color:#2D2D3A;"
                "letter-spacing:-0.5px;margin-bottom:8px'>Atticus</div>"
                "<div style='font-size:0.95rem;color:#667EEA;font-weight:500;margin-bottom:24px'>"
                "Legal Research Intelligence</div>"
                "<div style='font-size:0.88rem;color:#6B6B7A;line-height:1.8'>"
                "Ask about contracts, filings, transcripts,<br>"
                "or public-record collections across your uploaded documents."
                "</div>"
                "</div>"
            )

    # Conversation history
    for i, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_legal_ai_message(
                    msg["content"], msg.get("chunks"), image_index,
                    msg.get("scored_images"), key_suffix=f"il_{i}",
                )
            else:
                st.markdown(msg["content"])

    # Chat input with upload
    question, uploaded_files = _render_chat_input_with_upload(
        "Ask about your legal documents…", key="insightlens_input"
    )

    if question:
        # Input validation + rate limiting
        try:
            question = validate_query(question)
        except InputGuardError as exc:
            st.error(str(exc))
            st.stop()

        allowed, reason = check_rate_limit()
        if not allowed:
            st.warning(f"⏳ {reason}")
            st.stop()

        # Update chat name on first message
        _cid = st.session_state.current_chat_id
        if not messages:
            chat["name"] = question[:30].strip()
            if chat.get("persisted"):
                try:
                    with open_connection(cfg.db) as conn:
                        PersistentChatRepository(conn).update_chat_name(_cid, chat["name"])
                except Exception:
                    pass

        company_filter = None if company_choice == "All companies" else company_choice

        messages.append({"role": "user", "content": question})
        _save_message(_cid, "user", question)
        with st.chat_message("user"):
            st.markdown(question)

        retrieval_query = _contextualize_query(question, messages[:-1])

        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                corpus = _load_corpus(cfg)
                with open_connection(cfg.db) as conn:
                    repo = ChunkRepository(conn)
                    retrieval = HybridSearchService(
                        embedder=embedder,
                        repository=repo,
                        corpus_chunks=corpus,
                        reranker=reranker,
                    )
                    chunks = retrieval.retrieve(
                        RetrievalRequest(
                            query=retrieval_query,
                            top_k=top_k,
                            company_filter=company_filter,
                        )
                    )
                    if not company_filter:
                        chunks = _ensure_company_coverage(retrieval_query, chunks, retrieval)
                    query_vec = embedder.embed_query(retrieval_query)
                    from insightlens.storage.image_repository import ImageRepository as _IR
                    scored_images = _IR(conn).search_by_description(
                        query_vec, top_k=3, min_similarity=0.4,
                        company_filter=company_filter,
                    )

            user_prompt = build_user_prompt(question, chunks)
            with st.spinner("Analyzing documents…"):
                answer_text = "".join(llm.stream(SYSTEM_PROMPT, user_prompt))
            _render_legal_ai_message(answer_text, chunks, image_index, scored_images, key_suffix="il_new")

        messages.append({"role": "assistant", "content": answer_text, "chunks": chunks, "scored_images": scored_images})
        _save_message(_cid, "assistant", answer_text)

        _log_query_async(
            cfg, current_user(), "insightlens", question,
            len(chunks), cfg.generation_model, len(answer_text),
        )


# (date, title, body, category)
_TIMELINE_EVENTS: list[tuple[str, str, str, str]] = [
    ("1991–1994", "Early allegations surface",
     "Multiple women later testify that Epstein began abuse in the early 1990s at his Palm Beach estate and New York townhouse.",
     "Investigation"),
    ("1997", "New York townhouse acquired",
     "Epstein takes ownership of 9 East 71st Street — the largest private residence in Manhattan, valued at ~$77 million.",
     "Key Event"),
    ("2002", "Access to high-profile circles peaks",
     "Epstein is regularly seen with prominent financiers, politicians, and royalty. Virginia Giuffre (then Roberts) enters his orbit at Mar-a-Lago.",
     "Key Event"),
    ("2005", "Palm Beach Police investigation begins",
     "A parent reports to Palm Beach PD that their 14-year-old daughter was abused at Epstein's estate. Detective Joseph Recarey builds a case with 17 identified victims.",
     "Investigation"),
    ("2007–2008", "Controversial federal plea deal",
     "U.S. Attorney Alexander Acosta negotiates a non-prosecution agreement. Epstein pleads guilty to two Florida state charges, serves 13 months with work release. Victims not notified — potential Crime Victims' Rights Act violation.",
     "Ruling"),
    ("2009–2018", "Civil litigation and partial disclosures",
     "Multiple civil suits filed. Court-sealed documents from Giuffre v. Maxwell begin to surface. Miami Herald publishes 'Perversion of Justice' (Nov 2018), a landmark investigative series.",
     "Filing"),
    ("2019 Jul 6", "Arrested at Teterboro Airport",
     "Returning from France, Epstein is arrested by FBI/NYPD on federal sex-trafficking charges. SDNY unseals an indictment citing dozens of victims.",
     "Arrest"),
    ("2019 Jul 8", "Federal indictment unsealed — SDNY",
     "Two counts: sex trafficking of minors and conspiracy. Prosecutors allege a decade-long scheme across Manhattan and Palm Beach properties.",
     "Filing"),
    ("2019 Jul 23", "Found dead at MCC New York",
     "Epstein found unresponsive in his cell. Taken off suicide watch days earlier despite a prior incident. Cell inadequately monitored. Two guards later indicted for falsifying records.",
     "Key Event"),
    ("2019 Aug", "Medical examiner rules suicide",
     "NYC ME Dr. Barbara Sampson concludes death by hanging. Independent pathologist Dr. Michael Baden disputes findings, citing injuries more consistent with homicide by strangulation.",
     "Ruling"),
    ("2021 Jul 2", "Ghislaine Maxwell arrested",
     "Maxwell is indicted on six federal counts including sex trafficking of minors. Trial begins December 2021 after Maxwell's bail denied.",
     "Arrest"),
    ("2021 Dec 29", "Maxwell convicted on 5 of 6 counts",
     "Jury convicts Maxwell after six weeks of testimony. Four women describe abuse as minors. Maxwell maintains she was a scapegoat for a dead man.",
     "Ruling"),
    ("2022 Jun 28", "Maxwell sentenced to 20 years",
     "Judge Alison Nathan sentences Maxwell to 20 years federal prison. Maxwell's attorneys appeal on juror disclosure grounds.",
     "Ruling"),
    ("2023", "Bank settlements — institutional accountability",
     "JPMorgan settles with USVI for $75M and Giuffre for $290M over facilitation claims. Deutsche Bank settles with USVI for $75M. First major corporate accountability in the case.",
     "Settlement"),
    ("2024", "Sealed documents released — names emerge",
     "Federal courts release thousands of pages from Giuffre v. Maxwell civil case under Judge Preska's orders. Additional individuals named. Civil suits against estate continue.",
     "Filing"),
]

_KEY_FIGURES = [
    ("Jeffrey Epstein", "Financier & convicted sex offender. Born Jan 20, 1953. Died Aug 10, 2019, MCC New York.", "Defendant"),
    ("Ghislaine Maxwell", "British socialite, Epstein's associate and alleged procurer. Convicted 2021, sentenced 20 years.", "Convicted"),
    ("Virginia Giuffre", "Primary survivor-witness. Filed multiple civil suits; reached $12M settlement with Prince Andrew (2022).", "Survivor"),
    ("Alexander Acosta", "U.S. Attorney (SD Florida) who negotiated the 2008 non-prosecution agreement. Resigned as Labor Secretary July 2019.", "Prosecutor"),
    ("Prince Andrew", "Duke of York. Named in Giuffre's civil suit. Settled for $12M, withdrew from public duties.", "Named Party"),
    ("Alan Dershowitz", "Harvard Law professor, Epstein's attorney. Named by Giuffre in unsealed documents.", "Named Party"),
    ("Julie K. Brown", "Miami Herald investigative reporter. 'Perversion of Justice' series (2018) reignited federal scrutiny.", "Journalist"),
    ("Judge Loretta Preska", "SDNY. Ordered unsealing of Giuffre v. Maxwell documents in 2024.", "Judiciary"),
    ("Judge Alison Nathan", "Presided over Maxwell trial and sentencing. Now serves on Second Circuit.", "Judiciary"),
    ("Det. Joseph Recarey", "Palm Beach PD detective who built the original 2005 case with 17 identified victims.", "Law Enforcement"),
]

_KEY_LOCATIONS = [
    ("Palm Beach, FL", "Primary site of 2005 police investigation. Epstein's estate at 358 El Brillo Way."),
    ("9 E 71st St, Manhattan", "Largest private home in NYC. Site of alleged abuse and trafficking operations."),
    ("Little Saint James, USVI", "Private island leased from USVI. Center of USVI $190M lawsuit. 'Temple' structure under scrutiny."),
    ("New Mexico Ranch", "Zorro Ranch — 7,500-acre property. Additional allegations from multiple witnesses."),
    ("Paris Apartment", "Rue du Faubourg Saint-Honoré. Epstein departed Paris days before his 2019 arrest."),
    ("MCC New York", "Metropolitan Correctional Center. Epstein died here Aug 10, 2019. Severe lapses in monitoring documented."),
]

_KEY_INSIGHTS = [
    ("$440M+", "Total institutional settlements (JPMorgan $365M, Deutsche Bank $75M)", "#4f46e5"),
    ("50+", "Identified victims across investigations in US and Europe", "#dc2626"),
    ("13 months", "Time served under 2008 plea deal — widely criticized as inadequate", "#d97706"),
    ("6 counts", "Maxwell convicted on 5 of 6 federal charges at trial", "#16a34a"),
    ("~$1B", "Estimated Epstein estate value at time of death", "#64748b"),
    ("2,000+", "Pages released from Giuffre v. Maxwell in 2024 unsealing", "#4f46e5"),
]

_CATEGORY_COLORS = {
    "Investigation": "#7c3aed",
    "Key Event":     "#1e293b",
    "Ruling":        "#16a34a",
    "Filing":        "#2563eb",
    "Arrest":        "#dc2626",
    "Settlement":    "#d97706",
}


def _render_epstein_timeline() -> None:
    # ── Filter bar ────────────────────────────────────────────────────────────
    all_cats = ["All events", "Arrest", "Filing", "Ruling", "Investigation", "Settlement", "Key Event"]

    filter_col, search_col = st.columns([3, 2])
    with filter_col:
        selected_cat = st.pills(
            "Filter by category",
            options=all_cats,
            default="All events",
            key="timeline_filter_cat",
        )
    with search_col:
        search_q = st.text_input(
            "Search events",
            placeholder="Search events…",
            key="timeline_search",
        )

    # ── Filter logic ──────────────────────────────────────────────────────────
    filtered = []
    for ev in _TIMELINE_EVENTS:
        date, title, body, cat = ev
        if selected_cat and selected_cat != "All events" and cat != selected_cat:
            continue
        if search_q and search_q.lower() not in (date + title + body + cat).lower():
            continue
        filtered.append(ev)

    active_filter = selected_cat if (selected_cat and selected_cat != "All events") else None
    filter_text = f" · filtered by <strong>{html.escape(active_filter)}</strong>" if active_filter else ""
    st.html(
        f"<div style='font-size:0.75rem;color:#94a3b8;margin:4px 0 20px'>"
        f"Showing <strong>{len(filtered)}</strong> of {len(_TIMELINE_EVENTS)} events{filter_text}</div>"
    )

    if not filtered:
        st.html(
            "<div style='text-align:center;padding:40px 0;color:#94a3b8;"
            "font-size:0.9rem'>No events match your filter.</div>"
        )
        return

    # ── Zigzag timeline ───────────────────────────────────────────────────────
    def _left_event(date: str, title: str, body: str, cat: str) -> str:
        color = _CATEGORY_COLORS.get(cat, "#4f46e5")
        cat_chip = (
            f"<span style='font-size:0.62rem;font-weight:700;letter-spacing:0.04em;"
            f"background:{color}18;color:{color};border-radius:3px;"
            f"padding:1px 5px;margin-left:6px'>{html.escape(cat)}</span>"
        )
        return (
            "<div style='display:flex;align-items:flex-start;margin-bottom:28px'>"
            # left content
            "<div style='flex:1;text-align:right;padding-right:20px'>"
            f"<div style='font-size:0.68rem;font-weight:700;color:{color};"
            f"letter-spacing:0.04em;margin-bottom:2px'>{html.escape(date)}{cat_chip}</div>"
            f"<div style='font-size:0.88rem;font-weight:600;color:#1e293b;margin-bottom:4px'>{html.escape(title)}</div>"
            f"<div style='font-size:0.78rem;color:#475569;line-height:1.6'>{html.escape(body)}</div>"
            "</div>"
            # center dot
            "<div style='position:relative;width:20px;flex-shrink:0;display:flex;"
            "justify-content:center;padding-top:4px'>"
            f"<div style='width:13px;height:13px;border-radius:50%;background:{color};"
            "border:3px solid #ffffff;box-shadow:0 0 0 2px #e2e8f0;position:relative;z-index:1'></div>"
            "</div>"
            # right empty
            "<div style='flex:1;padding-left:20px'></div>"
            "</div>"
        )

    def _right_event(date: str, title: str, body: str, cat: str) -> str:
        color = _CATEGORY_COLORS.get(cat, "#4f46e5")
        cat_chip = (
            f"<span style='font-size:0.62rem;font-weight:700;letter-spacing:0.04em;"
            f"background:{color}18;color:{color};border-radius:3px;"
            f"padding:1px 5px;margin-right:6px'>{html.escape(cat)}</span>"
        )
        return (
            "<div style='display:flex;align-items:flex-start;margin-bottom:28px'>"
            # left empty
            "<div style='flex:1;padding-right:20px'></div>"
            # center dot
            "<div style='position:relative;width:20px;flex-shrink:0;display:flex;"
            "justify-content:center;padding-top:4px'>"
            f"<div style='width:13px;height:13px;border-radius:50%;background:{color};"
            "border:3px solid #ffffff;box-shadow:0 0 0 2px #e2e8f0;position:relative;z-index:1'></div>"
            "</div>"
            # right content
            "<div style='flex:1;text-align:left;padding-left:20px'>"
            f"<div style='font-size:0.68rem;font-weight:700;color:{color};"
            f"letter-spacing:0.04em;margin-bottom:2px'>{cat_chip}{html.escape(date)}</div>"
            f"<div style='font-size:0.88rem;font-weight:600;color:#1e293b;margin-bottom:4px'>{html.escape(title)}</div>"
            f"<div style='font-size:0.78rem;color:#475569;line-height:1.6'>{html.escape(body)}</div>"
            "</div>"
            "</div>"
        )

    total_h = 28 * len(filtered) + 200  # rough estimate for center line height
    zigzag = (
        f"<div style='position:relative;padding:0 4px'>"
        f"<div style='position:absolute;left:50%;top:0;bottom:0;width:2px;"
        f"background:#e2e8f0;transform:translateX(-50%);z-index:0'></div>"
    )
    for i, (date, title, body, cat) in enumerate(filtered):
        if i % 2 == 0:
            zigzag += _left_event(date, title, body, cat)
        else:
            zigzag += _right_event(date, title, body, cat)
    zigzag += "</div>"
    st.html(zigzag)


def _render_epstein_overall() -> None:
    st.html(
        "<div style='font-size:0.78rem;color:#64748b;margin-bottom:16px;line-height:1.6'>"
        "Case overview, key figures, financial impact, and locations. "
        "All information drawn from public court records and investigative reporting."
        "</div>"
    )

    # ── Key metrics ──────────────────────────────────────────────────────────
    st.html(
        "<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
        "letter-spacing:0.06em;margin-bottom:8px'>KEY METRICS</div>"
    )
    metrics_html = "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:20px'>"
    for value, label, color in _KEY_INSIGHTS:
        metrics_html += (
            f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
            f"padding:12px;text-align:center'>"
            f"<div style='font-size:1.3rem;font-weight:700;color:{color}'>{html.escape(value)}</div>"
            f"<div style='font-size:0.68rem;color:#64748b;margin-top:3px;line-height:1.4'>{html.escape(label)}</div>"
            f"</div>"
        )
    metrics_html += "</div>"
    st.html(metrics_html)

    # ── Case status grid ──────────────────────────────────────────────────────
    st.html(
        "<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
        "letter-spacing:0.06em;margin-bottom:8px'>PROCEEDINGS STATUS</div>"
    )
    statuses = [
        ("Epstein criminal case", "Closed — died in custody Aug 2019", "#64748b"),
        ("Ghislaine Maxwell", "Convicted · Serving 20-year sentence", "#16a34a"),
        ("SDNY federal investigation", "Closed, no additional indictments", "#64748b"),
        ("USVI civil suit", "Settled — JPMorgan $75M + Deutsche Bank $75M", "#16a34a"),
        ("Giuffre v. Maxwell civil", "Settled (sealed amount)", "#16a34a"),
        ("Giuffre v. Prince Andrew", "Settled — $12M (Feb 2022)", "#16a34a"),
        ("Estate civil suits (survivors)", "Ongoing — multiple plaintiffs", "#d97706"),
        ("Document unsealing (Preska)", "In progress — 2024 releases ongoing", "#d97706"),
        ("MCC guards criminal case", "Pled guilty to falsifying prison records", "#16a34a"),
    ]
    status_html = "<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:20px'>"
    for matter, status, color in statuses:
        status_html += (
            f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
            f"padding:9px 12px'>"
            f"<div style='font-size:0.75rem;font-weight:600;color:#1e293b;margin-bottom:2px'>{html.escape(matter)}</div>"
            f"<div style='font-size:0.71rem;color:{color};font-weight:500'>{html.escape(status)}</div>"
            f"</div>"
        )
    status_html += "</div>"
    st.html(status_html)

    # ── Key figures ──────────────────────────────────────────────────────────
    st.html(
        "<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
        "letter-spacing:0.06em;margin-bottom:8px'>KEY FIGURES</div>"
    )
    role_colors = {
        "Defendant": "#dc2626", "Convicted": "#d97706", "Survivor": "#7c3aed",
        "Prosecutor": "#2563eb", "Named Party": "#64748b",
        "Journalist": "#16a34a", "Judiciary": "#4f46e5", "Law Enforcement": "#0891b2",
    }
    for name, desc, role in _KEY_FIGURES:
        rc = role_colors.get(role, "#64748b")
        st.html(
            f"<div style='display:flex;align-items:flex-start;gap:10px;border-left:3px solid {rc};"
            f"padding:6px 10px;margin-bottom:5px;background:#f8fafc;border-radius:0 6px 6px 0'>"
            f"<div style='flex:1'>"
            f"<div style='display:flex;align-items:center;gap:8px'>"
            f"<span style='font-size:0.82rem;font-weight:600;color:#1e293b'>{html.escape(name)}</span>"
            f"<span style='font-size:0.62rem;font-weight:700;color:{rc};background:{rc}18;"
            f"border-radius:3px;padding:1px 5px'>{html.escape(role)}</span>"
            f"</div>"
            f"<div style='font-size:0.75rem;color:#475569;margin-top:2px'>{html.escape(desc)}</div>"
            f"</div></div>"
        )

    st.html("<div style='height:12px'></div>")

    # ── Key locations ────────────────────────────────────────────────────────
    st.html(
        "<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
        "letter-spacing:0.06em;margin-bottom:8px'>KEY LOCATIONS</div>"
    )
    loc_html = "<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px'>"
    for loc, desc in _KEY_LOCATIONS:
        loc_html += (
            f"<div style='border-left:3px solid #64748b;padding:7px 10px;"
            f"background:#f8fafc;border-radius:0 6px 6px 0'>"
            f"<div style='font-size:0.8rem;font-weight:600;color:#1e293b'>📍 {html.escape(loc)}</div>"
            f"<div style='font-size:0.73rem;color:#475569;margin-top:2px;line-height:1.5'>{html.escape(desc)}</div>"
            f"</div>"
        )
    loc_html += "</div>"
    st.html(loc_html)


def _render_epstein_post() -> None:
    import datetime as _dt
    from insightlens.storage.discussion_repository import DiscussionRepository

    if "lawyer_posts" not in st.session_state:
        st.session_state.lawyer_posts = []

    # ── Load: DB first, session-state fallback ────────────────────────────────
    db_available = False
    posts: list[dict] = []
    try:
        with open_connection(cfg.db) as conn:
            posts = DiscussionRepository(conn).list_posts()
            db_available = True
    except Exception:
        posts = st.session_state.lawyer_posts

    # ── Header ────────────────────────────────────────────────────────────────
    count_chip = (
        f"<span style='font-size:0.65rem;font-weight:700;background:#e0e7ff;"
        f"color:#4338ca;border-radius:99px;padding:1px 8px;margin-left:8px'>{len(posts)}</span>"
        if posts else ""
    )
    persistence_badge = (
        "<span style='color:#16a34a;font-weight:500'>● Live — posts persist across sessions.</span>"
        if db_available else
        "<span style='color:#d97706;font-weight:500'>● Session-only — database unavailable.</span>"
    )
    st.html(
        "<div style='padding:8px 0 4px'>"
        f"<span style='font-size:1rem;font-weight:600;color:#1e293b'>"
        f"Legal Discussion Board{count_chip}</span>"
        f"<div style='font-size:0.76rem;color:#94a3b8;margin-top:3px'>"
        f"A shared space for legal professionals to post findings, questions, and analysis "
        f"regarding the Epstein case. {persistence_badge}"
        "</div></div>"
    )
    st.divider()

    # ── Post display ──────────────────────────────────────────────────────────
    if not posts:
        st.html(
            "<div style='text-align:center;padding:48px 0;color:#94a3b8;"
            "font-size:0.9rem'>No posts yet. Be the first to add a finding or question below.</div>"
        )
    else:
        for post in posts:
            role_color = "#4f46e5" if post.get("type") == "Finding" else (
                "#d97706" if post.get("type") == "Question" else "#16a34a"
            )
            card_col, del_col = st.columns([10, 1])
            with card_col:
                st.html(
                    f"<div style='border:1px solid #e2e8f0;border-radius:10px;"
                    f"padding:12px 16px;margin-bottom:4px;background:#ffffff'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;"
                    f"margin-bottom:6px'>"
                    f"<div style='display:flex;align-items:center;gap:8px'>"
                    f"<span style='font-size:0.82rem;font-weight:600;color:#1e293b'>"
                    f"{html.escape(post['author'])}</span>"
                    f"<span style='font-size:0.65rem;font-weight:700;color:{role_color};"
                    f"background:{role_color}18;border-radius:3px;padding:1px 6px'>"
                    f"{html.escape(post.get('type', 'Note'))}</span>"
                    f"</div>"
                    f"<span style='font-size:0.68rem;color:#94a3b8'>{html.escape(post['time'])}</span>"
                    f"</div>"
                    f"<div style='font-size:0.83rem;color:#374151;line-height:1.65;white-space:pre-wrap'>"
                    f"{html.escape(post['content'])}</div>"
                    f"</div>"
                )
            with del_col:
                pid = post.get("post_id", post.get("time", ""))
                if db_available and pid:
                    if st.button("✕", key=f"del_post_{pid}",
                                 use_container_width=True):
                        try:
                            with open_connection(cfg.db) as conn:
                                DiscussionRepository(conn).delete_post(pid)
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

    # ── Compose ───────────────────────────────────────────────────────────────
    st.divider()
    st.html(
        "<div style='font-size:0.72rem;font-weight:600;color:#94a3b8;"
        "letter-spacing:0.05em;margin-bottom:6px'>NEW POST</div>"
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        author = st.text_input("Your name / firm", placeholder="e.g. Jane Smith, Smith & Assoc.",
                               key="post_author", label_visibility="collapsed")
    with c2:
        post_type = st.selectbox("Type", ["Finding", "Question", "Analysis"],
                                 key="post_type", label_visibility="collapsed")

    content = st.text_area("Post content", placeholder="Share a finding, legal question, or analysis…",
                           height=100, key="post_content", label_visibility="collapsed")

    pc1, pc2 = st.columns([1, 4])
    with pc1:
        if st.button("Post", type="primary", use_container_width=True, key="submit_post"):
            if author.strip() and content.strip():
                try:
                    author  = validate_text_input(author,  "Name",    max_length=120)
                    content = validate_text_input(content, "Content", max_length=8000)
                except InputGuardError as exc:
                    st.error(str(exc))
                    st.stop()
                if db_available:
                    try:
                        with open_connection(cfg.db) as conn:
                            DiscussionRepository(conn).add_post(
                                author.strip(), post_type, content.strip()
                            )
                    except Exception:
                        st.session_state.lawyer_posts.append({
                            "author":  author.strip(),
                            "type":    post_type,
                            "content": content.strip(),
                            "time":    _dt.datetime.now().strftime("%b %d, %H:%M"),
                        })
                else:
                    st.session_state.lawyer_posts.append({
                        "author":  author.strip(),
                        "type":    post_type,
                        "content": content.strip(),
                        "time":    _dt.datetime.now().strftime("%b %d, %H:%M"),
                    })
                st.rerun()
            else:
                st.error("Name and content are required.")
    with pc2:
        if posts:
            export_md = "\n\n---\n\n".join(
                f"**{p['author']}** ({p.get('type', 'Note')}) — {p['time']}\n\n{p['content']}"
                for p in posts
            )
            st.download_button("⬇ Export board", data=export_md,
                               file_name="epstein_discussion.md", mime="text/markdown",
                               use_container_width=True, key="export_board")


def _render_epstein_page() -> None:
    chat = _get_current_chat()
    if chat is None:
        return

    st.html(
        "<div style='padding:16px 0 4px'>"
        "<span style='font-size:1.4rem;font-weight:700;color:#1e293b'>"
        "Epstein's Case</span>"
        "</div>"
    )
    st.html(
        "<div style='background:#fef9c3;border:1px solid #fde047;border-radius:8px;"
        "padding:10px 16px;margin-bottom:12px;font-size:0.78rem;color:#713f12;"
        "line-height:1.6'>"
        "<strong>⚠ Legal Research Tool — Not Legal Advice.</strong> "
        "Responses are AI-generated from case documents and general legal knowledge. "
        "They do not constitute legal advice, do not create an attorney-client relationship, "
        "and must be independently verified by a licensed attorney before use in any "
        "legal proceeding or client matter."
        "</div>"
    )
    if _demo_mode:
        st.html(
            "<div style='background:#eef2ff;border:1px solid #c7d2fe;border-radius:8px;"
            "padding:10px 16px;margin-bottom:12px;font-size:0.78rem;color:#3730a3;"
            "line-height:1.6'>"
            "<strong>Public demo mode.</strong> You can chat with the shared public-record "
            "corpus and inspect timelines, but uploads, profile settings, and private "
            "matter features require sign-in."
            "</div>"
        )

    tab_chat, tab_timeline, tab_overall, tab_post = st.tabs(
        ["Chat", "Timeline", "Overall", "Post"]
    )

    with tab_chat:
        messages: list[dict] = chat["messages"]

        if not messages:
            st.markdown("<div style='height:12vh'></div>", unsafe_allow_html=True)
            _, mid, _ = st.columns([1, 2, 1])
            with mid:
                st.html(
                    "<div style='text-align:center;padding:28px 24px;background:#fafafa;"
                    "border:1px solid #e2e8f0;border-radius:12px'>"
                    "<div style='font-size:1.2rem;font-weight:700;color:#1e293b;"
                    "margin-bottom:6px'>Epstein's Case</div>"
                    "<div style='font-size:0.85rem;color:#64748b'>"
                    "Ask questions about case documents, evidence, or timeline."
                    "</div></div>"
                )

        for i, msg in enumerate(messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    _render_legal_ai_message(
                        msg["content"], msg.get("chunks"), image_index,
                        msg.get("scored_images"), key_suffix=f"ep_{i}",
                    )
                else:
                    st.markdown(msg["content"])

        question, uploaded_files = _render_chat_input_with_upload(
            "Ask about Epstein's case…", key="epstein_input"
        )
        if question:
            try:
                question = validate_query(question)
            except InputGuardError as exc:
                st.error(str(exc))
                st.stop()

            allowed, reason = check_rate_limit()
            if not allowed:
                st.warning(f"⏳ {reason}")
                st.stop()

            _ecid = st.session_state.current_chat_id
            if not messages:
                chat["name"] = question[:30].strip()
                if chat.get("persisted"):
                    try:
                        with open_connection(cfg.db) as conn:
                            PersistentChatRepository(conn).update_chat_name(_ecid, chat["name"])
                    except Exception:
                        pass

            messages.append({"role": "user", "content": question})
            _save_message(_ecid, "user", question)
            with st.chat_message("user"):
                st.markdown(question)

            retrieval_query = _contextualize_query(question, messages[:-1])

            with st.chat_message("assistant"):
                with st.spinner("Searching case documents…"):
                    corpus = _load_corpus_epstein(cfg)
                    with open_connection(cfg.db) as conn:
                        repo = ChunkRepository(conn)
                        retrieval = HybridSearchService(
                            embedder=embedder,
                            repository=repo,
                            corpus_chunks=corpus,
                            reranker=reranker,
                        )
                        chunks = retrieval.retrieve(
                            RetrievalRequest(
                                query=retrieval_query,
                                top_k=_DEFAULT_TOP_K,
                                company_filter="Epstein",
                            )
                        )
                        query_vec = embedder.embed_query(retrieval_query)
                        from insightlens.storage.image_repository import ImageRepository as _IR
                        scored_images = _IR(conn).search_by_description(
                            query_vec, top_k=3, min_similarity=0.4,
                            company_filter="Epstein",
                        )

                user_prompt = build_user_prompt(question, chunks)
                with st.spinner("Analyzing documents…"):
                    answer_text = "".join(llm.stream(CASE_SYSTEM_PROMPT, user_prompt))
                _render_legal_ai_message(answer_text, chunks, image_index, scored_images, key_suffix="ep_new")

            messages.append({"role": "assistant", "content": answer_text, "chunks": chunks, "scored_images": scored_images})
            _save_message(_ecid, "assistant", answer_text)

            _log_query_async(
                cfg, current_user(), "epstein", question,
                len(chunks), cfg.generation_model, len(answer_text),
            )

    with tab_timeline:
        _render_epstein_timeline()

    with tab_overall:
        _render_epstein_overall()

    with tab_post:
        _render_epstein_post()


# ── Main routing ───────────────────────────────────────────────────────────────
_page = st.session_state.active_page

if _page == "epstein":
    _render_epstein_page()
elif _page == "profile":
    render_profile_page(cfg, current_user())
elif _page == "data":
    render_data_page(cfg, current_user())
elif _page == "cases":
    render_cases_page(cfg, embedder, current_user(), llm=llm)
elif _page == "legal":
    render_legal_page(st.session_state.get("legal_tab", "terms"))
elif _page == "about":
    render_about_page()
elif _page == "team":
    render_org_page(cfg, current_user())
elif _page == "discussion":
    render_discussion_page(cfg, current_user())
else:
    _render_insightlens_page()

# ── Clear first-load skeleton after route rendering has completed ──────────────
if _sk_first_load:
    _sk_placeholder.empty()
    st.session_state._app_initialized = True
