"""Cases page — named document collections + document management."""
from __future__ import annotations

import html
import re
import tempfile
from pathlib import Path

import streamlit as st

from insightlens.billing import default_plan, format_limit_summary, max_upload_bytes
from insightlens.config import AppConfig
from insightlens.embeddings.embedder import Embedder
from insightlens.generation.llm_client import ClaudeClient
from insightlens.ingestion.cloud_storage import (
    CloudFile,
    CloudFolder,
    download_to_temp,
    make_provider,
)
from insightlens.ingestion.cloud_auth import (
    build_oauth_state,
    dropbox_auth_url,
    google_drive_auth_url,
    onedrive_auth_url,
    refresh_dropbox_token,
    refresh_google_drive_token,
    refresh_onedrive_token,
)
from insightlens.ingestion.ingest_service import IngestService
from insightlens.jobs.handlers import JOB_TYPE_EXTRACT_INSIGHTS
from insightlens.storage.cases_repository import CasesRepository
from insightlens.storage.chunk_repository import ChunkRepository
from insightlens.storage.cloud_credentials_repository import CloudCredentialsRepository
from insightlens.storage.insights_repository import InsightsRepository
from insightlens.storage.jobs_repository import JobsRepository
from insightlens.storage.snowflake_client import open_connection
from insightlens.storage.usage_repository import UsageRepository


def _fmt_name(file_name: str, maxlen: int = 55) -> str:
    import urllib.parse, re
    name = urllib.parse.unquote(file_name)
    stem = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    return (stem[:maxlen] + "…") if len(stem) > maxlen else stem


# ── Jurisdiction & Regulatory detection (rule-based, no LLM) ──────────────────

_JURISDICTION_PATTERNS: list[tuple[str, str, str]] = [
    ("Delaware",    "#dbeafe", "#1e40af", r"\bDelaware\b"),
    ("New York",    "#dbeafe", "#1e40af", r"\bNew York\b|\bSDNY\b|\bNDNY\b"),
    ("California",  "#dcfce7", "#166534", r"\bCalifornia\b|\bN\.D\. Cal\b"),
    ("Florida",     "#dcfce7", "#166534", r"\bFlorida\b|\bS\.D\. Fla\b"),
    ("Texas",       "#fef9c3", "#854d0e", r"\bTexas\b|\bN\.D\. Tex\b"),
    ("Federal US",  "#f3e8ff", "#6b21a8", r"\bfederal\b|\bU\.S\. District\b|\bUnited States District\b"),
    ("UK / English","#fce7f3", "#9d174d", r"\bUnited Kingdom\b|\bEnglish law\b|\bEngland and Wales\b"),
    ("EU",          "#fff7ed", "#9a3412", r"\bEuropean Union\b|\bEU law\b|\bEuropean Court\b"),
]

_REGULATORY_PATTERNS: list[tuple[str, str, str]] = [
    ("SEC",   "#dbeafe", "#1e40af", r"\bSEC\b|\bSecurities and Exchange\b|\bSecurities Act\b"),
    ("GDPR",  "#dcfce7", "#166534", r"\bGDPR\b|\bGeneral Data Protection\b"),
    ("SOX",   "#fef9c3", "#854d0e", r"\bSarbanes.Oxley\b|\bSOX\b"),
    ("FINRA", "#f3e8ff", "#6b21a8", r"\bFINRA\b|\bFinancial Industry Regulatory\b"),
    ("HIPAA", "#fce7f3", "#9d174d", r"\bHIPAA\b|\bHealth Insurance Portability\b"),
    ("FCPA",  "#fff7ed", "#9a3412", r"\bFCPA\b|\bForeign Corrupt Practices\b"),
    ("AML",   "#dbeafe", "#1e40af", r"\banti.money laundering\b|\bBank Secrecy Act\b|\bAML\b"),
]


def _detect_tags(text: str, patterns: list[tuple]) -> list[tuple[str, str, str]]:
    """Return list of (label, bg, color) for matched patterns."""
    found = []
    for label, bg, color, pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            found.append((label, bg, color))
    return found


def _tags_html(tags: list[tuple[str, str, str]]) -> str:
    return "".join(
        f"<span style='background:{bg};color:{color};font-size:0.6rem;font-weight:700;"
        f"padding:2px 7px;border-radius:3px;margin-right:4px;white-space:nowrap'>{label}</span>"
        for label, bg, color in tags
    )


# ── Cloud Import ─────────────────────────────────────────────────────────────────

_PROVIDER_META = {
    "google_drive": {
        "label": "Google Drive",
        "icon": "🗂",
        "color": "#4285f4",
        "auth_fn": google_drive_auth_url,
        "refresh_fn": refresh_google_drive_token,
    },
    "dropbox": {
        "label": "Dropbox",
        "icon": "📦",
        "color": "#0061fe",
        "auth_fn": dropbox_auth_url,
        "refresh_fn": refresh_dropbox_token,
    },
    "onedrive": {
        "label": "OneDrive",
        "icon": "☁",
        "color": "#0078d4",
        "auth_fn": onedrive_auth_url,
        "refresh_fn": refresh_onedrive_token,
    },
}


def _cloud_base_url() -> str:
    import os
    return "https://" + os.getenv("STREAMLIT_SERVER_BASE_URL", "app.atticus.ai")


def _get_valid_access_token(cfg: AppConfig, user_id: str, provider_key: str):
    """Fetch a valid access token, refreshing if expired."""
    with open_connection(cfg.db) as conn:
        cred = CloudCredentialsRepository(conn).get(user_id, provider_key)
    if not cred:
        return None
    from datetime import datetime, timezone, timedelta
    if cred.token_expires_at and cred.token_expires_at < datetime.now(timezone.utc) - timedelta(minutes=5):
        tokens = (
            refresh_google_drive_token(cred.refresh_token)
            if provider_key == "google_drive"
            else refresh_dropbox_token(cred.refresh_token)
            if provider_key == "dropbox"
            else refresh_onedrive_token(cred.refresh_token)
        )
        exp = datetime.now(timezone.utc) + timedelta(seconds=tokens.expires_in)
        with open_connection(cfg.db) as conn:
            CloudCredentialsRepository(conn).upsert(
                user_id, provider_key, tokens.refresh_token or cred.refresh_token,
                tokens.access_token, exp,
            )
        return tokens.access_token
    return cred.access_token


def _render_cloud_import_tab(cfg: AppConfig, embedder: Embedder, uid: str | None) -> None:
    if not uid:
        st.info("Sign in to import documents from cloud storage.")
        return

    st.html(
        "<div style='font-size:0.9rem;color:#475569;margin-bottom:16px'>"
        "Connect your cloud storage account to import PDFs directly. "
        "Files are downloaded on demand, parsed, and indexed in Atticus. "
        "Nothing is stored in your cloud account beyond what you ingest here."
        "</div>"
    )

    connected: dict[str, bool] = {}
    if uid:
        try:
            with open_connection(cfg.db) as conn:
                providers = CloudCredentialsRepository(conn).list_providers(uid)
            connected = {p: True for p in providers}
        except Exception:
            connected = {}

    cols = st.columns(3)
    provider_keys = list(_PROVIDER_META.keys())
    for idx, provider_key in enumerate(provider_keys):
        meta = _PROVIDER_META[provider_key]
        with cols[idx]:
            connected_now = connected.get(provider_key, False)
            st.html(
                f"<div style='border:1px solid {'#bbf7d0' if connected_now else '#e2e8f0'};"
                f"border-radius:12px;padding:20px;text-align:center;"
                f"background:{'#f0fdf4' if connected_now else '#f8fafc'}'>"
                f"<div style='font-size:2rem;margin-bottom:8px'>{meta['icon']}</div>"
                f"<div style='font-weight:600;font-size:0.9rem;color:#1e293b;margin-bottom:4px'>"
                f"{meta['label']}</div>"
                f"<div style='font-size:0.78rem;color:#64748b;margin-bottom:14px'>"
                f"{'Connected' if connected_now else 'Not connected'}</div>"
            )
            if connected_now:
                if st.button("Disconnect", key=f"dc_{provider_key}", use_container_width=True):
                    try:
                        with open_connection(cfg.db) as conn:
                            CloudCredentialsRepository(conn).delete(uid, provider_key)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to disconnect: {exc}")
                if st.button("Browse files", key=f"browse_{provider_key}", use_container_width=True):
                    st.session_state[f"_cloud_browse_{provider_key}"] = True
            else:
                env_key = {
                    "google_drive": "GOOGLE_DRIVE_CLIENT_ID",
                    "dropbox": "DROPBOX_APP_KEY",
                    "onedrive": "ONEDRIVE_CLIENT_ID",
                }[provider_key]
                if not os.getenv(env_key):
                    st.button(
                        f"Connect (setup required)",
                        key=f"conn_{provider_key}",
                        use_container_width=True,
                        disabled=True,
                    )
                    st.caption(f"Set {env_key} in .env to enable.")
                else:
                    state = build_oauth_state(provider_key, _cloud_base_url())
                    auth_url = meta["auth_fn"](_cloud_base_url(), state)
                    st.markdown(
                        f"[<button style='width:100%;padding:6px 0;border-radius:6px;"
                        f"border:none;background:{meta['color']};color:white;font-weight:600;"
                        f"cursor:pointer'>Connect {meta['label']}</button>"
                        f"](#{auth_url.replace('https://', '')})"
                        if False
                        else f"<a href='{auth_url}' target='_self'>"
                        f"<button style='width:100%;padding:6px 0;border-radius:6px;"
                        f"border:none;background:{meta['color']};color:white;font-weight:600;"
                        f"cursor:pointer'>Connect {meta['label']}</button></a>",
                        unsafe_allow_html=True,
                    )
            st.html("</div>")

    st.html("<hr style='border:none;border-top:1px solid #e2e8f0;margin:24px 0'>")

    for provider_key in provider_keys:
        if st.session_state.get(f"_cloud_browse_{provider_key}"):
            _render_cloud_file_browser(cfg, embedder, uid, provider_key)


def _render_cloud_file_browser(
    cfg: AppConfig, embedder: Embedder, uid: str, provider_key: str
) -> None:
    meta = _PROVIDER_META[provider_key]
    st.html(
        f"<div style='font-size:1rem;font-weight:600;color:#1e293b;margin-bottom:12px'>"
        f"{meta['icon']} {meta['label']} files</div>"
    )

    access_token = _get_valid_access_token(cfg, uid, provider_key)
    if not access_token:
        st.error("Session expired. Please reconnect your account.")
        return

    try:
        provider = make_provider(provider_key, access_token)
        folders = provider.list_folders()
        pdfs = provider.list_pdfs()
    except Exception as exc:
        st.error(f"Failed to list files: {exc}")
        return

    selected_folder = st.selectbox(
        "Folder", ["My Drive / OneDrive (root)"] + [f.name for f in folders],
        key=f"_folder_{provider_key}",
    )
    folder_id = None
    if selected_folder != "My Drive / OneDrive (root)":
        folder_id = next((f.id for f in folders if f.name == selected_folder), None)

    try:
        pdfs = provider.list_pdfs(folder_id) if folder_id else provider.list_pdfs()
    except Exception:
        pdfs = []

    if not pdfs:
        st.info("No PDF files found in this location.")
        return

    st.html(
        f"<div style='font-size:0.78rem;color:#64748b;margin-bottom:8px'>"
        f"{len(pdfs)} PDF file{'s' if len(pdfs) != 1 else ''} found</div>"
    )

    selected: list[CloudFile] = []
    for pf in pdfs:
        checked = st.checkbox(
            f"📄 {pf.name}  ({pf.size_bytes // 1024:.0f} KB)",
            value=False,
            key=f"_pf_{provider_key}_{pf.id}",
        )
        if checked:
            selected.append(pf)

    if not selected:
        st.info("Select files above, then click Import Selected.")
        return

    st.html(
        f"<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;"
        f"padding:10px 14px;font-size:0.85rem;color:#166534;margin-bottom:12px'>"
        f"<strong>{len(selected)} file{'s' if len(selected) != 1 else ''}</strong> selected "
        f"({sum(p.size_bytes for p in selected) // 1024:.0f} KB total)"
        "</div>"
    )

    if st.button("Import Selected", type="primary", key=f"_import_{provider_key}"):
        log_lines: list[str] = []
        ph = st.empty()

        def _prog(msg: str) -> None:
            log_lines.append(msg)
            ph.html(
                "<div style='font-size:0.82rem;color:#475569;font-family:monospace;"
                "background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;"
                "padding:10px 14px;white-space:pre-wrap'>"
                + "\n".join(log_lines[-8:]) + "</div>"
            )

        svc = IngestService(cfg=cfg, embedder=embedder)
        successes = 0
        errors = 0

        for pf in selected:
            _prog(f"Downloading {pf.name}…")
            try:
                tmp_path = download_to_temp(provider, pf.id)
                _prog(f"Processing {pf.name}…")
                result = svc.ingest(tmp_path, user_id=uid, progress=_prog)
                if result.error and not result.skipped:
                    _prog(f"Error: {result.error}")
                    errors += 1
                else:
                    successes += 1
                    try:
                        with open_connection(cfg.db) as conn:
                            UsageRepository(conn).log_upload(
                                user_id=uid,
                                document_id=result.document_id,
                                file_name=result.file_name,
                                file_size_bytes=pf.size_bytes,
                                page_count=result.page_count,
                                chunks_inserted=result.chunks_inserted,
                                estimated_cost_usd=result.estimated_cost_usd,
                            )
                    except Exception:
                        pass
                tmp_path.unlink(missing_ok=True)
            except Exception as exc:
                _prog(f"Failed to download {pf.name}: {exc}")
                errors += 1

        ph.empty()
        if errors == 0:
            st.success(
                f"Imported {successes} file{'s' if successes != 1 else ''} from {meta['label']}."
            )
        else:
            st.warning(
                f"Imported {successes}, {errors} failed. Check the log above."
            )
        st.cache_data.clear()

    if st.button("Clear selection", key=f"_clear_{provider_key}"):
        for pf in pdfs:
            if f"_pf_{provider_key}_{pf.id}" in st.session_state:
                del st.session_state[f"_pf_{provider_key}_{pf.id}"]
        st.rerun()


# ── Bulk summarization ─────────────────────────────────────────────────────────

_BULK_SUMMARIZE_CAP = 10  # stay within token budget


def _render_bulk_summarize(
    cfg: AppConfig, case_id: str, case_docs: list[dict], llm: ClaudeClient, uid: str
) -> None:
    st.html(
        "<div style='font-size:0.72rem;font-weight:600;color:#94a3b8;"
        "letter-spacing:0.05em;margin:18px 0 8px'>BULK DOCUMENT SUMMARIES</div>"
    )
    if not case_docs:
        st.caption("Add documents to this case first.")
        return

    docs_to_run = case_docs[:_BULK_SUMMARIZE_CAP]
    if len(case_docs) > _BULK_SUMMARIZE_CAP:
        st.caption(
            f"Rate-limited to first {_BULK_SUMMARIZE_CAP} documents "
            f"({len(case_docs)} total in case)."
        )

    if st.button("Summarize all documents", key=f"bulk_sum_{case_id}", type="primary"):
        doc_ids = [d["document_id"] for d in docs_to_run]
        with open_connection(cfg.db) as conn:
            all_chunks = ChunkRepository(conn).get_chunks_for_documents(doc_ids)
        chunks_by_doc: dict[str, list] = {}
        for c in all_chunks:
            chunks_by_doc.setdefault(c.document_id, []).append(c)

        summaries: dict[str, str] = {}
        progress = st.progress(0.0, text="Summarizing…")
        for i, doc in enumerate(docs_to_run):
            doc_chunks = chunks_by_doc.get(doc["document_id"], [])
            if not doc_chunks:
                continue
            sample = "\n\n".join(c.chunk_text for c in doc_chunks[:6])[:4000]
            try:
                summary = llm.generate(
                    "You are a concise legal document summarizer for attorneys.",
                    f"Summarize this legal document in exactly 2–3 sentences covering: "
                    f"document type, key parties, main obligations or findings, and jurisdiction if apparent.\n\n{sample}",
                )
            except Exception as exc:
                summary = f"Summary unavailable: {exc}"
            summaries[doc["document_id"]] = summary
            progress.progress((i + 1) / len(docs_to_run), text=f"Summarized {i + 1}/{len(docs_to_run)}…")
        progress.empty()

        # Persist each summary as a generated artifact
        try:
            with open_connection(cfg.db) as conn:
                repo = InsightsRepository(conn)
                for doc in docs_to_run:
                    s = summaries.get(doc["document_id"])
                    if s and not s.startswith("Summary unavailable"):
                        repo.create_artifact(
                            user_id=uid,
                            case_id=case_id,
                            artifact_type="bulk_summary",
                            title=_fmt_name(doc["file_name"], 120),
                            content=s,
                        )
        except Exception:
            pass

        st.session_state[f"_summaries_{case_id}"] = summaries
        st.rerun()

    stored = st.session_state.get(f"_summaries_{case_id}", {})
    if stored:
        for doc in case_docs:
            s = stored.get(doc["document_id"])
            if s:
                st.html(
                    f"<div style='border-left:3px solid #6366f1;background:#f8fafc;"
                    f"padding:8px 12px;margin-bottom:8px;border-radius:0 6px 6px 0'>"
                    f"<div style='font-size:0.75rem;font-weight:600;color:#1e293b;margin-bottom:4px'>"
                    f"📄 {html.escape(_fmt_name(doc['file_name']))}</div>"
                    f"<div style='font-size:0.78rem;color:#475569;line-height:1.6'>"
                    f"{html.escape(s)}</div>"
                    f"</div>"
                )


# ── Document comparison (diff) ─────────────────────────────────────────────────

def _render_document_compare(
    cfg: AppConfig, case_id: str, all_docs: list, llm: ClaudeClient
) -> None:
    st.html(
        "<div style='font-size:0.72rem;font-weight:600;color:#94a3b8;"
        "letter-spacing:0.05em;margin:18px 0 8px'>COMPARE DOCUMENTS</div>"
    )
    if len(all_docs) < 2:
        st.caption("Need at least 2 documents to compare.")
        return

    doc_map = {_fmt_name(d.file_name, 60): d for d in all_docs}
    names = list(doc_map.keys())
    c1, c2 = st.columns(2)
    with c1:
        doc_a_name = st.selectbox("Document A", names, key=f"cmp_a_{case_id}")
    with c2:
        remaining = [n for n in names if n != doc_a_name]
        doc_b_name = st.selectbox("Document B", remaining, key=f"cmp_b_{case_id}")

    if st.button("Compare →", key=f"cmp_run_{case_id}", type="primary"):
        doc_a = doc_map[doc_a_name]
        doc_b = doc_map[doc_b_name]
        with open_connection(cfg.db) as conn:
            repo = ChunkRepository(conn)
            chunks_a = repo.get_chunks_for_documents([doc_a.document_id])[:8]
            chunks_b = repo.get_chunks_for_documents([doc_b.document_id])[:8]

        text_a = "\n\n".join(c.chunk_text for c in chunks_a)[:3000]
        text_b = "\n\n".join(c.chunk_text for c in chunks_b)[:3000]

        with st.spinner("Comparing documents…"):
            try:
                result = llm.generate(
                    "You are an expert legal document analyst. Compare documents clearly and concisely.",
                    f"Compare these two legal documents side by side. Structure your response with these sections:\n"
                    f"1. Document Types & Purpose\n"
                    f"2. Key Parties\n"
                    f"3. Jurisdiction & Governing Law\n"
                    f"4. Material Differences (obligations, scope, risk, liability)\n"
                    f"5. Which document is more favorable and why\n\n"
                    f"DOCUMENT A — {doc_a_name}:\n{text_a}\n\n"
                    f"DOCUMENT B — {doc_b_name}:\n{text_b}",
                )
                st.session_state[f"_compare_{case_id}"] = (doc_a_name, doc_b_name, result)
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")

    stored = st.session_state.get(f"_compare_{case_id}")
    if stored:
        a_name, b_name, result = stored
        st.html(
            f"<div style='font-size:0.75rem;font-weight:600;color:#1e293b;margin-bottom:8px'>"
            f"📄 {html.escape(a_name)} vs 📄 {html.escape(b_name)}</div>"
        )
        st.markdown(result)
        dl_text = f"# {a_name} vs {b_name}\n\n{result}"
        st.download_button(
            "Download comparison",
            data=dl_text,
            file_name=f"comparison_{case_id[:8]}.md",
            mime="text/markdown",
            key=f"dl_compare_{case_id}",
        )


# ── Structured legal extraction ────────────────────────────────────────────────

def _render_legal_extraction(
    cfg: AppConfig, case_id: str, case_docs: list[dict], llm: ClaudeClient
) -> None:
    st.html(
        "<div style='font-size:0.72rem;font-weight:600;color:#94a3b8;"
        "letter-spacing:0.05em;margin:18px 0 8px'>STRUCTURED EXTRACTION</div>"
    )
    if not case_docs:
        st.caption("Add documents to this case first.")
        return

    mode = st.radio(
        "Extraction mode",
        ["Legal document", "Deposition transcript"],
        horizontal=True,
        key=f"extract_mode_{case_id}",
    )
    if mode == "Legal document":
        st.caption("Extracts parties, key dates, obligations, and risk clauses across all case documents.")
    else:
        st.caption("Extracts speakers, Q&A exchanges, admissions, and internal inconsistencies from a deposition.")

    if st.button("Extract legal structure", key=f"extract_{case_id}", type="primary"):
        doc_ids = [d["document_id"] for d in case_docs]
        with open_connection(cfg.db) as conn:
            all_chunks = ChunkRepository(conn).get_chunks_for_documents(doc_ids)
        sample_text = "\n\n---\n\n".join(c.chunk_text for c in all_chunks[:15])[:6000]

        if mode == "Deposition transcript":
            system_prompt = (
                "You are an expert litigation analyst specializing in deposition analysis. "
                "Extract structured information from deposition transcripts clearly and precisely."
            )
            user_prompt = (
                f"Analyze this deposition transcript and structure your output with these sections:\n"
                f"1. **Witnesses & Attorneys**: All named participants and their roles\n"
                f"2. **Key Admissions**: Statements by the witness that acknowledge facts unfavorable to their position\n"
                f"3. **Q&A Highlights**: The 5–8 most legally significant question/answer exchanges\n"
                f"4. **Internal Inconsistencies**: Statements that contradict each other or prior testimony\n"
                f"5. **Dates & Timeline**: All dates mentioned with context\n"
                f"6. **Exhibits Referenced**: Any documents or exhibits introduced\n\n"
                f"Transcript:\n{sample_text}"
            )
        else:
            system_prompt = "You are a legal data extraction specialist. Extract structured information clearly."
            user_prompt = (
                f"Extract the following from this document set and format as structured sections:\n"
                f"1. **Parties**: All named parties, their roles, and relationships\n"
                f"2. **Key Dates**: All dates with their significance\n"
                f"3. **Obligations**: What each party must do\n"
                f"4. **Risk Clauses**: Liability caps, indemnification, force majeure, termination triggers\n"
                f"5. **Jurisdiction & Governing Law**\n"
                f"6. **Financial Terms**: Any monetary amounts, fees, or penalties\n\n"
                f"Documents:\n{sample_text}"
            )

        with st.spinner("Extracting legal structure…"):
            try:
                result = llm.generate(system_prompt, user_prompt)
                st.session_state[f"_extraction_{case_id}"] = result
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")

    stored = st.session_state.get(f"_extraction_{case_id}")
    if stored:
        st.markdown(stored)
        st.download_button(
            "Download extraction",
            data=stored,
            file_name=f"legal_extraction_{case_id[:8]}.md",
            mime="text/markdown",
            key=f"dl_extraction_{case_id}",
        )


# ── Case insights (original) ───────────────────────────────────────────────────

def _render_case_insights(cfg: AppConfig, case_id: str, uid: str, case_docs: list[dict]) -> None:
    st.html(
        "<div style='font-size:0.72rem;font-weight:600;color:#94a3b8;"
        "letter-spacing:0.05em;margin:18px 0 8px'>CASE INSIGHTS</div>"
    )
    if not case_docs:
        st.caption("Add documents to this case before generating insights.")
        return

    # Check current job status for this case
    with open_connection(cfg.db) as conn:
        job = JobsRepository(conn).latest_job_for_case(case_id, JOB_TYPE_EXTRACT_INSIGHTS)

    job_running = job and job.status in ("queued", "running")
    job_failed  = job and job.status == "failed"

    col_btn, col_status = st.columns([2, 3])
    with col_btn:
        label = "Re-generate insights" if job and job.status == "completed" else "Generate insights"
        if st.button(label, key=f"gen_insights_{case_id}", type="primary", disabled=bool(job_running)):
            doc_ids = [doc["document_id"] for doc in case_docs]
            with open_connection(cfg.db) as conn:
                JobsRepository(conn).enqueue(
                    job_type=JOB_TYPE_EXTRACT_INSIGHTS,
                    user_id=uid,
                    case_id=case_id,
                    payload={"case_id": case_id, "doc_ids": doc_ids, "user_id": uid},
                )
            st.rerun()

    with col_status:
        if job_running:
            st.info(f"⏳ Insights {'queued' if job.status == 'queued' else 'running'}… refresh to check progress.")
        elif job_failed:
            st.error(f"Extraction failed: {(job.error or '')[:120]}")
        elif job and job.status == "completed" and job.finished_at:
            st.caption(f"Last run: {job.finished_at.strftime('%b %d %H:%M')}")

    try:
        with open_connection(cfg.db) as conn:
            repo = InsightsRepository(conn)
            timeline = repo.list_case_insights(case_id, "timeline", limit=20)
            entities = repo.list_case_insights(case_id, "entity", limit=20)
            contradictions = repo.list_case_insights(case_id, "contradiction", limit=20)
    except Exception:
        timeline, entities, contradictions = [], [], []

    i1, i2, i3 = st.columns(3)
    with i1:
        st.metric("Timeline items", len(timeline))
    with i2:
        st.metric("Entities", len(entities))
    with i3:
        st.metric("Contradictions", len(contradictions))

    if contradictions:
        st.html("<div style='font-size:0.75rem;font-weight:600;color:#991b1b;margin:10px 0 6px'>VERIFY THESE</div>")
        for item in contradictions[:5]:
            st.html(
                f"<div style='border-left:3px solid #dc2626;background:#fef2f2;"
                f"padding:8px 10px;margin-bottom:6px;border-radius:0 6px 6px 0'>"
                f"<div style='font-size:0.82rem;font-weight:600;color:#7f1d1d'>"
                f"{html.escape(item.title)}</div>"
                f"<div style='font-size:0.76rem;color:#475569;white-space:pre-wrap'>"
                f"{html.escape(item.body or '')}</div>"
                f"</div>"
            )

    if timeline:
        st.html("<div style='font-size:0.75rem;font-weight:600;color:#334155;margin:10px 0 6px'>TIMELINE PREVIEW</div>")
        for item in timeline[:8]:
            st.html(
                f"<div style='border-left:3px solid #4f46e5;background:#f8fafc;"
                f"padding:7px 10px;margin-bottom:5px;border-radius:0 6px 6px 0'>"
                f"<div style='font-size:0.8rem;font-weight:600;color:#1e293b'>{html.escape(item.title)}</div>"
                f"<div style='font-size:0.74rem;color:#475569'>{html.escape(item.body or '')}</div>"
                f"</div>"
            )

    if entities:
        entity_csv = "Entity,Details\n" + "\n".join(
            f"\"{item.title}\",\"{(item.body or '').replace('\"', '\"\"')}\""
            for item in entities
        )
        st.download_button(
            "Download entities CSV",
            data=entity_csv,
            file_name="case_entities.csv",
            mime="text/csv",
            key=f"entities_csv_{case_id}",
        )

    summary_md = "# Client Summary\n\nGenerate insights to refresh this export.\n"
    if timeline or contradictions:
        summary_md = "# Client Summary\n\n"
        summary_md += f"- Timeline items detected: {len(timeline)}\n"
        summary_md += f"- Entity mentions detected: {len(entities)}\n"
        summary_md += f"- Possible contradictions detected: {len(contradictions)}\n\n"
        if timeline:
            summary_md += "## Timeline Preview\n"
            for item in timeline[:10]:
                summary_md += f"- **{item.title}**: {item.body or ''}\n"
        if contradictions:
            summary_md += "\n## Items To Verify\n"
            for item in contradictions[:10]:
                summary_md += f"- **{item.title}**: {(item.body or '').splitlines()[0]}\n"
    st.download_button(
        "Download client summary",
        data=summary_md,
        file_name="client_summary.md",
        mime="text/markdown",
        key=f"client_summary_{case_id}",
    )


# ── Cases sub-page ─────────────────────────────────────────────────────────────

def _render_cases_tab(cfg: AppConfig, user: dict | None, llm: ClaudeClient | None) -> None:
    st.html(
        "<div style='font-size:0.85rem;color:#475569;margin-bottom:16px'>"
        "Group documents into named cases for focused research and multi-matter workspaces."
        "</div>"
    )

    if not user:
        st.info("Sign in to manage cases.")
        return

    uid = user["uid"]

    if "selected_case_id" not in st.session_state:
        st.session_state.selected_case_id = None

    # ── Create new case ───────────────────────────────────────────────────────
    with st.expander("＋ Create new case"):
        new_name = st.text_input("Case name", key="new_case_name", max_chars=200)
        new_desc = st.text_area("Description (optional)", key="new_case_desc", height=80)
        if st.button("Create case", type="primary", key="btn_create_case"):
            if not new_name.strip():
                st.error("Case name is required.")
            else:
                try:
                    with open_connection(cfg.db) as conn:
                        case_id = CasesRepository(conn).create_case(
                            user_id=uid,
                            case_name=new_name.strip(),
                            description=new_desc.strip() or None,
                        )
                    st.success("Case created.")
                    st.session_state.selected_case_id = case_id
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not create case: {exc}")

    st.divider()

    # ── Load cases ────────────────────────────────────────────────────────────
    cases = []
    with st.spinner("Loading cases…"):
        try:
            with open_connection(cfg.db) as conn:
                cases = CasesRepository(conn).list_cases(uid)
        except Exception as exc:
            st.error(f"Could not load cases: {exc}")
            return

    if not cases:
        st.html(
            "<div style='padding:40px 0;text-align:center;color:#94a3b8;"
            "font-size:0.9rem'>No cases yet. Create your first case above.</div>"
        )
        return

    # ── Two-panel layout ──────────────────────────────────────────────────────
    col_list, col_detail = st.columns([1, 2], gap="large")

    with col_list:
        st.html(
            "<div style='font-size:0.72rem;font-weight:600;color:#94a3b8;"
            "letter-spacing:0.05em;margin-bottom:6px'>YOUR CASES</div>"
        )
        for case in cases:
            is_active = case.case_id == st.session_state.selected_case_id
            border = "3px solid #4f46e5" if is_active else "1px solid #e2e8f0"
            bg = "#f0f0ff" if is_active else "#fff"
            st.html(
                f"<div style='background:{bg};border:{border};border-radius:8px;"
                f"padding:10px 14px;margin-bottom:6px'>"
                f"<div style='font-size:0.88rem;font-weight:600;color:#1e293b'>"
                f"{html.escape(case.case_name)}</div>"
                f"<div style='font-size:0.75rem;color:#64748b;margin-top:2px'>"
                f"{case.document_count} doc{'s' if case.document_count != 1 else ''}</div>"
                f"</div>"
            )
            if st.button(
                "View" if not is_active else "Selected",
                key=f"case_btn_{case.case_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.selected_case_id = case.case_id
                st.rerun()

    with col_detail:
        selected_id = st.session_state.selected_case_id
        if not selected_id:
            st.html(
                "<div style='padding:40px 0;text-align:center;color:#94a3b8;"
                "font-size:0.9rem'>Select a case to view details.</div>"
            )
            return

        selected_case = next((c for c in cases if c.case_id == selected_id), None)
        if not selected_case:
            st.session_state.selected_case_id = None
            st.rerun()
            return

        st.html(
            f"<div style='font-size:1.1rem;font-weight:700;color:#1e293b;margin-bottom:4px'>"
            f"{html.escape(selected_case.case_name)}</div>"
        )
        if selected_case.description:
            st.html(
                f"<div style='font-size:0.85rem;color:#475569;margin-bottom:12px'>"
                f"{html.escape(selected_case.description)}</div>"
            )

        st.html(
            "<div style='font-size:0.72rem;font-weight:600;color:#94a3b8;"
            "letter-spacing:0.05em;margin-bottom:8px'>DOCUMENTS IN THIS CASE</div>"
        )

        case_docs = []
        try:
            with open_connection(cfg.db) as conn:
                case_docs = CasesRepository(conn).get_case_documents_info(selected_id)
        except Exception:
            pass

        # Batch-load chunk samples for richer tag detection (3 chunks per doc max)
        _chunk_samples: dict[str, str] = {}
        if case_docs:
            try:
                with open_connection(cfg.db) as conn:
                    _raw = ChunkRepository(conn).get_chunks_for_documents(
                        [d["document_id"] for d in case_docs]
                    )
                for c in _raw:
                    existing = _chunk_samples.get(c.document_id, "")
                    if existing.count("\n\n") < 3:
                        _chunk_samples[c.document_id] = existing + "\n\n" + c.chunk_text
            except Exception:
                pass

        if not case_docs:
            st.html(
                "<div style='font-size:0.85rem;color:#94a3b8;margin-bottom:12px'>"
                "No documents yet. Add documents from the list below.</div>"
            )
        else:
            for doc in case_docs:
                tag_text = (
                    f"{doc.get('file_name', '')} {doc.get('company', '')} "
                    f"{_chunk_samples.get(doc['document_id'], '')[:2000]}"
                )
                j_tags = _detect_tags(tag_text, _JURISDICTION_PATTERNS)
                r_tags = _detect_tags(tag_text, _REGULATORY_PATTERNS)
                all_tags_html = _tags_html(j_tags) + _tags_html(r_tags)

                col_doc, col_rm = st.columns([4, 1])
                with col_doc:
                    st.html(
                        f"<div style='padding:6px 0'>"
                        f"<div style='font-size:0.85rem;color:#1e293b;margin-bottom:3px'>"
                        f"📄 {html.escape(_fmt_name(doc['file_name']))}"
                        + (f" <span style='color:#94a3b8'>· {doc['company']}</span>" if doc.get('company') else "")
                        + "</div>"
                        + (f"<div style='margin-top:3px'>{all_tags_html}</div>" if all_tags_html else "")
                        + "</div>"
                    )
                with col_rm:
                    if st.button("Remove", key=f"rm_{selected_id}_{doc['document_id']}",
                                 use_container_width=True):
                        try:
                            with open_connection(cfg.db) as conn:
                                CasesRepository(conn).remove_document_from_case(
                                    selected_id, doc["document_id"]
                                )
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

        # ── AI-powered insights tabs ──────────────────────────────────────────
        with st.expander("Insights: contradictions, timeline, entities, exports", expanded=False):
            _render_case_insights(cfg, selected_id, uid, case_docs)

        if llm:
            with st.expander("AI Summaries — bulk summarize all documents", expanded=False):
                _render_bulk_summarize(cfg, selected_id, case_docs, llm, uid)

            with st.expander("Structured extraction — parties, dates, obligations, risks", expanded=False):
                _render_legal_extraction(cfg, selected_id, case_docs, llm)

        st.html("<div style='height:16px'></div>")
        st.html(
            "<div style='font-size:0.72rem;font-weight:600;color:#94a3b8;"
            "letter-spacing:0.05em;margin-bottom:8px'>ADD DOCUMENTS TO CASE</div>"
        )

        all_docs = []
        try:
            with open_connection(cfg.db) as conn:
                repo = ChunkRepository(conn)
                user_docs = repo.list_documents(user_id=uid) if uid else []
                sys_docs_for_case = repo.list_documents(user_id=None)
                all_docs = user_docs + sys_docs_for_case
        except Exception:
            pass

        in_case_ids = {d["document_id"] for d in case_docs}
        available = [d for d in all_docs if d.document_id not in in_case_ids]

        if not available:
            st.html(
                "<div style='font-size:0.82rem;color:#94a3b8'>"
                "All available documents are already in this case.</div>"
            )
        else:
            doc_options = {_fmt_name(d.file_name): d.document_id for d in available}
            chosen = st.multiselect(
                "Select documents to add",
                options=list(doc_options.keys()),
                key=f"add_docs_{selected_id}",
            )
            if chosen and st.button("Add selected", key=f"add_btn_{selected_id}", type="primary"):
                try:
                    with open_connection(cfg.db) as conn:
                        repo = CasesRepository(conn)
                        for name in chosen:
                            repo.add_document_to_case(selected_id, doc_options[name])
                    st.success(f"Added {len(chosen)} document(s).")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not add documents: {exc}")

        if llm and all_docs:
            st.html("<div style='height:16px'></div>")
            with st.expander("Compare documents side by side", expanded=False):
                _render_document_compare(cfg, selected_id, all_docs, llm)

        st.html("<div style='height:16px'></div>")
        with st.expander("⚠ Delete this case"):
            st.warning(
                "This will delete the case and remove all document associations. "
                "Documents themselves are not deleted."
            )
            if st.button("Delete case", key=f"del_case_{selected_id}", type="primary"):
                try:
                    with open_connection(cfg.db) as conn:
                        CasesRepository(conn).delete_case(selected_id)
                    st.session_state.selected_case_id = None
                    st.success("Case deleted.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not delete case: {exc}")


# ── Documents sub-page ─────────────────────────────────────────────────────────

def _render_documents_tab(cfg: AppConfig, embedder: Embedder, user: dict | None) -> None:
    uid = user["uid"] if user else None

    tab_upload, tab_mine, tab_system, tab_cloud = st.tabs(
        ["Upload", "My Documents", "System Documents", "Cloud Import"]
    )

    # ── Upload ────────────────────────────────────────────────────────────────
    with tab_upload:
        plan = default_plan()
        st.html(
            "<div style='font-size:0.9rem;color:#475569;margin-bottom:16px'>"
            "Upload a PDF to make it searchable in Atticus. "
            "Ingestion may take 1–3 minutes depending on document size."
            "</div>"
        )
        st.caption(f"Current plan guardrails: {format_limit_summary(plan)}.")
        uploaded = st.file_uploader(
            "Choose a PDF file", type=["pdf"], accept_multiple_files=False,
            help=f"Max {plan.max_upload_mb} MB. PDFs only.",
        )
        if uploaded:
            upload_too_large = uploaded.size > max_upload_bytes(plan)
            upload_count_blocked = False
            uploads_this_month = 0
            if uid:
                try:
                    with open_connection(cfg.db) as conn:
                        uploads_this_month = UsageRepository(conn).count_uploads_this_month(uid)
                    upload_count_blocked = uploads_this_month >= plan.monthly_upload_limit
                except Exception:
                    upload_count_blocked = False
            st.html(
                f"<div style='background:{'#fef2f2' if upload_too_large else '#f0fdf4'};"
                f"border:1px solid {'#fecaca' if upload_too_large else '#bbf7d0'};"
                f"border-radius:8px;padding:10px 14px;font-size:0.85rem;"
                f"color:{'#991b1b' if upload_too_large else '#166534'}'>"
                f"Ready to ingest: <strong>{uploaded.name}</strong> "
                f"({uploaded.size / 1024:.0f} KB)</div>"
            )
            if not uid:
                st.warning("Sign in to upload documents.")
            elif upload_count_blocked:
                st.error(
                    f"The {plan.name} plan currently allows "
                    f"{plan.monthly_upload_limit} uploads per month. "
                    f"You have used {uploads_this_month} this month."
                )
            elif upload_too_large:
                st.error(
                    f"This file is larger than the {plan.max_upload_mb} MB "
                    f"{plan.name} limit. Split it or compress it before uploading."
                )
            elif st.button("Ingest document", type="primary", key="btn_ingest"):
                log_lines: list[str] = []
                placeholder = st.empty()

                def _progress(msg: str) -> None:
                    log_lines.append(msg)
                    placeholder.html(
                        "<div style='font-size:0.82rem;color:#475569;font-family:monospace;"
                        "background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;"
                        "padding:10px 14px;white-space:pre-wrap'>"
                        + "\n".join(log_lines[-8:]) + "</div>"
                    )

                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp) / uploaded.name
                    tmp_path.write_bytes(uploaded.read())
                    svc = IngestService(cfg=cfg, embedder=embedder)
                    result = svc.ingest(tmp_path, user_id=uid, progress=_progress)

                placeholder.empty()
                if result.error and not result.skipped:
                    st.error(f"Ingestion failed: {result.error}")
                elif result.skipped:
                    st.warning(f"Document skipped: {result.error}")
                else:
                    try:
                        with open_connection(cfg.db) as conn:
                            UsageRepository(conn).log_upload(
                                user_id=uid,
                                document_id=result.document_id,
                                file_name=result.file_name,
                                file_size_bytes=uploaded.size,
                                page_count=result.page_count,
                                chunks_inserted=result.chunks_inserted,
                                estimated_cost_usd=result.estimated_cost_usd,
                            )
                    except Exception:
                        pass
                    st.success(
                        f"Ingested **{result.file_name}** — "
                        f"{result.page_count} pages, {result.chunks_inserted} chunks, "
                        f"{result.images_found} images. Estimated variable cost: "
                        f"${result.estimated_cost_usd:.4f}."
                    )
                    st.cache_data.clear()

    # ── Cloud Import ──────────────────────────────────────────────────────────
    with tab_cloud:
        _render_cloud_import_tab(cfg, embedder, uid)

    # ── My Documents ─────────────────────────────────────────────────────────
    with tab_mine:
        if not uid:
            st.info("Sign in to see your uploaded documents.")
        else:
            docs = []
            with st.spinner("Loading your documents…"):
                try:
                    with open_connection(cfg.db) as conn:
                        docs = ChunkRepository(conn).list_documents(user_id=uid)
                except Exception as exc:
                    st.error(f"Could not load documents: {exc}")

            if not docs:
                st.html(
                    "<div style='padding:40px 0;text-align:center;color:#94a3b8;"
                    "font-size:0.9rem'>You haven't uploaded any documents yet.</div>"
                )
            else:
                st.html(
                    f"<div style='font-size:0.78rem;color:#64748b;margin-bottom:8px'>"
                    f"{len(docs)} document{'s' if len(docs) != 1 else ''} uploaded</div>"
                )
                for doc in docs:
                    # Jurisdiction/regulatory tags from filename + doc metadata
                    tag_src = f"{doc.file_name} {doc.company or ''} {doc.document_type or ''}"
                    j_tags = _detect_tags(tag_src, _JURISDICTION_PATTERNS)
                    r_tags = _detect_tags(tag_src, _REGULATORY_PATTERNS)
                    tags_html_str = _tags_html(j_tags) + _tags_html(r_tags)

                    with st.expander(f"📄 {_fmt_name(doc.file_name, 60)}"):
                        ci, cd = st.columns([4, 1])
                        with ci:
                            meta_html = "<div style='font-size:0.8rem;color:#475569;line-height:1.8'>"
                            if doc.company:
                                meta_html += f"Company: {doc.company}<br>"
                            if doc.document_type:
                                meta_html += f"Type: {doc.document_type}<br>"
                            if doc.version_label:
                                meta_html += f"Version: {doc.version_label}<br>"
                            meta_html += f"Pages: {doc.page_count}</div>"
                            if tags_html_str:
                                meta_html += f"<div style='margin-top:6px'>{tags_html_str}</div>"
                            st.html(meta_html)
                        with cd:
                            if st.button("Delete", key=f"del_doc_{doc.document_id}",
                                         type="secondary", use_container_width=True):
                                try:
                                    with open_connection(cfg.db) as conn:
                                        ChunkRepository(conn).delete_document(doc.document_id, uid)
                                    st.success(f"Deleted {doc.file_name}")
                                    st.cache_data.clear()
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Delete failed: {exc}")

    # ── System Documents ──────────────────────────────────────────────────────
    with tab_system:
        st.html(
            "<div style='font-size:0.85rem;color:#475569;margin-bottom:12px'>"
            "Shared documents available to all users."
            "</div>"
        )
        sys_docs = []
        with st.spinner("Loading system documents…"):
            try:
                with open_connection(cfg.db) as conn:
                    sys_docs = ChunkRepository(conn).list_documents(user_id=None)
            except Exception as exc:
                st.error(f"Could not load system documents: {exc}")

        if not sys_docs:
            st.html(
                "<div style='padding:40px 0;text-align:center;color:#94a3b8;"
                "font-size:0.9rem'>No system documents found.</div>"
            )
        else:
            st.html(
                f"<div style='font-size:0.78rem;color:#64748b;margin-bottom:8px'>"
                f"{len(sys_docs)} system document{'s' if len(sys_docs) != 1 else ''}</div>"
            )
            by_company: dict[str, list] = {}
            for doc in sys_docs:
                by_company.setdefault(doc.company or "Unknown", []).append(doc)
            for company, cdocs in sorted(by_company.items()):
                st.html(
                    f"<div style='font-size:0.78rem;font-weight:600;color:#64748b;"
                    f"letter-spacing:0.04em;margin-top:12px;margin-bottom:4px'>"
                    f"{company.upper()} ({len(cdocs)} docs)</div>"
                )
                for doc in cdocs:
                    st.html(
                        f"<div style='font-size:0.82rem;color:#1e293b;padding:4px 8px;"
                        f"border-left:2px solid #e2e8f0;margin-bottom:2px'>"
                        f"{_fmt_name(doc.file_name, 60)}"
                        + (f" <span style='color:#94a3b8'>· {doc.version_label}</span>" if doc.version_label else "")
                        + f" <span style='color:#94a3b8'>· {doc.page_count}pp</span>"
                        + "</div>"
                    )


# ── Main entry point ───────────────────────────────────────────────────────────

def render_cases_page(
    cfg: AppConfig,
    embedder: Embedder,
    user: dict | None,
    llm: ClaudeClient | None = None,
) -> None:
    st.html(
        "<div style='padding:16px 0 4px'>"
        "<span style='font-size:1.4rem;font-weight:700;color:#1e293b'>Cases</span>"
        "</div>"
    )

    tab_cases, tab_docs = st.tabs(["My Cases", "Documents"])

    with tab_cases:
        _render_cases_tab(cfg, user, llm)

    with tab_docs:
        _render_documents_tab(cfg, embedder, user)
