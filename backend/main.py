"""FastAPI backend — thin REST layer over existing insightlens Python packages."""
from __future__ import annotations

import json as _json
import logging
import logging.handlers
import os
import re
import tempfile
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import firebase_admin
import firebase_admin.auth
import firebase_admin.credentials
from fastapi import FastAPI, Header, HTTPException, Depends, UploadFile, File, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

# ── Bootstrap insightlens ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parents[1]
_SRC_PATH  = str(_REPO_ROOT / "src")
import sys
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

# ── Logging — rotating error.log in project root ──────────────────────────────
_LOG_PATH = _REPO_ROOT / "error.log"
_log_handler = logging.handlers.RotatingFileHandler(
    _LOG_PATH,
    maxBytes=10 * 1024 * 1024,   # 10 MB per file
    backupCount=5,
    encoding="utf-8",
)
_log_handler.setLevel(logging.WARNING)
_log_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_root_logger.addHandler(_log_handler)
# Also emit to stdout so docker/uvicorn logs stay visible
_stdout_handler = logging.StreamHandler()
_stdout_handler.setLevel(logging.INFO)
_root_logger.addHandler(_stdout_handler)

logger = logging.getLogger("atticus.api")

_IMAGE_STOP_WORDS = {
    "about", "after", "again", "against", "also", "and", "any", "are", "case",
    "demo", "did", "does", "documents", "files", "for", "from", "has", "have",
    "how", "image", "images", "into", "its", "photo", "photos", "related",
    "show", "tell", "that", "the", "this", "was", "were", "what", "when",
    "where", "which", "who", "with",
}

_BROAD_LEGAL_QUERY_RE = re.compile(
    r"\b(?:key\s+legal\s+issues|legal\s+issues|case\s+theory|case\s+strategy|"
    r"strongest\s+arguments|weaknesses|risks|timeline|chronology|summarize|"
    r"overview|memo|brief|discovery|contradictions|unresolved\s+issues)\b",
    re.IGNORECASE,
)

_SOURCE_SNIPPET_STOP_WORDS = {
    "about", "after", "again", "against", "also", "and", "any", "are", "case",
    "cases", "document", "documents", "does", "epstein", "evidence", "file",
    "files", "for", "from", "give", "has", "have", "issue", "issues", "key",
    "legal", "matter", "matters", "record", "records", "related", "show",
    "source", "sources", "summarize", "summary", "tell", "that", "the", "this",
    "was", "were", "what", "when", "where", "which", "who", "why", "with",
}
_SUPPORTED_UPLOAD_EXTENSIONS = (".pdf", ".pptx")

_JURISDICTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("S.D.N.Y.", re.compile(r"\b(?:sdny|s\.d\.n\.y\.|southern district of new york|giuffre v\. maxwell)\b", re.IGNORECASE)),
    ("S.D. Fla.", re.compile(r"\b(?:s\.d\. fla\.|southern district of florida|acosta|non[- ]prosecution agreement|npa)\b", re.IGNORECASE)),
    ("Florida state", re.compile(r"\b(?:palm beach|florida state|state prosecution|solicitation)\b", re.IGNORECASE)),
    ("USVI", re.compile(r"\b(?:usvi|u\.s\. virgin islands|virgin islands|little saint james)\b", re.IGNORECASE)),
    ("UK", re.compile(r"\b(?:prince andrew|duke of york|uk|united kingdom)\b", re.IGNORECASE)),
]

from insightlens.config import load_config, ConfigError
from insightlens.embeddings.embedder import Embedder
from insightlens.generation.llm_client import ClaudeClient
from insightlens.retrieval.hybrid_search import HybridSearchService
from insightlens.retrieval.reranker import Reranker
from insightlens.retrieval.vector_search import RetrievalRequest
from insightlens.storage.chunk_repository import ChunkRepository, RetrievedChunk
from insightlens.storage.image_repository import ImageRecord, ImageRepository
from insightlens.storage.audit_repository import AuditRepository
from insightlens.storage.chat_repository_persistent import PersistentChatRepository
from insightlens.storage.consent_repository import ConsentRepository
from insightlens.storage.cases_repository import CasesRepository
from insightlens.storage.discussion_repository import DiscussionRepository
from insightlens.storage.org_repository import OrgRepository
from insightlens.storage.usage_repository import UsageRepository
from insightlens.storage.snowflake_client import open_connection
from insightlens.storage.user_repository import UserRepository
from insightlens.storage.jobs_repository import JobsRepository
from insightlens.memory.zep_memory import ZepActor, ZepMemory
from insightlens.billing import estimate_query_cost_usd, default_plan, max_upload_bytes
from insightlens.demo.epstein_people import epstein_people_context_text
try:
    from insightlens.ui.input_guard import (
        validate_query,
        validate_text_input,
        InputGuardError,
    )
except ImportError:
    # Graceful fallback if input_guard module is absent
    def validate_query(q: str) -> str:  # type: ignore[misc]
        return q.strip()
    def validate_text_input(t: str, field: str = "", max_length: int = 10000) -> str:  # type: ignore[misc]
        return t.strip()[:max_length]
    class InputGuardError(Exception):  # type: ignore[misc]
        pass

try:
    from backend.rate_limiter import check_query_rate_limit, check_upload_rate_limit, check_demo_rate_limit
except ModuleNotFoundError:
    from rate_limiter import check_query_rate_limit, check_upload_rate_limit, check_demo_rate_limit

# ── Firebase Admin SDK ────────────────────────────────────────────────────────
_fb_app: firebase_admin.app.App | None = None


def _init_firebase() -> None:
    global _fb_app
    if _fb_app is not None:
        return
    # Three ways to supply the service account, checked in order:
    #   1. FIREBASE_SERVICE_ACCOUNT_JSON — the raw JSON content pasted directly
    #      into one env var. This is the path of least friction on platforms
    #      (Render, Railway, etc.) where "secret files" are extra setup —
    #      paste the whole downloaded JSON as a single env var value instead.
    #   2. FIREBASE_SERVICE_ACCOUNT_PATH — a file path on disk (Docker COPY,
    #      a mounted secret file, or local dev with the file sitting in the repo root).
    #   3. Application Default Credentials — only works on GCP infra; not viable
    #      on Render/Railway/Vercel, kept as a last-resort fallback.
    cred_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    if cred_json:
        try:
            cred = firebase_admin.credentials.Certificate(_json.loads(cred_json))
        except (_json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                "FIREBASE_SERVICE_ACCOUNT_JSON is set but is not valid JSON. "
                "Paste the full contents of the downloaded service-account file."
            ) from exc
    elif cred_path and Path(cred_path).exists():
        cred = firebase_admin.credentials.Certificate(cred_path)
    else:
        cred = firebase_admin.credentials.ApplicationDefault()
    _fb_app = firebase_admin.initialize_app(cred)


def _verify_firebase_token(id_token: str) -> dict:
    """Verify a Firebase ID token; raises 401 on any failure."""
    _init_firebase()
    try:
        return firebase_admin.auth.verify_id_token(id_token, app=_fb_app)
    except firebase_admin.auth.InvalidIdTokenError as exc:
        logger.warning("Invalid Firebase token: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    except Exception as exc:
        logger.error("Firebase token verification error", exc_info=True)
        raise HTTPException(status_code=401, detail="Token verification failed.")


# ── Auth dependency (applied to every protected endpoint) ─────────────────────
bearer = HTTPBearer(auto_error=False)


def current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict:
    """FastAPI dependency — verify Firebase JWT and return decoded user claims.

    Every protected endpoint must declare `user: dict = Depends(require_user)`.
    The frontend is NEVER trusted to gate access; all authorisation happens here.
    """
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    claims = _verify_firebase_token(creds.credentials)
    return {
        "uid":          claims["uid"],
        "email":        claims.get("email", ""),
        "display_name": claims.get("name", ""),
    }


def require_user(user: dict = Depends(current_user)) -> dict:
    return user


# ── Pydantic models ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query:          str
    chat_id:        str | None = None
    page:           str = "insightlens"
    company_filter: str | None = None
    top_k:          int = 8
    case_id:        str | None = None


class DemoQueryRequest(BaseModel):
    query: str
    top_k: int = 8


class PlanUpdateRequest(BaseModel):
    plan: str


_VALID_PLANS = frozenset({"trial", "starter", "pro", "enterprise"})

# ── Confidence scoring ────────────────────────────────────────────────────────
_CONFIDENCE_RE = re.compile(r"<CONFIDENCE>(.*?)</CONFIDENCE>", re.DOTALL)
_CONFIDENCE_RATINGS = {5: "High", 4: "Good", 3: "Moderate", 2: "Low", 1: "Unreliable"}


def _extract_confidence(text: str) -> tuple[str, dict | None]:
    """Strip <CONFIDENCE>...</CONFIDENCE> from LLM output and return (clean_text, conf_dict).

    Returns (original_text, None) if no block found or JSON is malformed.
    """
    m = _CONFIDENCE_RE.search(text)
    if not m:
        return text, None
    try:
        raw = _json.loads(m.group(1).strip())
        score = int(raw.get("score", 3))
        score = max(1, min(5, score))
        conf = {
            "score": score,
            "rating": _CONFIDENCE_RATINGS.get(score, "Moderate"),
            "rationale": str(raw.get("rationale", "")),
        }
        clean = _CONFIDENCE_RE.sub("", text).rstrip()
        return clean, conf
    except Exception:
        # Malformed JSON — strip the block but don't break the response
        clean = _CONFIDENCE_RE.sub("", text).rstrip()
        return clean, None


def _user_events_thread(uid: str) -> str:
    """Stable per-user Zep thread for system events (account, plan, uploads, cases)."""
    return f"{uid}:events"


def _query_memory_thread_id(uid: str, page: str, chat_id: str | None, case_id: str | None) -> str:
    """Scope chat memory by workspace so case and demo corpora do not cross-pollinate."""
    workspace = f"case:{case_id}" if case_id else f"page:{page}"
    chat = chat_id or "default"
    return f"{uid}:{workspace}:{chat}"


# ── Demo-corpus guard helpers ─────────────────────────────────────────────────
def _assert_document_not_demo(conn, doc_id: str) -> None:
    """Raise 403 if the document is marked is_demo = TRUE."""
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT is_demo FROM documents WHERE document_id = %s", (doc_id,)
        )
        row = cur.fetchone()
        if row and row[0]:
            raise HTTPException(
                status_code=403,
                detail="Demo documents are read-only and cannot be deleted or modified.",
            )
    except HTTPException:
        raise
    except Exception:
        pass  # column may not exist yet (pre-migration); fail open
    finally:
        cur.close()


def _assert_case_not_demo(conn, case_id: str) -> None:
    """Raise 403 if the case is marked is_demo = TRUE."""
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT is_demo FROM cases WHERE case_id = %s", (case_id,)
        )
        row = cur.fetchone()
        if row and row[0]:
            raise HTTPException(
                status_code=403,
                detail="Demo cases are read-only and cannot be modified.",
            )
    except HTTPException:
        raise
    except Exception:
        pass
    finally:
        cur.close()


# ── Ownership / IDOR guard helpers ────────────────────────────────────────────
def _assert_case_owned_by(conn, case_id: str, uid: str) -> None:
    """Raise 403/404 if case_id does not exist or is not owned by uid."""
    cur = conn.cursor()
    try:
        cur.execute("SELECT user_id FROM cases WHERE case_id = %s", (case_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Case not found.")
        if row[0] != uid:
            raise HTTPException(status_code=403, detail="Access denied.")
    except HTTPException:
        raise
    except Exception:
        logger.error("_assert_case_owned_by: DB error case_id=%s", case_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        cur.close()


def _assert_document_owned_by(conn, doc_id: str, uid: str) -> None:
    """Raise 403/404 unless doc_id is an uploaded document owned by uid."""
    cur = conn.cursor()
    try:
        cur.execute("SELECT user_id FROM documents WHERE document_id = %s", (doc_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        if row[0] != uid:
            raise HTTPException(
                status_code=403,
                detail="Document does not belong to the requesting user.",
            )
    except HTTPException:
        raise
    except Exception:
        logger.error("_assert_document_owned_by: DB error doc_id=%s", doc_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        cur.close()


def _assert_chat_owned_by(conn, chat_id: str, uid: str) -> None:
    """Raise 403/404 if chat_id does not exist or is not owned by uid."""
    cur = conn.cursor()
    try:
        cur.execute("SELECT user_id FROM chats WHERE chat_id = %s", (chat_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Chat not found.")
        if row[0] != uid:
            raise HTTPException(status_code=403, detail="Access denied.")
    except HTTPException:
        raise
    except Exception:
        logger.error("_assert_chat_owned_by: DB error chat_id=%s", chat_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        cur.close()


def _assert_chat_workspace_matches(
    conn,
    chat_id: str | None,
    uid: str,
    page: str,
    case_id: str | None,
) -> None:
    """A chat belongs to exactly one page/case workspace once created."""
    if not chat_id:
        return
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT page, case_id FROM chats WHERE chat_id = %s AND user_id = %s",
            (chat_id, uid),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Chat not found.")
        chat_page, chat_case_id = row
        if chat_case_id:
            if case_id != chat_case_id:
                raise HTTPException(
                    status_code=409,
                    detail="This chat is locked to a different case. Start a new chat to ask about another case.",
                )
        elif case_id or chat_page != page:
            raise HTTPException(
                status_code=409,
                detail="This chat is locked to a different workspace. Start a new chat to switch cases.",
            )
    except HTTPException:
        raise
    except Exception:
        logger.error("_assert_chat_workspace_matches: DB error chat_id=%s", chat_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        cur.close()


def _assert_image_is_demo(conn, image_id: str) -> None:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT d.user_id
            FROM images i
            JOIN documents d ON i.document_id = d.document_id
            WHERE i.image_id = %s
            """,
            (image_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Image not found.")
        if row[0] is not None:
            raise HTTPException(status_code=403, detail="Image is not part of the public demo corpus.")
    finally:
        cur.close()


def _image_file_path(record: ImageRecord) -> Path:
    raw = Path(record.file_path)
    path = raw if raw.is_absolute() else _REPO_ROOT / raw
    resolved = path.resolve()
    images_root = (_REPO_ROOT / "data" / "images").resolve()
    if images_root not in resolved.parents:
        raise HTTPException(status_code=403, detail="Image path is outside the image store.")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Image file not found.")
    return resolved


def _image_payload(record: ImageRecord, source: str) -> dict:
    return {
        "image_id": record.image_id,
        "url": f"/api/demo/images/{record.image_id}",
        "source": source,
        "document_id": record.document_id,
        "page_number": record.page_number,
        "image_index": record.image_index,
        "width": record.width,
        "height": record.height,
        "description": record.ai_description,
    }


def _is_image_gallery_query(query: str) -> bool:
    q = query.lower()
    return any(
        term in q
        for term in (
            "show images",
            "show photos",
            "display images",
            "display photos",
            "extracted images",
            "extracted photos",
            "image gallery",
            "photo gallery",
            "any images",
            "any photos",
        )
    )


def _image_query_terms(query: str) -> list[str]:
    terms = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'-]{2,}", query.lower())
    return [
        term
        for term in dict.fromkeys(terms)
        if term not in _IMAGE_STOP_WORDS
    ]


def _sample_demo_images(conn, limit: int) -> list[ImageRecord]:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT i.image_id, i.document_id, i.page_number, i.image_index,
                   i.file_path, i.media_type, i.width, i.height, i.ai_description
            FROM images i
            JOIN documents d ON i.document_id = d.document_id
            WHERE d.user_id IS NULL
              AND COALESCE(i.ai_description, '') NOT ILIKE '%%nude%%'
              AND COALESCE(i.ai_description, '') NOT ILIKE '%%naked%%'
              AND COALESCE(i.ai_description, '') NOT ILIKE '%%sexual%%'
              AND COALESCE(i.ai_description, '') NOT ILIKE '%%minor%%'
              AND COALESCE(i.ai_description, '') NOT ILIKE '%%child%%'
              AND COALESCE(i.ai_description, '') NOT ILIKE '%%kiro%%'
              AND COALESCE(i.ai_description, '') NOT ILIKE '%%I appreciate%%'
            ORDER BY (i.ai_description IS NULL), i.image_id
            LIMIT %s
            """,
            (limit,),
        )
        return [
            ImageRecord(
                image_id=row[0],
                document_id=row[1],
                page_number=row[2],
                image_index=row[3],
                file_path=row[4],
                media_type=row[5],
                width=row[6],
                height=row[7],
                ai_description=row[8],
            )
            for row in cur.fetchall()
        ]
    finally:
        cur.close()


def _collect_demo_images(conn, sources, query_embedding, query: str, top_k: int = 10) -> list[dict]:
    image_repo = ImageRepository(conn)
    seen: set[str] = set()
    images: list[dict] = []

    for source_index, chunk in enumerate(sources, start=1):
        for image in image_repo.get_images_for_page(chunk.document_id, chunk.page_number):
            if image.image_id in seen:
                continue
            seen.add(image.image_id)
            images.append(_image_payload(image, f"Source {source_index} page image"))
            if len(images) >= top_k:
                return images

    try:
        for image in image_repo.search_by_text_terms(
            _image_query_terms(query),
            top_k=top_k,
            system_only=True,
        ):
            if image.image_id in seen:
                continue
            try:
                _image_file_path(image)
            except HTTPException:
                continue
            seen.add(image.image_id)
            images.append(_image_payload(image, "Image metadata match"))
            if len(images) >= top_k:
                return images
    except Exception:
        logger.warning("Failed to collect image metadata matches", exc_info=True)

    try:
        if query_embedding:
            for image in image_repo.search_by_description(query_embedding, top_k=top_k):
                if image.image_id in seen:
                    continue
                _assert_image_is_demo(conn, image.image_id)
                seen.add(image.image_id)
                images.append(_image_payload(image, "Image description match"))
                if len(images) >= top_k:
                    break
    except Exception:
        logger.warning("Failed to collect image-description matches", exc_info=True)

    if not images and _is_image_gallery_query(query):
        for image in _sample_demo_images(conn, top_k):
            if image.image_id in seen:
                continue
            try:
                _image_file_path(image)
            except HTTPException:
                continue
            seen.add(image.image_id)
            images.append(_image_payload(image, "Sample extracted demo image"))
            if len(images) >= top_k:
                break

    return images


def _image_availability_note(sources, images: list[dict]) -> str | None:
    if images:
        return None

    image_terms = ("image", "images", ".jpg", ".jpeg", ".png", ".tif", ".tiff", "photo", "photograph")
    mentions_images = any(
        any(term in chunk.chunk_text.lower() for term in image_terms)
        for chunk in sources
    )
    if not mentions_images:
        return None

    return (
        "These sources reference image filenames or image indexes, but the actual image files "
        "are not present in the extracted image store for the cited pages."
    )


def _source_query_terms(query: str) -> list[str]:
    terms = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'-]{2,}", query.lower())
    return [
        term
        for term in dict.fromkeys(terms)
        if term not in _SOURCE_SNIPPET_STOP_WORDS
    ]


def _sentence_windows(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|(?:\n\s*){2,}", clean)
        if sentence.strip()
    ]
    if not sentences:
        return [clean]

    windows: list[str] = []
    for index, sentence in enumerate(sentences):
        if len(sentence.split()) < 8:
            continue
        windows.append(sentence)
        if index + 1 < len(sentences):
            combined = f"{sentence} {sentences[index + 1]}".strip()
            if len(combined.split()) <= 120:
                windows.append(combined)
    return windows or sentences[:3]


def _best_source_excerpt(text: str, query: str, max_chars: int = 900) -> str:
    windows = _sentence_windows(text)
    if not windows:
        return text[:max_chars].strip()

    terms = _source_query_terms(query)
    if not terms:
        excerpt = windows[0]
    else:
        query_phrases = [
            " ".join(terms[i : i + 2])
            for i in range(len(terms) - 1)
        ]
        scored = []
        for window in windows:
            lower = window.lower()
            term_hits = sum(1 for term in terms if term in lower)
            phrase_hits = sum(2 for phrase in query_phrases if phrase in lower)
            exact_hits = sum(lower.count(term) for term in terms)
            length_penalty = max(0, len(window.split()) - 90) / 90
            score = term_hits * 3 + phrase_hits + min(exact_hits, 6) - length_penalty
            scored.append((score, term_hits, -abs(len(window.split()) - 45), window))
        scored.sort(reverse=True)
        best_score, best_hits, _, excerpt = scored[0]
        if best_score <= 0 or best_hits == 0:
            excerpt = windows[0]

    excerpt = excerpt.strip()
    if len(excerpt) <= max_chars:
        return excerpt
    trimmed = excerpt[:max_chars].rsplit(" ", 1)[0].rstrip()
    return f"{trimmed}..."


def _normalized_source_name(file_name: str) -> str:
    name = unquote(file_name or "").strip().lower()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\s*\(\d+\)(?=(?:\.[a-z0-9]+)?$)", "", name)
    return name


def _display_source_name(file_name: str) -> str:
    name = unquote(file_name or "").strip()
    name = re.sub(r"\s*\(\d+\)(?=(?:\.[A-Za-z0-9]+)?$)", "", name)
    name = re.sub(r"\.[Pp][Dd][Ff]$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "Source"


def _infer_jurisdiction(source) -> str | None:
    text = " ".join(
        str(part or "")
        for part in (
            getattr(source, "file_name", ""),
            getattr(source, "document_type", ""),
            getattr(source, "section_header", ""),
            getattr(source, "chunk_text", "")[:1200],
        )
    )
    for label, pattern in _JURISDICTION_PATTERNS:
        if pattern.search(text):
            return label
    return None


def _citation_label(source) -> str:
    source_type = getattr(source, "source_type", "document") or "document"
    page = getattr(source, "page_number", None)
    page_text = f", at {page}" if page else ""
    if source_type == "demo_summary":
        return f"Atticus demo Epstein matter summary{page_text}"
    if source_type == "case_overview":
        return f"AI-generated case overview{page_text}"
    if source_type == "case_timeline":
        return f"AI-generated case timeline{page_text}"
    title = _display_source_name(getattr(source, "file_name", ""))
    return f"{title}{page_text}"


def _workspace_note(query: str, sources) -> str:
    profile = _evidence_profile(query, sources)
    jurisdictions = sorted({
        jurisdiction
        for source in sources
        if (jurisdiction := _infer_jurisdiction(source))
    })
    jurisdiction_text = ", ".join(jurisdictions) if jurisdictions else "not tagged from retrieved sources"
    return (
        f"Workspace/source universe for this answer: {profile['primary_count']} primary source chunk(s) "
        f"from {profile['unique_primary_docs']} unique primary document(s), plus "
        f"{profile['secondary_count']} secondary/generated context source(s). "
        f"Jurisdiction tags detected: {jurisdiction_text}."
    )


def _dedupe_comparable_sources(sources) -> list:
    """Collapse duplicate copies of the same source page before prompting/display."""
    seen: set[tuple] = set()
    deduped = []
    for source in sources:
        source_type = getattr(source, "source_type", "document") or "document"
        if source_type == "document":
            key = (
                "document-page",
                _normalized_source_name(getattr(source, "file_name", "")),
                getattr(source, "page_number", None),
            )
        else:
            key = (
                source_type,
                getattr(source, "document_id", "") or getattr(source, "file_name", ""),
                getattr(source, "page_number", None),
            )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _source_payload(sources, query: str | None = None) -> list[dict]:
    return [
        {
            "index": index,
            "label": f"{chunk.file_name} · p.{chunk.page_number}",
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "file_name": chunk.file_name,
            "page_number": chunk.page_number,
            "section_header": chunk.section_header,
            "chunk_text": chunk.chunk_text,
            "excerpt": _best_source_excerpt(chunk.chunk_text, query or ""),
            "source_type": getattr(chunk, "source_type", "document") or "document",
            "citation_label": _citation_label(chunk),
            "jurisdiction": _infer_jurisdiction(chunk),
        }
        for index, chunk in enumerate(sources, start=1)
    ]


def _evidence_profile(query: str, sources) -> dict:
    primary = [
        source for source in sources
        if (getattr(source, "source_type", "document") or "document") == "document"
    ]
    secondary = [source for source in sources if source not in primary]
    primary_docs = {
        getattr(source, "document_id", "") or getattr(source, "file_name", "")
        for source in primary
    }
    primary_pages = {
        (getattr(source, "document_id", "") or getattr(source, "file_name", ""), getattr(source, "page_number", None))
        for source in primary
    }
    return {
        "is_broad_legal": bool(_BROAD_LEGAL_QUERY_RE.search(query)),
        "primary_count": len(primary),
        "secondary_count": len(secondary),
        "unique_primary_docs": len(primary_docs),
        "unique_primary_pages": len(primary_pages),
    }


def _confidence_rating(score: int) -> str:
    return {5: "High", 4: "Good", 3: "Moderate", 2: "Low", 1: "Unreliable"}.get(score, "Moderate")


def _cap_confidence_for_coverage(query: str, sources, confidence: dict | None) -> dict | None:
    if not confidence:
        return None
    profile = _evidence_profile(query, sources)
    cap = 5
    reasons: list[str] = []

    if profile["primary_count"] == 0:
        cap = min(cap, 3 if profile["secondary_count"] else 1)
        reasons.append("the answer is not supported by primary document evidence")
    if profile["is_broad_legal"] and profile["unique_primary_docs"] <= 1:
        cap = min(cap, 3)
        reasons.append("the question asks for broad legal synthesis but retrieval found one or fewer primary documents")
    elif profile["is_broad_legal"] and profile["unique_primary_docs"] < 3:
        cap = min(cap, 4)
        reasons.append("the question asks for broad legal synthesis and retrieval coverage is limited")
    if profile["secondary_count"] and profile["primary_count"] == 0:
        cap = min(cap, 3)
        reasons.append("the answer relies on secondary generated/demo context")

    score = max(1, min(5, int(confidence.get("score", 3))))
    if score <= cap:
        return confidence

    capped = dict(confidence)
    capped["score"] = cap
    capped["rating"] = _confidence_rating(cap)
    existing = str(capped.get("rationale", "")).strip()
    cap_reason = "; ".join(dict.fromkeys(reasons))
    capped["rationale"] = (
        f"{existing} Confidence capped at {cap}/5 because {cap_reason}."
    ).strip()
    return capped


def _scope_note_for_coverage(query: str, sources) -> str | None:
    profile = _evidence_profile(query, sources)
    if not profile["is_broad_legal"]:
        return None
    if profile["unique_primary_docs"] >= 3:
        return None
    if profile["primary_count"] == 0:
        return (
            "**Scope note:** No primary document evidence was retrieved for this broad legal question. "
            "The answer below should be treated as secondary context, not a complete legal assessment.\n\n"
        )
    return (
        f"**Scope note:** This is a limited synthesis based on {profile['unique_primary_docs']} "
        f"primary document(s) and {profile['unique_primary_pages']} page-level source(s) retrieved for this question. "
        "It should not be treated as a complete issue list until the underlying filings and related records are reviewed.\n\n"
    )


def _ensure_lawyer_followups(answer: str, query: str, sources) -> str:
    if re.search(r"actionable follow[- ]?up searches", answer, re.IGNORECASE) or re.search(
        r"(?m)^\s*-\s*(?:Search for|Search corpus for|Pull|Verify):", answer
    ):
        return answer
    profile = _evidence_profile(query, sources)
    if not profile["is_broad_legal"]:
        return answer
    jurisdictions = sorted({
        jurisdiction
        for source in sources
        if (jurisdiction := _infer_jurisdiction(source))
    })
    jurisdiction_text = ", ".join(jurisdictions) if jurisdictions else "the detected forum"
    return (
        f"{answer.rstrip()}\n\n"
        "## Actionable follow-up searches\n"
        "- Pull: the complete underlying document for each cited page and verify the quoted or summarized proposition.\n"
        f"- Search corpus for: \"victim notification\" AND \"non-prosecution agreement\" within {jurisdiction_text} materials.\n"
        "- Search for: Giuffre v. Maxwell, No. 15-cv-07433 (S.D.N.Y.), if survivor testimony or unsealed civil filings matter to this issue.\n"
        "- Verify: whether any cited generated summary is supported by a primary filing, transcript, order, or investigative report."
    )


def _ensure_workspace_note(answer: str, query: str, sources) -> str:
    if re.search(r"workspace note|workspace/source universe|source universe", answer, re.IGNORECASE):
        return answer
    profile = _evidence_profile(query, sources)
    if not profile["is_broad_legal"]:
        return answer
    return f"{answer.rstrip()}\n\n## Workspace note\n{_workspace_note(query, sources)}"


def _case_context_chunks(conn, case_id: str, uid: str) -> list[RetrievedChunk]:
    """Return generated case overview/timeline as secondary context chunks."""
    from insightlens.analysis.case_intelligence import build_case_overview, build_case_timeline

    _assert_case_owned_by(conn, case_id, uid)
    chunks: list[RetrievedChunk] = []
    cur = conn.cursor()
    case_name = "Uploaded Case"
    try:
        cur.execute("SELECT case_name FROM cases WHERE case_id = %s", (case_id,))
        case_row = cur.fetchone()
        if case_row and case_row[0]:
            case_name = case_row[0]

        cur.execute(
            """SELECT summary, parties, key_issues, jurisdiction, matter_type, generated_at
               FROM case_overviews
               WHERE case_id = %s""",
            (case_id,),
        )
        row = cur.fetchone()
        if row:
            parties = row[1] if row[1] else []
            key_issues = row[2] if row[2] else []
            text = "\n".join(
                [
                    "AI-generated case overview. Treat as secondary context, not primary evidence.",
                    f"Summary: {row[0] or ''}",
                    f"Jurisdiction: {row[3] or 'Unknown'}",
                    f"Matter type: {row[4] or 'Unknown'}",
                    "Parties:",
                    *[
                        f"- {party.get('name', '')}: {party.get('role', '')}"
                        for party in parties
                        if isinstance(party, dict)
                    ],
                    "Key issues:",
                    *[f"- {issue}" for issue in key_issues],
                ]
            )
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"{case_id}-overview",
                    document_id=case_id,
                    file_name="AI-generated case overview",
                    company="Case",
                    version_label="generated",
                    page_number=1,
                    chunk_text=text,
                    similarity=0.0,
                    section_header="Overview",
                    chunk_type="overview",
                    document_type="case-overview",
                    source_type="case_overview",
                )
            )

        cur.execute(
            """SELECT events, generated_at
               FROM case_timelines
               WHERE case_id = %s""",
            (case_id,),
        )
        row = cur.fetchone()
        if row:
            events = row[0] if row[0] else []
            event_lines = []
            for event in events:
                if isinstance(event, dict):
                    event_lines.append(
                        f"- {event.get('date', 'Unknown date')}: {event.get('title', 'Untitled')} — {event.get('description', '')}"
                    )
            if event_lines:
                text = "\n".join(
                    [
                        "AI-generated case timeline. Treat as secondary context, not primary evidence.",
                        *event_lines,
                    ]
                )
                chunks.append(
                    RetrievedChunk(
                        chunk_id=f"{case_id}-timeline",
                        document_id=case_id,
                        file_name="AI-generated case timeline",
                        company="Case",
                        version_label="generated",
                        page_number=1,
                        chunk_text=text,
                        similarity=0.0,
                        section_header="Timeline",
                        chunk_type="timeline",
                        document_type="case-timeline",
                        source_type="case_timeline",
                    )
                )
    finally:
        cur.close()

    if not chunks:
        # Cost-controlled fallback: if Celery/LLM overview generation has not
        # completed, derive a small case map locally from the selected case's
        # chunks. This improves broad questions without an extra model call.
        primary_chunks = ChunkRepository(conn).get_chunks_for_case(case_id, uid, limit=40)
        if primary_chunks:
            overview = build_case_overview(primary_chunks, case_name)
            overview_text = "\n".join(
                [
                    "Deterministic case overview generated from uploaded chunks. Treat as secondary context, not primary evidence.",
                    f"Summary: {overview.get('summary', '')}",
                    f"Jurisdiction: {overview.get('jurisdiction') or 'Unknown'}",
                    f"Matter type: {overview.get('matter_type') or 'Unknown'}",
                    "Parties:",
                    *[
                        f"- {party.get('name', '')}: {party.get('role', '')}"
                        for party in overview.get("parties", [])
                        if isinstance(party, dict)
                    ],
                    "Key issues:",
                    *[f"- {issue}" for issue in overview.get("key_issues", [])],
                ]
            )
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"{case_id}-deterministic-overview",
                    document_id=case_id,
                    file_name="Deterministic case overview",
                    company="Case",
                    version_label="generated",
                    page_number=1,
                    chunk_text=overview_text,
                    similarity=0.0,
                    section_header="Overview",
                    chunk_type="overview",
                    document_type="case-overview",
                    source_type="case_overview",
                )
            )
            timeline = build_case_timeline(primary_chunks, limit=10)
            if timeline:
                timeline_text = "\n".join(
                    [
                        "Deterministic case timeline generated from uploaded chunks. Treat as secondary context, not primary evidence.",
                        *[
                            f"- {event.get('date', 'Unknown date')}: {event.get('title', 'Untitled')} — {event.get('description', '')}"
                            for event in timeline
                        ],
                    ]
                )
                chunks.append(
                    RetrievedChunk(
                        chunk_id=f"{case_id}-deterministic-timeline",
                        document_id=case_id,
                        file_name="Deterministic case timeline",
                        company="Case",
                        version_label="generated",
                        page_number=1,
                        chunk_text=timeline_text,
                        similarity=0.0,
                        section_header="Timeline",
                        chunk_type="timeline",
                        document_type="case-timeline",
                        source_type="case_timeline",
                    )
                )
    return chunks


def _epstein_people_context_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="epstein-public-people-index",
        document_id="epstein-public-people-index",
        file_name="Public reference people index for Epstein matter",
        company="Epstein",
        version_label="public-reference",
        page_number=1,
        chunk_text=epstein_people_context_text(),
        similarity=1.0,
        document_type="public_reference",
        source_type="public_context",
    )


def _zep_actor_from_user(user: dict) -> ZepActor:
    return ZepActor(
        user_id=user["uid"],
        email=user.get("email", ""),
        display_name=user.get("display_name", ""),
    )


def _demo_zep_actor() -> ZepActor:
    return ZepActor(
        user_id="atticus-demo-user",
        email="demo@atticus.local",
        display_name="Atticus Demo User",
    )


def _system_prompt_with_zep_context(system_prompt: str, zep_context: str) -> str:
    if not zep_context:
        return system_prompt
    return (
        f"{system_prompt}\n\n"
        "Zep memory context is available for this conversation. Use it for continuity, "
        "user preferences, and prior conversation facts. Do not treat Zep memory as a "
        "document source: factual claims about uploaded documents still require inline "
        "[Source N] citations from the retrieved source list.\n\n"
        f"{zep_context}"
    )


_ROLE_RANK: dict[str, int] = {"owner": 3, "admin": 2, "member": 1}


def _assert_org_role(conn, org_id: str, uid: str, min_role: str = "member") -> None:
    """Raise 403 if uid is not at least min_role in org_id."""
    role = OrgRepository(conn).get_member_role(org_id, uid)
    if role is None:
        raise HTTPException(status_code=403, detail="You are not a member of this organization.")
    if _ROLE_RANK.get(role, 0) < _ROLE_RANK.get(min_role, 0):
        raise HTTPException(
            status_code=403,
            detail=f"This action requires '{min_role}' role or higher.",
        )


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(title="Atticus API", version="1.0.0")

    configured_origins = []
    for value in (os.getenv("APP_URL", ""), os.getenv("CORS_ORIGINS", "")):
        configured_origins.extend(origin.strip() for origin in value.split(",") if origin.strip())

    _raw_origins = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://localhost:8501",
        *configured_origins,
    ]
    cors_origins = list(dict.fromkeys(o.rstrip("/") for o in _raw_origins if o))
    cors_origin_regex = os.getenv("CORS_ORIGIN_REGEX", "").strip() or None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=cors_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Bootstrap services (run once at startup) ──────────────────────────────
    try:
        _cfg      = load_config()
        _embedder = Embedder(model=_cfg.embedding_model)
        _llm      = ClaudeClient(api_key=_cfg.anthropic_api_key, model=_cfg.generation_model)
        _reranker = Reranker()
        _zep      = ZepMemory(api_key=_cfg.zep_api_key, enabled=_cfg.zep_enabled)
        logger.info("Atticus API services bootstrapped successfully.")
    except ConfigError as exc:
        logger.error("Config error at startup: %s", exc)
        _cfg = _embedder = _llm = _reranker = _zep = None  # type: ignore[assignment]
    except Exception as exc:
        # Previously only ConfigError was caught here — any other startup
        # failure (e.g. EmbeddingError from Embedder, a bad ANTHROPIC_API_KEY,
        # Reranker/ZepMemory init failure) propagated out of create_app(),
        # which would crash the whole ASGI app at import time with no useful
        # log line distinguishing it from any other boot failure. Widened to
        # log the real exception and degrade gracefully instead, same as the
        # ConfigError path, so `_cfg is None` always has a traceback explaining why.
        logger.error("Unexpected error bootstrapping services at startup: %s", exc, exc_info=True)
        _cfg = _embedder = _llm = _reranker = _zep = None  # type: ignore[assignment]

    def require_active_subscription(user: dict = Depends(current_user)) -> dict:
        """Block queries / uploads once the trial period expires."""
        if not _cfg:
            return user
        try:
            with open_connection(_cfg.db) as conn:
                expired = UserRepository(conn).is_trial_expired(user["uid"])
        except Exception:
            return user  # fail-open: never block on DB errors
        if expired:
            raise HTTPException(status_code=402, detail="TRIAL_EXPIRED")
        return user

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get("/api/health")
    def health():
        return {"status": "ok", "service": "atticus-api"}

    @app.get("/api/demo/images/{image_id}", tags=["query"])
    def get_demo_image(image_id: str):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                _assert_image_is_demo(conn, image_id)
                image = ImageRepository(conn).get_image(image_id)
                if image is None:
                    raise HTTPException(status_code=404, detail="Image not found.")
                path = _image_file_path(image)
                return FileResponse(path, media_type=image.media_type or "image/png")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("get_demo_image image_id=%s", image_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    # ── Auth ──────────────────────────────────────────────────────────────────
    @app.post("/api/auth/session", tags=["auth"])
    def create_session_token(user: dict = Depends(require_user)):
        return {
            "uid":          user["uid"],
            "email":        user["email"],
            "display_name": user["display_name"],
        }

    @app.post("/api/auth/refresh", tags=["auth"])
    def refresh_id_token(body: dict):
        import requests as _requests
        api_key = os.getenv("FIREBASE_API_KEY") or os.getenv("FIREBASE_WEB_API_KEY", "")
        url = f"https://securetoken.googleapis.com/v1/token?key={api_key}"
        try:
            resp = _requests.post(
                url,
                json={"grant_type": "refresh_token", "refresh_token": body.get("refresh_token", "")},
                timeout=10,
            )
        except Exception as exc:
            logger.error("Token refresh request failed", exc_info=True)
            raise HTTPException(status_code=502, detail="Token refresh upstream failed.")
        if not resp.ok:
            raise HTTPException(status_code=401, detail="Token refresh failed.")
        data = resp.json()
        return {"id_token": data["id_token"], "refresh_token": data["refresh_token"]}

    # ── User registration / sync ──────────────────────────────────────────────
    @app.post("/api/auth/register", tags=["auth"])
    def register_user(user: dict = Depends(require_user)):
        """Idempotently register or sync a Firebase user in Postgres and Zep.

        Called by the frontend on every sign-in. Safe to call repeatedly —
        upsert only updates email/display_name for existing users, and only
        sets the Firebase custom claim + logs a Zep event for brand-new ones.
        """
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        uid          = user["uid"]
        email        = user.get("email", "")
        display_name = user.get("display_name", "")

        try:
            with open_connection(_cfg.db) as conn:
                repo    = UserRepository(conn)
                is_new  = repo.upsert_user(uid, email, display_name, plan="trial")
                if is_new:
                    # Set a 4-day trial window for brand-new accounts.
                    trial_expires = datetime.now(timezone.utc).replace(microsecond=0)
                    from datetime import timedelta
                    trial_expires = trial_expires + timedelta(days=4)
                    repo.set_trial_expires_at(uid, trial_expires)
                record  = repo.get_user(uid)
                plan    = record.plan if record else "trial"
        except Exception:
            logger.error("register_user: DB error uid=%s", uid, exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to register user.")

        if is_new:
            # Set Firebase custom claim so the JWT carries the plan on next refresh
            try:
                _init_firebase()
                firebase_admin.auth.set_custom_user_claims(
                    uid,
                    {"plan": plan, "plan_updated_at": datetime.now(timezone.utc).isoformat()},
                    app=_fb_app,
                )
            except Exception:
                logger.warning("register_user: failed to set Firebase custom claim uid=%s", uid, exc_info=True)

            # Log account creation to Zep
            if _zep and _zep.enabled:
                actor = _zep_actor_from_user(user)
                thread_id = _user_events_thread(uid)
                event_text = (
                    f"Account created on {datetime.now(timezone.utc).strftime('%Y-%m-%d')} "
                    f"with plan: {plan}. Email: {email}."
                )
                try:
                    _zep.add_system_event(thread_id, event_text, actor)
                except Exception:
                    logger.warning("register_user: Zep event failed uid=%s", uid, exc_info=True)

        return {"uid": uid, "plan": plan, "is_new": is_new}

    # ── Admin: update user plan ───────────────────────────────────────────────
    @app.post("/api/admin/users/{uid}/plan", tags=["admin"])
    def admin_update_plan(
        uid: str,
        req: PlanUpdateRequest,
        x_admin_key: str | None = Header(None, alias="x-admin-key"),
    ):
        """Update a user's plan. Requires the X-Admin-Key header.

        Updates:
          1. PostgreSQL users.plan + plan_updated_at
          2. Firebase custom claim { plan, plan_updated_at }
          3. Zep system event on the user's events thread
        """
        admin_key = os.getenv("ADMIN_API_KEY", "")
        if not admin_key or x_admin_key != admin_key:
            raise HTTPException(status_code=403, detail="Invalid or missing admin key.")

        new_plan = req.plan.lower().strip()
        if new_plan not in _VALID_PLANS:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid plan '{new_plan}'. Valid plans: {', '.join(sorted(_VALID_PLANS))}.",
            )

        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")

        try:
            with open_connection(_cfg.db) as conn:
                old_plan, _ = UserRepository(conn).update_plan(uid, new_plan)
        except Exception:
            logger.error("admin_update_plan: DB error uid=%s", uid, exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to update plan in database.")

        # Firebase custom claim
        try:
            _init_firebase()
            firebase_admin.auth.set_custom_user_claims(
                uid,
                {"plan": new_plan, "plan_updated_at": datetime.now(timezone.utc).isoformat()},
                app=_fb_app,
            )
        except Exception:
            logger.warning("admin_update_plan: Firebase claim update failed uid=%s", uid, exc_info=True)

        # Zep event
        if _zep and _zep.enabled:
            try:
                # We don't have a full actor here (no email/name), so build a minimal one
                actor = ZepActor(user_id=uid)
                thread_id = _user_events_thread(uid)
                event_text = (
                    f"Plan changed from {old_plan} to {new_plan} on "
                    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}."
                )
                _zep.add_system_event(thread_id, event_text, actor)
            except Exception:
                logger.warning("admin_update_plan: Zep event failed uid=%s", uid, exc_info=True)

        return {"uid": uid, "old_plan": old_plan, "new_plan": new_plan}

    # ── Chats ─────────────────────────────────────────────────────────────────
    @app.get("/api/chats", tags=["chats"])
    def list_chats(user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                chats = PersistentChatRepository(conn).list_chats(user["uid"])
            return [
                {
                    "chat_id": c.chat_id,
                    "name": c.chat_name or "Chat",
                    "page": c.page,
                    "case_id": c.case_id,
                    "chat_type": c.chat_type,
                }
                for c in chats
            ]
        except Exception as exc:
            logger.error("list_chats uid=%s", user["uid"], exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.post("/api/chats", tags=["chats"])
    def create_chat(payload: dict, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                case_id = payload.get("case_id")
                if case_id:
                    _assert_case_owned_by(conn, case_id, user["uid"])
                chat_id = PersistentChatRepository(conn).create_chat(
                    user["uid"],
                    payload.get("page", "insightlens"),
                    payload.get("name", "New Chat"),
                    chat_id=payload.get("chat_id"),
                    case_id=case_id,
                )
            if _zep and _zep.enabled:
                _zep.ensure_thread(chat_id, _zep_actor_from_user(user))
            return {"chat_id": chat_id}
        except Exception as exc:
            logger.error("create_chat uid=%s", user["uid"], exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.get("/api/chats/{chat_id}/messages", tags=["chats"])
    def get_chat_messages(chat_id: str, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                msgs = PersistentChatRepository(conn).load_messages(chat_id, user["uid"])
            return [{"role": m.role, "content": m.content} for m in msgs]
        except Exception as exc:
            logger.error("get_chat_messages chat_id=%s", chat_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.post("/api/chats/{chat_id}/messages", tags=["chats"])
    def save_message(chat_id: str, payload: dict, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                _assert_chat_owned_by(conn, chat_id, user["uid"])
                PersistentChatRepository(conn).save_message(
                    chat_id, payload.get("role", "user"), payload.get("content", "")
                )
            return {"ok": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("save_message chat_id=%s", chat_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.delete("/api/chats/{chat_id}", tags=["chats"])
    def delete_chat(chat_id: str, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                PersistentChatRepository(conn).delete_chat(chat_id, user["uid"])
            return {"ok": True}
        except Exception as exc:
            logger.error("delete_chat chat_id=%s", chat_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.patch("/api/chats/{chat_id}", tags=["chats"])
    def rename_chat(chat_id: str, payload: dict, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                PersistentChatRepository(conn).update_chat_name(
                    chat_id, payload.get("name", "Chat"), user["uid"]
                )
            return {"ok": True}
        except Exception as exc:
            logger.error("rename_chat chat_id=%s", chat_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    # ── Companies ─────────────────────────────────────────────────────────────
    @app.get("/api/companies", tags=["data"])
    def list_companies(user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                return ChunkRepository(conn).list_companies(user_id=user["uid"])
        except Exception:
            logger.error("list_companies", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    # ── Query (rate-limited + input-guarded) ──────────────────────────────────
    @app.post("/api/query", tags=["query"])
    def run_query(req: QueryRequest, user: dict = Depends(require_active_subscription)):
        if not all([_cfg, _embedder, _llm, _reranker]):
            raise HTTPException(status_code=503, detail="Service not ready.")

        # ── Input guard ───────────────────────────────────────────────────────
        try:
            clean_query = validate_query(req.query)
        except InputGuardError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        # ── Rate limit (check monthly from DB, then apply sliding windows) ────
        try:
            with open_connection(_cfg.db) as conn:
                stats = AuditRepository(conn).get_user_stats(user["uid"])
        except Exception as exc:
            logger.error("run_query: failed to load usage stats uid=%s", user["uid"], exc_info=True)
            stats = {"queries_this_month": 0}

        plan = default_plan()
        check_query_rate_limit(
            user["uid"],
            queries_this_month=int(stats.get("queries_this_month", 0)),
            monthly_limit=plan.monthly_query_limit,
        )

        # ── Execute retrieval + generation ────────────────────────────────────
        from insightlens.generation.prompts import SYSTEM_PROMPT, build_user_prompt
        try:
            top_k = max(1, min(req.top_k, 12))
            if req.chat_id:
                with open_connection(_cfg.db) as conn:
                    _assert_chat_workspace_matches(conn, req.chat_id, user["uid"], req.page, req.case_id)
            zep_thread_id = _query_memory_thread_id(user["uid"], req.page, req.chat_id, req.case_id)
            zep_actor = _zep_actor_from_user(user)
            zep_context = ""
            if _zep and _zep.enabled:
                zep_context = _zep.add_message_get_context(zep_thread_id, "user", clean_query, zep_actor)

            # P2: scope corpus — case_id takes priority, then page-based scoping
            _is_epstein = (req.page == "epstein")
            if req.case_id:
                _scope_system_only = False
                _scope_user_only   = False
            else:
                _scope_system_only = _is_epstein
                _scope_user_only   = not _is_epstein

            try:
                with open_connection(_cfg.db) as conn:
                    if req.case_id:
                        _assert_case_owned_by(conn, req.case_id, user["uid"])
                    chunk_repo = ChunkRepository(conn)
                    corpus = chunk_repo.get_all_chunks(
                        company_filter=req.company_filter,
                        user_id=user["uid"],
                        system_only=_scope_system_only,
                        user_only=_scope_user_only,
                        case_id=req.case_id,
                    )
                    hybrid_svc = HybridSearchService(
                        _embedder,
                        chunk_repo,
                        corpus,
                        _reranker,
                    )
                    sources = hybrid_svc.retrieve(
                        RetrievalRequest(
                            query=clean_query,
                            top_k=top_k,
                            company_filter=req.company_filter,
                            user_id=user["uid"],
                            system_only=_scope_system_only,
                            user_only=_scope_user_only,
                            case_id=req.case_id,
                        )
                    )
                    sources = list(sources)
                    sources = _dedupe_comparable_sources(sources)
                    if req.case_id:
                        sources.extend(_case_context_chunks(conn, req.case_id, user["uid"]))
                    elif _is_epstein:
                        sources.append(_epstein_people_context_chunk())
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))

            prompt   = build_user_prompt(clean_query, sources)
            response = _llm.generate(
                _system_prompt_with_zep_context(SYSTEM_PROMPT, zep_context),
                prompt,
            )
            clean_response, confidence = _extract_confidence(response)
            scope_note = _scope_note_for_coverage(clean_query, sources)
            if scope_note and "Scope note:" not in clean_response:
                clean_response = scope_note + clean_response.lstrip()
            clean_response = _ensure_lawyer_followups(clean_response, clean_query, sources)
            clean_response = _ensure_workspace_note(clean_response, clean_query, sources)
            confidence = _cap_confidence_for_coverage(clean_query, sources, confidence)
            if _zep and _zep.enabled:
                _zep.add_message(zep_thread_id, "assistant", clean_response, zep_actor)

            try:
                with open_connection(_cfg.db) as conn:
                    AuditRepository(conn).log_query(
                        user_id=user["uid"],
                        page=req.page,
                        query_text=clean_query,
                        chunks_retrieved=len(sources),
                        model_used=_cfg.generation_model,
                        response_length=len(clean_response),
                        estimated_cost_usd=estimate_query_cost_usd(
                            query_text=clean_query,
                            response_text=clean_response,
                            chunks_retrieved=len(sources),
                        ),
                    )
            except Exception:
                logger.warning("run_query: audit log failed uid=%s", user["uid"], exc_info=True)

            result = {
                "answer": clean_response,
                "sources": [
                    f"{s.file_name} · p.{s.page_number}"
                    for s in sources
                ],
                "source_details": _source_payload(sources, clean_query),
            }
            if confidence:
                result["confidence"] = confidence
            return result

        except HTTPException:
            raise
        except Exception:
            logger.error("run_query: LLM/retrieval error uid=%s", user["uid"], exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    # ── Streaming query ───────────────────────────────────────────────────────
    @app.post("/api/query/stream", tags=["query"])
    def run_query_stream(req: QueryRequest, user: dict = Depends(require_active_subscription)):
        """SSE streaming variant of /api/query.

        Emits newline-delimited JSON events:
          {"token": "..."} — incremental text token
          {"done": true, "sources": [...], "source_details": [...]} — final metadata
          {"error": "..."} — unrecoverable error (no further events follow)
        """
        if not all([_cfg, _embedder, _llm, _reranker]):
            raise HTTPException(status_code=503, detail="Service not ready.")

        try:
            clean_query = validate_query(req.query)
        except InputGuardError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        # Rate limit checked before streaming begins so we can still return 429
        try:
            with open_connection(_cfg.db) as conn:
                stats = AuditRepository(conn).get_user_stats(user["uid"])
        except Exception:
            logger.error("run_query_stream: failed to load stats uid=%s", user["uid"], exc_info=True)
            stats = {"queries_this_month": 0}

        plan = default_plan()
        check_query_rate_limit(
            user["uid"],
            queries_this_month=int(stats.get("queries_this_month", 0)),
            monthly_limit=plan.monthly_query_limit,
        )

        if req.chat_id:
            try:
                with open_connection(_cfg.db) as conn:
                    _assert_chat_workspace_matches(conn, req.chat_id, user["uid"], req.page, req.case_id)
            except HTTPException:
                raise

        from insightlens.generation.prompts import SYSTEM_PROMPT, build_user_prompt

        def _event(payload: dict) -> str:
            return f"data: {_json.dumps(payload)}\n\n"

        def event_stream():
            zep_thread_id = _query_memory_thread_id(user["uid"], req.page, req.chat_id, req.case_id)
            zep_actor     = _zep_actor_from_user(user)
            zep_context   = ""
            if _zep and _zep.enabled:
                zep_context = _zep.add_message_get_context(
                    zep_thread_id, "user", clean_query, zep_actor
                )

            # P2: scope corpus — case_id takes priority, then page-based scoping
            _is_epstein = (req.page == "epstein")
            if req.case_id:
                _scope_system_only = False
                _scope_user_only   = False
            else:
                _scope_system_only = _is_epstein
                _scope_user_only   = not _is_epstein

            # Retrieval
            try:
                top_k = max(1, min(req.top_k, 12))
                with open_connection(_cfg.db) as conn:
                    if req.case_id:
                        _assert_case_owned_by(conn, req.case_id, user["uid"])
                    chunk_repo  = ChunkRepository(conn)
                    corpus      = chunk_repo.get_all_chunks(
                        company_filter=req.company_filter,
                        user_id=user["uid"],
                        system_only=_scope_system_only,
                        user_only=_scope_user_only,
                        case_id=req.case_id,
                    )
                    hybrid_svc  = HybridSearchService(_embedder, chunk_repo, corpus, _reranker)
                    sources     = hybrid_svc.retrieve(
                        RetrievalRequest(
                            query=clean_query,
                            top_k=top_k,
                            company_filter=req.company_filter,
                            user_id=user["uid"],
                            system_only=_scope_system_only,
                            user_only=_scope_user_only,
                            case_id=req.case_id,
                        )
                    )
                    sources = list(sources)
                    sources = _dedupe_comparable_sources(sources)
                    if req.case_id:
                        sources.extend(_case_context_chunks(conn, req.case_id, user["uid"]))
                    elif _is_epstein:
                        sources.append(_epstein_people_context_chunk())
            except ValueError as exc:
                yield _event({"error": str(exc)})
                return
            except Exception:
                logger.error("run_query_stream: retrieval failed uid=%s", user["uid"], exc_info=True)
                yield _event({"error": "Retrieval failed. Please try again."})
                return

            # Stream generation
            try:
                prompt          = build_user_prompt(clean_query, sources)
                system          = _system_prompt_with_zep_context(SYSTEM_PROMPT, zep_context)
                full_parts: list[str] = []

                for token in _llm.stream(system, prompt):
                    full_parts.append(token)
                    yield _event({"token": token})

                full_response = "".join(full_parts)
                clean_response, confidence = _extract_confidence(full_response)
                scope_note = _scope_note_for_coverage(clean_query, sources)
                if scope_note and "Scope note:" not in clean_response:
                    clean_response = scope_note + clean_response.lstrip()
                clean_response = _ensure_lawyer_followups(clean_response, clean_query, sources)
                clean_response = _ensure_workspace_note(clean_response, clean_query, sources)
                confidence = _cap_confidence_for_coverage(clean_query, sources, confidence)

                done_event: dict = {
                    "done": True,
                    "text":           clean_response,
                    "sources":        [f"{s.file_name} · p.{s.page_number}" for s in sources],
                    "source_details": _source_payload(sources, clean_query),
                }
                if confidence:
                    done_event["confidence"] = confidence
                yield _event(done_event)

                if _zep and _zep.enabled:
                    _zep.add_message(zep_thread_id, "assistant", clean_response, zep_actor)

                try:
                    with open_connection(_cfg.db) as conn:
                        AuditRepository(conn).log_query(
                            user_id=user["uid"],
                            page=req.page,
                            query_text=clean_query,
                            chunks_retrieved=len(sources),
                            model_used=_cfg.generation_model,
                            response_length=len(clean_response),
                            estimated_cost_usd=estimate_query_cost_usd(
                                query_text=clean_query,
                                response_text=clean_response,
                                chunks_retrieved=len(sources),
                            ),
                        )
                except Exception:
                    logger.warning("run_query_stream: audit log failed uid=%s", user["uid"], exc_info=True)

            except Exception:
                logger.error("run_query_stream: generation failed uid=%s", user["uid"], exc_info=True)
                yield _event({"error": "Generation failed. Please try again."})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Usage / Audit ─────────────────────────────────────────────────────────
    @app.get("/api/usage", tags=["usage"])
    def get_usage(user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                stats = AuditRepository(conn).get_user_stats(user["uid"])
            return stats
        except Exception as exc:
            logger.error("get_usage uid=%s", user["uid"], exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    # ── Consent ───────────────────────────────────────────────────────────────
    @app.post("/api/consent", tags=["auth"])
    def log_consent(user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                ConsentRepository(conn).log_acceptance(user["uid"])
            return {"ok": True}
        except Exception as exc:
            logger.error("log_consent uid=%s", user["uid"], exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    # ── Cases ─────────────────────────────────────────────────────────────────
    @app.get("/api/cases", tags=["cases"])
    def list_cases(user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                cases = CasesRepository(conn).list_cases(user["uid"])
            return [
                {
                    "case_id":        c.case_id,
                    "case_name":      c.case_name,
                    "description":    c.description,
                    "document_count": c.document_count,
                }
                for c in cases
            ]
        except Exception as exc:
            logger.error("list_cases uid=%s", user["uid"], exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.post("/api/cases", tags=["cases"])
    def create_case(payload: dict, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        name = (payload.get("case_name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="case_name is required.")
        try:
            validated_name = validate_text_input(name, field="Case name", max_length=200)
        except InputGuardError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        try:
            desc_raw = (payload.get("description") or "").strip() or None
            if desc_raw:
                try:
                    desc_raw = validate_text_input(desc_raw, field="Description", max_length=1000)
                except InputGuardError as exc:
                    raise HTTPException(status_code=422, detail=str(exc))
            with open_connection(_cfg.db) as conn:
                # Enforce trial case limit (max 2 cases for non-subscribers)
                try:
                    sub = UserRepository(conn).get_user_subscription(user["uid"])
                    case_count = UserRepository(conn).count_user_cases(user["uid"])
                    if case_count >= 2 and not sub.get("subscription_active"):
                        raise HTTPException(status_code=403, detail="TRIAL_CASE_LIMIT")
                except HTTPException:
                    raise
                except Exception:
                    pass  # fail-open on DB errors

                case_id = CasesRepository(conn).create_case(
                    user_id=user["uid"],
                    case_name=validated_name,
                    description=desc_raw,
                )
                # Auto-create 3 chat tabs (Chat / Timeline / Overview)
                try:
                    PersistentChatRepository(conn).create_case_chats(user["uid"], case_id, validated_name)
                except Exception:
                    logger.warning("create_case: create_case_chats failed case_id=%s", case_id, exc_info=True)

            if _zep and _zep.enabled:
                try:
                    actor = _zep_actor_from_user(user)
                    _zep.add_system_event(
                        _user_events_thread(user["uid"]),
                        f"Created case: '{validated_name}' on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.",
                        actor,
                    )
                except Exception:
                    logger.warning("create_case: Zep event failed uid=%s", user["uid"], exc_info=True)

            return {"case_id": case_id}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("create_case uid=%s", user["uid"], exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.delete("/api/cases/{case_id}", tags=["cases"])
    def delete_case(case_id: str, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                _assert_case_owned_by(conn, case_id, user["uid"])
                _assert_case_not_demo(conn, case_id)
                CasesRepository(conn).delete_case(case_id)
            return {"ok": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("delete_case case_id=%s", case_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.get("/api/cases/{case_id}/documents", tags=["cases"])
    def get_case_documents(case_id: str, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                _assert_case_owned_by(conn, case_id, user["uid"])
                docs = CasesRepository(conn).get_case_documents_info(case_id)
            return docs
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("get_case_documents case_id=%s", case_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.post("/api/cases/{case_id}/documents", tags=["cases"])
    def add_document_to_case(case_id: str, payload: dict, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        doc_id = (payload.get("document_id") or "").strip()
        if not doc_id:
            raise HTTPException(status_code=400, detail="document_id is required.")
        try:
            with open_connection(_cfg.db) as conn:
                _assert_case_owned_by(conn, case_id, user["uid"])
                _assert_case_not_demo(conn, case_id)
                _assert_document_owned_by(conn, doc_id, user["uid"])
                CasesRepository(conn).add_document_to_case(case_id, doc_id)
            return {"ok": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("add_document_to_case case_id=%s doc_id=%s", case_id, doc_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.delete("/api/cases/{case_id}/documents/{doc_id}", tags=["cases"])
    def remove_document_from_case(case_id: str, doc_id: str, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                _assert_case_owned_by(conn, case_id, user["uid"])
                _assert_case_not_demo(conn, case_id)
                CasesRepository(conn).remove_document_from_case(case_id, doc_id)
            return {"ok": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("remove_document_from_case case_id=%s", case_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    # ── Bulk upload ────────────────────────────────────────────────────────────
    @app.post("/api/cases/bulk-upload", tags=["cases"])
    async def bulk_upload_case(
        files: list[UploadFile] = File(...),
        case_name: str = "Untitled Case",
        matter_type: str = "",
        jurisdiction: str = "",
        client_name: str = "",
        notes: str = "",
        user: dict = Depends(require_active_subscription),
    ):
        """Create a case and queue background ingestion jobs for multiple files."""
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")

        uid = user["uid"]
        description_parts = [p for p in [matter_type, jurisdiction, client_name, notes] if p]
        description = " · ".join(description_parts) or None

        BULK_TMP = Path(tempfile.gettempdir()) / "atticus_bulk"
        BULK_TMP.mkdir(parents=True, exist_ok=True)

        try:
            with open_connection(_cfg.db) as conn:
                _safe_case_name = case_name.strip() or "Untitled Case"
                case_id = CasesRepository(conn).create_case(uid, _safe_case_name, description)
                try:
                    PersistentChatRepository(conn).create_case_chats(uid, case_id, _safe_case_name)
                except Exception:
                    logger.warning("bulk_upload_case: create_case_chats failed case_id=%s", case_id, exc_info=True)
        except HTTPException:
            raise
        except Exception:
            logger.error("bulk_upload_case: create_case failed uid=%s", uid, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

        job_ids: list[str] = []
        skipped: list[str] = []
        plan = default_plan()

        for upload in files:
            fname = upload.filename or "upload.pdf"
            if not fname.lower().endswith(_SUPPORTED_UPLOAD_EXTENSIONS):
                skipped.append(fname)
                continue

            try:
                contents = await upload.read()
            except Exception:
                skipped.append(fname)
                continue

            if len(contents) > max_upload_bytes(plan):
                skipped.append(fname)
                continue

            safe_name = fname.replace("/", "_").replace("\\", "_")
            tmp_path = BULK_TMP / f"{_uuid.uuid4().hex}_{safe_name}"
            try:
                tmp_path.write_bytes(contents)
            except Exception:
                skipped.append(fname)
                continue

            try:
                with open_connection(_cfg.db) as conn:
                    job_id = JobsRepository(conn).enqueue(
                        job_type="ingest_pdf",
                        user_id=uid,
                        case_id=case_id,
                        payload={"file_path": str(tmp_path), "file_name": fname, "case_id": case_id, "user_id": uid},
                    )
                # Dispatch to Celery worker (persistent, survives restarts)
                from insightlens.jobs.tasks import ingest_pdf_task
                ingest_pdf_task.delay(job_id, str(tmp_path), case_id, uid, fname)
                job_ids.append(job_id)
            except Exception:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                skipped.append(fname)

        if not job_ids:
            try:
                with open_connection(_cfg.db) as conn:
                    CasesRepository(conn).delete_case(case_id)
            except Exception:
                logger.warning("bulk_upload_case: failed to roll back empty case=%s", case_id, exc_info=True)
            detail = "No supported files could be queued. Upload PDF or PPTX files."
            if skipped:
                detail += " " + "; ".join(skipped[:3])
            raise HTTPException(status_code=422, detail=detail)

        return {"case_id": case_id, "job_ids": job_ids, "total_files": len(job_ids), "skipped": skipped}

    @app.get("/api/cases/{case_id}/jobs", tags=["cases"])
    def get_case_jobs(case_id: str, user: dict = Depends(require_user)):
        """Return aggregate job status for all ingest_pdf jobs on a case."""
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                _assert_case_owned_by(conn, case_id, user["uid"])
                jobs = JobsRepository(conn).get_jobs_for_case(case_id, job_type="ingest_pdf")

            status_counts: dict[str, int] = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
            errors: list[str] = []
            for j in jobs:
                status_counts[j.status] = status_counts.get(j.status, 0) + 1
                if j.status == "failed" and j.error:
                    fname = j.payload.get("file_name", j.job_id)
                    errors.append(f"{fname}: {j.error[:200]}")

            return {
                "total":       len(jobs),
                "completed":   status_counts.get("completed", 0),
                "failed":      status_counts.get("failed", 0),
                "in_progress": status_counts.get("running", 0),
                "pending":     status_counts.get("queued", 0),
                "errors":      errors,
            }
        except HTTPException:
            raise
        except Exception:
            logger.error("get_case_jobs case_id=%s uid=%s", case_id, user["uid"], exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    # ── Documents ─────────────────────────────────────────────────────────────
    @app.get("/api/documents", tags=["documents"])
    def list_documents(user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                repo = ChunkRepository(conn)
                user_docs = repo.list_documents(user_id=user["uid"])
                sys_docs  = repo.list_documents(user_id=None)

            def _serialize(docs, is_system: bool):
                return [
                    {
                        "document_id":   d.document_id,
                        "file_name":     d.file_name,
                        "company":       d.company,
                        "document_type": getattr(d, "document_type", None),
                        "page_count":    d.page_count,
                        "version_label": getattr(d, "version_label", None),
                        "is_system":     is_system,
                    }
                    for d in docs
                ]

            return _serialize(user_docs, False) + _serialize(sys_docs, True)
        except Exception as exc:
            logger.error("list_documents uid=%s", user["uid"], exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.delete("/api/documents/{doc_id}", tags=["documents"])
    def delete_document(doc_id: str, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                _assert_document_not_demo(conn, doc_id)
                ChunkRepository(conn).delete_document(doc_id, user["uid"])
            return {"ok": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("delete_document doc_id=%s uid=%s", doc_id, user["uid"], exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.post("/api/upload", tags=["documents"])
    async def upload_document(
        file: UploadFile = File(...),
        user: dict = Depends(require_user),
    ):
        """Upload and ingest a supported document.

        Rate-limited per user (upload count) and file-size capped.
        Always associates the document with the authenticated user — never with
        the demo corpus (which has user_id = NULL and is_demo = TRUE).
        """
        if not all([_cfg, _embedder]):
            raise HTTPException(status_code=503, detail="Service not ready.")

        if not file.filename or not file.filename.lower().endswith(_SUPPORTED_UPLOAD_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Only PDF and PPTX files are supported.")

        # Read the file first so we know the size for rate-limit checks
        contents = await file.read()

        # ── Upload rate limit ─────────────────────────────────────────────────
        plan = default_plan()
        try:
            with open_connection(_cfg.db) as conn:
                uploads_month = UsageRepository(conn).count_uploads_this_month(user["uid"])
        except Exception:
            uploads_month = 0

        check_upload_rate_limit(
            user["uid"],
            uploads_this_month=uploads_month,
            monthly_upload_limit=plan.monthly_upload_limit,
            file_size_bytes=len(contents),
            max_upload_bytes=max_upload_bytes(plan),
        )

        # ── Ingest ────────────────────────────────────────────────────────────
        from insightlens.ingestion.ingest_service import IngestService
        import tempfile

        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp) / (file.filename or "upload.pdf")
                tmp_path.write_bytes(contents)
                svc = IngestService(cfg=_cfg, embedder=_embedder)
                # user_id is always set from the JWT — never None/demo
                result = svc.ingest(tmp_path, user_id=user["uid"])

            if result.error and not result.skipped:
                logger.error(
                    "upload_document: ingest error uid=%s file=%s err=%s",
                    user["uid"], file.filename, result.error,
                )
                raise HTTPException(status_code=422, detail=result.error)

            try:
                with open_connection(_cfg.db) as conn:
                    UsageRepository(conn).log_upload(
                        user_id=user["uid"],
                        document_id=result.document_id,
                        file_name=result.file_name,
                        file_size_bytes=len(contents),
                        page_count=result.page_count,
                        chunks_inserted=result.chunks_inserted,
                        estimated_cost_usd=result.estimated_cost_usd,
                    )
            except Exception:
                logger.warning("upload_document: usage log failed uid=%s", user["uid"], exc_info=True)

            if _zep and _zep.enabled:
                try:
                    actor = _zep_actor_from_user(user)
                    _zep.add_system_event(
                        _user_events_thread(user["uid"]),
                        f"Uploaded document: '{result.file_name}' ({result.page_count} pages) "
                        f"on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.",
                        actor,
                    )
                except Exception:
                    logger.warning("upload_document: Zep event failed uid=%s", user["uid"], exc_info=True)

            return {
                "document_id":    result.document_id,
                "file_name":      result.file_name,
                "page_count":     result.page_count,
                "chunks_inserted": result.chunks_inserted,
                "skipped":        result.skipped,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("upload_document: error uid=%s", user["uid"], exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    # ── Discussion (disabled) ─────────────────────────────────────────────────
    _DISCUSSION_UNAVAILABLE = HTTPException(
        status_code=503, detail="Discussion is not available yet"
    )

    @app.get("/api/discussion", tags=["discussion"])
    def list_posts(user: dict = Depends(require_user)):
        raise _DISCUSSION_UNAVAILABLE

    @app.post("/api/discussion", tags=["discussion"])
    def create_post(payload: dict, user: dict = Depends(require_user)):
        raise _DISCUSSION_UNAVAILABLE

    @app.delete("/api/discussion/{post_id}", tags=["discussion"])
    def delete_post(post_id: str, user: dict = Depends(require_user)):
        raise _DISCUSSION_UNAVAILABLE

    # ── Organizations (input-guarded) ─────────────────────────────────────────
    @app.get("/api/orgs", tags=["orgs"])
    def list_orgs(user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                orgs = OrgRepository(conn).list_user_orgs(user["uid"])
            return [
                {"org_id": o.org_id, "org_name": o.org_name, "owner_id": o.owner_id}
                for o in orgs
            ]
        except Exception as exc:
            logger.error("list_orgs uid=%s", user["uid"], exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.post("/api/orgs", tags=["orgs"])
    def create_org(payload: dict, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        name = (payload.get("org_name") or "").strip()
        try:
            name = validate_text_input(name, field="Organization name", max_length=200)
        except InputGuardError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        try:
            with open_connection(_cfg.db) as conn:
                OrgRepository(conn).create_org(user["uid"], name)
            return {"ok": True}
        except Exception as exc:
            logger.error("create_org uid=%s", user["uid"], exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.patch("/api/orgs/{org_id}", tags=["orgs"])
    def rename_org(org_id: str, payload: dict, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        name = (payload.get("org_name") or "").strip()
        try:
            name = validate_text_input(name, field="Organization name", max_length=200)
        except InputGuardError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        try:
            with open_connection(_cfg.db) as conn:
                _assert_org_role(conn, org_id, user["uid"], min_role="owner")
                OrgRepository(conn).rename_org(org_id, name)
            return {"ok": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("rename_org org_id=%s", org_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.get("/api/orgs/{org_id}/members", tags=["orgs"])
    def list_members(org_id: str, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                _assert_org_role(conn, org_id, user["uid"], min_role="member")
                members = OrgRepository(conn).list_members(org_id)
            return [
                {
                    "user_id":   m["user_id"],
                    "role":      m["role"],
                    "joined_at": m["joined_at"].isoformat() if m.get("joined_at") else None,
                }
                for m in members
            ]
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("list_members org_id=%s", org_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.post("/api/orgs/{org_id}/members", tags=["orgs"])
    def add_member(org_id: str, payload: dict, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        uid  = (payload.get("user_id") or "").strip()
        role = (payload.get("role") or "member").strip()
        if not uid:
            raise HTTPException(status_code=400, detail="user_id is required.")
        try:
            with open_connection(_cfg.db) as conn:
                _assert_org_role(conn, org_id, user["uid"], min_role="admin")
                OrgRepository(conn).add_member(org_id, uid, role)
            return {"ok": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("add_member org_id=%s", org_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.delete("/api/orgs/{org_id}/members/{member_id}", tags=["orgs"])
    def remove_member(org_id: str, member_id: str, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                _assert_org_role(conn, org_id, user["uid"], min_role="admin")
                OrgRepository(conn).remove_member(org_id, member_id)
            return {"ok": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("remove_member org_id=%s", org_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    # ── Analytics ─────────────────────────────────────────────────────────────
    @app.get("/api/analytics", tags=["analytics"])
    def get_analytics(days: int = 30, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        days = max(1, min(days, 365))  # clamp to sane range
        try:
            with open_connection(_cfg.db) as conn:
                audit         = AuditRepository(conn)
                stats         = audit.get_user_stats(user["uid"])
                daily         = audit.get_daily_counts(user["uid"], days=days)
                by_page       = audit.get_page_breakdown(user["uid"], days=days)
                recent        = audit.get_recent_queries(user["uid"], limit=25)
                usage_repo    = UsageRepository(conn)
                uploads_month = usage_repo.count_uploads_this_month(user["uid"])
                upload_cost   = usage_repo.estimated_upload_cost_this_month(user["uid"])
            return {
                "stats":                stats,
                "daily":                daily,
                "by_page":              by_page,
                "recent":               recent,
                "uploads_this_month":   uploads_month,
                "upload_cost_this_month": float(upload_cost),
            }
        except Exception as exc:
            logger.error("get_analytics uid=%s", user["uid"], exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    # ── Subscription status ───────────────────────────────────────────────────
    @app.get("/api/subscription/status", tags=["auth"])
    def get_subscription_status(user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                sub = UserRepository(conn).get_user_subscription(user["uid"])
                case_count = UserRepository(conn).count_user_cases(user["uid"])
        except Exception:
            logger.error("get_subscription_status uid=%s", user["uid"], exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

        trial_expires_at = sub.get("trial_expires_at")
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        seconds_remaining = 0
        if trial_expires_at:
            exp = trial_expires_at
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            delta = exp - now
            seconds_remaining = max(0, int(delta.total_seconds()))

        days_remaining  = seconds_remaining // 86400
        hours_remaining = (seconds_remaining % 86400) // 3600

        return {
            "plan":               sub.get("plan", "trial"),
            "trial_expires_at":   trial_expires_at.isoformat() if trial_expires_at else None,
            "days_remaining":     days_remaining,
            "hours_remaining":    hours_remaining,
            "is_trial_expired":   sub.get("is_trial_expired", False),
            "subscription_active": sub.get("subscription_active", False),
            "case_count":         case_count,
            "trial_case_limit":   2,
        }

    # ── Case chats / overview / timeline ─────────────────────────────────────
    @app.get("/api/cases/{case_id}/chats", tags=["cases"])
    def get_case_chats(case_id: str, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                _assert_case_owned_by(conn, case_id, user["uid"])
                chats = PersistentChatRepository(conn).list_case_chats(case_id, user["uid"])
            return chats
        except HTTPException:
            raise
        except Exception:
            logger.error("get_case_chats case_id=%s", case_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.get("/api/cases/{case_id}/overview", tags=["cases"])
    def get_case_overview(case_id: str, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                _assert_case_owned_by(conn, case_id, user["uid"])
                cur = conn.cursor()
                cur.execute(
                    "SELECT summary, parties, key_issues, jurisdiction, matter_type, generated_at "
                    "FROM case_overviews WHERE case_id = %s",
                    (case_id,),
                )
                row = cur.fetchone()
                cur.close()
            if row is None:
                return {"pending": True}
            return {
                "summary":      row[0],
                "parties":      row[1] if row[1] else [],
                "key_issues":   row[2] if row[2] else [],
                "jurisdiction": row[3],
                "matter_type":  row[4],
                "generated_at": row[5].isoformat() if row[5] else None,
                "pending":      False,
            }
        except HTTPException:
            raise
        except Exception:
            logger.error("get_case_overview case_id=%s", case_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    @app.get("/api/cases/{case_id}/timeline", tags=["cases"])
    def get_case_timeline(case_id: str, user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        try:
            with open_connection(_cfg.db) as conn:
                _assert_case_owned_by(conn, case_id, user["uid"])
                cur = conn.cursor()
                cur.execute(
                    "SELECT events, generated_at FROM case_timelines WHERE case_id = %s",
                    (case_id,),
                )
                row = cur.fetchone()
                cur.close()
            if row is None:
                return {"pending": True}
            return {
                "events":       row[0] if row[0] else [],
                "generated_at": row[1].isoformat() if row[1] else None,
                "pending":      False,
            }
        except HTTPException:
            raise
        except Exception:
            logger.error("get_case_timeline case_id=%s", case_id, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    # ── Profile / GDPR delete ─────────────────────────────────────────────────
    @app.delete("/api/profile", tags=["auth"])
    def delete_profile(user: dict = Depends(require_user)):
        if not _cfg:
            raise HTTPException(status_code=503, detail="Service not ready.")
        uid = user["uid"]
        try:
            with open_connection(_cfg.db) as conn:
                AuditRepository(conn).delete_user_logs(uid)
                PersistentChatRepository(conn).delete_user_chats(uid)
                CasesRepository(conn).delete_user_cases(uid)
                ChunkRepository(conn).delete_user_documents(uid)
                ConsentRepository(conn).delete_user_consents(uid)
            logger.info("GDPR delete completed uid=%s", uid)
            return {"ok": True}
        except Exception as exc:
            logger.error("delete_profile uid=%s", uid, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error.")

    # ── Demo portal router ────────────────────────────────────────────────────
    try:
        from backend.demo_router import router as demo_router, seed_demo_users
    except ModuleNotFoundError:
        from demo_router import router as demo_router, seed_demo_users
    app.include_router(demo_router)
    if _cfg:
        try:
            seed_demo_users(_cfg)
        except Exception as _seed_exc:
            logger.warning("seed_demo_users failed at startup: %s", _seed_exc, exc_info=True)
    else:
        # Previously silent: if service bootstrap failed above, _cfg is None
        # and seed_demo_users is never even called — meaning demo access
        # codes never get hashed into demo.sessions, with zero log evidence
        # of why. Now it's explicit, and points back to the bootstrap error
        # logged above.
        logger.warning(
            "seed_demo_users skipped at startup because service config failed to "
            "load (_cfg is None) — see the 'Config error at startup' / "
            "'Unexpected error bootstrapping services' log line above for the cause."
        )

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
