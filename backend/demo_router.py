"""Demo portal API — access-code auth, Epstein-scoped chat, cost tracking."""
from __future__ import annotations

import logging
import os
import re
import time
import tempfile
import uuid as _uuid
import json as _json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import bcrypt
import jwt
from fastapi import APIRouter, Header, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel
from insightlens.demo.epstein_people import epstein_people_context_text

logger = logging.getLogger("atticus.demo")

router = APIRouter(prefix="/api/demo", tags=["demo"])
_REPO_ROOT = Path(__file__).parents[1]
_CONFIDENCE_RE = re.compile(r"<CONFIDENCE>(.*?)</CONFIDENCE>", re.DOTALL)
_CONFIDENCE_RATINGS = {5: "High", 4: "Good", 3: "Moderate", 2: "Low", 1: "Unreliable"}
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

# ── Cost constants ────────────────────────────────────────────────────────────
COST_PER_1M: dict[str, dict[str, float]] = {
    "claude-haiku-3-5":         {"input": 0.80,  "output": 4.00},
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-5":        {"input": 3.00,  "output": 15.00},
    "claude-sonnet-4-6":        {"input": 3.00,  "output": 15.00},
    "claude-opus-4-8":          {"input": 15.00, "output": 75.00},
    "voyage-law-2":             {"input": 0.12,  "output": 0.0},
    "all-MiniLM-L6-v2":        {"input": 0.0,   "output": 0.0},
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_PER_1M.get(model, {"input": 3.00, "output": 15.00})
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


# ── JWT helpers ───────────────────────────────────────────────────────────────
def _secret_key() -> str:
    # SECURITY: previously fell back to the hardcoded literal "demo-secret-key"
    # if neither SECRET_KEY nor ADMIN_API_KEY was set. Anyone who read this
    # source (or guessed it) could forge a valid demo JWT for any user_slug.
    # Fail loudly instead of silently signing tokens with a public string.
    key = os.getenv("SECRET_KEY") or os.getenv("ADMIN_API_KEY")
    if not key:
        raise RuntimeError(
            "SECRET_KEY (or ADMIN_API_KEY) must be set to sign demo session tokens. "
            "Refusing to start with a hardcoded fallback secret."
        )
    return key


def _sign_token(user_slug: str) -> str:
    payload = {
        "user_slug": user_slug,
        "exp": datetime.now(timezone.utc) + timedelta(hours=8),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _secret_key(), algorithm="HS256")


def _verify_token(token: str) -> str:
    try:
        payload = jwt.decode(token, _secret_key(), algorithms=["HS256"])
        return payload["user_slug"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Demo session expired.")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid demo session.")


def _get_demo_user(authorization: str | None = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Demo session required.")
    return _verify_token(authorization[7:])


def _demo_user_id(user_slug: str) -> str:
    return f"demo:{user_slug}"


def _assert_demo_case_owned(conn, case_id: str, user_slug: str) -> None:
    cur = conn.cursor()
    try:
        cur.execute("SELECT user_id FROM cases WHERE case_id = %s", (case_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Demo case not found.")
        if row[0] != _demo_user_id(user_slug):
            raise HTTPException(status_code=403, detail="Demo case does not belong to this session.")
    finally:
        cur.close()


# ── Rate limiter for demo (20 queries/hour) ───────────────────────────────────
_DEMO_RATE_LIMIT = 20
_DEMO_WINDOW_S   = 3600


def _check_demo_rate(user_slug: str) -> None:
    try:
        try:
            from backend.rate_limiter import _redis_check
        except ModuleNotFoundError:
            from rate_limiter import _redis_check
        if not _redis_check(f"demo_rl:{user_slug}", _DEMO_RATE_LIMIT, _DEMO_WINDOW_S):
            raise HTTPException(status_code=429, detail="Demo rate limit reached (20 queries/hour).")
    except HTTPException:
        raise
    except Exception:
        pass  # fail-open if Redis unavailable


# ── Pydantic models ───────────────────────────────────────────────────────────
class AuthRequest(BaseModel):
    access_code: str


class DemoQueryRequest(BaseModel):
    question: str
    chat_history: list[dict] = []
    case_id: str | None = None


_FALLBACK_TIMELINE_EVENTS: list[dict[str, str]] = [
    {
        "date": "1991-1994",
        "title": "Early allegations surface",
        "description": (
            "Multiple women later testify that Epstein began abuse in the early "
            "1990s at his Palm Beach estate and New York townhouse."
        ),
    },
    {
        "date": "1997",
        "title": "New York townhouse acquired",
        "description": (
            "Epstein takes ownership of 9 East 71st Street, one of the largest "
            "private residences in Manhattan."
        ),
    },
    {
        "date": "2005",
        "title": "Palm Beach Police investigation begins",
        "description": (
            "A parent reports abuse allegations to Palm Beach Police, leading "
            "Detective Joseph Recarey to build a case with multiple identified victims."
        ),
    },
    {
        "date": "2007-2008",
        "title": "Controversial federal plea deal",
        "description": (
            "Federal prosecutors negotiate a non-prosecution agreement. Epstein "
            "pleads guilty to two Florida state charges and serves 13 months."
        ),
    },
    {
        "date": "2019 Jul 6",
        "title": "Arrested at Teterboro Airport",
        "description": (
            "Epstein is arrested after returning from France. Federal prosecutors "
            "later unseal sex-trafficking charges in New York."
        ),
    },
    {
        "date": "2019 Jul 8",
        "title": "Federal indictment unsealed",
        "description": (
            "The SDNY indictment alleges sex trafficking of minors and conspiracy "
            "involving conduct in Manhattan and Palm Beach."
        ),
    },
    {
        "date": "2019 Aug 10",
        "title": "Found dead at MCC New York",
        "description": (
            "Epstein is found unresponsive in federal custody after being taken "
            "off suicide watch. Monitoring failures later draw scrutiny."
        ),
    },
    {
        "date": "2021 Dec 29",
        "title": "Ghislaine Maxwell convicted",
        "description": (
            "A federal jury convicts Maxwell on five of six counts after testimony "
            "from survivors and witnesses."
        ),
    },
    {
        "date": "2022 Jun 28",
        "title": "Maxwell sentenced to 20 years",
        "description": (
            "Judge Alison Nathan sentences Maxwell to 20 years in federal prison."
        ),
    },
    {
        "date": "2023-2024",
        "title": "Civil settlements and document releases",
        "description": (
            "Major financial institutions settle related claims, and federal courts "
            "release additional records from civil litigation."
        ),
    },
]


_FALLBACK_OVERVIEW: dict[str, Any] = {
    "summary": (
        "This demo workspace covers the public-record Epstein matter, including "
        "the Palm Beach investigation, federal charging decisions, civil litigation, "
        "Ghislaine Maxwell proceedings, institutional settlements, and later record "
        "unsealing. It is intended as a shared baseline for all demo users."
    ),
    "parties": [
        {"role": "Defendant", "name": "Jeffrey Epstein"},
        {"role": "Associate", "name": "Ghislaine Maxwell"},
        {"role": "Survivor", "name": "Virginia Giuffre"},
        {"role": "Prosecutor", "name": "Alexander Acosta"},
        {"role": "Court", "name": "SDNY and related federal courts"},
        {"role": "Investigator", "name": "Palm Beach Police Department"},
    ],
    "key_issues": [
        "Whether early investigations and charging decisions adequately protected victims.",
        "The scope of alleged trafficking conduct across jurisdictions.",
        "The effect and legality of the 2007-2008 non-prosecution agreement.",
        "Civil accountability for individuals, the estate, and financial institutions.",
        "Public access to sealed records and witness-related filings.",
    ],
    "jurisdiction": "Florida, New York, USVI, and related federal proceedings",
    "matter_type": "Public-record criminal, civil, and investigative corpus",
}


_DEMO_PARTY_CONTEXT: dict[str, str] = {
    "Jeffrey Epstein": "Financier and convicted sex offender. Born January 20, 1953; died August 10, 2019, at MCC New York.",
    "Ghislaine Maxwell": "British socialite and Epstein associate. Convicted in 2021 and sentenced to 20 years in federal prison.",
    "Virginia Giuffre": "Primary survivor-witness in the public Epstein record. She filed multiple civil suits and reached a reported settlement with Prince Andrew in 2022.",
    "Alexander Acosta": "Former U.S. Attorney for the Southern District of Florida who negotiated the 2007-2008 non-prosecution agreement.",
    "SDNY and related federal courts": "Federal courts and prosecutors tied to the New York criminal case, Maxwell prosecution, civil litigation, and later record unsealing.",
    "Palm Beach Police Department": "Local law-enforcement agency whose 2005 investigation identified multiple alleged victims.",
}


_TIMELINE_IMAGE_STOP_WORDS = {
    "about", "after", "and", "are", "case", "civil", "demo", "event", "events",
    "federal", "from", "has", "have", "into", "later", "matter", "multiple",
    "public", "record", "records", "related", "the", "this", "was", "were",
    "with",
}


def _timeline_image_terms(event: dict[str, Any]) -> list[str]:
    text = f"{event.get('date', '')} {event.get('title', '')} {event.get('description', '')}"
    terms = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'-]{2,}", text.lower())
    return [
        term
        for term in dict.fromkeys(terms)
        if term not in _TIMELINE_IMAGE_STOP_WORDS
    ][:10]


def _demo_image_file_exists(record: Any) -> bool:
    try:
        raw = Path(record.file_path)
        path = raw if raw.is_absolute() else _REPO_ROOT / raw
        resolved = path.resolve()
        images_root = (_REPO_ROOT / "data" / "images").resolve()
        return images_root in resolved.parents and resolved.exists()
    except Exception:
        return False


def _demo_image_payload(record: Any, source: str) -> dict[str, Any]:
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


def _sample_demo_images(conn: Any, limit: int) -> list[Any]:
    from insightlens.storage.image_repository import ImageRecord

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


def _attach_timeline_images(conn: Any, events: list[Any]) -> list[dict[str, Any]]:
    from insightlens.storage.image_repository import ImageRepository

    repo = ImageRepository(conn)
    enriched: list[dict[str, Any]] = []
    used: set[str] = set()
    samples = [image for image in _sample_demo_images(conn, len(events)) if _demo_image_file_exists(image)]

    for index, raw_event in enumerate(events):
        event = dict(raw_event) if isinstance(raw_event, dict) else {"title": str(raw_event)}
        if event.get("image"):
            enriched.append(event)
            continue

        matched = None
        try:
            for image in repo.search_by_text_terms(
                _timeline_image_terms(event),
                top_k=6,
                system_only=True,
            ):
                if image.image_id in used or not _demo_image_file_exists(image):
                    continue
                matched = image
                break
        except Exception:
            logger.warning("demo_timeline: image metadata match failed", exc_info=True)

        if matched is None:
            for image in samples:
                if image.image_id not in used:
                    matched = image
                    break

        if matched is not None:
            used.add(matched.image_id)
            event["image"] = _demo_image_payload(matched, "Closest demo image match")

        enriched.append(event)

    return enriched


def _demo_overview_source_chunk() -> Any:
    from insightlens.storage.chunk_repository import RetrievedChunk

    party_lines = [
        f"- {party['name']}: {party['role']}. {_DEMO_PARTY_CONTEXT.get(party['name'], '')}".rstrip()
        for party in _FALLBACK_OVERVIEW["parties"]
    ]
    issue_lines = [f"- {issue}" for issue in _FALLBACK_OVERVIEW["key_issues"]]
    timeline_lines = [
        f"- {event['date']}: {event['title']} — {event['description']}"
        for event in _FALLBACK_TIMELINE_EVENTS
    ]
    text = "\n".join(
        [
            "Shared demo overview for the public-record Epstein matter.",
            "",
            f"Summary: {_FALLBACK_OVERVIEW['summary']}",
            f"Matter type: {_FALLBACK_OVERVIEW['matter_type']}",
            f"Jurisdiction: {_FALLBACK_OVERVIEW['jurisdiction']}",
            "",
            "Key parties:",
            *party_lines,
            "",
            "Key issues:",
            *issue_lines,
            "",
            "Shared timeline:",
            *timeline_lines,
        ]
    )
    return RetrievedChunk(
        chunk_id="demo-overview-timeline",
        document_id="demo-overview-timeline",
        file_name="Shared demo overview and timeline",
        company="Epstein",
        version_label="demo",
        page_number=1,
        chunk_text=text,
        similarity=1.0,
        section_header="Overview",
        chunk_type="overview",
        document_type="demo-summary",
        source_type="demo_summary",
    )


def _epstein_people_source_chunk() -> Any:
    from insightlens.storage.chunk_repository import RetrievedChunk

    return RetrievedChunk(
        chunk_id="demo-epstein-public-people-index",
        document_id="demo-epstein-public-people-index",
        file_name="Public reference people index for Epstein matter",
        company="Epstein",
        version_label="public-reference",
        page_number=1,
        chunk_text=epstein_people_context_text(),
        similarity=1.0,
        section_header="People index",
        chunk_type="public_reference",
        document_type="public_reference",
        source_type="public_context",
    )


def _extract_confidence(text: str) -> tuple[str, dict[str, Any] | None]:
    match = _CONFIDENCE_RE.search(text)
    if not match:
        return text, None
    clean = _CONFIDENCE_RE.sub("", text).rstrip()
    try:
        raw = _json.loads(match.group(1).strip())
        score = max(1, min(5, int(raw.get("score", 3))))
        return clean, {
            "score": score,
            "rating": _CONFIDENCE_RATINGS.get(score, "Moderate"),
            "rationale": str(raw.get("rationale", "")),
        }
    except Exception:
        return clean, None


def _cap_secondary_only_confidence(sources: list[Any], confidence: dict[str, Any] | None) -> dict[str, Any] | None:
    if not confidence:
        return None
    has_document_evidence = any(
        (getattr(source, "source_type", "document") or "document") == "document"
        for source in sources
    )
    if has_document_evidence or int(confidence.get("score", 3)) <= 3:
        return confidence
    confidence = dict(confidence)
    confidence["score"] = 3
    confidence["rating"] = _CONFIDENCE_RATINGS[3]
    confidence["rationale"] = (
        f"{confidence.get('rationale', '').rstrip()} "
        "Confidence is capped because the answer relies on secondary demo summary context rather than primary document evidence."
    ).strip()
    return confidence


def _demo_evidence_profile(query: str, sources: list[Any]) -> dict[str, Any]:
    primary = [
        source for source in sources
        if (getattr(source, "source_type", "document") or "document") == "document"
    ]
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
        "unique_primary_docs": len(primary_docs),
        "unique_primary_pages": len(primary_pages),
        "secondary_count": len(sources) - len(primary),
    }


def _cap_confidence_for_coverage(query: str, sources: list[Any], confidence: dict[str, Any] | None) -> dict[str, Any] | None:
    confidence = _cap_secondary_only_confidence(sources, confidence)
    if not confidence:
        return None
    profile = _demo_evidence_profile(query, sources)
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
    score = max(1, min(5, int(confidence.get("score", 3))))
    if score <= cap:
        return confidence
    capped = dict(confidence)
    capped["score"] = cap
    capped["rating"] = _CONFIDENCE_RATINGS[cap]
    capped["rationale"] = (
        f"{capped.get('rationale', '').rstrip()} "
        f"Confidence capped at {cap}/5 because {'; '.join(dict.fromkeys(reasons))}."
    ).strip()
    return capped


def _scope_note_for_coverage(query: str, sources: list[Any]) -> str | None:
    profile = _demo_evidence_profile(query, sources)
    if not profile["is_broad_legal"] or profile["unique_primary_docs"] >= 3:
        return None
    if profile["primary_count"] == 0:
        return (
            "**Scope note:** No primary document evidence was retrieved for this broad legal question. "
            "The answer below is based on secondary demo context and should not be treated as a complete legal assessment.\n\n"
        )
    return (
        f"**Scope note:** This is a limited synthesis based on {profile['unique_primary_docs']} "
        f"primary document(s) and {profile['unique_primary_pages']} page-level source(s) retrieved for this question. "
        "It should not be treated as a complete issue list until the underlying filings and related records are reviewed.\n\n"
    )


def _ensure_lawyer_followups(answer: str, query: str, sources: list[Any]) -> str:
    if re.search(r"actionable follow[- ]?up searches", answer, re.IGNORECASE) or re.search(
        r"(?m)^\s*-\s*(?:Search for|Search corpus for|Pull|Verify):", answer
    ):
        return answer
    if not _demo_evidence_profile(query, sources)["is_broad_legal"]:
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


def _ensure_workspace_note(answer: str, query: str, sources: list[Any]) -> str:
    if re.search(r"workspace note|workspace/source universe|source universe", answer, re.IGNORECASE):
        return answer
    if not _demo_evidence_profile(query, sources)["is_broad_legal"]:
        return answer
    return f"{answer.rstrip()}\n\n## Workspace note\n{_workspace_note(query, sources)}"


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
        query_phrases = [" ".join(terms[i : i + 2]) for i in range(len(terms) - 1)]
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


def _infer_jurisdiction(source: Any) -> str | None:
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


def _citation_label(source: Any) -> str:
    source_type = getattr(source, "source_type", "document") or "document"
    page = getattr(source, "page_number", None)
    page_text = f", at {page}" if page else ""
    if source_type == "demo_summary":
        return f"Atticus demo Epstein matter summary{page_text}"
    if source_type == "case_overview":
        return f"AI-generated case overview{page_text}"
    if source_type == "case_timeline":
        return f"AI-generated case timeline{page_text}"
    return f"{_display_source_name(getattr(source, 'file_name', ''))}{page_text}"


def _workspace_note(query: str, sources: list[Any]) -> str:
    profile = _demo_evidence_profile(query, sources)
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


def _dedupe_comparable_sources(sources: list[Any]) -> list[Any]:
    seen: set[tuple] = set()
    deduped: list[Any] = []
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


def _demo_case_context_chunks(conn, case_id: str, user_slug: str) -> list[Any]:
    """Return cached or deterministic case context for uploaded demo cases."""
    from insightlens.analysis.case_intelligence import build_case_overview, build_case_timeline
    from insightlens.storage.chunk_repository import ChunkRepository, RetrievedChunk

    uid = _demo_user_id(user_slug)
    _assert_demo_case_owned(conn, case_id, user_slug)
    chunks: list[Any] = []
    cur = conn.cursor()
    case_name = "Uploaded Case"
    try:
        cur.execute("SELECT case_name FROM cases WHERE case_id = %s", (case_id,))
        case_row = cur.fetchone()
        if case_row and case_row[0]:
            case_name = case_row[0]

        cur.execute(
            """SELECT summary, parties, key_issues, jurisdiction, matter_type
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
            "SELECT events FROM case_timelines WHERE case_id = %s",
            (case_id,),
        )
        row = cur.fetchone()
        if row and row[0]:
            event_lines = [
                f"- {event.get('date', 'Unknown date')}: {event.get('title', 'Untitled')} — {event.get('description', '')}"
                for event in row[0]
                if isinstance(event, dict)
            ]
            if event_lines:
                chunks.append(
                    RetrievedChunk(
                        chunk_id=f"{case_id}-timeline",
                        document_id=case_id,
                        file_name="AI-generated case timeline",
                        company="Case",
                        version_label="generated",
                        page_number=1,
                        chunk_text="\n".join(
                            [
                                "AI-generated case timeline. Treat as secondary context, not primary evidence.",
                                *event_lines,
                            ]
                        ),
                        similarity=0.0,
                        section_header="Timeline",
                        chunk_type="timeline",
                        document_type="case-timeline",
                        source_type="case_timeline",
                    )
                )
    finally:
        cur.close()

    if chunks:
        return chunks

    primary_chunks = ChunkRepository(conn).get_chunks_for_case(case_id, uid, limit=40)
    if not primary_chunks:
        return []
    overview = build_case_overview(primary_chunks, case_name)
    timeline = build_case_timeline(primary_chunks, limit=10)
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
    if timeline:
        chunks.append(
            RetrievedChunk(
                chunk_id=f"{case_id}-deterministic-timeline",
                document_id=case_id,
                file_name="Deterministic case timeline",
                company="Case",
                version_label="generated",
                page_number=1,
                chunk_text="\n".join(
                    [
                        "Deterministic case timeline generated from uploaded chunks. Treat as secondary context, not primary evidence.",
                        *[
                            f"- {event.get('date', 'Unknown date')}: {event.get('title', 'Untitled')} — {event.get('description', '')}"
                            for event in timeline
                        ],
                    ]
                ),
                similarity=0.0,
                section_header="Timeline",
                chunk_type="timeline",
                document_type="case-timeline",
                source_type="case_timeline",
            )
        )
    return chunks


def _store_demo_case_intelligence(conn, case_id: str, user_slug: str) -> None:
    """Persist a no-LLM overview/timeline for demo uploads."""
    import json
    from insightlens.analysis.case_intelligence import build_case_overview, build_case_timeline
    from insightlens.storage.chunk_repository import ChunkRepository

    uid = _demo_user_id(user_slug)
    cur = conn.cursor()
    try:
        cur.execute("SELECT case_name FROM cases WHERE case_id = %s", (case_id,))
        row = cur.fetchone()
        case_name = row[0] if row and row[0] else "Uploaded Case"
        chunks = ChunkRepository(conn).get_chunks_for_case(case_id, uid, limit=80)
        if not chunks:
            return
        overview = build_case_overview(chunks, case_name)
        timeline = build_case_timeline(chunks, limit=16)
        cur.execute(
            """INSERT INTO case_overviews
                   (case_id, user_id, summary, parties, key_issues, jurisdiction, matter_type)
               VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
               ON CONFLICT (case_id) DO UPDATE SET
                   summary=EXCLUDED.summary,
                   parties=EXCLUDED.parties,
                   key_issues=EXCLUDED.key_issues,
                   jurisdiction=EXCLUDED.jurisdiction,
                   matter_type=EXCLUDED.matter_type,
                   generated_at=NOW()""",
            (
                case_id,
                uid,
                overview.get("summary", ""),
                json.dumps(overview.get("parties", [])),
                json.dumps(overview.get("key_issues", [])),
                overview.get("jurisdiction"),
                overview.get("matter_type"),
            ),
        )
        cur.execute(
            """INSERT INTO case_timelines (case_id, user_id, events)
               VALUES (%s, %s, %s::jsonb)
               ON CONFLICT (case_id) DO UPDATE SET events=EXCLUDED.events, generated_at=NOW()""",
            (case_id, uid, json.dumps(timeline)),
        )
        cur.execute(
            "UPDATE cases SET overview_generated = TRUE, timeline_generated = TRUE WHERE case_id = %s",
            (case_id,),
        )
        conn.commit()
    except Exception:
        logger.warning("demo case intelligence failed case=%s", case_id, exc_info=True)
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        cur.close()


# ── Seed demo users ───────────────────────────────────────────────────────────
def seed_demo_users(cfg: Any) -> None:
    """Insert demo sessions if they don't exist. Called at startup."""
    from insightlens.storage.snowflake_client import open_connection
    slugs_codes = [
        ("user1", os.getenv("DEMO_ACCESS_CODE_USER1", "")),
        ("user2", os.getenv("DEMO_ACCESS_CODE_USER2", "")),
        ("user3", os.getenv("DEMO_ACCESS_CODE_USER3", "")),
    ]
    try:
        with open_connection(cfg.db) as conn:
            cur = conn.cursor()
            # Ensure schema/tables exist
            cur.execute("""
                CREATE SCHEMA IF NOT EXISTS demo;
                CREATE TABLE IF NOT EXISTS demo.sessions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_slug TEXT NOT NULL UNIQUE,
                    access_code_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_active TIMESTAMPTZ,
                    query_count INT DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS demo.usage (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_slug TEXT NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    query_type TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INT DEFAULT 0,
                    output_tokens INT DEFAULT 0,
                    cost_usd NUMERIC(10,6) DEFAULT 0,
                    question TEXT
                );
            """)

            for slug, code in slugs_codes:
                normalized_code = code.strip().upper()
                if not normalized_code:
                    logger.warning("seed_demo_users: DEMO_ACCESS_CODE_%s not set, skipping", slug.upper())
                    continue
                cur.execute("SELECT access_code_hash FROM demo.sessions WHERE user_slug = %s", (slug,))
                row = cur.fetchone()
                if row is None:
                    hashed = bcrypt.hashpw(normalized_code.encode(), bcrypt.gensalt()).decode()
                    cur.execute(
                        "INSERT INTO demo.sessions (user_slug, access_code_hash) VALUES (%s, %s)",
                        (slug, hashed),
                    )
                    logger.info("seed_demo_users: seeded %s", slug)
                elif not bcrypt.checkpw(normalized_code.encode(), row[0].encode()):
                    hashed = bcrypt.hashpw(normalized_code.encode(), bcrypt.gensalt()).decode()
                    cur.execute(
                        "UPDATE demo.sessions SET access_code_hash = %s WHERE user_slug = %s",
                        (hashed, slug),
                    )
                    logger.info("seed_demo_users: updated %s access code", slug)
                else:
                    logger.info("seed_demo_users: %s already exists", slug)

            conn.commit()
            cur.close()
        logger.info("Demo users seeded successfully.")
    except Exception as exc:
        logger.error("seed_demo_users failed: %s", exc, exc_info=True)


# ── POST /api/demo/auth ───────────────────────────────────────────────────────
@router.post("/auth")
def auth_demo(req: AuthRequest):
    from insightlens.config import load_config
    from insightlens.storage.snowflake_client import open_connection
    cfg = load_config()

    raw_code = req.access_code.strip().upper()
    if not raw_code:
        raise HTTPException(status_code=401, detail="Access code required.")

    try:
        with open_connection(cfg.db) as conn:
            cur = conn.cursor()
            cur.execute("SELECT user_slug, access_code_hash FROM demo.sessions")
            rows = cur.fetchall()
            cur.close()
    except Exception as exc:
        logger.error("auth_demo: DB error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

    for user_slug, stored_hash in rows:
        try:
            if bcrypt.checkpw(raw_code.encode(), stored_hash.encode()):
                token = _sign_token(user_slug)
                return {"user_slug": user_slug, "token": token, "ok": True}
        except Exception:
            continue

    raise HTTPException(status_code=401, detail="Invalid access code.")


# ── GET /api/demo/me ──────────────────────────────────────────────────────────
@router.get("/me")
def demo_me(user_slug: str = Depends(_get_demo_user)):
    from insightlens.config import load_config
    from insightlens.storage.snowflake_client import open_connection
    cfg = load_config()
    try:
        with open_connection(cfg.db) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT query_count, last_active FROM demo.sessions WHERE user_slug = %s",
                (user_slug,),
            )
            row = cur.fetchone()
            cur.close()
        return {
            "user_slug": user_slug,
            "query_count": row[0] if row else 0,
            "last_active": row[1].isoformat() if row and row[1] else None,
        }
    except Exception as exc:
        logger.error("demo_me: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")


# ── Demo cases: lightweight upload path for trial/demo users ─────────────────
@router.get("/cases")
def demo_cases(user_slug: str = Depends(_get_demo_user)):
    from insightlens.config import load_config
    from insightlens.storage.cases_repository import CasesRepository
    from insightlens.storage.snowflake_client import open_connection

    cfg = load_config()
    try:
        with open_connection(cfg.db) as conn:
            cases = CasesRepository(conn).list_cases(_demo_user_id(user_slug))
        return [
            {
                "case_id": case.case_id,
                "case_name": case.case_name,
                "description": case.description,
                "document_count": case.document_count,
            }
            for case in cases
        ]
    except Exception:
        logger.error("demo_cases failed user=%s", user_slug, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load demo cases.")


@router.post("/cases/upload")
async def demo_upload_case(
    files: list[UploadFile] = File(...),
    case_name: str = Form("Uploaded Case"),
    user_slug: str = Depends(_get_demo_user),
):
    """Create a demo-owned case and synchronously ingest a small document folder."""
    from insightlens.config import load_config
    from insightlens.embeddings.embedder import Embedder
    from insightlens.ingestion.ingest_service import IngestService
    from insightlens.storage.cases_repository import CasesRepository
    from insightlens.storage.snowflake_client import open_connection

    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one PDF or PPTX file.")
    if len(files) > 8:
        raise HTTPException(status_code=413, detail="Demo upload is limited to 8 files.")

    cfg = load_config()
    uid = _demo_user_id(user_slug)
    safe_case_name = (case_name or "Uploaded Case").strip()[:200] or "Uploaded Case"

    try:
        with open_connection(cfg.db) as conn:
            existing_cases = CasesRepository(conn).list_cases(uid)
            if existing_cases:
                raise HTTPException(
                    status_code=409,
                    detail="Demo sessions can upload one full case only. Use the existing uploaded case or restart the demo session.",
                )
            case_id = CasesRepository(conn).create_case(
                uid,
                safe_case_name,
                "Demo-uploaded case. Uploaded files are user-provided and separate from the public Epstein corpus.",
            )
    except HTTPException:
        raise
    except Exception:
        logger.error("demo_upload_case: create case failed user=%s", user_slug, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create demo case.")

    embedder = Embedder(model=cfg.embedding_model)
    ingest = IngestService(cfg=cfg, embedder=embedder)
    temp_dir = Path(tempfile.gettempdir()) / "atticus_demo_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    ingested: list[dict[str, Any]] = []
    skipped: list[str] = []

    for upload in files:
        original_name = (upload.filename or "upload.pdf").replace("\\", "/")
        if not original_name.lower().endswith(_SUPPORTED_UPLOAD_EXTENSIONS):
            skipped.append(f"{original_name}: not a supported file type")
            continue
        try:
            contents = await upload.read()
        except Exception:
            skipped.append(f"{original_name}: could not read file")
            continue
        if len(contents) > 25 * 1024 * 1024:
            skipped.append(f"{original_name}: demo limit is 25 MB per file")
            continue

        tmp_path = temp_dir / f"{_uuid.uuid4().hex}_{Path(original_name).name}"
        try:
            tmp_path.write_bytes(contents)
            result = ingest.ingest(tmp_path, user_id=uid, original_file_name=original_name)
            if result.error:
                skipped.append(f"{original_name}: {result.error}")
                continue
            if result.document_id:
                with open_connection(cfg.db) as conn:
                    CasesRepository(conn).add_document_to_case(case_id, result.document_id)
            ingested.append(
                {
                    "document_id": result.document_id,
                    "file_name": result.file_name,
                    "page_count": result.page_count,
                    "chunks_inserted": result.chunks_inserted,
                }
            )
        except Exception as exc:
            logger.warning("demo_upload_case: ingest failed file=%s err=%s", original_name, exc, exc_info=True)
            skipped.append(f"{original_name}: ingest failed")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    if not ingested:
        # The case row was created up front (line ~925), before any file was
        # read. If every file in this attempt failed (wrong extension, over
        # the size limit, parser error, etc.), that case would otherwise be
        # left behind empty — and since the demo only allows one case per
        # session (`existing_cases` check above), a single failed attempt
        # would permanently disable the upload button / drag-and-drop for
        # the rest of the session even though nothing was ever uploaded.
        # Roll it back so the user can simply try again.
        try:
            with open_connection(cfg.db) as conn:
                CasesRepository(conn).delete_case(case_id)
        except Exception:
            logger.warning("demo_upload_case: failed to roll back empty case=%s", case_id, exc_info=True)
        raise HTTPException(status_code=422, detail="No files could be ingested. " + "; ".join(skipped[:3]))

    try:
        with open_connection(cfg.db) as conn:
            _store_demo_case_intelligence(conn, case_id, user_slug)
    except Exception:
        logger.warning("demo_upload_case: case intelligence skipped case=%s", case_id, exc_info=True)

    return {
        "case_id": case_id,
        "case_name": safe_case_name,
        "documents": ingested,
        "skipped": skipped,
    }


# ── POST /api/demo/query ──────────────────────────────────────────────────────
@router.post("/query")
def demo_query(req: DemoQueryRequest, user_slug: str = Depends(_get_demo_user)):
    from anthropic import Anthropic
    from insightlens.config import load_config
    from insightlens.embeddings.embedder import Embedder
    from insightlens.generation.prompts import SYSTEM_PROMPT, build_user_prompt
    from insightlens.retrieval.hybrid_search import HybridSearchService
    from insightlens.retrieval.reranker import Reranker
    from insightlens.retrieval.vector_search import RetrievalRequest
    from insightlens.storage.chunk_repository import ChunkRepository
    from insightlens.storage.snowflake_client import open_connection

    _check_demo_rate(user_slug)

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    cfg = load_config()

    # ── Retrieve from selected demo corpus ────────────────────────────────────
    try:
        embedder = Embedder(model=cfg.embedding_model)
        reranker = Reranker()

        with open_connection(cfg.db) as conn:
            chunk_repo = ChunkRepository(conn)
            if req.case_id:
                _assert_demo_case_owned(conn, req.case_id, user_slug)
            corpus = chunk_repo.get_all_chunks(
                system_only=not bool(req.case_id),
                user_id=_demo_user_id(user_slug) if req.case_id else None,
                case_id=req.case_id,
            )
            hybrid_svc = HybridSearchService(embedder, chunk_repo, corpus, reranker)
            sources = hybrid_svc.retrieve(
                RetrievalRequest(
                    query=question,
                    top_k=8,
                    system_only=not bool(req.case_id),
                    user_id=_demo_user_id(user_slug) if req.case_id else None,
                    case_id=req.case_id,
                )
            )
            sources = list(sources)
            sources = _dedupe_comparable_sources(sources)
            if req.case_id:
                sources = _dedupe_comparable_sources(
                    sources + _demo_case_context_chunks(conn, req.case_id, user_slug)
                )
                # Uploaded demo cases are often tiny and user questions are
                # naturally broad ("brief the case", "who is the suspect?").
                # If hybrid search finds no lexical/vector match, still ground
                # the answer in representative chunks from that selected case
                # instead of returning a misleading "no source material" answer.
                if not any((getattr(s, "source_type", "document") or "document") == "document" for s in sources):
                    sources = _dedupe_comparable_sources(
                        sources
                        + chunk_repo.get_chunks_for_case(req.case_id, _demo_user_id(user_slug), limit=8)
                    )
            else:
                sources.append(_demo_overview_source_chunk())
                sources.append(_epstein_people_source_chunk())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("demo_query: retrieval failed user=%s: %s", user_slug, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Retrieval failed.")

    # Estimate embedding tokens for cost tracking
    embed_input_tokens = max(1, int(len(question.split()) * 1.3))
    embed_cost = compute_cost(cfg.embedding_model, embed_input_tokens, 0)

    # ── Generate with Claude (direct API for token counts) ────────────────────
    prompt = build_user_prompt(question, sources)
    messages_payload = []
    for msg in req.chat_history[-6:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages_payload.append({"role": role, "content": content})
    messages_payload.append({"role": "user", "content": prompt})

    try:
        anthropic_client = Anthropic(api_key=cfg.anthropic_api_key)
        response = anthropic_client.messages.create(
            model=cfg.generation_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages_payload,
        )
    except Exception as exc:
        logger.error("demo_query: LLM failed user=%s: %s", user_slug, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Generation failed.")

    answer = "\n".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
    answer, confidence = _extract_confidence(answer)
    scope_note = _scope_note_for_coverage(question, sources)
    if scope_note and "Scope note:" not in answer:
        answer = scope_note + answer.lstrip()
    answer = _ensure_lawyer_followups(answer, question, sources)
    answer = _ensure_workspace_note(answer, question, sources)
    confidence = _cap_confidence_for_coverage(question, sources, confidence)

    input_tokens  = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    llm_cost      = compute_cost(cfg.generation_model, input_tokens, output_tokens)

    # ── Write usage rows ──────────────────────────────────────────────────────
    try:
        with open_connection(cfg.db) as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO demo.usage (user_slug, query_type, model, input_tokens, output_tokens, cost_usd, question)
                   VALUES (%s, 'chat', %s, %s, %s, %s, %s)""",
                (user_slug, cfg.generation_model, input_tokens, output_tokens, llm_cost, question[:200]),
            )
            cur.execute(
                """INSERT INTO demo.usage (user_slug, query_type, model, input_tokens, output_tokens, cost_usd)
                   VALUES (%s, 'embedding', %s, %s, 0, %s)""",
                (user_slug, cfg.embedding_model, embed_input_tokens, embed_cost),
            )
            cur.execute(
                """UPDATE demo.sessions
                   SET last_active = NOW(), query_count = query_count + 1
                   WHERE user_slug = %s""",
                (user_slug,),
            )
            conn.commit()
            cur.close()
    except Exception as exc:
        logger.warning("demo_query: usage write failed user=%s: %s", user_slug, exc)

    source_details = [
        {
            "index": i + 1,
            "label": f"{s.file_name} · p.{s.page_number}",
            "chunk_id": s.chunk_id,
            "document_id": s.document_id,
            "file_name": s.file_name,
            "page_number": s.page_number,
            "section_header": s.section_header,
            "chunk_text": s.chunk_text,
            "excerpt": _best_source_excerpt(s.chunk_text, question),
            "source_type": getattr(s, "source_type", "document") or "document",
            "citation_label": _citation_label(s),
            "jurisdiction": _infer_jurisdiction(s),
        }
        for i, s in enumerate(sources)
    ]

    result = {
        "answer": answer,
        "sources": [f"{s.file_name} · p.{s.page_number}" for s in sources],
        "source_details": source_details,
        "cost_usd": round(llm_cost + embed_cost, 6),
    }
    if confidence:
        result["confidence"] = confidence
    return result


# ── GET /api/demo/timeline ────────────────────────────────────────────────────
@router.get("/timeline")
def demo_timeline(user_slug: str = Depends(_get_demo_user)):
    from insightlens.config import load_config
    from insightlens.storage.snowflake_client import open_connection
    cfg = load_config()
    try:
        with open_connection(cfg.db) as conn:
            cur = conn.cursor()
            cur.execute(
                """SELECT ct.events, ct.generated_at
                   FROM case_timelines ct
                   JOIN cases c ON ct.case_id = c.case_id
                   WHERE COALESCE(c.is_demo, FALSE) = TRUE OR c.user_id IS NULL
                   ORDER BY COALESCE(c.is_demo, FALSE) DESC, ct.generated_at DESC
                   LIMIT 1"""
            )
            row = cur.fetchone()
            cur.close()
        if row is None:
            return {
                "pending": False,
                "events": _attach_timeline_images(conn, _FALLBACK_TIMELINE_EVENTS),
                "generated_at": None,
                "note": "Using the shared demo timeline.",
            }
        events = row[0] if row[0] else []
        if not events:
            events = _FALLBACK_TIMELINE_EVENTS
        return {
            "pending": False,
            "events": _attach_timeline_images(conn, events),
            "generated_at": row[1].isoformat() if row[1] else None,
        }
    except Exception as exc:
        logger.error("demo_timeline: %s", exc, exc_info=True)
        return {
            "pending": False,
            "events": _FALLBACK_TIMELINE_EVENTS,
            "generated_at": None,
            "note": "Using the shared demo timeline.",
        }


# ── GET /api/demo/overview ────────────────────────────────────────────────────
@router.get("/overview")
def demo_overview(user_slug: str = Depends(_get_demo_user)):
    from insightlens.config import load_config
    from insightlens.storage.snowflake_client import open_connection
    cfg = load_config()
    try:
        with open_connection(cfg.db) as conn:
            cur = conn.cursor()
            cur.execute(
                """SELECT co.summary, co.parties, co.key_issues, co.jurisdiction, co.matter_type, co.generated_at
                   FROM case_overviews co
                   JOIN cases c ON co.case_id = c.case_id
                   WHERE COALESCE(c.is_demo, FALSE) = TRUE OR c.user_id IS NULL
                   ORDER BY COALESCE(c.is_demo, FALSE) DESC, co.generated_at DESC
                   LIMIT 1"""
            )
            row = cur.fetchone()
            cur.close()
        if row is None:
            return {
                "pending": False,
                **_FALLBACK_OVERVIEW,
                "generated_at": None,
                "note": "Using the shared demo overview.",
            }
        return {
            "pending": False,
            "summary": row[0] or _FALLBACK_OVERVIEW["summary"],
            "parties": row[1] if row[1] else _FALLBACK_OVERVIEW["parties"],
            "key_issues": row[2] if row[2] else _FALLBACK_OVERVIEW["key_issues"],
            "jurisdiction": row[3] or _FALLBACK_OVERVIEW["jurisdiction"],
            "matter_type": row[4] or _FALLBACK_OVERVIEW["matter_type"],
            "generated_at": row[5].isoformat() if row[5] else None,
        }
    except Exception as exc:
        logger.error("demo_overview: %s", exc, exc_info=True)
        return {
            "pending": False,
            **_FALLBACK_OVERVIEW,
            "generated_at": None,
            "note": "Using the shared demo overview.",
        }


# ── GET /api/demo/admin/costs ─────────────────────────────────────────────────
@router.get("/admin/costs")
def admin_costs(x_admin_key: str | None = Header(None, alias="x-admin-key")):
    admin_key = os.getenv("ADMIN_API_KEY", "")
    if not admin_key or x_admin_key != admin_key:
        raise HTTPException(status_code=403, detail="Invalid or missing admin key.")

    from insightlens.config import load_config
    from insightlens.storage.snowflake_client import open_connection
    cfg = load_config()
    try:
        with open_connection(cfg.db) as conn:
            cur = conn.cursor()

            # Total cost + query count
            cur.execute("SELECT COALESCE(SUM(cost_usd), 0), COUNT(*) FROM demo.usage WHERE query_type = 'chat'")
            row = cur.fetchone()
            total_cost = float(row[0])
            total_queries = int(row[1])

            # Per-user
            # NOTE: select s.user_slug (the LEFT side), not u.user_slug — a session
            # that hasn't made any query yet has no matching demo.usage row, so the
            # joined u.user_slug is NULL even though the session itself is real.
            # Selecting u.user_slug here previously caused every never-queried user
            # to show up as user_slug: null, which (a) hid them under one bucket in
            # this dashboard and (b) gave the frontend table a duplicate `null` React
            # key for every such user.
            cur.execute(
                """SELECT s.user_slug,
                          s.query_count,
                          COALESCE(SUM(u.cost_usd), 0) AS cost_usd,
                          s.last_active
                   FROM demo.sessions s
                   LEFT JOIN demo.usage u ON s.user_slug = u.user_slug
                   GROUP BY s.user_slug, s.query_count, s.last_active
                   ORDER BY s.user_slug"""
            )
            by_user = [
                {
                    "user_slug": r[0],
                    "query_count": r[1] or 0,
                    "cost_usd": round(float(r[2]), 6),
                    "last_active": r[3].isoformat() if r[3] else None,
                }
                for r in cur.fetchall()
            ]

            # Per-model
            cur.execute(
                """SELECT model,
                          SUM(input_tokens) AS input_tokens,
                          SUM(output_tokens) AS output_tokens,
                          SUM(cost_usd) AS cost_usd
                   FROM demo.usage
                   GROUP BY model ORDER BY cost_usd DESC"""
            )
            by_model = [
                {
                    "model": r[0],
                    "input_tokens": int(r[1] or 0),
                    "output_tokens": int(r[2] or 0),
                    "cost_usd": round(float(r[3] or 0), 6),
                }
                for r in cur.fetchall()
            ]

            # Recent queries (last 20 chat queries)
            cur.execute(
                """SELECT user_slug, timestamp, question, cost_usd
                   FROM demo.usage
                   WHERE query_type = 'chat'
                   ORDER BY timestamp DESC LIMIT 20"""
            )
            recent_queries = [
                {
                    "user_slug": r[0],
                    "timestamp": r[1].isoformat() if r[1] else None,
                    "question": r[2] or "",
                    "cost_usd": round(float(r[3] or 0), 6),
                }
                for r in cur.fetchall()
            ]

            cur.close()

        return {
            "total_cost_usd": round(total_cost, 6),
            "total_queries": total_queries,
            "by_user": by_user,
            "by_model": by_model,
            "recent_queries": recent_queries,
        }
    except Exception as exc:
        logger.error("admin_costs: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")
