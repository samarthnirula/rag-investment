"""Heuristic case insight extraction with optional LLM verification.

Pass `llm_client` to `extract_case_insights` to enrich results with Claude.
Without it (or when the API key is invalid) the deterministic pass still runs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from insightlens.storage.chunk_repository import RetrievedChunk

if TYPE_CHECKING:
    from insightlens.generation.llm_client import ClaudeClient

_log = logging.getLogger(__name__)

# ── LLM call cache ──────────────────────────────────────────────────────────────
# Bounded LRU cache for LLM-verified insights. Key = (stage, data_hash).
# Prevents redundant API calls when the same data is re-analyzed.
_llm_cache: dict[tuple[str, str], list[dict]] = {}
_LLM_CACHE_MAX = 256


def _cache_key(stage: str, items: list[dict]) -> tuple[str, str]:
    """Return a hashable cache key for LLM results."""
    # Stable JSON ensures same data always produces same hash
    stable = json.dumps(items, sort_keys=True, ensure_ascii=True)
    h = hashlib.sha1(stable.encode()).hexdigest()[:24]
    return (stage, h)


_DATE_RE = re.compile(
    r"\b(?:"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\b20\d{2}\b"
    r")\b",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(r"\$[\d,]+(?:\.\d+)?\s*(?:million|billion|m|bn)?", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?%\b")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")
_STOP_ENTITIES = {
    "United States",
    "New York",
    "Palm Beach",
    "This Agreement",
    "The Court",
    "The Company",
}


@dataclass(frozen=True)
class CaseAnalysisResult:
    timeline: list[dict]
    entities: list[dict]
    contradictions: list[dict]
    client_summary: str


def _sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in raw if len(s.strip()) >= 20]


def _snippet(sentence: str, limit: int = 260) -> str:
    clean = " ".join(sentence.split())
    return clean[: limit - 1] + "…" if len(clean) > limit else clean


def extract_case_insights(
    chunks: list[RetrievedChunk],
    llm_client: ClaudeClient | None = None,
) -> CaseAnalysisResult:
    timeline = _extract_timeline(chunks)
    entities = _extract_entities(chunks)
    contradictions = _extract_contradictions(chunks)

    if llm_client:
        try:
            timeline = _llm_classify_timeline(timeline, llm_client)
        except Exception as exc:
            _log.warning("LLM timeline classification failed, using heuristic: %s", exc)
        try:
            contradictions = _llm_verify_contradictions(contradictions, llm_client)
        except Exception as exc:
            _log.warning("LLM contradiction verification failed, using heuristic: %s", exc)
        try:
            entities = _llm_classify_entities(entities, llm_client)
        except Exception as exc:
            _log.warning("LLM entity classification failed, using heuristic: %s", exc)

    return CaseAnalysisResult(
        timeline=timeline,
        entities=entities,
        contradictions=contradictions,
        client_summary=_build_client_summary(timeline, entities, contradictions),
    )


def _extract_timeline(chunks: list[RetrievedChunk], limit: int = 80) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        for sentence in _sentences(chunk.chunk_text):
            match = _DATE_RE.search(sentence)
            if not match:
                continue
            date_text = match.group(0)
            title = _snippet(sentence, 90)
            key = (date_text.lower(), title.lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "title": date_text,
                    "body": _snippet(sentence),
                    "severity": "info",
                    "document_id": chunk.document_id,
                    "page_number": chunk.page_number,
                    "metadata_json": json.dumps(
                        {"file_name": chunk.file_name, "event_text": title}
                    ),
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def _extract_entities(chunks: list[RetrievedChunk], limit: int = 80) -> list[dict]:
    counts: dict[str, dict] = {}
    for chunk in chunks:
        for match in _ENTITY_RE.finditer(chunk.chunk_text):
            entity = match.group(0).strip()
            if entity in _STOP_ENTITIES or len(entity) > 80:
                continue
            current = counts.setdefault(
                entity,
                {
                    "title": entity,
                    "count": 0,
                    "document_id": chunk.document_id,
                    "page_number": chunk.page_number,
                    "file_name": chunk.file_name,
                },
            )
            current["count"] += 1

    ranked = sorted(counts.values(), key=lambda item: item["count"], reverse=True)[:limit]
    return [
        {
            "title": item["title"],
            "body": f"Mentioned {item['count']} time(s). First seen in {item['file_name']} p.{item['page_number']}.",
            "severity": "info",
            "document_id": item["document_id"],
            "page_number": item["page_number"],
            "metadata_json": json.dumps({"mentions": item["count"], "file_name": item["file_name"]}),
        }
        for item in ranked
    ]


def _topic_key(sentence: str) -> str:
    words = [
        re.sub(r"[^a-z0-9]", "", w.lower())
        for w in sentence.split()
        if (
            len(w) > 3
            and not _DATE_RE.fullmatch(w)
            and not _MONEY_RE.fullmatch(w)
            and not _NUMBER_RE.fullmatch(w)
            and not re.search(r"\d", w)
        )
    ]
    return " ".join(words[:6])


def _extract_contradictions(chunks: list[RetrievedChunk], limit: int = 50) -> list[dict]:
    buckets: dict[str, list[tuple[str, RetrievedChunk, str]]] = defaultdict(list)
    for chunk in chunks:
        for sentence in _sentences(chunk.chunk_text):
            values = _MONEY_RE.findall(sentence) + _NUMBER_RE.findall(sentence)
            if not values:
                continue
            key = _topic_key(sentence)
            if len(key) < 12:
                continue
            buckets[key].append(("; ".join(sorted(set(values))), chunk, sentence))

    results: list[dict] = []
    for key, rows in buckets.items():
        values = {row[0] for row in rows}
        if len(values) < 2:
            continue
        first_value, first_chunk, first_sentence = rows[0]
        other = next(row for row in rows[1:] if row[0] != first_value)
        other_value, other_chunk, other_sentence = other
        results.append(
            {
                "title": f"Possible conflicting values: {first_value} vs {other_value}",
                "body": (
                    f"One source says: {_snippet(first_sentence)}\n\n"
                    f"Another source says: {_snippet(other_sentence)}"
                ),
                "severity": "medium",
                "document_id": first_chunk.document_id,
                "page_number": first_chunk.page_number,
                "metadata_json": json.dumps(
                    {
                        "topic_key": key,
                        "first_source": f"{first_chunk.file_name} p.{first_chunk.page_number}",
                        "second_source": f"{other_chunk.file_name} p.{other_chunk.page_number}",
                    }
                ),
            }
        )
        if len(results) >= limit:
            break
    return results


_TIMELINE_CLASSIFY_PROMPT = """\
You are a legal research assistant. Classify each of the following timeline events from a case document.
For each item, output ONLY a JSON array (same length as input) where each element has:
  "severity": one of "high" | "medium" | "info"
  "category": one of "filing" | "ruling" | "hearing" | "settlement" | "deposition" | "arrest" | "plea" | "other"
  "title": cleaned-up date label (keep brief)

Input events (JSON):
{events_json}

Output ONLY the JSON array, no explanation."""

_CONTRADICTION_VERIFY_PROMPT = """\
You are a legal research assistant. Review these possible contradictions found in a case document.
Determine if each is a genuine contradiction or a false positive (e.g., same number in different contexts).
Output ONLY a JSON array (same length) where each element has:
  "severity": "high" | "medium" | "low" | "false_positive"
  "verified": true | false
  "note": one sentence explaining your verdict (max 120 chars)

Input contradictions (JSON):
{contradictions_json}

Output ONLY the JSON array, no explanation."""

_ENTITY_CLASSIFY_PROMPT = """\
You are a legal research assistant. Classify each of these entities from a case document.
Output ONLY a JSON array (same length) where each element has:
  "entity_type": "person" | "organization" | "location" | "unknown"
  "role": brief role description in the case context (max 60 chars, e.g. "defendant", "law firm", "court")

Input entities (JSON):
{entities_json}

Output ONLY the JSON array, no explanation."""


def _llm_classify_timeline(
    timeline: list[dict], llm: ClaudeClient, batch: int = 20
) -> list[dict]:
    if not timeline:
        return timeline

    # Check cache
    key = _cache_key("timeline", timeline)
    if key in _llm_cache:
        return _llm_cache[key]

    out = []
    for start in range(0, len(timeline), batch):
        chunk = timeline[start : start + batch]
        prompt = _TIMELINE_CLASSIFY_PROMPT.format(
            events_json=json.dumps([{"title": e["title"], "body": e["body"]} for e in chunk])
        )
        raw = llm.generate(
            "You classify legal case events. Respond only with a JSON array.",
            prompt,
        )
        try:
            classifications = json.loads(raw)
            for event, cls in zip(chunk, classifications):
                out.append({
                    **event,
                    "severity": cls.get("severity", event.get("severity", "info")),
                    "category": cls.get("category", "other"),
                    "title": cls.get("title", event["title"]),
                })
        except Exception:
            out.extend(chunk)

    # Evict oldest entry if cache is full
    if len(_llm_cache) >= _LLM_CACHE_MAX:
        evict_key = next(iter(_llm_cache))
        del _llm_cache[evict_key]
    _llm_cache[key] = out
    return out


def _llm_verify_contradictions(
    contradictions: list[dict], llm: ClaudeClient, batch: int = 10
) -> list[dict]:
    if not contradictions:
        return contradictions

    # Check cache
    key = _cache_key("contradictions", contradictions)
    if key in _llm_cache:
        return _llm_cache[key]

    out = []
    for start in range(0, len(contradictions), batch):
        chunk = contradictions[start : start + batch]
        prompt = _CONTRADICTION_VERIFY_PROMPT.format(
            contradictions_json=json.dumps(
                [{"title": c["title"], "body": c["body"]} for c in chunk]
            )
        )
        raw = llm.generate(
            "You verify legal document contradictions. Respond only with a JSON array.",
            prompt,
        )
        try:
            verifications = json.loads(raw)
            for item, v in zip(chunk, verifications):
                if v.get("severity") == "false_positive":
                    continue
                out.append({
                    **item,
                    "severity": v.get("severity", item.get("severity", "medium")),
                    "llm_note": v.get("note", ""),
                    "llm_verified": v.get("verified", False),
                })
        except Exception:
            out.extend(chunk)

    # Evict oldest entry if cache is full
    if len(_llm_cache) >= _LLM_CACHE_MAX:
        evict_key = next(iter(_llm_cache))
        del _llm_cache[evict_key]
    _llm_cache[key] = out
    return out


def _llm_classify_entities(
    entities: list[dict], llm: ClaudeClient, batch: int = 30
) -> list[dict]:
    if not entities:
        return entities

    # Check cache
    key = _cache_key("entities", entities)
    if key in _llm_cache:
        return _llm_cache[key]

    out = []
    for start in range(0, len(entities), batch):
        chunk = entities[start : start + batch]
        prompt = _ENTITY_CLASSIFY_PROMPT.format(
            entities_json=json.dumps([{"title": e["title"]} for e in chunk])
        )
        raw = llm.generate(
            "You classify legal case entities. Respond only with a JSON array.",
            prompt,
        )
        try:
            classifications = json.loads(raw)
            for entity, cls in zip(chunk, classifications):
                out.append({
                    **entity,
                    "entity_type": cls.get("entity_type", "unknown"),
                    "role": cls.get("role", ""),
                })
        except Exception:
            out.extend(chunk)

    # Evict oldest entry if cache is full
    if len(_llm_cache) >= _LLM_CACHE_MAX:
        evict_key = next(iter(_llm_cache))
        del _llm_cache[evict_key]
    _llm_cache[key] = out
    return out


def _build_client_summary(
    timeline: list[dict],
    entities: list[dict],
    contradictions: list[dict],
) -> str:
    lines = [
        "# Client Summary",
        "",
        "This summary is AI-assisted and should be reviewed by an attorney before sending.",
        "",
        "## What We Found",
        f"- {len(timeline)} timeline event(s) were detected from the document set.",
        f"- {len(entities)} notable person/entity mention(s) were detected.",
        f"- {len(contradictions)} possible contradiction or conflicting-value flag(s) were detected.",
    ]
    if timeline[:5]:
        lines.extend(["", "## Key Timeline Items"])
        for item in timeline[:5]:
            lines.append(f"- **{item['title']}**: {item['body']}")
    if contradictions[:5]:
        lines.extend(["", "## Items To Verify"])
        for item in contradictions[:5]:
            lines.append(f"- **{item['title']}**: {item['body'].splitlines()[0]}")
    return "\n".join(lines)
