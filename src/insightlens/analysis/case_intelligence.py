"""Cost-controlled case intelligence derived from uploaded chunks.

This module intentionally starts with deterministic extraction. It gives chat a
stable case map without adding an LLM call to every user question. Higher-cost
LLM enrichment can still overwrite these rows later through the existing
overview/timeline generation tasks.
"""
from __future__ import annotations

import re
import json
from collections import Counter
from typing import Any

from insightlens.analysis.case_insights import extract_case_insights
from insightlens.storage.chunk_repository import RetrievedChunk


_ROLE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Victim", re.compile(r"\b(victim|decedent|injured|killed|survivor)\b", re.IGNORECASE)),
    ("Defendant / suspect", re.compile(r"\b(defendant|suspect|accused|charged|arrested|person of interest)\b", re.IGNORECASE)),
    ("Plaintiff / claimant", re.compile(r"\b(plaintiff|claimant|petitioner)\b", re.IGNORECASE)),
    ("Prosecutor", re.compile(r"\b(prosecutor|district attorney|state attorney|u\.s\. attorney)\b", re.IGNORECASE)),
    ("Witness", re.compile(r"\b(witness|testified|statement|interviewed)\b", re.IGNORECASE)),
    ("Investigator / agency", re.compile(r"\b(police|detective|investigator|sheriff|department)\b", re.IGNORECASE)),
]

_JURISDICTION_RE = re.compile(
    r"\b(?:S\.D\.N\.Y\.|S\.D\. Fla\.|District Court|Superior Court|County|Police Department|"
    r"District Attorney|State of [A-Z][a-z]+|United States)\b"
)

_DATE_RE = re.compile(
    r"\b(?:"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|20\d{2}"
    r")\b",
    re.IGNORECASE,
)


def build_case_overview(chunks: list[RetrievedChunk], case_name: str = "Uploaded Case") -> dict[str, Any]:
    """Return a small structured overview from existing chunks without an LLM."""
    analysis = extract_case_insights(chunks)
    parties = _parties_from_entities(chunks, analysis.entities)
    key_issues = _key_issues(chunks)
    summary = _summary(chunks, case_name, parties, key_issues)
    return {
        "summary": summary,
        "parties": parties[:12],
        "key_issues": key_issues[:8],
        "jurisdiction": _jurisdiction(chunks),
        "matter_type": _matter_type(chunks),
    }


def build_case_timeline(chunks: list[RetrievedChunk], limit: int = 12) -> list[dict[str, Any]]:
    """Return timeline events shaped like case_timelines.events."""
    events: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for insight in extract_case_insights(chunks).timeline:
        body = str(insight.get("body") or "").strip()
        date_text = str(insight.get("title") or "").strip()
        if not body or not date_text:
            continue
        key = (date_text.lower(), body[:80].lower())
        if key in seen:
            continue
        seen.add(key)
        events.append(
            {
                "date": date_text,
                "title": _clip(body, 90),
                "description": _clip(body, 260),
                "source_doc": _metadata_value(insight, "file_name"),
                "page": insight.get("page_number"),
            }
        )
        if len(events) >= limit:
            break
    return events


def _parties_from_entities(chunks: list[RetrievedChunk], entities: list[dict]) -> list[dict[str, str]]:
    contexts = _entity_contexts(chunks)
    parties: list[dict[str, str]] = []
    for entity in entities[:30]:
        name = str(entity.get("title") or "").strip()
        if not name or len(name.split()) > 5:
            continue
        context = contexts.get(name.lower(), "")
        role = _role_for_context(context)
        parties.append({"name": name, "role": role})
    return _dedupe_parties(parties)


def _entity_contexts(chunks: list[RetrievedChunk]) -> dict[str, str]:
    contexts: dict[str, str] = {}
    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", chunk.chunk_text)
        for sentence in sentences:
            if len(sentence) < 20:
                continue
            for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", sentence):
                name = match.group(0)
                contexts.setdefault(name.lower(), sentence[:500])
    return contexts


def _role_for_context(context: str) -> str:
    for role, pattern in _ROLE_PATTERNS:
        if pattern.search(context):
            return role
    return "Person / entity mentioned"


def _dedupe_parties(parties: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for party in parties:
        key = party["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(party)
    return result


def _key_issues(chunks: list[RetrievedChunk]) -> list[str]:
    text = "\n".join(chunk.chunk_text[:2500] for chunk in chunks).lower()
    issues: list[str] = []
    if re.search(r"\b(charge|charged|manslaughter|murder|reckless|negligent|criminal)\b", text):
        issues.append("Identify the charged or alleged offense and the evidence supporting each element.")
    if re.search(r"\b(accident|collision|vehicle|crash|bicycle|injured|killed)\b", text):
        issues.append("Reconstruct the incident timeline and compare it against physical and witness evidence.")
    if re.search(r"\b(witness|statement|interview|testif)\b", text):
        issues.append("Assess witness statements for consistency, credibility, and missing follow-up questions.")
    if re.search(r"\b(photo|image|portrait|visual|diagram|map|vision extraction)\b", text):
        issues.append("Review visual evidence and confirm what each image can and cannot prove.")
    if re.search(r"\b(report|police|detective|investigator|department)\b", text):
        issues.append("Validate investigative reports against underlying source documents and exhibits.")
    if not issues:
        issues.append("Summarize the parties, core facts, evidence, and unresolved questions from the uploaded documents.")
    return issues


def _summary(
    chunks: list[RetrievedChunk],
    case_name: str,
    parties: list[dict[str, str]],
    key_issues: list[str],
) -> str:
    important_sentences = _rank_sentences(chunks)
    lead = f"{case_name} is an uploaded matter represented by {len(chunks)} searchable document chunk(s)."
    if parties:
        named = ", ".join(f"{p['name']} ({p['role']})" for p in parties[:4])
        lead += f" Key people/entities detected include {named}."
    if important_sentences:
        lead += " " + " ".join(important_sentences[:3])
    if key_issues:
        lead += " Main review focus: " + key_issues[0]
    return lead


def _rank_sentences(chunks: list[RetrievedChunk]) -> list[str]:
    scored: list[tuple[int, str]] = []
    signals = re.compile(
        r"\b(victim|defendant|suspect|charged|charge|killed|injured|accident|collision|"
        r"witness|police|report|evidence|photo|statement)\b",
        re.IGNORECASE,
    )
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk.chunk_text):
            clean = " ".join(sentence.split())
            if not (40 <= len(clean) <= 360):
                continue
            score = len(signals.findall(clean)) * 5
            score += len(_DATE_RE.findall(clean))
            if score:
                scored.append((score, clean))
    scored.sort(key=lambda item: item[0], reverse=True)
    result: list[str] = []
    seen: set[str] = set()
    for _, sentence in scored:
        key = sentence[:90].lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(_clip(sentence, 280))
        if len(result) >= 5:
            break
    return result


def _jurisdiction(chunks: list[RetrievedChunk]) -> str | None:
    candidates: Counter[str] = Counter()
    for chunk in chunks:
        for match in _JURISDICTION_RE.finditer(chunk.chunk_text[:3000]):
            candidates[match.group(0)] += 1
    return candidates.most_common(1)[0][0] if candidates else None


def _matter_type(chunks: list[RetrievedChunk]) -> str:
    text = "\n".join(chunk.chunk_text[:1500] for chunk in chunks).lower()
    if re.search(r"\b(criminal|manslaughter|murder|reckless driving|charged|prosecutor)\b", text):
        return "Criminal / investigative matter"
    if re.search(r"\b(contract|agreement|clause|indemnif|termination)\b", text):
        return "Contract review"
    if re.search(r"\b(complaint|plaintiff|defendant|motion|court)\b", text):
        return "Civil litigation"
    return "Document review"


def _metadata_value(item: dict, key: str) -> Any:
    metadata = item.get("metadata_json")
    if isinstance(metadata, dict):
        return metadata.get(key)
    if isinstance(metadata, str) and metadata:
        try:
            parsed = json.loads(metadata)
            if isinstance(parsed, dict):
                return parsed.get(key)
        except json.JSONDecodeError:
            return None
    return None


def _clip(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "…"
