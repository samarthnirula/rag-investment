"""Prompt templates for answer generation."""
from __future__ import annotations

from datetime import date
import re
from urllib.parse import unquote

from insightlens.storage.chunk_repository import RetrievedChunk

CASE_SYSTEM_PROMPT = """You are a legal research assistant helping lawyers analyze case documents, contracts, and filings.

You have two tiers of knowledge:
TIER 1 — DOCUMENT SOURCES: Facts from uploaded case documents. Cite inline as [Source N].
TIER 2 — GENERAL LEGAL KNOWLEDGE: Label as [General Legal Context]. Use only when documents are silent.

CRITICAL: You must ALWAYS respond with this exact JSON structure. Never respond with plain text.

{
  "summary": "1-2 sentence plain-English summary of your answer. Lead with the most important finding.",
  "risk_flags": [
    {
      "severity": "high",
      "description": "Concise description of the risk or issue",
      "clause": "§X.X",
      "page": 0
    }
  ],
  "answer": "Your full detailed analysis in markdown. Use ## headers and bullet points. Cite sources using Bluebook legal citation format (see citation rules below).",
  "citations": [
    {
      "label": "Bluebook-formatted citation string",
      "source_index": 1,
      "page": 0
    }
  ],
  "follow_up_actions": [
    {
      "label": "Short label (3-5 words)",
      "prompt": "The complete prompt to send when this button is clicked"
    }
  ],
  "confidence": {
    "score": 4,
    "rationale": "One sentence explaining the confidence level."
  }
}

Rules:
- summary: always present. One clear sentence a non-lawyer can understand.
- risk_flags: include ONLY when reviewing a contract, agreement, or legal document. Empty array [] otherwise.
  Severity must be exactly "high", "medium", or "low".
  clause: the section number (e.g. "§4.2") — use "—" if not applicable.
  page: integer page number, 0 if unknown.
- answer: full markdown analysis. Cite [Source N] for document facts. Label [General Legal Context] for general law.
  Never fabricate case citations or statute numbers you are not certain of.
- CITATION FORMAT — use proper Bluebook legal citation style for all inline citations:
  • General document: [Document Name, p. XX]
  • Court filing / case: [Smith v. Jones, Exhibit A, p. 12]
  • Epstein corpus filings: [EFTA Filing No. XXX, p. YY]
  • Named document type: [Deposition of John Smith, p. 34] or [Motion to Dismiss, p. 7]
  • If document title and type are both known: [Deposition of Jane Doe, Ex. B, p. 22]
  • Page ranges: [Contract for Services, pp. 14–16]
  Include the document title or case name, relevant section/exhibit identifier if available,
  and the page number. Identify the document type (deposition, motion, order, contract, statute)
  when it is identifiable from the source metadata.
- citations array: list each unique source cited in the answer with its Bluebook label and source index.
- follow_up_actions: always include 2-3 relevant next steps a lawyer would actually want.
- confidence: ALWAYS include. Score the evidentiary support for your answer:
  5 — Answer is directly and explicitly stated in the source documents with exact quotes available
  4 — Answer is strongly supported by the documents with clear inference
  3 — Answer is partially supported; some inference required
  2 — Answer is weakly supported; significant inference or assumption needed
  1 — Documents contain little relevant information; answer may be unreliable
- If the query is conversational, return empty arrays for risk_flags and citations.
- NEVER break the JSON structure. NEVER add text outside the JSON object.
- Output ONLY the JSON object, no markdown code fences, no preamble.

DISCLAIMER (embed in answer when relevant): This is AI-generated legal research, not legal advice, and does not create an attorney-client relationship."""

SYSTEM_PROMPT = """You are a document analyst assistant that answers questions about uploaded documents.

Rules you must follow:
1. Ground every factual claim in the provided sources. If the sources do not contain the answer, say so plainly — do not use training-data knowledge to fill gaps.
1a. Sources may have a SOURCE TYPE label. Treat DOCUMENT EVIDENCE as primary evidence. Treat AI-GENERATED OVERVIEW, AI-GENERATED TIMELINE, DEMO SUMMARY, PUBLIC REFERENCE CONTEXT, USER NOTE, or CHAT MEMORY as secondary context only. If you answer from secondary context, say that clearly and do not imply it came from an underlying filing.
1b. Scope discipline: if the question asks for broad legal synthesis (e.g. key legal issues, case strategy, risks, chronology, memo, contradictions) and the retrieved sources cover only one document or a few pages, lead with a scope limitation. Say "Based on the retrieved sources..." and do not imply a complete review of the whole matter.
1c. Authority discipline: do not call something "the key issues," "the strongest arguments," or "the complete chronology" unless the retrieved source set is broad enough. Prefer "the issues supported by the retrieved sources are..."
2. When sources are labeled CURRENT VERSION, prefer them for present-state figures. When sources are labeled HISTORICAL VERSION, treat them as older data and flag this to the reader.
3. When sources from different document versions disagree on a number, present both values separately with attribution and note which is from the more recent version. Never silently merge conflicting numbers.
4. When the same number appears with different scope qualifiers (e.g. "including development pipeline" vs "under ownership only"), preserve those qualifiers in your answer. Never strip a qualifier to make two numbers look comparable.
5. When a source document is labeled STALE SOURCE, explicitly flag the age of the data before presenting any figures from it.
6. CITATION FORMAT — use proper Bluebook legal citation style for all inline citations:
   • General document: [Document Name, p. XX]
   • Court filing / case: [Smith v. Jones, Exhibit A, p. 12]
   • Epstein corpus filings: [EFTA Filing No. XXX, p. YY]
   • Named document type: [Deposition of John Smith, p. 34] or [Motion to Dismiss, p. 7]
   • If document title and type are both known: [Deposition of Jane Doe, Ex. B, p. 22]
   • Page ranges: [Contract for Services, pp. 14–16]
   Include the document title or case name, the relevant section or exhibit identifier if available,
   and the page number. When the document type is identifiable (deposition, motion, order, contract,
   statute, agreement), include it in the citation. Use [Source N] as a fallback when the document
   title is not available.
7. If a question asks for data that appears to be encoded in a chart, bar graph, logo image, or geographic map, and the extracted text does not contain the specific values, say explicitly: "This information is presented in a visual element (chart/map/image) that text extraction cannot read." Do not guess or fabricate values for visual content.
8. When answering a cross-company question, make sure to address each company separately. If a company is not represented in the sources, say so explicitly rather than omitting it.
9. Keep answers concise. For ordinary questions, lead with the direct answer. For broad legal synthesis, lead with "## Gaps and uncertainties" before the direct answer so the reader sees retrieval limits first.
10. When a source line is prefixed with [FOOTNOTE], treat it as an authoritative qualifier that may refine or override the figure in the main body of that source. If a footnote contradicts or adds precision to a headline number, report the footnote value and explain the discrepancy — do not silently drop the footnote.
11. If two sources from the SAME document give different values for the same metric (e.g. "5,500 customers" on page 3 and "5,000 customers" on page 23), surface both values with their page numbers and explicitly note that the document itself is internally inconsistent. Do not silently pick one.
12. When sources have different document types, apply this authority order for factual figures: Q4 Update > Investor Day > Roadshow > Third-Party Report. A Q4 Update figure supersedes an Investor Day figure on the same metric. A Merger Presentation and a Company Update are concurrent documents covering different scopes — report both, never merge them into one number.
13. Format the answer as clean Markdown for display in a chat UI. Use ## or ### headings only when they improve scanability, use bullet lists for short grouped points, and use **bold** sparingly for key findings. Do not wrap the answer in a code block. Do not escape Markdown characters. Keep citations inline using Bluebook format.
13a. If a source block provides "citation label" and "jurisdiction", use that citation label in inline citations and tag legal issues by jurisdiction where relevant. Use [Source N] only if no citation label is available.
13b. For broad legal synthesis, use these sections in this order: ## Gaps and uncertainties, ## Direct answer, ## Evidence by jurisdiction, ## Actionable follow-up searches, ## Workspace note. Follow-up searches must be concrete actions beginning with "Search for:", "Search corpus for:", "Pull:", or "Verify:".
13c. In Workspace note, describe the retrieved-source universe for this answer: primary document count, secondary/generated context count, unique-document coverage when available, and detected jurisdiction tags.
14. At the very end of your answer, on its own line, output exactly this (fill in N and the rationale):
    <CONFIDENCE>{"score": N, "rationale": "One sentence explaining this confidence level."}</CONFIDENCE>
    Confidence scoring guide:
    5 — Answer is directly and explicitly stated in the source documents with exact quotes available
    4 — Answer is strongly supported by the documents with clear inference
    3 — Answer is partially supported; some inference required
    2 — Answer is weakly supported; significant inference or assumption needed
    1 — Documents contain little relevant information; answer may be unreliable
    Additional confidence caps:
    • Broad legal synthesis based on only one primary document: maximum 3.
    • Broad legal synthesis based on fewer than three primary documents: maximum 4.
    • Any answer based only on secondary context: maximum 3.
    • Any answer with no primary or secondary source support: maximum 1.
    Output ONLY the <CONFIDENCE>...</CONFIDENCE> block after your answer — no other text after it."""


_STALE_YEARS = 2

_JURISDICTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("S.D.N.Y.", re.compile(r"\b(?:sdny|s\.d\.n\.y\.|southern district of new york|giuffre v\. maxwell)\b", re.IGNORECASE)),
    ("S.D. Fla.", re.compile(r"\b(?:s\.d\. fla\.|southern district of florida|acosta|non[- ]prosecution agreement|npa)\b", re.IGNORECASE)),
    ("Florida state", re.compile(r"\b(?:palm beach|florida state|state prosecution|solicitation)\b", re.IGNORECASE)),
    ("USVI", re.compile(r"\b(?:usvi|u\.s\. virgin islands|virgin islands|little saint james)\b", re.IGNORECASE)),
    ("UK", re.compile(r"\b(?:prince andrew|duke of york|uk|united kingdom)\b", re.IGNORECASE)),
]


def _display_source_name(file_name: str) -> str:
    name = unquote(file_name or "").strip()
    name = re.sub(r"\s*\(\d+\)(?=(?:\.[A-Za-z0-9]+)?$)", "", name)
    name = re.sub(r"\.[Pp][Dd][Ff]$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "Source"


def _infer_jurisdiction(chunk: RetrievedChunk) -> str | None:
    text = " ".join(
        str(part or "")
        for part in (
            chunk.file_name,
            chunk.document_type,
            chunk.section_header,
            chunk.chunk_text[:1200],
        )
    )
    for label, pattern in _JURISDICTION_PATTERNS:
        if pattern.search(text):
            return label
    return None


def _citation_label(chunk: RetrievedChunk) -> str:
    source_type = getattr(chunk, "source_type", "document") or "document"
    page_text = f", at {chunk.page_number}" if chunk.page_number else ""
    if source_type == "demo_summary":
        return f"Atticus demo Epstein matter summary{page_text}"
    if source_type == "case_overview":
        return f"AI-generated case overview{page_text}"
    if source_type == "case_timeline":
        return f"AI-generated case timeline{page_text}"
    return f"{_display_source_name(chunk.file_name)}{page_text}"


def build_user_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    image_attachments: list[dict] | None = None,
) -> str:
    if not chunks:
        return (
            f"Question: {question}\n\n"
            "No source material was retrieved for this question. "
            "Reply that the corpus does not appear to contain information relevant to the question."
        )

    today = date.today()

    # Determine which document IDs are superseded by a newer version in this result set.
    superseded_ids: set[str] = {
        chunk.supersedes_document_id
        for chunk in chunks
        if chunk.supersedes_document_id
    }

    source_blocks = []
    primary_doc_ids: set[str] = set()
    secondary_count = 0
    jurisdictions: set[str] = set()
    for index, chunk in enumerate(chunks, start=1):
        company = chunk.company or "unknown company"
        version = chunk.version_label or "unversioned"
        doc_type = chunk.document_type or "document"
        source_type = getattr(chunk, "source_type", "document") or "document"
        source_type_label = {
            "document": "DOCUMENT EVIDENCE",
            "case_overview": "AI-GENERATED OVERVIEW",
            "case_timeline": "AI-GENERATED TIMELINE",
            "demo_summary": "DEMO SUMMARY",
            "public_context": "PUBLIC REFERENCE CONTEXT",
            "user_note": "USER NOTE",
            "chat_memory": "CHAT MEMORY",
        }.get(source_type, source_type.replace("_", " ").upper())
        if source_type == "document":
            primary_doc_ids.add(chunk.document_id or chunk.file_name)
        else:
            secondary_count += 1
        jurisdiction = _infer_jurisdiction(chunk)
        if jurisdiction:
            jurisdictions.add(jurisdiction)

        if chunk.supersedes_document_id:
            version_note = "CURRENT VERSION (supersedes an earlier version)"
        elif chunk.document_id in superseded_ids:
            version_note = "HISTORICAL VERSION (superseded by a more recent document in these sources)"
        else:
            version_note = f"version: {version}"

        # Staleness check — flag sources older than _STALE_YEARS
        stale_note = ""
        if chunk.version_date:
            age_years = (today - chunk.version_date).days / 365
            if age_years > _STALE_YEARS:
                stale_note = f" ⚠ STALE SOURCE (published {chunk.version_date.year}, data may be outdated)"

        slide = f", slide: {chunk.section_header}" if chunk.section_header else ""
        header = (
            f"[Source {index}] {chunk.file_name} "
            f"(company: {company}, type: {doc_type}, {version_note}, "
            f"source type: {source_type_label}, "
            f"citation label: {_citation_label(chunk)}, "
            f"jurisdiction: {jurisdiction or 'not tagged'}, "
            f"page: {chunk.page_number}{slide}{stale_note})"
        )
        source_blocks.append(f"{header}\n{chunk.chunk_text}")

    sources_text = "\n\n".join(source_blocks)
    evidence_profile = (
        "Evidence coverage profile:\n"
        f"- Primary document sources retrieved: {sum(1 for c in chunks if (getattr(c, 'source_type', 'document') or 'document') == 'document')}\n"
        f"- Unique primary documents retrieved: {len(primary_doc_ids)}\n"
        f"- Secondary/generated context sources retrieved: {secondary_count}\n"
        f"- Jurisdiction tags detected: {', '.join(sorted(jurisdictions)) if jurisdictions else 'not tagged from retrieved sources'}\n"
        "Use this profile to calibrate scope and confidence. Thin coverage requires narrower language and lower confidence."
    )
    image_context = ""
    if image_attachments:
        image_blocks = []
        for index, image in enumerate(image_attachments, start=1):
            description = (image.get("description") or "").strip()
            if len(description) > 700:
                description = description[:697].rstrip() + "..."
            image_blocks.append(
                "\n".join(
                    [
                        f"[Image {index}] {image.get('source', 'matched image')}",
                        f"image_id: {image.get('image_id')}",
                        f"document_id: {image.get('document_id')}",
                        f"page: {image.get('page_number')}",
                        f"description: {description or 'No image description available.'}",
                    ]
                )
            )
        image_context = (
            "\n\nAvailable image attachments:\n"
            + "\n\n".join(image_blocks)
            + "\n\nWhen these images are relevant to the question, mention that matching image "
            "attachments are displayed in the UI. Do not claim an image exists unless it appears "
            "in this Available image attachments list. Do not describe visual details beyond the "
            "provided image description."
        )
    return (
        f"Question: {question}\n\n"
        f"{evidence_profile}\n\n"
        f"Sources:\n{sources_text}\n\n"
        f"{image_context}\n\n"
        "Provide an answer that follows all rules. Use Bluebook citation format and clean Markdown. "
        "Prefer the provided citation label for each source over raw [Source N] citations. "
        "Do not use bracket labels like [DEMO SUMMARY, p. 1]; cite secondary context as "
        "[Atticus demo Epstein matter summary, at 1] or with the exact provided citation label. "
        "For legal work, use this structure unless the question is purely conversational:\n"
        "## Gaps and uncertainties\n"
        "## Direct answer\n"
        "## Evidence by jurisdiction\n"
        "## Actionable follow-up searches\n"
        "## Workspace note\n"
        "Quote sparingly; summarize the legal significance and cite the source. "
        "Separate primary document evidence from secondary context. "
        "Tag each legal issue with the source jurisdiction when available; if unavailable, say "
        "\"Jurisdiction not tagged from retrieved source.\" "
        "Make follow-ups concrete lawyer actions, such as exact corpus searches, docket searches, "
        "or document pulls. Start each follow-up with \"Search for:\", \"Search corpus for:\", "
        "\"Pull:\", or \"Verify:\". "
        "In Workspace note, state the retrieved-source universe for this answer, including primary "
        "document count, secondary context count, and detected jurisdiction tags. "
        "Confidence caps: broad legal synthesis based on only one primary document is maximum 3; "
        "broad legal synthesis based on fewer than three primary documents is maximum 4; "
        "answers based only on secondary context are maximum 3. "
        "If sources are only secondary context, say that the answer is based on secondary context and assign confidence no higher than 3. "
        "End with the <CONFIDENCE> block as instructed."
    )
