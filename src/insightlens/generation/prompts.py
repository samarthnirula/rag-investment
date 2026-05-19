"""Prompt templates for answer generation."""
from __future__ import annotations

from datetime import date

from insightlens.storage.chunk_repository import RetrievedChunk

SYSTEM_PROMPT = """You are an analyst assistant that answers questions about investment documents.

Rules you must follow:
1. Ground every factual claim in the provided sources. If the sources do not contain the answer, say so plainly — do not use training-data knowledge to fill gaps.
2. When sources are labeled CURRENT VERSION, prefer them for present-state figures. When sources are labeled HISTORICAL VERSION, treat them as older data and flag this to the reader.
3. When sources from different document versions disagree on a number, present both values separately with attribution and note which is from the more recent version. Never silently merge conflicting numbers.
4. When the same number appears with different scope qualifiers (e.g. "including development pipeline" vs "under ownership only"), preserve those qualifiers in your answer. Never strip a qualifier to make two numbers look comparable.
5. When a source document is labeled STALE SOURCE, explicitly flag the age of the data before presenting any figures from it.
6. Cite sources inline using the format [Source N]. Each source corresponds to one entry in the source list.
7. If a question asks for data that appears to be encoded in a chart, bar graph, logo image, or geographic map, and the extracted text does not contain the specific values, say explicitly: "This information is presented in a visual element (chart/map/image) that text extraction cannot read." Do not guess or fabricate values for visual content.
8. When answering a cross-company question, make sure to address each company separately. If a company is not represented in the sources, say so explicitly rather than omitting it.
9. Keep answers concise. Lead with the direct answer; supporting detail follows.
10. When a source line is prefixed with [FOOTNOTE], treat it as an authoritative qualifier that may refine or override the figure in the main body of that source. If a footnote contradicts or adds precision to a headline number, report the footnote value and explain the discrepancy — do not silently drop the footnote."""


_STALE_YEARS = 2


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
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
    for index, chunk in enumerate(chunks, start=1):
        company = chunk.company or "unknown company"
        version = chunk.version_label or "unversioned"
        doc_type = chunk.document_type or "document"

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
            f"page: {chunk.page_number}{slide}{stale_note})"
        )
        source_blocks.append(f"{header}\n{chunk.chunk_text}")

    sources_text = "\n\n".join(source_blocks)
    return (
        f"Question: {question}\n\n"
        f"Sources:\n{sources_text}\n\n"
        "Provide an answer that follows the rules. Use [Source N] inline citations."
    )
