"""Prompt templates for answer generation."""
from __future__ import annotations

from insightlens.storage.chunk_repository import RetrievedChunk

SYSTEM_PROMPT = """You are an analyst assistant that answers questions about investment documents.

Rules you must follow:
1. Ground every factual claim in the provided sources. If the sources do not contain the answer, say so plainly.
2. When sources are labeled HISTORICAL VERSION, treat them as older data. Prefer CURRENT VERSION sources for present-state figures.
3. When sources from different document versions disagree on a number, present both values separately with attribution and note which is from the more recent version. Never silently merge conflicting numbers.
4. Cite sources inline using the format [Source N]. Each source corresponds to one entry in the source list.
5. If a question asks about charts, images, or visual content, acknowledge that visual extraction is limited and answer from any available text.
6. Keep answers concise. Lead with the direct answer; supporting detail follows."""


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return (
            f"Question: {question}\n\n"
            "No source material was retrieved for this question. "
            "Reply that the corpus does not appear to contain information relevant to the question."
        )

    # Determine which document IDs are superseded by a newer version in this result set.
    # A chunk with supersedes_document_id=X means its document is newer than document X.
    superseded_ids: set[str] = {
        chunk.supersedes_document_id
        for chunk in chunks
        if chunk.supersedes_document_id
    }

    source_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        company = chunk.company or "unknown company"
        version = chunk.version_label or "unversioned"

        if chunk.supersedes_document_id:
            version_note = "CURRENT VERSION (supersedes an earlier version)"
        elif chunk.document_id in superseded_ids:
            version_note = "HISTORICAL VERSION (superseded by a more recent document in these sources)"
        else:
            version_note = f"version: {version}"

        slide = f", slide: {chunk.section_header}" if chunk.section_header else ""
        header = (
            f"[Source {index}] {chunk.file_name} "
            f"(company: {company}, {version_note}, page: {chunk.page_number}{slide})"
        )
        source_blocks.append(f"{header}\n{chunk.chunk_text}")

    sources_text = "\n\n".join(source_blocks)
    return (
        f"Question: {question}\n\n"
        f"Sources:\n{sources_text}\n\n"
        "Provide an answer that follows the rules. Use [Source N] inline citations."
    )
