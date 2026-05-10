"""Prompt templates for answer generation."""
from __future__ import annotations

from insightlens.storage.chunk_repository import RetrievedChunk

SYSTEM_PROMPT = """You are an analyst assistant that answers questions about investment documents.

Rules you must follow:
1. Ground every factual claim in the provided sources. If the sources do not contain the answer, say so plainly.
2. When sources from different document versions disagree, present each view separately with attribution. Never silently merge conflicting numbers.
3. Cite sources inline using the format [Source N]. Each source corresponds to one entry in the source list.
4. If a question asks about charts, images, or visual content, acknowledge that visual extraction is limited and answer from any available text.
5. Keep answers concise. Lead with the direct answer; supporting detail follows."""


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return (
            f"Question: {question}\n\n"
            "No source material was retrieved for this question. "
            "Reply that the corpus does not appear to contain information relevant to the question."
        )

    source_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        version = chunk.version_label or "unversioned"
        company = chunk.company or "unknown company"
        header = f"[Source {index}] {chunk.file_name} (company: {company}, version: {version}, page: {chunk.page_number})"
        source_blocks.append(f"{header}\n{chunk.chunk_text}")

    sources_text = "\n\n".join(source_blocks)
    return (
        f"Question: {question}\n\n"
        f"Sources:\n{sources_text}\n\n"
        "Provide an answer that follows the rules. Use [Source N] inline citations."
    )
