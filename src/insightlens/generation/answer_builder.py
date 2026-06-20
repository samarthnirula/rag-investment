"""Combines retrieval + generation and packages the response with citations."""
from __future__ import annotations

import re
from dataclasses import dataclass

from insightlens.generation.llm_client import ClaudeClient
from insightlens.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from insightlens.retrieval.vector_search import RetrievalRequest, VectorSearchService
from insightlens.storage.chunk_repository import RetrievedChunk

_CITATION_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)
_NOISE_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "what", "who",
    "where", "when", "how", "why", "and", "or", "but", "if", "in",
    "on", "at", "to", "for", "of", "with", "by", "not", "no",
})


def _parse_cited_indices(answer_text: str) -> set[int]:
    """Return the set of 1-based Source indices the LLM actually cited."""
    return {int(m.group(1)) for m in _CITATION_RE.finditer(answer_text)}


@dataclass(frozen=True)
class Citation:
    label: str
    file_name: str
    company: str | None
    version_label: str | None
    page_number: int
    similarity: float


@dataclass(frozen=True)
class AnswerWithSources:
    answer_text: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]


class AnswerService:
    """End-to-end question answering."""

    def __init__(
        self,
        retrieval: VectorSearchService,
        llm: ClaudeClient,
        default_top_k: int,
    ) -> None:
        self._retrieval = retrieval
        self._llm = llm
        self._default_top_k = default_top_k

    def answer(self, question: str, company_filter: str | None = None) -> AnswerWithSources:
        # Minimum query quality guard
        meaningful = [t for t in question.lower().split() if len(t) > 2 and t not in _NOISE_WORDS]
        if len(meaningful) < 2:
            raise ValueError(
                "Query is too short or contains only common words. "
                "Please ask a more specific question."
            )

        chunks = self._retrieval.retrieve(
            RetrievalRequest(
                query=question,
                top_k=self._default_top_k,
                company_filter=company_filter,
            )
        )

        user_prompt = build_user_prompt(question, chunks)
        answer_text = self._llm.generate(SYSTEM_PROMPT, user_prompt)

        # Filter to only chunks the LLM actually cited — fall back to all if none found.
        cited_indices = _parse_cited_indices(answer_text)
        indexed_chunks = list(enumerate(chunks, start=1))
        if cited_indices:
            indexed_chunks = [(i, c) for i, c in indexed_chunks if i in cited_indices]

        citations = [
            Citation(
                label=f"Source {index}",
                file_name=chunk.file_name,
                company=chunk.company,
                version_label=chunk.version_label,
                page_number=chunk.page_number,
                similarity=chunk.similarity,
            )
            for index, chunk in indexed_chunks
        ]

        return AnswerWithSources(
            answer_text=answer_text,
            citations=citations,
            retrieved_chunks=[chunk for _, chunk in indexed_chunks],
        )
