"""Combines retrieval + generation and packages the response with citations."""
from __future__ import annotations

from dataclasses import dataclass

from insightlens.generation.llm_client import ClaudeClient
from insightlens.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from insightlens.retrieval.vector_search import RetrievalRequest, VectorSearchService
from insightlens.storage.chunk_repository import RetrievedChunk


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
        chunks = self._retrieval.retrieve(
            RetrievalRequest(
                query=question,
                top_k=self._default_top_k,
                company_filter=company_filter,
            )
        )

        user_prompt = build_user_prompt(question, chunks)
        answer_text = self._llm.generate(SYSTEM_PROMPT, user_prompt)

        citations = [
            Citation(
                label=f"Source {index}",
                file_name=chunk.file_name,
                company=chunk.company,
                version_label=chunk.version_label,
                page_number=chunk.page_number,
                similarity=chunk.similarity,
            )
            for index, chunk in enumerate(chunks, start=1)
        ]

        return AnswerWithSources(
            answer_text=answer_text,
            citations=citations,
            retrieved_chunks=chunks,
        )
