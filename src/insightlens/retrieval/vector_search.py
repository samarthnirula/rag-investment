"""High-level retrieval orchestration: embed query -> search -> return ranked chunks."""
from __future__ import annotations

from dataclasses import dataclass

from insightlens.embeddings.embedder import Embedder
from insightlens.storage.chunk_repository import ChunkRepository, RetrievedChunk


class RetrievalError(Exception):
    """Raised when the retrieval pipeline fails."""


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    top_k: int
    company_filter: str | None = None


class VectorSearchService:
    """Coordinates query embedding and vector lookup."""

    def __init__(self, embedder: Embedder, repository: ChunkRepository) -> None:
        self._embedder = embedder
        self._repository = repository

    def retrieve(self, request: RetrievalRequest) -> list[RetrievedChunk]:
        if not request.query.strip():
            raise RetrievalError("Query is empty.")
        if request.top_k <= 0:
            raise RetrievalError(f"top_k must be positive, got {request.top_k}")

        query_vector = self._embedder.embed_query(request.query)
        return self._repository.search_similar(
            query_embedding=query_vector,
            top_k=request.top_k,
            company_filter=request.company_filter,
        )
