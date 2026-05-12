"""Cross-encoder reranker: re-scores retrieval candidates given the exact query.

Why this exists: cosine similarity scores each chunk independently of the question.
A cross-encoder reads (query, chunk) as a pair and judges relevance jointly,
which catches cases where a chunk is topically similar but not actually the
best evidence for the specific question asked.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from insightlens.storage.chunk_repository import RetrievedChunk

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Re-scores query-chunk pairs using a cross-encoder for improved precision."""

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model = CrossEncoder(model)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, chunk.chunk_text) for chunk in chunks]
        scores = self._model.predict(pairs).tolist()
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]
