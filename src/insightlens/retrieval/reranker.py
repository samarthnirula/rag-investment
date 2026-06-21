"""Cross-encoder reranker: re-scores retrieval candidates given the exact query.

Why this exists: cosine similarity scores each chunk independently of the question.
A cross-encoder reads (query, chunk) as a pair and judges relevance jointly,
which catches cases where a chunk is topically similar but not actually the
best evidence for the specific question asked.
"""
from __future__ import annotations

from insightlens.storage.chunk_repository import RetrievedChunk

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_RERANK_THRESHOLD = 0.3  # chunks scoring below this are dropped as irrelevant


class Reranker:
    """Re-scores query-chunk pairs using a cross-encoder for improved precision."""

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model_name = model
        self._model = None

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
        threshold: float = _RERANK_THRESHOLD,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if self._model is None:
            # Deferred: importing sentence_transformers pulls in torch, which at
            # startup (before uvicorn binds its port) was racing Render's
            # port-scan timeout and OOM-killing the container on the 512Mi free
            # tier — same issue as Embedder, see insightlens.embeddings.embedder.
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        pairs = [(query, chunk.chunk_text) for chunk in chunks]
        scores = self._model.predict(pairs).tolist()
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in ranked[:top_k] if score >= threshold]
