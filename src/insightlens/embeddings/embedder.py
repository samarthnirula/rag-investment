"""Local sentence-transformer embedding client — no API key required."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sentence_transformers import SentenceTransformer


VECTOR_DIM = 384  # dimension produced by all-MiniLM-L6-v2


class EmbeddingError(Exception):
    """Raised when embeddings cannot be generated."""


@dataclass(frozen=True)
class EmbeddingResult:
    text: str
    vector: list[float]


class Embedder:
    """Wraps a local SentenceTransformer model with batched encoding."""

    def __init__(self, model: str, batch_size: int = 64) -> None:
        if not model:
            raise EmbeddingError("Embedding model name is empty.")
        if batch_size <= 0:
            raise EmbeddingError(f"batch_size must be positive, got {batch_size}")
        try:
            self._model = SentenceTransformer(model)
        except Exception as exc:
            raise EmbeddingError(f"Failed to load embedding model '{model}': {exc}") from exc
        self._batch_size = batch_size

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        results: list[EmbeddingResult] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            results.extend(self._embed_batch(batch))
        return results

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        return self._embed_batch([query])[0].vector

    def _embed_batch(self, batch: list[str]) -> list[EmbeddingResult]:
        try:
            vectors = self._model.encode(batch, convert_to_numpy=True)
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed for batch of {len(batch)}: {exc}") from exc
        return [
            EmbeddingResult(text=text, vector=vec.tolist())
            for text, vec in zip(batch, vectors)
        ]
