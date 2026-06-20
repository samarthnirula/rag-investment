"""Embedding client — uses Voyage AI (voyage-law-2) when VOYAGE_API_KEY is set,
falls back to local sentence-transformers (all-MiniLM-L6-v2) otherwise.

Voyage AI: input_type="document" for indexing, "query" for queries.
Batching: max 128 texts per Voyage API call (well within the 1,000-text limit).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Sequence

logger = logging.getLogger(__name__)

# voyage-law-2 outputs 1024-dim; all-MiniLM-L6-v2 outputs 384-dim
VOYAGE_DIM = 1024
LOCAL_DIM = 384

# Safe batch size for Voyage AI (rate limit: 1,000 texts per request, 120K tokens/min)
_VOYAGE_BATCH = 128

# Public constant callers can check
VECTOR_DIM: int  # set in Embedder.__init__ as a class-level property


class EmbeddingError(Exception):
    """Raised when embeddings cannot be generated."""


@dataclass(frozen=True)
class EmbeddingResult:
    text: str
    vector: list[float]


class Embedder:
    """Wraps either the Voyage AI API or a local SentenceTransformer model.

    Public methods:
        embed_texts(texts)   -> list[EmbeddingResult]   (for indexing documents)
        embed_documents(texts) -> list[EmbeddingResult] (alias for embed_texts)
        embed_query(text)    -> list[float]              (for querying)

    The backend is selected at init time based on VOYAGE_API_KEY env var:
        - Non-empty VOYAGE_API_KEY  → Voyage AI voyage-law-2 (1024-dim)
        - Empty / unset             → local SentenceTransformer (384-dim)
    """

    def __init__(self, model: str, batch_size: int = 64) -> None:
        if not model:
            raise EmbeddingError("Embedding model name is empty.")

        voyage_key = os.getenv("VOYAGE_API_KEY", "").strip()
        if voyage_key:
            try:
                import voyageai
                self._voyage_client = voyageai.Client(api_key=voyage_key)
                self._backend = "voyage"
                self._voyage_model = "voyage-law-2"
                self._dim = VOYAGE_DIM
                self._batch_size = _VOYAGE_BATCH
                logger.info(
                    "Embedder: using Voyage AI (model=voyage-law-2, dim=%d)", VOYAGE_DIM
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to initialise Voyage AI client: {exc}"
                ) from exc
        else:
            try:
                # Local SentenceTransformer pulls in joblib's "loky" multiprocessing
                # backend, which spawns background worker processes. Under uvicorn
                # --reload, each file-save restart kills the process before those
                # workers shut down cleanly, producing a harmless but noisy
                # "leaked semaphore objects" warning on exit. Capping loky to a
                # single process avoids spawning the pool at all — no behavior
                # change for embedding correctness, just removes the dev-only noise.
                os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(model)
                self._backend = "local"
                self._local_model_name = model
                self._dim = LOCAL_DIM
                self._batch_size = batch_size
                logger.info(
                    "Embedder: using local SentenceTransformer (model=%s, dim=%d)",
                    model,
                    LOCAL_DIM,
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load embedding model '{model}': {exc}"
                ) from exc

    @property
    def vector_dim(self) -> int:
        return self._dim

    # ── Public interface ──────────────────────────────────────────────────────

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        """Embed a list of document texts (for indexing). Batches automatically."""
        text_list = list(texts)
        results: list[EmbeddingResult] = []
        for start in range(0, len(text_list), self._batch_size):
            batch = text_list[start : start + self._batch_size]
            results.extend(self._embed_batch(batch, input_type="document"))
        return results

    def embed_documents(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        """Alias for embed_texts — use this name when indexing documents."""
        return self.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query. Uses query-optimised input_type for Voyage."""
        if not query.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        return self._embed_batch([query], input_type="query")[0].vector

    # ── Backend dispatch ──────────────────────────────────────────────────────

    def _embed_batch(
        self, batch: list[str], input_type: str = "document"
    ) -> list[EmbeddingResult]:
        if self._backend == "voyage":
            return self._embed_voyage(batch, input_type)
        return self._embed_local(batch)

    def _embed_voyage(self, batch: list[str], input_type: str) -> list[EmbeddingResult]:
        try:
            response = self._voyage_client.embed(
                batch,
                model=self._voyage_model,
                input_type=input_type,
            )
            return [
                EmbeddingResult(text=text, vector=vec)
                for text, vec in zip(batch, response.embeddings)
            ]
        except Exception as exc:
            raise EmbeddingError(
                f"Voyage AI embedding failed for batch of {len(batch)}: {exc}"
            ) from exc

    def _embed_local(self, batch: list[str]) -> list[EmbeddingResult]:
        try:
            vectors = self._st_model.encode(batch, convert_to_numpy=True)
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding failed for batch of {len(batch)}: {exc}"
            ) from exc
        return [
            EmbeddingResult(text=text, vector=vec.tolist())
            for text, vec in zip(batch, vectors)
        ]
