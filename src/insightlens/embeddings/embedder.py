"""Embedding client — uses Voyage AI (voyage-law-2) when VOYAGE_API_KEY is set,
falls back to local sentence-transformers (all-MiniLM-L6-v2) otherwise.

Voyage AI: input_type="document" for indexing, "query" for queries.
Batching: max 128 texts per Voyage API call (well within the 1,000-text limit).
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import requests

logger = logging.getLogger(__name__)

# voyage-law-2 outputs 1024-dim; all-MiniLM-L6-v2 outputs 384-dim
VOYAGE_DIM = 1024
LOCAL_DIM = 384

# Batch size for Voyage AI. A payment method is now on file, so the account
# uses Voyage's standard tier-1 limits (2000 RPM / 4M TPM for voyage-law-2)
# instead of the unpaid 3 RPM / 10K TPM cap. 64 chunks * 800 tokens = ~51K
# tokens per batch, comfortably under the 4M TPM ceiling.
_VOYAGE_BATCH = 64
_VOYAGE_EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings"
_VOYAGE_TIMEOUT_SECONDS = 30

# Retry transient failures (429 rate limit, 5xx) with exponential backoff.
# Demo uploads can fire several batches back-to-back, which is enough to trip
# Voyage's per-minute limit even on small documents; without a retry, a single
# transient 429 used to fail the whole file.
_VOYAGE_MAX_RETRIES = 5
_VOYAGE_BACKOFF_BASE_SECONDS = 1.0

# Pause between consecutive batch requests so a multi-batch document doesn't
# burst past Voyage's requests/tokens-per-minute limit before any 429 occurs.
_VOYAGE_INTER_BATCH_DELAY_SECONDS = 0.5

# Global cap on Voyage requests/minute, shared across every FastAPI worker and
# Celery process via Redis. Several documents can embed concurrently (Celery
# runs with --concurrency=4), so a per-document delay alone doesn't stop the
# combined request rate from exceeding the account's RPM limit.
#
# With a payment method on file, voyage-law-2's standard tier-1 limit is
# 2000 RPM. Capped well below that — this app's own concurrency (4 Celery
# workers) is nowhere near enough to need more than a few hundred RPM, and
# staying conservative leaves headroom for other Voyage usage on the account.
_VOYAGE_GLOBAL_RPM_LIMIT = 120
_VOYAGE_GLOBAL_RPM_KEY = "voyage:rpm:global"


# ── Redis-backed global throttle (shared across processes) ────────────────────

_redis_lock = threading.Lock()
_redis_client: Optional["redis.Redis"] = None  # type: ignore[name-defined]
_redis_unavailable = False
_redis_retry_at: float = 0.0


def _get_redis():
    """Return a connected Redis client, or None if Redis is unreachable."""
    global _redis_client, _redis_unavailable, _redis_retry_at

    if _redis_client is not None:
        return _redis_client

    now = time.monotonic()
    if _redis_unavailable and now < _redis_retry_at:
        return None

    with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        try:
            import redis as _redis_mod
            url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            client = _redis_mod.Redis.from_url(
                url, socket_connect_timeout=1, socket_timeout=1, decode_responses=True,
            )
            client.ping()
            _redis_client = client
            _redis_unavailable = False
        except Exception as exc:
            _redis_unavailable = True
            _redis_retry_at = time.monotonic() + 60
            logger.warning("Voyage throttle: Redis unavailable, falling back to local pacing only: %s", exc)
        return _redis_client


def _throttle_voyage_request() -> None:
    """Block until a slot is free in the shared per-minute Voyage budget.

    Uses a Redis sorted set as a sliding window shared by every process that
    calls Voyage. Fails open (no wait) if Redis is unreachable, since the
    per-batch delay and retry/backoff still provide some protection.
    """
    rc = _get_redis()
    if rc is None:
        return

    while True:
        now = time.time()
        member = f"{now:.6f}:{random.getrandbits(32)}"
        try:
            pipe = rc.pipeline()
            pipe.zremrangebyscore(_VOYAGE_GLOBAL_RPM_KEY, 0, now - 60)
            pipe.zcard(_VOYAGE_GLOBAL_RPM_KEY)
            pipe.zadd(_VOYAGE_GLOBAL_RPM_KEY, {member: now})
            pipe.expire(_VOYAGE_GLOBAL_RPM_KEY, 61)
            _, count, _, _ = pipe.execute()
        except Exception as exc:
            logger.warning("Voyage throttle: Redis check failed, proceeding without wait: %s", exc)
            return

        if int(count) < _VOYAGE_GLOBAL_RPM_LIMIT:
            return

        # Over budget: undo our provisional slot and wait before retrying.
        try:
            rc.zrem(_VOYAGE_GLOBAL_RPM_KEY, member)
        except Exception:  # nosec B110 - best-effort cleanup; safe to skip if Redis is down
            pass
        time.sleep(1.0)

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
            # Use Voyage's REST API directly. Importing the official Python SDK
            # currently imports sentence-transformers, torch, transformers,
            # pandas, and numpy even though remote embeddings do not need them.
            # That consumes roughly 300+ MiB at process startup and exceeds the
            # Render free-tier memory limit once FastAPI/Firebase are loaded.
            self._voyage_key = voyage_key
            self._backend = "voyage"
            self._voyage_model = "voyage-law-2"
            self._dim = VOYAGE_DIM
            self._batch_size = _VOYAGE_BATCH
            logger.info(
                "Embedder: using Voyage AI REST API (model=voyage-law-2, dim=%d)",
                VOYAGE_DIM,
            )
        else:
            # Defer the actual SentenceTransformer/torch import+load until the
            # first embed call instead of doing it here in __init__. __init__
            # runs during app startup, before uvicorn binds its port — on
            # memory-constrained hosts (e.g. Render free tier), loading torch
            # eagerly at boot was racing the platform's port-scan timeout and
            # causing an OOM kill before the server ever came up, even on
            # deploys that never end up using local embeddings. No behavior
            # change for embedding correctness, just moves *when* the load
            # happens.
            self._st_model = None
            self._backend = "local"
            self._local_model_name = model
            self._dim = LOCAL_DIM
            self._batch_size = batch_size
            logger.info(
                "Embedder: using local SentenceTransformer (model=%s, dim=%d, lazy-loaded)",
                model,
                LOCAL_DIM,
            )

    @property
    def vector_dim(self) -> int:
        return self._dim

    # ── Public interface ──────────────────────────────────────────────────────

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        """Embed a list of document texts (for indexing). Batches automatically."""
        text_list = list(texts)
        results: list[EmbeddingResult] = []
        for start in range(0, len(text_list), self._batch_size):
            if start > 0 and self._backend == "voyage":
                time.sleep(_VOYAGE_INTER_BATCH_DELAY_SECONDS)
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
        last_exc: Exception | None = None
        for attempt in range(_VOYAGE_MAX_RETRIES + 1):
            try:
                _throttle_voyage_request()
                response = requests.post(
                    _VOYAGE_EMBEDDINGS_URL,
                    headers={
                        "Authorization": f"Bearer {self._voyage_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "input": batch,
                        "model": self._voyage_model,
                        "input_type": input_type,
                    },
                    timeout=_VOYAGE_TIMEOUT_SECONDS,
                )

                if response.status_code == 429 or response.status_code >= 500:
                    last_exc = requests.HTTPError(
                        f"{response.status_code} {response.reason} for url: {response.url}",
                        response=response,
                    )
                    if attempt < _VOYAGE_MAX_RETRIES:
                        retry_after = response.headers.get("Retry-After")
                        delay = (
                            float(retry_after)
                            if retry_after and retry_after.replace(".", "", 1).isdigit()
                            else _VOYAGE_BACKOFF_BASE_SECONDS * (2 ** attempt)
                        )
                        logger.warning(
                            "Voyage AI returned %s for batch of %d (attempt %d/%d), retrying in %.1fs",
                            response.status_code,
                            len(batch),
                            attempt + 1,
                            _VOYAGE_MAX_RETRIES,
                            delay,
                        )
                        time.sleep(delay)
                        continue
                    break

                response.raise_for_status()
                payload = response.json()
                data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
                vectors = [item["embedding"] for item in data]
                if len(vectors) != len(batch):
                    raise ValueError(
                        f"Voyage returned {len(vectors)} embeddings for {len(batch)} inputs."
                    )
                return [
                    EmbeddingResult(text=text, vector=vec)
                    for text, vec in zip(batch, vectors)
                ]
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt < _VOYAGE_MAX_RETRIES and isinstance(exc, requests.ConnectionError):
                    delay = _VOYAGE_BACKOFF_BASE_SECONDS * (2 ** attempt)
                    logger.warning(
                        "Voyage AI connection error for batch of %d (attempt %d/%d), retrying in %.1fs: %s",
                        len(batch),
                        attempt + 1,
                        _VOYAGE_MAX_RETRIES,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue
                break

        raise EmbeddingError(
            f"Voyage AI embedding failed for batch of {len(batch)}: {last_exc}"
        ) from last_exc

    def _embed_local(self, batch: list[str]) -> list[EmbeddingResult]:
        try:
            if self._st_model is None:
                # See __init__: load is deferred to here, on first actual use.
                os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(self._local_model_name)
            vectors = self._st_model.encode(batch, convert_to_numpy=True)
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding failed for batch of {len(batch)}: {exc}"
            ) from exc
        return [
            EmbeddingResult(text=text, vector=vec.tolist())
            for text, vec in zip(batch, vectors)
        ]
