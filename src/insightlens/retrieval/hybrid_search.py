"""Hybrid retrieval: BM25 keyword search + vector similarity, fused with RRF.

Pipeline (in order):
1. Vector search      — top-N candidates ranked by cosine similarity.
2. BM25 search        — same corpus ranked by exact-token overlap.
3. RRF fusion         — score = 1/(k + rank_vector) + 1/(k + rank_bm25).
                        Chunks near the top of both lists score highest.
4. Similarity floor   — vector candidates below the cosine threshold are dropped.
5. Version scoring    — chunks from the CURRENT document version get a boost;
                        chunks from a SUPERSEDED version get a penalty.
6. Chunk-type scoring — financial_table chunks get a boost for numeric/financial
                        queries; body chunks get a small boost for narrative
                        queries.  This lets structured artifacts (tables, data
                        slides) surface ahead of prose when the question is
                        asking for a specific figure or metric.
7. Cross-encoder rerank — final pass reads (query, chunk) as a joint pair,
                        catching cases where a chunk is topically similar but
                        not the best evidence for the specific question asked.
8. Deduplication      — keeps only the highest-ranked chunk per (document, page).
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from insightlens.embeddings.embedder import Embedder
from insightlens.retrieval.reranker import Reranker
from insightlens.retrieval.vector_search import RetrievalRequest
from insightlens.storage.chunk_repository import ChunkRepository, RetrievedChunk

_RRF_K = 60                   # dampens rank differences; 60 is the standard choice
_SIMILARITY_THRESHOLD = 0.35  # cosine floor — vector candidates below this are dropped
_CANDIDATE_MULTIPLIER = 3     # over-fetch before reranking

# Version-currency score multipliers.
_VERSION_BOOST = 1.15
_VERSION_PENALTY = 0.80

# Chunk-type score multipliers applied when the query is classified as numeric.
# financial_table gets the largest boost — it's the most authoritative source
# for specific figures.  chart_caption sits in between — numbers present but
# less structured.  body gets a small penalty for numeric queries because a
# paragraph that *mentions* FFO is less authoritative than the actual table.
# For narrative queries the adjustment is reversed: body chunks are preferred.
_TABLE_BOOST_NUMERIC = 1.25    # financial_table  on a numeric  query
_CHART_BOOST_NUMERIC = 1.08    # chart_caption    on a numeric  query
_BODY_PENALTY_NUMERIC = 0.88   # body             on a numeric  query
_BODY_BOOST_NARRATIVE = 1.10   # body             on a narrative query
_TABLE_PENALTY_NARRATIVE = 0.95  # financial_table on a narrative query (slight)

# Signals that mark a query as asking for a specific financial figure or metric.
_NUMERIC_QUERY_RE = re.compile(
    r"\b(?:"
    # financial metrics and their abbreviations
    r"revenue|earnings|income|ebitda|noi|ffo|affo|ebit|ebita|"
    r"dividend|yield|per\s+share|occupancy|leased|"
    r"growth|guidance|target|forecast|projection|outlook|"
    r"margin|ratio|rate|return|cap\s+rate|spread|"
    r"million|billion|trillion|quarterly|annual|fiscal|"
    r"metric|kpi|financial|figure|number|amount|total|"
    r"basis\s+points|bps|percent|percentage|"
    # question patterns that signal a numeric answer
    r"how\s+much|how\s+many|"
    r"what\s+(?:are|were)\s+(?:the\s+)?(?:key\s+)?(?:operating|financial|"
    r"performance)\s+metrics"
    r")\b"
    r"|[\$\%]"           # explicit currency or percent symbol
    r"|\d+\.?\d*[xX]",   # multiplier notation like 3.5x
    re.IGNORECASE,
)


class HybridSearchService:
    """BM25 + vector search fused with RRF, version-scored, then cross-encoder reranked."""

    def __init__(
        self,
        embedder: Embedder,
        repository: ChunkRepository,
        corpus_chunks: list[RetrievedChunk],
        reranker: Reranker | None = None,
    ) -> None:
        self._embedder = embedder
        self._repository = repository
        self._reranker = reranker

        # Build BM25 index once from the preloaded corpus.
        self._corpus = corpus_chunks
        tokenized = [c.chunk_text.lower().split() for c in corpus_chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def retrieve(self, request: RetrievalRequest) -> list[RetrievedChunk]:
        candidate_k = request.top_k * _CANDIDATE_MULTIPLIER

        # ── 1. Vector search ──────────────────────────────────────────────────
        query_vector = self._embedder.embed_query(request.query)
        vector_candidates = self._repository.search_similar(
            query_embedding=query_vector,
            top_k=candidate_k,
            company_filter=request.company_filter,
        )
        vector_candidates = [
            c for c in vector_candidates if c.similarity >= _SIMILARITY_THRESHOLD
        ]

        # ── 2. BM25 search ────────────────────────────────────────────────────
        bm25_candidates = self._bm25_search(request.query, request.company_filter, candidate_k)

        # ── 3. RRF fusion ─────────────────────────────────────────────────────
        fused_scored = self._rrf_fuse(vector_candidates, bm25_candidates)

        # ── 4. Version-aware re-scoring ───────────────────────────────────────
        fused_scored = self._apply_version_scores(fused_scored)

        # ── 5. Chunk-type re-scoring ──────────────────────────────────────────
        # Boosts structured chunks (tables) for numeric queries and prose chunks
        # for narrative queries.  The caller can override auto-detection by
        # setting preferred_chunk_types on the request.
        fused_scored = self._apply_chunk_type_scores(fused_scored, request)

        # Strip scores — reranker and deduplicator work on chunk lists.
        fused = [chunk for chunk, _ in fused_scored]

        # ── 6 & 7. Rerank → deduplicate ───────────────────────────────────────
        if self._reranker and fused:
            reranked = self._reranker.rerank(request.query, fused, request.top_k * 2)
            return self._deduplicate(reranked)[: request.top_k]
        return self._deduplicate(fused)[: request.top_k]

    # ── Internals ──────────────────────────────────────────────────────────────

    def _bm25_search(
        self, query: str, company_filter: str | None, top_k: int
    ) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results: list[RetrievedChunk] = []
        for idx, score in ranked:
            if score <= 0:
                break
            chunk = self._corpus[idx]
            if company_filter and (chunk.company or "").upper() != company_filter.upper():
                continue
            results.append(chunk)
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _is_numeric_query(query: str) -> bool:
        """Return True when the query is asking for a specific financial figure."""
        return bool(_NUMERIC_QUERY_RE.search(query))

    def _apply_chunk_type_scores(
        self,
        scored: list[tuple[RetrievedChunk, float]],
        request: RetrievalRequest,
    ) -> list[tuple[RetrievedChunk, float]]:
        """Adjust scores so structured chunks surface for numeric questions.

        Query classification (auto, overridable):
          numeric  — mentions a financial metric, currency symbol, or percentage.
                     financial_table gets the highest boost; body gets a penalty.
          narrative — qualitative / strategy questions.
                     body chunks get a small boost; table chunks a slight penalty.

        If the caller sets preferred_chunk_types on the request, those chunk
        types receive the numeric-query boost unconditionally, bypassing
        auto-detection.  This lets a UI filter (e.g. "search tables only")
        propagate into scoring without a separate code path.
        """
        # Determine effective chunk-type preferences.
        if request.preferred_chunk_types:
            # Explicit caller override: boost requested types, no penalties.
            result = []
            for chunk, score in scored:
                if chunk.chunk_type in request.preferred_chunk_types:
                    result.append((chunk, score * _TABLE_BOOST_NUMERIC))
                else:
                    result.append((chunk, score))
            result.sort(key=lambda x: x[1], reverse=True)
            return result

        is_numeric = self._is_numeric_query(request.query)

        result = []
        for chunk, score in scored:
            ct = chunk.chunk_type
            if is_numeric:
                if ct == "financial_table":
                    result.append((chunk, score * _TABLE_BOOST_NUMERIC))
                elif ct == "chart_caption":
                    result.append((chunk, score * _CHART_BOOST_NUMERIC))
                else:  # body
                    result.append((chunk, score * _BODY_PENALTY_NUMERIC))
            else:
                # Narrative query: prefer prose, mildly de-prioritise raw tables.
                if ct == "body":
                    result.append((chunk, score * _BODY_BOOST_NARRATIVE))
                elif ct == "financial_table":
                    result.append((chunk, score * _TABLE_PENALTY_NARRATIVE))
                else:
                    result.append((chunk, score))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def _rrf_fuse(
        self,
        vector_chunks: list[RetrievedChunk],
        bm25_chunks: list[RetrievedChunk],
    ) -> list[tuple[RetrievedChunk, float]]:
        """Compute RRF scores and return (chunk, score) pairs sorted descending.

        Returning the raw scores (not just the ranked list) lets the version
        scorer apply meaningful multipliers rather than working from ordinal ranks.
        """
        scores: dict[str, float] = {}
        seen: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(vector_chunks, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
            seen[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(bm25_chunks, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
            if chunk.chunk_id not in seen:
                seen[chunk.chunk_id] = chunk

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(seen[cid], score) for cid, score in ranked]

    def _apply_version_scores(
        self,
        scored: list[tuple[RetrievedChunk, float]],
    ) -> list[tuple[RetrievedChunk, float]]:
        """Adjust RRF scores based on document version currency.

        Definitions (derived entirely from the candidate pool — no extra DB call):

          CURRENT   — this document supersedes an older one AND nothing in the
                      pool supersedes it in turn.  Gets _VERSION_BOOST.

          SUPERSEDED — another document in the pool explicitly lists this
                      document's ID as the one it supersedes.
                      Gets _VERSION_PENALTY.

          UNVERSIONED — no supersession relationship detected in the pool.
                      Score unchanged.  This is the common case when only one
                      version of a document exists.

        Multi-hop correctness (V1 → V2 → V3):
          superseded_doc_ids = {V1, V2}
          V2: supersedes_document_id="V1" ✓ but document_id="V2" ∈ superseded_doc_ids
              → not CURRENT, is SUPERSEDED → penalty ✓
          V3: supersedes_document_id="V2" ✓ and document_id="V3" ∉ superseded_doc_ids
              → CURRENT → boost ✓
        """
        # Collect all document IDs that are explicitly superseded within this pool.
        superseded_doc_ids: set[str] = {
            chunk.supersedes_document_id
            for chunk, _ in scored
            if chunk.supersedes_document_id
        }

        result: list[tuple[RetrievedChunk, float]] = []
        for chunk, score in scored:
            is_current = (
                chunk.supersedes_document_id is not None           # supersedes something
                and chunk.document_id not in superseded_doc_ids    # nothing supersedes this
            )
            is_superseded = chunk.document_id in superseded_doc_ids

            if is_current:
                result.append((chunk, score * _VERSION_BOOST))
            elif is_superseded:
                result.append((chunk, score * _VERSION_PENALTY))
            else:
                result.append((chunk, score))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def _deduplicate(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Keep only the highest-ranked chunk per (document, page) pair.

        The SlideAwareChunker produces both a text chunk and a table chunk per
        page. When both rank highly, this prevents the same page appearing twice
        in results — wasting a retrieval slot that should go to a different source.
        """
        seen: set[tuple[str, int]] = set()
        result: list[RetrievedChunk] = []
        for chunk in chunks:
            key = (chunk.document_id, chunk.page_number)
            if key not in seen:
                seen.add(key)
                result.append(chunk)
        return result
