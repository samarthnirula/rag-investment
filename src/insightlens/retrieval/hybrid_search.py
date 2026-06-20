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

# ── BM25 preprocessing ────────────────────────────────────────────────────────
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "and", "or", "but", "if",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
    "about", "than", "as", "it", "its", "he", "she", "they", "we", "you",
    "i", "me", "my", "our", "your", "his", "her", "their", "this", "that",
    "these", "those", "what", "which", "who", "where", "when", "how", "why",
    "not", "no", "so", "out", "into", "over", "after", "then", "there",
    "just", "also", "any", "all", "each", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "too", "very", "get", "got",
    "per", "its", "use", "used", "using", "make", "made", "given", "show",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace, remove stopwords and tokens under 3 chars."""
    return [t for t in text.lower().split() if len(t) >= 3 and t not in _STOPWORDS]


# ── Compound query splitting ───────────────────────────────────────────────────
_MULTI_Q_RE = re.compile(r"\?\s+")


def _split_compound_query(query: str) -> list[str]:
    """Split a compound multi-question query at '?' boundaries.

    Only treats the input as compound if every resulting part has at least
    3 words; single questions that happen to contain a '?' mid-sentence are
    left intact.
    """
    parts = _MULTI_Q_RE.split(query.strip())
    clean = [p.strip() for p in parts if len(p.strip().split()) >= 3]
    if len(clean) <= 1:
        return [query]
    return [p if p.endswith("?") else p + "?" for p in clean]


from insightlens.embeddings.embedder import Embedder
from insightlens.retrieval.reranker import Reranker
from insightlens.retrieval.vector_search import RetrievalRequest
from insightlens.storage.chunk_repository import ChunkRepository, RetrievedChunk

_RRF_K = 60                   # dampens rank differences; 60 is the standard choice
_SIMILARITY_THRESHOLD = 0.35  # cosine floor — vector candidates below this are dropped
_CANDIDATE_MULTIPLIER = 3     # over-fetch before reranking
_MAX_CHUNKS_PER_DOC = 3       # cross-company cap: no single document dominates top-K

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

_QUERY_EXPANSIONS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b(?:brief|summarize|summary|overview|what(?:'s| is) this case about)\b", re.IGNORECASE),
        "case summary facts parties claims charges allegations evidence procedural posture key issues",
    ),
    (
        re.compile(r"\b(?:suspect|perpetrator|who did it|person of interest)\b", re.IGNORECASE),
        "suspect defendant accused charged arrested person of interest responsible",
    ),
    (
        re.compile(r"\b(?:victim|decedent|injured|survivor)\b", re.IGNORECASE),
        "victim decedent injured killed survivor complainant",
    ),
    (
        re.compile(r"\b(?:image|images|photo|photos|visual|picture|diagram|map)\b", re.IGNORECASE),
        "image photo visual evidence portrait diagram map AI vision extraction caption",
    ),
    (
        re.compile(r"\b(?:timeline|chronology|when|dates?)\b", re.IGNORECASE),
        "timeline chronology date time incident event report",
    ),
    (
        re.compile(r"\b(?:evidence|proof|exhibits?|support)\b", re.IGNORECASE),
        "evidence exhibit report witness statement photo record document",
    ),
]


def _expand_query(query: str) -> str:
    additions = [
        expansion
        for pattern, expansion in _QUERY_EXPANSIONS
        if pattern.search(query)
    ]
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


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
        tokenized = [_tokenize(c.chunk_text) for c in corpus_chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def retrieve(self, request: RetrievalRequest) -> list[RetrievedChunk]:
        # ── Query quality guard ───────────────────────────────────────────────
        meaningful = _tokenize(request.query)
        if len(meaningful) < 2:
            raise ValueError(
                "Query is too short or contains only common words. "
                "Please ask a more specific question."
            )

        # ── Compound query: split, retrieve per sub-question, merge ──────────
        sub_queries = _split_compound_query(request.query)
        if len(sub_queries) > 1:
            seen: set[str] = set()
            merged: list[RetrievedChunk] = []
            for sub_q in sub_queries:
                sub_req = RetrievalRequest(
                    query=sub_q,
                    top_k=request.top_k,
                    company_filter=request.company_filter,
                    user_id=request.user_id,
                    org_member_ids=request.org_member_ids,
                    system_only=request.system_only,
                    user_only=request.user_only,
                    case_id=request.case_id,
                    preferred_chunk_types=request.preferred_chunk_types,
                )
                for chunk in self._retrieve_single(sub_req):
                    if chunk.chunk_id not in seen:
                        seen.add(chunk.chunk_id)
                        merged.append(chunk)
            return merged[: request.top_k]

        return self._retrieve_single(request)

    def _retrieve_single(self, request: RetrievalRequest) -> list[RetrievedChunk]:
        """Core 8-stage retrieval pipeline for a single query."""
        candidate_k = request.top_k * _CANDIDATE_MULTIPLIER
        expanded_query = _expand_query(request.query)

        # ── 1. Vector search ──────────────────────────────────────────────────
        query_vector = self._embedder.embed_query(expanded_query)
        vector_candidates = self._repository.search_similar(
            query_embedding=query_vector,
            top_k=candidate_k,
            company_filter=request.company_filter,
            user_id=request.user_id,
            org_member_ids=request.org_member_ids,
            system_only=request.system_only,
            user_only=request.user_only,
            case_id=request.case_id,
        )
        vector_candidates = [
            c for c in vector_candidates if c.similarity >= _SIMILARITY_THRESHOLD
        ]

        # ── 2. BM25 search ────────────────────────────────────────────────────
        bm25_candidates = self._bm25_search(expanded_query, request.company_filter, candidate_k)

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

        # ── 6. Per-document quota (cross-company queries only) ─────────────────
        # When no company filter is set, one document can dominate all top-K
        # slots (e.g. Digital Realty with 100+ AI mentions on cross-sector
        # queries). Cap each document to _MAX_CHUNKS_PER_DOC before reranking.
        if not request.company_filter:
            fused = self._apply_per_doc_quota(fused)

        # ── 7 & 8. Rerank → deduplicate ───────────────────────────────────────
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
        scores = self._bm25.get_scores(_tokenize(query))
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

    def _apply_per_doc_quota(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Cap each document to _MAX_CHUNKS_PER_DOC to prevent one corpus-dominant
        document from filling all top-K slots on cross-company queries."""
        counts: dict[str, int] = {}
        result: list[RetrievedChunk] = []
        for chunk in chunks:
            n = counts.get(chunk.document_id, 0)
            if n < _MAX_CHUNKS_PER_DOC:
                result.append(chunk)
                counts[chunk.document_id] = n + 1
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
