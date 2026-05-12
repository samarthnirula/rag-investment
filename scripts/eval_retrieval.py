"""Retrieval evaluation: measures Hit@K and MRR against a ground-truth query set.

Run:
    python scripts/eval_retrieval.py

What this measures:
  Hit@K  — fraction of queries where at least one relevant chunk appears in the top K results.
  MRR    — Mean Reciprocal Rank: average of 1/rank_of_first_relevant_chunk.
            MRR=1.0 means the best chunk is always rank 1. MRR=0.5 means it's rank 2 on average.

Ground truth:
  Each test case defines a query and a set of "relevant signals" — keywords or phrases that
  MUST appear in at least one retrieved chunk for the result to count as a hit.
  This is a keyword-presence proxy for relevance; it avoids needing human-labeled data
  while still being meaningful (if "FFO" and "per share" don't appear in any retrieved chunk
  for the FFO-per-share query, the retrieval clearly failed).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from insightlens.config import load_config
from insightlens.embeddings.embedder import Embedder
from insightlens.retrieval.hybrid_search import HybridSearchService
from insightlens.retrieval.reranker import Reranker
from insightlens.retrieval.vector_search import RetrievalRequest
from insightlens.storage.chunk_repository import ChunkRepository
from insightlens.storage.snowflake_client import open_connection

# ── Ground-truth test cases ────────────────────────────────────────────────────
# Each entry: (query, [list of keyword signals], company_filter or None)
# A result is a HIT when at least one retrieved chunk contains ALL signals in any
# single signal group (inner list = AND, outer list = OR between groups).
TEST_CASES: list[tuple[str, list[list[str]], str | None]] = [
    # VICI — operating metrics
    (
        "What are the key operating metrics for VICI Properties?",
        [["occupancy"], ["same-store"], ["rent"], ["FFO"], ["AFFO"]],
        "VICI",
    ),
    # VICI — FFO
    (
        "What was VICI's FFO per share?",
        [["FFO", "per share"], ["FFO", "diluted"], ["funds from operations"]],
        "VICI",
    ),
    # VICI — dividend / yield
    (
        "What is VICI's dividend yield or dividend per share?",
        [["dividend"], ["yield"], ["per share"]],
        "VICI",
    ),
    # VICI — portfolio composition
    (
        "What properties does VICI own and where are they located?",
        [["Las Vegas"], ["Caesars"], ["MGM"], ["portfolio"], ["properties"]],
        "VICI",
    ),
    # VICI — investment thesis / strategy
    (
        "What is VICI's investment thesis and competitive advantage?",
        [["experiential"], ["triple net"], ["NNN"], ["long-term"], ["lease"]],
        "VICI",
    ),
    # BXP — NOI
    (
        "What is BXP's net operating income?",
        [["NOI"], ["net operating income"], ["operating income"]],
        "BXP",
    ),
    # BXP — leasing activity
    (
        "What is BXP's occupancy rate and leasing activity?",
        [["occupancy"], ["leased"], ["square feet"], ["leasing"]],
        "BXP",
    ),
    # Cross-company — version currency check
    (
        "What is the latest guidance for 2025?",
        [["guidance"], ["2025"], ["outlook"], ["forecast"]],
        None,
    ),
]

TOP_K_VALUES = [1, 3, 5]


def _is_hit(chunks, signals: list[list[str]]) -> tuple[bool, int]:
    """Return (hit, rank) where rank is 1-indexed position of first matching chunk."""
    for rank, chunk in enumerate(chunks, start=1):
        text_lower = chunk.chunk_text.lower()
        for signal_group in signals:
            if all(kw.lower() in text_lower for kw in signal_group):
                return True, rank
    return False, 0


def main() -> None:
    cfg = load_config()
    embedder = Embedder(model=cfg.embedding_model)
    reranker = Reranker()

    print("Loading corpus for BM25…", flush=True)
    with open_connection(cfg.snowflake) as conn:
        repo = ChunkRepository(conn)
        corpus = repo.get_all_chunks()
        companies = repo.list_companies()

    print(f"Corpus size: {len(corpus)} chunks")
    print(f"Companies in DB: {companies}\n")

    results_by_k: dict[int, list[bool]] = {k: [] for k in TOP_K_VALUES}
    reciprocal_ranks: list[float] = []

    for i, (query, signals, company_filter) in enumerate(TEST_CASES, start=1):
        with open_connection(cfg.snowflake) as conn:
            repo = ChunkRepository(conn)
            service = HybridSearchService(
                embedder=embedder,
                repository=repo,
                corpus_chunks=corpus,
                reranker=reranker,
            )
            chunks = service.retrieve(
                RetrievalRequest(
                    query=query,
                    top_k=max(TOP_K_VALUES),
                    company_filter=company_filter,
                )
            )

        hit, rank = _is_hit(chunks, signals)
        rr = 1.0 / rank if hit else 0.0
        reciprocal_ranks.append(rr)

        for k in TOP_K_VALUES:
            results_by_k[k].append(hit and rank <= k)

        status = f"HIT  (rank {rank})" if hit else "MISS"
        company_label = company_filter or "ALL"
        print(f"[{i:02d}] [{company_label:4s}] [{status}]  {query[:70]}")

    n = len(TEST_CASES)
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    for k in TOP_K_VALUES:
        hits = sum(results_by_k[k])
        print(f"  Hit@{k}  : {hits}/{n}  ({hits/n:.0%})")
    mrr = sum(reciprocal_ranks) / n
    print(f"  MRR    : {mrr:.3f}")
    print("=" * 60)
    print()
    print("Interpretation:")
    print("  Hit@5 > 80%  → retrieval is finding the right documents")
    print("  MRR   > 0.60 → the best chunk is typically in the top 2 results")
    print("  MRR   < 0.40 → retrieval is struggling; check chunk size / similarity threshold")


if __name__ == "__main__":
    main()
