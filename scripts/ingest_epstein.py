"""Ingest Epstein case documents from the HuggingFace CSV into PostgreSQL/pgvector.

CSV: data/epstein_data/dataset_text_extract - dataset_text_extract.csv
Columns: id (PDF filename), text (extracted text for one page/row)

Each unique `id` becomes one DOCUMENT. Each CSV row is one page of that document,
chunked individually so page numbers are preserved accurately.

Run:
  python scripts/ingest_epstein.py           # incremental (skips existing docs)
  python scripts/ingest_epstein.py --reset   # wipe Epstein data and re-ingest fresh
"""
from __future__ import annotations

import csv
import hashlib
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv()

from insightlens.config import ConfigError, load_config
from insightlens.embeddings.embedder import Embedder, EmbeddingError
from insightlens.ingestion.chunker import RecursiveTokenChunker
from insightlens.storage.chunk_repository import (
    ChunkRecord,
    ChunkRepository,
    DocumentRecord,
    RepositoryError,
)
from insightlens.storage.snowflake_client import SnowflakeConnectionError, open_connection

CSV_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "epstein_data"
    / "dataset_text_extract - dataset_text_extract.csv"
)
COMPANY   = "Epstein"
DOC_TYPE  = "case_document"
DOC_BATCH = 500   # documents to accumulate before a bulk write


def _doc_id(filename: str) -> str:
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:16]


def _chunk_id(document_id: str, idx: int) -> str:
    return f"{document_id}-{idx:05d}"


def _load_csv() -> dict[str, list[str]]:
    """Return {filename: [page1_text, page2_text, ...]} in row order."""
    print(f"[epstein] Reading {CSV_PATH.name} …")
    docs: dict[str, list[str]] = defaultdict(list)
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            if text:
                docs[row["id"]].append(text)
    print(f"[epstein] {len(docs):,} unique documents loaded")
    return dict(docs)


def _reset_epstein(conn) -> None:
    """Delete all existing Epstein data from PostgreSQL."""
    import psycopg2
    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM chunks WHERE document_id IN "
            "(SELECT document_id FROM documents WHERE UPPER(company) = 'EPSTEIN')"
        )
        cur.execute("DELETE FROM documents WHERE UPPER(company) = 'EPSTEIN'")
        print("[epstein] Existing Epstein data cleared")
    except psycopg2.Error as exc:
        print(f"[epstein] Reset failed: {exc}", file=sys.stderr)
    finally:
        cur.close()


def _existing_doc_ids(conn) -> set[str]:
    import psycopg2
    cur = conn.cursor()
    try:
        cur.execute("SELECT document_id FROM documents WHERE UPPER(company) = 'EPSTEIN'")
        return {r[0] for r in cur.fetchall()}
    except psycopg2.Error:
        return set()
    finally:
        cur.close()


def main() -> int:
    reset = "--reset" in sys.argv

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"[epstein] Config error: {exc}", file=sys.stderr)
        return 1

    if not CSV_PATH.exists():
        print(f"[epstein] CSV not found: {CSV_PATH}", file=sys.stderr)
        print("[epstein] Run:  python scripts/fetch_epstein_data.py", file=sys.stderr)
        return 1

    docs = _load_csv()

    print("[epstein] Loading embedding model …")
    embedder = Embedder(model=cfg.embedding_model)
    chunker  = RecursiveTokenChunker(
        chunk_size_tokens=cfg.chunk_size_tokens,
        overlap_tokens=cfg.chunk_overlap_tokens,
    )

    total_docs   = 0
    total_chunks = 0
    skipped      = 0
    start_time   = time.time()

    try:
        with open_connection(cfg.db) as conn:
            repo = ChunkRepository(conn)

            if reset:
                _reset_epstein(conn)
                already_done: set[str] = set()
            else:
                already_done = _existing_doc_ids(conn)

            pending       = {k: v for k, v in docs.items() if _doc_id(k) not in already_done}
            total_pending = len(pending)
            print(f"[epstein] {len(already_done):,} already in DB — skipping")
            print(f"[epstein] {total_pending:,} documents to ingest\n")

            doc_buffer:   list[tuple[str, list[str]]] = []  # (filename, pages)
            chunk_buffer: list[ChunkRecord]           = []

            for i, (filename, pages) in enumerate(pending.items(), start=1):
                doc_id = _doc_id(filename)

                all_chunks = []
                running_index = 0
                for page_num, page_text in enumerate(pages, start=1):
                    if not page_text.strip():
                        continue
                    raw = chunker.chunk_page(
                        page_text,
                        page_number=page_num,
                        starting_chunk_index=running_index,
                    )
                    all_chunks.extend(raw)
                    running_index += len(raw)

                if not all_chunks:
                    skipped += 1
                    continue

                doc_buffer.append((filename, pages))
                for chunk in all_chunks:
                    chunk_buffer.append(ChunkRecord(
                        chunk_id=_chunk_id(doc_id, chunk.chunk_index),
                        document_id=doc_id,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                        chunk_text=chunk.text,
                        token_count=chunk.token_count,
                        embedding=[],   # filled after embed_texts below
                        section_header=filename,
                        chunk_type="body",
                    ))

                if len(doc_buffer) >= DOC_BATCH or i == total_pending:
                    texts = [c.chunk_text for c in chunk_buffer]
                    try:
                        results = embedder.embed_texts(texts)
                    except EmbeddingError as exc:
                        print(f"  [warn] Embedding batch failed: {exc}", file=sys.stderr)
                        skipped += len(doc_buffer)
                        doc_buffer.clear()
                        chunk_buffer.clear()
                        continue

                    # Attach embeddings — ChunkRecord is frozen so rebuild
                    chunk_buffer = [
                        ChunkRecord(
                            chunk_id=c.chunk_id,
                            document_id=c.document_id,
                            page_number=c.page_number,
                            chunk_index=c.chunk_index,
                            chunk_text=c.chunk_text,
                            token_count=c.token_count,
                            embedding=r.vector,
                            section_header=c.section_header,
                            chunk_type=c.chunk_type,
                        )
                        for c, r in zip(chunk_buffer, results)
                    ]

                    for fname, fpages in doc_buffer:
                        fid = _doc_id(fname)
                        try:
                            repo.upsert_document(DocumentRecord(
                                document_id=fid,
                                file_name=fname,
                                company=COMPANY,
                                document_type=DOC_TYPE,
                                version_label=None,
                                version_date=None,
                                page_count=len(fpages),
                            ))
                        except RepositoryError as exc:
                            print(f"  [warn] Doc upsert failed for {fname}: {exc}", file=sys.stderr)

                    try:
                        inserted = repo.insert_chunks(chunk_buffer)
                    except RepositoryError as exc:
                        print(f"  [warn] Chunk insert failed: {exc}", file=sys.stderr)
                        inserted = 0

                    total_docs   += len(doc_buffer)
                    total_chunks += inserted

                    elapsed = time.time() - start_time
                    rate    = total_docs / elapsed if elapsed > 0 else 0
                    eta_min = (total_pending - total_docs) / rate / 60 if rate > 0 else 0
                    print(
                        f"  [{total_docs:>7,}/{total_pending:,}]  "
                        f"{total_chunks:,} chunks  "
                        f"{rate:.1f} docs/s  "
                        f"~{eta_min:.0f} min left"
                    )

                    doc_buffer.clear()
                    chunk_buffer.clear()

    except SnowflakeConnectionError as exc:
        print(f"[epstein] DB connection error: {exc}", file=sys.stderr)
        return 1

    # P12: mark all system docs (user_id IS NULL) as is_demo=TRUE
    try:
        with open_connection(cfg.db) as conn:
            cur = conn.cursor()
            cur.execute("UPDATE documents SET is_demo = TRUE WHERE user_id IS NULL")
            cur.close()
    except Exception as exc:
        print(f"[epstein] Warning: could not set is_demo flag: {exc}", file=sys.stderr)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Documents ingested : {total_docs:,}")
    print(f"  Chunks inserted    : {total_chunks:,}")
    print(f"  Skipped            : {skipped:,}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
