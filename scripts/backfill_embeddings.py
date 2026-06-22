#!/usr/bin/env python3
"""Backfill missing Voyage embeddings for document chunks and image descriptions.

Run after migrations/007_vector_dim_1024.sql:
    python scripts/backfill_embeddings.py

Progress is committed after every batch, so the script is safe to rerun after
an interruption.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import psycopg2
from psycopg2.extras import execute_batch

from insightlens.embeddings.embedder import Embedder
from insightlens.storage.chunk_repository import _safe_vector_str

CHUNK_BATCH = 64
IMAGE_BATCH = 64


def _connection() -> psycopg2.extensions.connection:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set.")
    return psycopg2.connect(database_url)


def _pending_count(conn, table: str, column: str, text_column: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT count(*)
            FROM {table}
            WHERE {column} IS NULL
              AND COALESCE({text_column}, '') <> ''
            """
        )
        return int(cur.fetchone()[0])


def _backfill_chunks(conn, embedder: Embedder) -> None:
    total = _pending_count(conn, "chunks", "embedding", "chunk_text")
    print(f"[backfill] {total} chunk embeddings pending")
    completed = 0

    while True:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, chunk_text
                FROM chunks
                WHERE embedding IS NULL
                  AND chunk_text <> ''
                ORDER BY chunk_id
                LIMIT %s
                """,
                (CHUNK_BATCH,),
            )
            rows = cur.fetchall()

        if not rows:
            break

        results = embedder.embed_texts([row[1] for row in rows])
        updates = [
            (_safe_vector_str(result.vector, dim=1024), chunk_id)
            for (chunk_id, _), result in zip(rows, results, strict=True)
        ]
        with conn.cursor() as cur:
            execute_batch(
                cur,
                "UPDATE chunks SET embedding = %s::vector WHERE chunk_id = %s",
                updates,
                page_size=CHUNK_BATCH,
            )
        conn.commit()
        completed += len(rows)
        print(f"[backfill] chunks {completed}/{total}")


def _backfill_images(conn, embedder: Embedder) -> None:
    total = _pending_count(
        conn, "images", "description_embedding", "ai_description"
    )
    print(f"[backfill] {total} image embeddings pending")
    completed = 0

    while True:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT image_id, ai_description
                FROM images
                WHERE description_embedding IS NULL
                  AND COALESCE(ai_description, '') <> ''
                ORDER BY image_id
                LIMIT %s
                """,
                (IMAGE_BATCH,),
            )
            rows = cur.fetchall()

        if not rows:
            break

        results = embedder.embed_texts([row[1] for row in rows])
        updates = [
            (_safe_vector_str(result.vector, dim=1024), image_id)
            for (image_id, _), result in zip(rows, results, strict=True)
        ]
        with conn.cursor() as cur:
            execute_batch(
                cur,
                """
                UPDATE images
                SET description_embedding = %s::vector
                WHERE image_id = %s
                """,
                updates,
                page_size=IMAGE_BATCH,
            )
        conn.commit()
        completed += len(rows)
        print(f"[backfill] images {completed}/{total}")


def main() -> int:
    if not os.getenv("VOYAGE_API_KEY", "").strip():
        print("ERROR: VOYAGE_API_KEY is not set.", file=sys.stderr)
        return 1

    model = os.getenv("EMBEDDING_MODEL", "voyage-law-2")
    embedder = Embedder(model=model)
    if embedder.vector_dim != 1024:
        print(
            f"ERROR: expected a 1024-dimensional embedder, got {embedder.vector_dim}.",
            file=sys.stderr,
        )
        return 1

    with _connection() as conn:
        _backfill_chunks(conn, embedder)
        _backfill_images(conn, embedder)

    print("[backfill] All available embeddings are current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
