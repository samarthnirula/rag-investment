"""Embed ai_description for images that are missing description_embedding.

Run:
  python scripts/backfill_image_embeddings.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv()

from insightlens.config import ConfigError, load_config
from insightlens.embeddings.embedder import Embedder
from insightlens.storage.image_repository import ImageRepository
from insightlens.storage.snowflake_client import open_connection

BATCH = 32


def main() -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"[backfill] Config error: {exc}", file=sys.stderr)
        return 1

    print("[backfill] Loading embedding model…")
    embedder = Embedder(model=cfg.embedding_model)

    with open_connection(cfg.db) as conn:
        repo = ImageRepository(conn)
        pending = repo.get_images_missing_embedding()

        if not pending:
            print("[backfill] All images already have embeddings.")
            return 0

        print(f"[backfill] {len(pending)} images need embeddings")

        for start in range(0, len(pending), BATCH):
            batch = pending[start : start + BATCH]
            texts = [r.ai_description for r in batch]
            results = embedder.embed_texts(texts)
            for record, result in zip(batch, results):
                repo.update_description_embedding(record.image_id, result.vector)
            print(f"  [{min(start + BATCH, len(pending))}/{len(pending)}] done")

    print(f"[backfill] Embedded {len(pending)} image descriptions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
