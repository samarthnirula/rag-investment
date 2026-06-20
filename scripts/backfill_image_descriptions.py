"""Backfill AI descriptions for images already stored in Snowflake.

Run this after topping up Anthropic API credits. Only processes images
where ai_description IS NULL — safe to re-run multiple times.

Run:
  python scripts/backfill_image_descriptions.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv()

from insightlens.config import ConfigError, load_config
from insightlens.ingestion.image_extractor import _describe_image, _EXT_TO_MEDIA
from insightlens.storage.image_repository import ImageRepository
from insightlens.storage.snowflake_client import SnowflakeConnectionError, open_connection
from snowflake.connector.errors import ProgrammingError


def _update_description(conn, image_id: str, description: str) -> None:
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE IMAGES SET ai_description = %s WHERE image_id = %s",
            (description, image_id),
        )
    finally:
        cur.close()


def main() -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"[backfill] Config error: {exc}", file=sys.stderr)
        return 1

    total = updated = failed = 0
    start = time.time()

    try:
        with open_connection(cfg.db) as conn:
            repo = ImageRepository(conn)
            all_images = repo.get_all_image_metadata()
            pending = [img for img in all_images if not img.ai_description]

            print(f"[backfill] {len(all_images)} total images, {len(pending)} missing descriptions")

            for img in pending:
                total += 1
                fp = Path(img.file_path)
                if not fp.exists():
                    print(f"  [{total}] SKIP  {fp.name} — file not on disk")
                    failed += 1
                    continue

                ext = fp.suffix.lower()
                media_type = img.media_type or _EXT_TO_MEDIA.get(ext, "image/png")

                print(f"  [{total}/{len(pending)}] {fp.name} …", end=" ", flush=True)
                desc = _describe_image(fp.read_bytes(), media_type, cfg.anthropic_api_key)

                if desc:
                    _update_description(conn, img.image_id, desc)
                    print(f"ok ({len(desc)} chars)")
                    updated += 1
                else:
                    print("failed (check credits)")
                    failed += 1
                    # Stop early if we keep failing — likely still out of credits
                    if failed >= 3:
                        print("\n[backfill] 3 consecutive failures — stopping. Check API credits.")
                        break

    except SnowflakeConnectionError as exc:
        print(f"[backfill] Snowflake error: {exc}", file=sys.stderr)
        return 1

    elapsed = time.time() - start
    print(f"\n[backfill] Done in {elapsed:.0f}s — {updated} updated, {failed} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
