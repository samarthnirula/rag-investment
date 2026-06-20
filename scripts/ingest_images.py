"""Ingest standalone image files downloaded by scrape_epstein_files.py.

Walks data/epstein_raw/ for JPEG/PNG/TIF files, creates a DOCUMENTS entry
for each one, generates an AI description via Claude Haiku, and stores the
record in the IMAGES table.

Run:
  python scripts/ingest_images.py           # incremental (skips existing)
  python scripts/ingest_images.py --reset   # wipe and re-ingest

Images already referenced in the IMAGES table are skipped on re-run.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv()

from insightlens.config import ConfigError, load_config
from insightlens.ingestion.image_extractor import (
    SUPPORTED_EXTENSIONS,
    describe_standalone_image,
    image_id_from_path,
)
from insightlens.storage.image_repository import ImageRecord, ImageRepository
from insightlens.storage.chunk_repository import ChunkRepository, DocumentRecord
from insightlens.storage.snowflake_client import SnowflakeConnectionError, open_connection

PROJECT_ROOT = Path(__file__).parents[1]
IMAGE_ROOT   = PROJECT_ROOT / "data" / "images"
COMPANY      = "Epstein"
DOC_TYPE     = "standalone_image"


def _doc_id(file_path: Path) -> str:
    return hashlib.sha1(str(file_path.resolve()).encode()).hexdigest()[:16]


def _existing_image_ids(conn) -> set[str]:
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT i.image_id FROM IMAGES i "
            "JOIN DOCUMENTS d ON i.document_id = d.document_id "
            "WHERE UPPER(d.company) = 'EPSTEIN'"
        )
        return {r[0] for r in cur.fetchall()}
    finally:
        cur.close()


def _reset_standalone_images(conn) -> None:
    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM IMAGES WHERE document_id IN "
            "(SELECT document_id FROM DOCUMENTS "
            "WHERE UPPER(company) = 'EPSTEIN' AND document_type = %s)",
            (DOC_TYPE,),
        )
        cur.execute(
            "DELETE FROM DOCUMENTS WHERE UPPER(company) = 'EPSTEIN' AND document_type = %s",
            (DOC_TYPE,),
        )
        print("[images] Existing standalone image data cleared")
    finally:
        cur.close()


def main() -> int:
    reset = "--reset" in sys.argv

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"[images] Config error: {exc}", file=sys.stderr)
        return 1

    if not IMAGE_ROOT.exists():
        print(f"[images] Image directory not found: {IMAGE_ROOT}", file=sys.stderr)
        print("[images] Run:  python scripts/scrape_epstein_files.py", file=sys.stderr)
        return 1

    image_files = [
        p for p in IMAGE_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"[images] No image files found under {IMAGE_ROOT}", file=sys.stderr)
        return 0

    print(f"[images] Found {len(image_files):,} image file(s) to consider")

    total = 0
    skipped = 0
    start_time = time.time()

    try:
        with open_connection(cfg.db) as conn:
            if reset:
                _reset_standalone_images(conn)
                existing: set[str] = set()
            else:
                existing = _existing_image_ids(conn)

            chunk_repo = ChunkRepository(conn)
            image_repo = ImageRepository(conn)

            for img_path in image_files:
                img_id = image_id_from_path(img_path)
                if img_id in existing:
                    skipped += 1
                    continue

                doc_id = _doc_id(img_path)

                chunk_repo.upsert_document(
                    DocumentRecord(
                        document_id=doc_id,
                        file_name=img_path.name,
                        company=COMPANY,
                        document_type=DOC_TYPE,
                        version_label=None,
                        version_date=None,
                        page_count=1,
                    )
                )

                print(f"  [{total + 1}] {img_path.name} — generating AI description…")
                description = describe_standalone_image(img_path, api_key=cfg.anthropic_api_key)

                from PIL import Image as _PILImage
                width, height = None, None
                try:
                    with _PILImage.open(img_path) as pil_img:
                        width, height = pil_img.size
                except Exception:
                    pass

                ext        = img_path.suffix.lower()
                media_type = {
                    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".tif": "image/tiff",  ".tiff": "image/tiff",
                }.get(ext, "image/jpeg")

                record = ImageRecord(
                    image_id=img_id,
                    document_id=doc_id,
                    page_number=1,
                    image_index=0,
                    file_path=str(img_path.resolve()),
                    media_type=media_type,
                    width=width,
                    height=height,
                    ai_description=description or None,
                )
                image_repo.insert_image(record)
                total += 1

    except SnowflakeConnectionError as exc:
        print(f"[images] Snowflake error: {exc}", file=sys.stderr)
        return 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Images ingested : {total:,}")
    print(f"  Skipped         : {skipped:,}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
