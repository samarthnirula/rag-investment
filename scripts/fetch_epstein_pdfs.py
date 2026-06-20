"""Download the full Epstein raw PDF archive from Google Drive.

Source: https://drive.google.com/drive/folders/18tIY9QEGUZe0q_AFAxoPnnVBCWbqHm2p
(linked from the FULL_EPSTEIN_INDEX HuggingFace dataset README)

Downloads into data/raw_pdfs/ so ingest_documents.py picks them up automatically.

Run:
  python scripts/fetch_epstein_pdfs.py

After downloading, run:
  python scripts/ingest_documents.py
"""
from __future__ import annotations

import sys
from pathlib import Path

FOLDER_ID  = "18tIY9QEGUZe0q_AFAxoPnnVBCWbqHm2p"
OUTPUT_DIR = Path(__file__).parents[1] / "data" / "raw_pdfs"


def main() -> int:
    try:
        import gdown
    except ImportError:
        print("[fetch] gdown not installed. Run:  pip install gdown", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[fetch] Downloading Epstein PDF archive from Google Drive…")
    print(f"[fetch] Destination: {OUTPUT_DIR}")
    print(f"[fetch] This is a large folder (~20,000 pages). It may take several minutes.\n")

    try:
        gdown.download_folder(
            id=FOLDER_ID,
            output=str(OUTPUT_DIR),
            quiet=False,
            use_cookies=False,
        )
    except Exception as exc:
        print(f"\n[fetch] Download error: {exc}", file=sys.stderr)
        print(
            "\n[fetch] If you see a quota/permission error, Google Drive has rate-limited "
            "the folder. Wait a few minutes and re-run — gdown resumes where it left off.",
            file=sys.stderr,
        )
        return 1

    pdfs = list(OUTPUT_DIR.rglob("*.pdf"))
    print(f"\n[fetch] Done. {len(pdfs):,} PDF(s) in {OUTPUT_DIR}")
    print("\nNext step:")
    print("  python scripts/ingest_documents.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
