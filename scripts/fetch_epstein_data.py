"""Download the full Epstein document index from Hugging Face.

Dataset: https://huggingface.co/datasets/B00MEMBERX/FULL_EPSTEIN_INDEX

Run:
  pip install huggingface_hub
  python scripts/fetch_epstein_data.py

Output:
  data/epstein_data/   — all dataset files from the HF repo
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

OUTPUT_DIR = Path(__file__).parents[1] / "data" / "epstein_data"
REPO_ID    = "B00MEMBERX/FULL_EPSTEIN_INDEX"


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed.")
        print("Run:  pip install huggingface_hub")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  InsightLens — Epstein Case Data Fetcher")
    print(f"  Source : {REPO_ID}")
    print(f"  Output : {OUTPUT_DIR}")
    print("=" * 60)

    start = time.time()

    try:
        local_dir = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(OUTPUT_DIR),
        )
    except Exception as exc:
        print(f"\n[error] Download failed: {exc}")
        return 1

    elapsed = time.time() - start
    files   = list(Path(local_dir).rglob("*"))
    files   = [f for f in files if f.is_file()]

    print("\n" + "=" * 60)
    print(f"  Done in {elapsed:.0f}s")
    print(f"  Files  : {len(files)}")
    print(f"  Path   : {local_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
