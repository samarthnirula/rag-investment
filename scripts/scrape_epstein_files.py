"""Scrape Epstein-related documents from the DOJ website.

Downloads PDFs, JPGs, PNGs, TIFs, and TIFFs that are actual case documents —
NOT website chrome like favicons or touch icons.

Run:
  python scripts/scrape_epstein_files.py           # scrape all known DOJ pages
  python scripts/scrape_epstein_files.py --search  # keyword search on justice.gov

Requirements:
  pip install playwright
  playwright install chromium
"""
from __future__ import annotations

import asyncio
import re
import sys
import urllib.parse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
OUTPUT_DIR   = PROJECT_ROOT / "data" / "epstein_raw"

# ── Known DOJ Epstein document release pages ───────────────────────────────────
# These are the pages that actually link to case PDFs.
DOJ_SEED_URLS = [
    # FBI January 2025 Epstein release
    "https://www.fbi.gov/contact-us/field-offices/miami/news/fbi-miami-releases-epstein-related-files",
    # DOJ main Epstein documents index
    "https://www.justice.gov/opa/pr/epstein-related-documents",
    # House Oversight Nov 2025 release
    "https://oversight.house.gov/release/epstein-related-documents",
    # SDNY case documents
    "https://www.justice.gov/usao-sdny/pr/jeffrey-epstein-indicted-sex-trafficking",
]

# These filename patterns are website chrome (favicons, icons) — skip them.
_SKIP_PATTERNS = re.compile(
    r"apple-touch-icon|metatag-image|favicon|logo\d+x\d+|icon[-_]\d+",
    re.IGNORECASE,
)

_DOWNLOADABLE_EXTS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _safe_filename(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    name = Path(path).name
    name = urllib.parse.unquote(name)
    name = re.sub(r'[^\w\-_\. ]', '_', name)
    return name or "unnamed_file"


def _is_case_document(url: str, filename: str) -> bool:
    """Return True only for actual case documents, not website chrome."""
    if _SKIP_PATTERNS.search(filename):
        return False
    # PDFs are almost always documents; images only if not a tiny icon
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return True
    # For images, require the URL to come from a /storage/ or /files/ path
    # (DOJ document hosting) rather than /sites/default/files/images/ (site chrome)
    url_lower = url.lower()
    if any(p in url_lower for p in ["/storage/", "/files/doc", "/documents/"]):
        return True
    if any(p in url_lower for p in ["/sites/default/files/images", "/core/misc/", "/themes/"]):
        return False
    # Default: keep images from non-theme paths
    return ext in {".jpg", ".jpeg", ".tif", ".tiff"}


async def _download(page, url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  [skip] {dest.name}")
        return True
    try:
        resp = await page.request.get(url, timeout=60_000)
        if resp.ok:
            dest.write_bytes(await resp.body())
            print(f"  [ok]   {dest.name}  ({dest.stat().st_size // 1024} KB)")
            return True
        print(f"  [err]  HTTP {resp.status}  {url}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"  [err]  {url}  {exc}", file=sys.stderr)
        return False


async def _scrape_page(browser, url: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx  = await browser.new_context()
    page = await ctx.new_page()
    downloaded = 0

    try:
        await page.goto(url, wait_until="networkidle", timeout=60_000)

        # Collect all anchor href links
        anchors = await page.query_selector_all("a[href]")
        links: list[str] = []
        for a in anchors:
            href = await a.get_attribute("href")
            if not href:
                continue
            abs_href = urllib.parse.urljoin(url, href)
            ext = Path(urllib.parse.urlparse(abs_href).path).suffix.lower()
            if ext in _DOWNLOADABLE_EXTS:
                links.append(abs_href)

        links = list(dict.fromkeys(links))  # deduplicate

        case_docs = [lnk for lnk in links if _is_case_document(lnk, _safe_filename(lnk))]

        print(f"\n[scraper] {url}")
        print(f"[scraper] {len(links)} downloadable links found, {len(case_docs)} case documents")

        for file_url in case_docs:
            fname = _safe_filename(file_url)
            await _download(page, file_url, out_dir / fname)
            downloaded += 1

    except Exception as exc:
        print(f"[scraper] Failed to load {url}: {exc}", file=sys.stderr)
    finally:
        await ctx.close()

    return downloaded


async def _search_mode(browser, keyword: str, out_dir: Path) -> None:
    search_url = (
        "https://search.justice.gov/search?query="
        + urllib.parse.quote(f"epstein {keyword}")
        + "&op=Search&affiliate=justice"
    )
    print(f"\n[scraper] Search: {search_url}")
    await _scrape_page(browser, search_url, out_dir)


async def main() -> int:
    search_mode = "--search" in sys.argv

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print(
            "[scraper] playwright not installed.\n"
            "  pip install playwright\n"
            "  playwright install chromium",
            file=sys.stderr,
        )
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            if search_mode:
                keyword = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or "documents"
                out_dir = OUTPUT_DIR / "search_results"
                await _search_mode(browser, keyword, out_dir)
            else:
                for i, seed_url in enumerate(DOJ_SEED_URLS, start=1):
                    out_dir = OUTPUT_DIR / f"dataset_{i}"
                    total += await _scrape_page(browser, seed_url, out_dir)
        finally:
            await browser.close()

    print(f"\n[scraper] Done. {total} case document(s) downloaded.")
    print(f"[scraper] Files saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Move any downloaded PDFs to data/raw_pdfs/")
    print("  2. Run: python scripts/ingest_documents.py")
    print("  3. Run: python scripts/ingest_images.py  (for standalone JPG/TIF files)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
