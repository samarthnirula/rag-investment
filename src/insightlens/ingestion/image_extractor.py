"""Extract raster images from PDFs and standalone image files.

For PDFs: uses fitz (PyMuPDF) to pull embedded images from each page.
For standalone files: reads bytes directly (JPEG/PNG/TIF downloaded by scraper).
Both paths optionally call Claude Haiku vision to produce a searchable description.
"""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif", ".bmp"}

_EXT_TO_MEDIA = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".tif":  "image/tiff",
    ".tiff": "image/tiff",
    ".gif":  "image/gif",
    ".bmp":  "image/bmp",
}

_VISION_PROMPT = """\
Describe this image from a legal case document in detail. Focus on:
- What is shown (photograph, diagram, scanned document, chart, map, signature page, etc.)
- Key people, locations, objects, dates, or text visible
- Any identifying information that would help a lawyer find this image when researching the case

Be concise but thorough. If text is visible, transcribe the most important parts.\
"""


def _image_id_from_doc(document_id: str, page_number: int, image_index: int) -> str:
    raw = f"{document_id}-p{page_number}-i{image_index}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def image_id_from_path(file_path: Path) -> str:
    return hashlib.sha1(str(file_path.resolve()).encode()).hexdigest()[:16]


def _describe_image(img_bytes: bytes, media_type: str, api_key: str) -> str:
    """Send image to Claude Haiku for a natural-language description."""
    if not api_key or not img_bytes:
        return ""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        img_b64 = base64.standard_b64encode(img_bytes).decode()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": _VISION_PROMPT},
                ],
            }],
        )
        return response.content[0].text
    except Exception:
        return ""


def extract_images_from_pdf(
    pdf_path: Path,
    document_id: str,
    output_dir: Path,
    api_key: str = "",
    min_size_px: int = 50,
) -> list[dict]:
    """Extract embedded raster images from a PDF and save them to output_dir.

    Returns a list of dicts with keys matching ImageRecord fields.
    Skips images smaller than min_size_px in either dimension (bullets, icons).
    """
    try:
        import fitz
    except ImportError:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    doc = fitz.open(str(pdf_path))
    try:
        for page_num in range(len(doc)):
            page_number = page_num + 1
            image_index = 0

            for img_info in doc[page_num].get_images(full=True):
                xref = img_info[0]
                try:
                    base_img = doc.extract_image(xref)
                except Exception:
                    continue

                width = base_img.get("width", 0)
                height = base_img.get("height", 0)
                if width < min_size_px or height < min_size_px:
                    continue

                ext = base_img.get("ext", "png").lower()
                if ext not in ("jpg", "jpeg", "png", "tiff", "bmp"):
                    ext = "png"

                img_bytes: bytes = base_img["image"]
                img_id = _image_id_from_doc(document_id, page_number, image_index)
                fname = f"{img_id}.{ext}"
                file_path = output_dir / fname
                file_path.write_bytes(img_bytes)

                media_type = _EXT_TO_MEDIA.get(f".{ext}", f"image/{ext}")
                description = _describe_image(img_bytes, media_type, api_key)

                records.append({
                    "image_id":      img_id,
                    "document_id":   document_id,
                    "page_number":   page_number,
                    "image_index":   image_index,
                    "file_path":     str(file_path.resolve()),
                    "media_type":    media_type,
                    "width":         width,
                    "height":        height,
                    "ai_description": description or None,
                })
                image_index += 1
    finally:
        doc.close()

    return records


def describe_standalone_image(file_path: Path, api_key: str = "") -> str:
    """Generate an AI description for a standalone image file on disk."""
    ext = file_path.suffix.lower()
    media_type = _EXT_TO_MEDIA.get(ext, "image/jpeg")
    try:
        img_bytes = file_path.read_bytes()
        return _describe_image(img_bytes, media_type, api_key)
    except Exception:
        return ""


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS
