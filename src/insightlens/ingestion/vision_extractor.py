"""Vision-based content extraction for chart, map, and logo slides.

Called during ingestion for pages with sparse text (< 300 chars) where the
meaningful content is encoded in bar charts, geographic maps, tenant logo
tables, or infographic layouts that text extraction cannot read.
"""
from __future__ import annotations

import base64
import os

_VISION_PROMPT = """\
Extract ALL data visible in this investment presentation slide image. Be exhaustive and precise.

For each type of visual element:
- Bar / column charts: every bar label AND its exact value or percentage
- Pie / donut charts: every segment name AND its percentage
- Geographic maps: every labeled region, state, country, or city with its data value and percentage
- Tables containing company logos: every company name (even if shown as a logo image) with every associated number
- Line charts: trend direction plus key labeled data points with dates
- Text callouts, KPI boxes, metric tiles: transcribe every label and value exactly
- Footnotes or fine-print annotations: transcribe word for word

Format: use "Label: Value" or bullet points — structured text, not prose.
If a specific value is genuinely unclear in the image write "value unclear" — do not guess or invent numbers.\
"""


def extract_visual_content(fitz_page: object) -> str:
    """Render a PDF page to PNG and extract all visible content via Claude vision.

    Parameters
    ----------
    fitz_page:
        A ``fitz.Page`` object (PyMuPDF).  Typed as ``object`` to avoid a
        hard import dependency — callers that don't use vision still work.

    Returns
    -------
    str
        Extracted text describing all visual elements, or empty string if the
        API key is missing or the call fails.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ""

    try:
        import anthropic
        import fitz as _fitz  # noqa: F401 — confirm fitz is available
    except ImportError:
        return ""

    # Render at 150 DPI: sharp enough to read chart labels, small enough to keep
    # the base64 payload under ~400 KB for a typical slide.
    try:
        mat = __import__("fitz").Matrix(150 / 72, 150 / 72)
        pix = fitz_page.get_pixmap(matrix=mat)
        img_b64 = base64.standard_b64encode(pix.tobytes("png")).decode()
    except Exception:
        return ""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": _VISION_PROMPT},
                    ],
                }
            ],
        )
        return response.content[0].text
    except Exception:
        return ""
