"""PDF helpers for the extraction agent: page image and text layer (pypdfium2, Apache-2.0)."""

from __future__ import annotations

import io
from pathlib import Path

import pypdfium2 as pdfium

RENDER_DPI = 150  # A4 at 150 DPI is about 1240 x 1754 pixels


def render_page_png(pdf_path: Path | str, dpi: int = RENDER_DPI, page_index: int = 0) -> bytes:
    """Page ``page_index`` (default the first) as PNG bytes."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_index]
        image = page.render(scale=dpi / 72).to_pil().convert("RGB")
        page.close()
    finally:
        pdf.close()
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def pdf_text_layer(pdf_path: Path | str) -> str:
    """All text the PDF carries as text (empty for a pure scan). Untrusted: scan it, never
    show it to a planner."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        parts = []
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            parts.append(textpage.get_text_range())
            textpage.close()
            page.close()
        return "\n".join(parts)
    finally:
        pdf.close()
