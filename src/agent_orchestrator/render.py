"""PDF helpers for the extraction agent: page image and text layer (pypdfium2, Apache-2.0).

**One PDFium thread.** PDFium is not thread-safe. In the graph, invoice branches run in worker
threads, and calling PDFium from them, even one at a time behind a lock, crashed the process
(an access violation, once a segfault) and sometimes rendered a page slightly differently
(Task 13). So every PDFium call runs on a single dedicated thread, and callers wait for it. A
page then always renders to the same bytes, which also keeps the paid-call ledger's request
hash (it covers the image) stable across a crash and resume.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pypdfium2 as pdfium

RENDER_DPI = 150  # A4 at 150 DPI is about 1240 x 1754 pixels
_PDFIUM = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdfium")


def _render(pdf_path: str, dpi: int, page_index: int) -> bytes:
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        page = pdf[page_index]
        bitmap = page.render(scale=dpi / 72)
        # to_pil() shares the bitmap's memory: copy before the bitmap is closed.
        image = bitmap.to_pil().convert("RGB").copy()
        bitmap.close()
        page.close()
    finally:
        pdf.close()
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _text(pdf_path: str) -> str:
    pdf = pdfium.PdfDocument(pdf_path)
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


def render_page_png(pdf_path: Path | str, dpi: int = RENDER_DPI, page_index: int = 0) -> bytes:
    """Page ``page_index`` (default the first) as PNG bytes."""
    return _PDFIUM.submit(_render, str(pdf_path), dpi, page_index).result()


def pdf_text_layer(pdf_path: Path | str) -> str:
    """All text the PDF carries as text (empty for a pure scan). Untrusted: scan it, never
    show it to a planner."""
    return _PDFIUM.submit(_text, str(pdf_path)).result()


def on_pdfium_thread(fn, *args, **kwargs):
    """Run any other PDFium work (the eval's scan-noise renderer) on the same thread."""
    return _PDFIUM.submit(fn, *args, **kwargs).result()
