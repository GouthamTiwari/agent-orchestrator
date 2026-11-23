"""Make a clean invoice PDF look like a phone or office scan (docs/SPEC.md section 15.1).

Page 1 is rasterised at 150 DPI, then: a small rotation (up to +/-3 degrees), a Gaussian blur
(sigma 0.6 to 1.2), a contrast change, and JPEG compression at quality 30 to 50. Every
parameter is drawn from a random generator seeded by the file's identity, so the same input
always gives the same image bytes.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter


def rasterise(pdf_path: Path, dpi: int = 150) -> Image.Image:
    """Render page 1 of a PDF to an RGB image."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[0]
        image = page.render(scale=dpi / 72).to_pil().convert("RGB")
        page.close()
    finally:
        pdf.close()
    return image


def noise_params(cfg: dict[str, Any], seed_key: str) -> dict[str, float | int]:
    n = cfg["noise"]
    rng = random.Random(f"{cfg['seed']}:noise:{seed_key}")
    return {
        "angle": round(rng.uniform(-n["rotate_degrees"], n["rotate_degrees"]), 2),
        "sigma": round(rng.uniform(*n["blur_sigma"]), 2),
        "contrast": round(rng.uniform(*n["contrast"]), 2),
        "quality": rng.randint(*n["jpeg_quality"]),
    }


def add_noise(image: Image.Image, params: dict[str, float | int]) -> bytes:
    """Apply rotation, blur, contrast and JPEG compression; return the JPEG bytes."""
    out = image.rotate(
        float(params["angle"]),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )
    out = out.filter(ImageFilter.GaussianBlur(radius=float(params["sigma"])))
    out = ImageEnhance.Contrast(out).enhance(float(params["contrast"]))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=int(params["quality"]), optimize=False)
    return buf.getvalue()


def render_noisy_scan(pdf_path: Path, out_path: Path, cfg: dict[str, Any], seed_key: str) -> dict:
    """Write a noisy JPEG scan of ``pdf_path`` to ``out_path``; return the parameters used."""
    params = noise_params(cfg, seed_key)
    data = add_noise(rasterise(pdf_path, cfg["noise"]["render_dpi"]), params)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return params
