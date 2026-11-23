"""Model prices (USD per 1M tokens) and cost helpers (docs/SPEC.md section 2.2).

Bedrock prices come from the official AWS price-list file for ap-south-1 (publication
2026-11-11); the Claude price from the Claude API price table. An unknown model id raises:
a call must never be counted as costing $0 by accident.
"""

from __future__ import annotations

import datetime as dt
import math

PRICES_CHECKED_ON = dt.date(2026, 11, 23)

# model id -> (input USD per 1M tokens, output USD per 1M tokens)
PRICES: dict[str, tuple[float, float]] = {
    "global.amazon.nova-2-lite-v1:0": (0.35, 2.95),
    "apac.amazon.nova-pro-v1:0": (0.94, 3.76),
    "claude-haiku-5-5": (0.10, 0.50),
    "fake": (0.0, 0.0),
}

# Input tokens charged per image, for estimates before a call. Nova bills a flat 230 tokens per
# image. Claude bills by pixels (about width x height / 750, after it scales the long edge to at
# most 1568 px), so an A4 page at 150 DPI is about 1,600 tokens.
IMAGE_TOKENS: dict[str, int] = {
    "global.amazon.nova-2-lite-v1:0": 230,
    "apac.amazon.nova-pro-v1:0": 230,
    "claude-haiku-5-5": 1600,
    "fake": 0,
}

CHARS_PER_TOKEN = 3  # conservative for English + numbers: over-estimates, never under


class UnknownModel(KeyError):
    """No price is known for this model id."""


def price(model_id: str) -> tuple[float, float]:
    try:
        return PRICES[model_id]
    except KeyError:
        raise UnknownModel(f"no price for model {model_id!r}; add it to llm/pricing.py") from None


def cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = price(model_id)
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


def estimate_input_tokens(model_id: str, text_chars: int, images: int) -> int:
    price(model_id)  # raises for unknown models
    return math.ceil(text_chars / CHARS_PER_TOKEN) + images * IMAGE_TOKENS.get(model_id, 2000)


def estimate_cost(model_id: str, text_chars: int, images: int, max_output_tokens: int) -> float:
    """Worst-case cost before a call: every allowed output token is assumed to be used."""
    tokens_in = estimate_input_tokens(model_id, text_chars, images)
    return cost_usd(model_id, tokens_in, max_output_tokens)
