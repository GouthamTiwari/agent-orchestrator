"""Shared building blocks for the domain models."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, Strict

# Integer paise. Strict: a float such as 4257.0 is rejected instead of silently converted,
# so money can only enter a model through `money.parse`.
Paise = Annotated[int, Strict()]
NonNegativePaise = Annotated[int, Strict(), Field(ge=0)]

# "YYYY-MM", e.g. "2026-10".
Period = Annotated[str, Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class Frozen(BaseModel):
    """Base for every message passed between agents.

    ``extra="forbid"``: a misspelt field name fails loudly instead of being dropped.
    ``frozen=True``: once created, a record cannot be changed by a later step by accident.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
