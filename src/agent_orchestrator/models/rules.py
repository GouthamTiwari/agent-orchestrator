"""Rule text the agents may cite (spec sections 6.4 and 12)."""

from __future__ import annotations

import datetime as dt

from pydantic import Field

from agent_orchestrator.models.base import Frozen
from agent_orchestrator.models.cases import MismatchType


class RuleSnippet(Frozen):
    snippet_id: str  # stable, e.g. "cgst_act_s16#2aa"
    source_id: str  # e.g. "cgst_act_s16"
    title: str
    url: str
    section: str  # e.g. "16(2)(aa)"
    text: str
    checked_on: dt.date


class RuleResearch(Frozen):
    """A plain-English summary of the rules for one mismatch type, citing snippet ids."""

    mismatch_type: MismatchType
    summary: str
    snippet_ids: list[str] = Field(default_factory=list)
    cached: bool = False
