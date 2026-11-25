"""The policy file's schema and its tiny condition language (docs/SPEC.md section 7.1).

A condition is data, never code: ``{all: [...]}``, ``{any: [...]}``, ``{not: ...}`` or a leaf
``{fact: case.mismatch_type, op: eq, value: missing_in_2b}``. Only the facts listed in
``FACTS`` exist; an unknown fact name, op or value type fails when the file is loaded, not when
a case is decided. There is no ``eval`` anywhere.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_orchestrator.models import ACTION_KINDS, MISMATCH_TYPES

Outcome = Literal["auto_allow", "require_approval", "block"]
Kind = Literal["legal", "safety", "data_quality", "business"]
Op = Literal["eq", "ne", "in", "not_in", "lt", "le", "gt", "ge", "contains"]

# Every fact a rule may read, with its type. Money, flags and confidence come only from
# deterministic code (intake, matcher, extraction checks). The proposal contributes only the
# action it chose, who proposed it, whether that action is allowed (computed by the engine)
# and whether the planner fell back after failing validation (set by code, not by the model).
FACTS: dict[str, type] = {
    "case.mismatch_type": str,
    "case.status": str,
    "case.match_pass": str,
    "case.itc_at_risk_paise": int,
    "case.has_book_line": bool,
    "case.has_pdf": bool,
    "case.injection": bool,
    "case.pdf_disagrees": bool,
    "case.gstin_invalid": bool,
    "case.fuzzy_match": bool,
    "case.ambiguous_match": bool,
    "case.flags": list,
    "proposal.action": str,
    "proposal.proposer": str,
    "proposal.allowed": bool,
    "proposal.fallback": bool,
    "extraction.present": bool,
    "extraction.failed": bool,
    "extraction.min_confidence": float,
    "extraction.checks": list,
    "run.budget_stopped": bool,
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class Leaf(_Strict):
    fact: str
    op: Op
    value: Any

    @model_validator(mode="after")
    def _known_fact_and_type(self) -> Leaf:
        if self.fact not in FACTS:
            raise ValueError(f"unknown fact {self.fact!r}")
        t = FACTS[self.fact]
        v = self.value
        if self.op in ("in", "not_in"):
            if not isinstance(v, list) or not v:
                raise ValueError(f"{self.fact} {self.op} needs a non-empty list")
        elif self.op == "contains":
            if t is not list:
                raise ValueError(f"'contains' only works on list facts, not {self.fact}")
        elif self.op in ("lt", "le", "gt", "ge"):
            if t not in (int, float) or isinstance(v, bool) or not isinstance(v, int | float):
                raise ValueError(f"{self.op} needs a numeric fact and value ({self.fact})")
        elif t is bool and not isinstance(v, bool):
            raise ValueError(f"{self.fact} is true/false; got {v!r}")
        if self.fact == "case.mismatch_type":
            values = v if isinstance(v, list) else [v]
            bad = [x for x in values if x not in MISMATCH_TYPES]
            if bad:
                raise ValueError(f"unknown mismatch type(s) {bad}")
        if self.fact == "proposal.action":
            values = v if isinstance(v, list) else [v]
            bad = [x for x in values if x not in ACTION_KINDS]
            if bad:
                raise ValueError(f"unknown action(s) {bad}")
        return self


class All(_Strict):
    all: list[Condition] = Field(min_length=1)


class AnyOf(_Strict):
    any: list[Condition] = Field(min_length=1)


class Not(_Strict):
    not_: Condition = Field(alias="not")


Condition = Annotated[Union[All, AnyOf, Not, Leaf], Field(union_mode="left_to_right")]  # noqa: UP007
All.model_rebuild()
AnyOf.model_rebuild()
Not.model_rebuild()


class Source(_Strict):
    id: str
    section: str
    url: str = Field(pattern=r"^https://")
    checked_on: dt.date


class Rule(_Strict):
    id: str = Field(pattern=r"^[A-Z]+(-[A-Z0-9]+)+$")
    kind: Kind
    description: str
    when: Condition
    outcome: Outcome
    unless: list[str] = Field(default_factory=list)  # documentation only
    sources: list[Source] = Field(default_factory=list)
    design_ref: str | None = None  # for non-legal rules: where the choice is explained

    @model_validator(mode="after")
    def _citations(self) -> Rule:
        if self.kind == "legal" and not self.sources:
            raise ValueError(
                f"legal rule {self.id} needs at least one source with url and checked_on"
            )
        if self.kind != "legal" and not (self.design_ref or self.sources):
            raise ValueError(f"rule {self.id} needs design_ref (or sources)")
        return self


class Defaults(_Strict):
    outcome_if_no_rule_fires: Outcome
    auto_hold_max_tax_paise: int = Field(ge=0)
    min_extraction_confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("outcome_if_no_rule_fires")
    @classmethod
    def _fail_closed(cls, v: str) -> str:
        if v == "auto_allow":
            raise ValueError("the default outcome may never be auto_allow (fail closed)")
        return v


class Policy(_Strict):
    version: str
    checked_on: dt.date
    defaults: Defaults
    allowed_actions: dict[str, list[str]]
    rules: list[Rule] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Policy:
        ids = [r.id for r in self.rules]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate rule ids {sorted(dupes)}")
        missing = set(MISMATCH_TYPES) - set(self.allowed_actions)
        extra = set(self.allowed_actions) - set(MISMATCH_TYPES)
        if missing or extra:
            raise ValueError(
                f"allowed_actions must list every mismatch type exactly "
                f"(missing {sorted(missing)}, unknown {sorted(extra)})"
            )
        for mt, actions in self.allowed_actions.items():
            bad = [a for a in actions if a not in ACTION_KINDS]
            if bad or not actions:
                raise ValueError(f"allowed_actions[{mt}] has unknown or no actions: {bad}")
        known = set(ids)
        for r in self.rules:
            unknown = [u for u in r.unless if u not in known]
            if unknown:
                raise ValueError(f"{r.id}.unless names unknown rules {unknown}")
        return self
