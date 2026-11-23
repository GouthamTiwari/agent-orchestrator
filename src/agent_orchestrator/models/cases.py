"""Reconciliation cases and the closed vocabularies every agent shares (spec section 5)."""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import Field

from agent_orchestrator.models.base import Frozen, NonNegativePaise, Paise, Period

MismatchType = Literal[
    "none",
    "rounding_within_tolerance",
    "invoice_no_format_difference",
    "missing_in_2b",
    "in_2b_not_in_books",
    "taxable_value_mismatch",
    "tax_amount_mismatch",
    "gstin_typo_in_books",
    "tax_head_mismatch",
    "duplicate_in_books",
    "date_mismatch_other_period",
]

CaseStatus = Literal[
    "matched",
    "matched_after_normalisation",
    "mismatch",
    "unmatched",
    "duplicate",
    "needs_human",
]

ActionKind = Literal[
    "none",
    "chase_supplier_and_hold_itc",
    "verify_with_supplier",
    "correct_books",
    "escalate_human",
]

MISMATCH_TYPES: tuple[str, ...] = get_args(MismatchType)
CASE_STATUSES: tuple[str, ...] = get_args(CaseStatus)
ACTION_KINDS: tuple[str, ...] = get_args(ActionKind)


class Case(Frozen):
    """One invoice identity after matching: the unit that gets a proposal and a decision."""

    case_key: str = Field(pattern=r"^[0-9a-f]{16}$")  # sha256(business, period, gstin, inv_key)
    business_id: str
    period: Period
    mismatch_type: MismatchType
    status: CaseStatus
    book_lines: list[int] = Field(default_factory=list)
    portal_index: int | None = None
    pdf_file: str | None = None
    match_pass: str  # "exact", "normalised", "no_year", "gstin_repair", "fuzzy", "none"
    match_score: float = Field(ge=0.0, le=100.0)
    diffs: dict[str, Paise] = Field(default_factory=dict)  # book minus portal, per field
    itc_at_risk_paise: NonNegativePaise = 0
    facts_hash: str  # hash of the facts the decision depends on (idempotency keys use it)
    flags: list[str] = Field(default_factory=list)  # e.g. "injection", "low_confidence"
