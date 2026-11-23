"""What the vision agent read from an invoice image (spec section 6.2)."""

from __future__ import annotations

from pydantic import Field

from agent_orchestrator.models.base import Confidence, Frozen


class FieldValue(Frozen):
    """One extracted field: the text as printed, and the final confidence (0 to 1) after the
    deterministic checks (checksum, arithmetic) have raised or lowered it."""

    value: str | None
    confidence: Confidence


class InjectionResult(Frozen):
    """Result of scanning untrusted document text for instructions aimed at the model."""

    flagged: bool = False
    sources: list[str] = Field(default_factory=list)  # "text_layer", "other_text", ...
    rule_ids: list[str] = Field(default_factory=list)


class ExtractedInvoice(Frozen):
    pdf_file: str
    model_id: str
    escalated: bool  # re-read by the stronger model because of low confidence
    supplier_name: FieldValue
    supplier_gstin: FieldValue
    buyer_gstin: FieldValue
    invoice_no: FieldValue
    invoice_date: FieldValue
    taxable_value: FieldValue
    rate: FieldValue
    igst: FieldValue
    cgst: FieldValue
    sgst: FieldValue
    total: FieldValue
    # Every other visible text, verbatim. Untrusted: it is scanned for injection and shown to a
    # person inside a labelled box, but NEVER passed to the planner or communicator prompts.
    other_text: str = ""
    checks: list[str] = Field(default_factory=list)  # e.g. "gstin_checksum_ok"
    injection: InjectionResult = Field(default_factory=InjectionResult)
    min_critical_confidence: Confidence
