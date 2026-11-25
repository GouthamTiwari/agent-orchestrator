"""Extraction agent: read one untrusted invoice image into checked, typed fields (spec 6.2).

Steps:
1. Render page 1 to PNG at 150 DPI (or use a supplied scan image, for the noisy eval set).
2. Free injection scan of the PDF text layer.
3. Ask the vision model for JSON; the prompt carries no document text, only the image.
4. Schema guard: validate; on failure re-ask with the errors, at most 2 retries; after that
   the result is empty with check ``extraction_failed``.
5. Normalise values and run deterministic checks that can only LOWER confidence.
6. If the lowest critical confidence is below the threshold, re-read once with the stronger
   model and keep the better result (``escalated=True``).
7. Injection scan over ``other_text`` and every extracted string field.

``other_text`` stays inside the result: it goes to the scanner, the store and the escaped UI,
never into another prompt.
"""

from __future__ import annotations

import datetime as dt
import re
from decimal import Decimal
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agent_orchestrator import gstin, money
from agent_orchestrator.interfaces.protocols import Guard
from agent_orchestrator.llm.budget import BudgetExceeded
from agent_orchestrator.llm.client import ConfigError, LLMClient
from agent_orchestrator.llm.pricing import UnknownModel
from agent_orchestrator.llm.recording import RecordingMissing
from agent_orchestrator.llm.types import ImagePart, Message
from agent_orchestrator.models import ExtractedInvoice, FieldValue, InjectionResult
from agent_orchestrator.render import pdf_text_layer, render_page_png

FIELDS = (
    "supplier_name",
    "supplier_gstin",
    "buyer_gstin",
    "invoice_no",
    "invoice_date",
    "taxable_value",
    "rate",
    "igst",
    "cgst",
    "sgst",
    "total",
)
CRITICAL = ("supplier_gstin", "invoice_no", "invoice_date", "taxable_value", "igst", "cgst", "sgst")
MONEY_FIELDS = ("taxable_value", "igst", "cgst", "sgst", "total")
TAX_HEADS = ("igst", "cgst", "sgst")
THRESHOLD = 0.85
MAX_SCHEMA_RETRIES = 2
MAX_OUTPUT_TOKENS = 1500


def _prompt(name: str) -> str:
    return (
        resources.files("agent_orchestrator.llm.prompts").joinpath(name).read_text("utf-8").strip()
    )


SYSTEM_PROMPT = _prompt("extract_system.txt")
USER_PROMPT = _prompt("extract_user.txt")


# --- what the model must return ---------------------------------------------------------------


class _AnswerField(BaseModel):
    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)
    value: str | None
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    supplier_name: _AnswerField
    supplier_gstin: _AnswerField
    buyer_gstin: _AnswerField
    invoice_no: _AnswerField
    invoice_date: _AnswerField
    taxable_value: _AnswerField
    rate: _AnswerField
    igst: _AnswerField
    cgst: _AnswerField
    sgst: _AnswerField
    total: _AnswerField
    other_text: str = ""


# --- normalisation ----------------------------------------------------------------------------

_CURRENCY = re.compile(r"(?i)(rs\.?|inr|₹)")
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%d-%b-%Y",
    "%b %d, %Y",
)


def normalise_money(raw: str | None) -> str | None:
    """'Rs 47,300.00' -> '47300.00'. None if it is not an amount."""
    if raw is None:
        return None
    text = _CURRENCY.sub("", raw).replace(",", "").replace(" ", "").strip()
    if not text:
        return None
    if text.lower() in ("-", "nil"):
        return "0.00"
    try:
        paise = money.parse(text)
    except money.MoneyError:
        return None
    return f"{Decimal(paise) / 100:.2f}"


def normalise_rate(raw: str | None) -> str | None:
    if raw is None:
        return None
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%?\s*", raw)
    if not m:
        return None
    d = Decimal(m.group(1))
    return str(int(d)) if d == d.to_integral_value() else str(d.normalize())


def normalise_date(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = " ".join(raw.split())
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


# --- the agent --------------------------------------------------------------------------------


def _empty(pdf_file: str, model_id: str, check: str, injection: InjectionResult,
           other_text: str = "") -> ExtractedInvoice:  # fmt: skip
    none = FieldValue(value=None, confidence=0.0)
    return ExtractedInvoice(
        pdf_file=pdf_file, model_id=model_id, escalated=False, **dict.fromkeys(FIELDS, none),
        other_text=other_text, checks=[check], injection=injection, min_critical_confidence=0.0,
    )  # fmt: skip


def _ask(
    client: LLMClient, guard: Guard, image: ImagePart, case_key: str | None, purpose: str
) -> tuple[ExtractionAnswer | None, list[str]]:
    """One schema-guarded extraction: up to 1 + MAX_SCHEMA_RETRIES calls."""
    messages = [Message.user(image, USER_PROMPT)]
    errors: list[str] = []
    for attempt in range(1 + MAX_SCHEMA_RETRIES):
        result = client.complete(
            purpose if attempt == 0 else f"{purpose}-fix",
            messages,
            system=SYSTEM_PROMPT,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.0,
            case_key=case_key,
        )
        checked = guard.validate(result.text, ExtractionAnswer)
        if checked.ok:
            return checked.value, errors
        errors = checked.errors
        # Re-ask with the field names and rules that failed; never with document text.
        feedback = "; ".join(errors[:8])[:600]
        messages = [
            *messages,
            Message.assistant(result.text[:4000]),
            Message.user(
                "Your reply did not match the required JSON. Problems: "
                f"{feedback}. Reply again with the corrected JSON object only."
            ),
        ]
    return None, errors


def _apply_checks(
    answer: ExtractionAnswer, buyer_gstin: str | None
) -> tuple[dict[str, FieldValue], list[str]]:
    """Normalise values and lower confidence where deterministic checks fail."""
    values: dict[str, str | None] = {}
    conf: dict[str, float] = {}
    checks: list[str] = []
    for name in FIELDS:
        f = getattr(answer, name)
        values[name], conf[name] = f.value, f.confidence

    def cap(name: str, ceiling: float) -> None:
        conf[name] = min(conf[name], ceiling)

    for name in ("supplier_gstin", "buyer_gstin"):
        if values[name] is not None:
            values[name] = gstin.normalise(values[name])
            if gstin.checksum_valid(values[name]):
                checks.append(f"{name}_checksum_ok")
            else:
                checks.append(f"{name}_checksum_failed")
                cap(name, 0.2)
    for name in MONEY_FIELDS:
        raw = values[name]
        values[name] = normalise_money(raw)
        if raw is not None and values[name] is None:
            checks.append(f"{name}_unreadable")
            cap(name, 0.3)
    raw_rate, raw_date = values["rate"], values["invoice_date"]
    values["rate"], values["invoice_date"] = normalise_rate(raw_rate), normalise_date(raw_date)
    if raw_rate is not None and values["rate"] is None:
        checks.append("rate_unreadable")
        cap("rate", 0.3)
    if raw_date is not None and values["invoice_date"] is None:
        checks.append("invoice_date_unreadable")
        cap("invoice_date", 0.3)
    if values["invoice_no"] is not None:
        values["invoice_no"] = values["invoice_no"].strip()
    # A tax head that is not printed is zero when another head is (CGST+SGST xor IGST).
    if any(values[h] is not None for h in TAX_HEADS):
        for h in TAX_HEADS:
            if values[h] is None:
                values[h] = "0.00"

    paise = {n: money.parse(values[n]) if values[n] else 0 for n in MONEY_FIELDS}
    tax = paise["igst"] + paise["cgst"] + paise["sgst"]
    t = money.TOLERANCE_PAISE
    if values["taxable_value"] and values["rate"]:
        expected = Decimal(paise["taxable_value"]) * Decimal(values["rate"]) / 100
        if abs(Decimal(tax) - expected) > t:
            checks.append("tax_arithmetic_failed")
            for h in TAX_HEADS:
                cap(h, 0.5)
        else:
            checks.append("tax_arithmetic_ok")
    s_state = (values["supplier_gstin"] or "")[:2]
    b_state = (values["buyer_gstin"] or buyer_gstin or "")[:2]
    if s_state and s_state == b_state and paise["cgst"] != paise["sgst"]:
        checks.append("cgst_sgst_unequal")
        cap("cgst", 0.5)
        cap("sgst", 0.5)
    if values["total"] and values["taxable_value"]:
        if abs(paise["total"] - (paise["taxable_value"] + tax)) > t:
            checks.append("total_failed")
            cap("total", 0.5)
        else:
            checks.append("total_ok")
    if buyer_gstin and values["buyer_gstin"] and values["buyer_gstin"] != buyer_gstin:
        checks.append("buyer_gstin_mismatch")
    fields = {n: FieldValue(value=values[n], confidence=round(conf[n], 4)) for n in FIELDS}
    return fields, checks


def _build(
    pdf_file: str, model_id: str, escalated: bool, answer: ExtractionAnswer,
    buyer_gstin: str | None, injection: InjectionResult,
) -> ExtractedInvoice:  # fmt: skip
    fields, checks = _apply_checks(answer, buyer_gstin)
    return ExtractedInvoice(
        pdf_file=pdf_file,
        model_id=model_id,
        escalated=escalated,
        **fields,
        other_text=answer.other_text,
        checks=checks,
        injection=injection,
        min_critical_confidence=min(fields[n].confidence for n in CRITICAL),
    )


def _merge(a: InjectionResult, b: InjectionResult) -> InjectionResult:
    sources = list(dict.fromkeys(a.sources + b.sources))
    return InjectionResult(
        flagged=a.flagged or b.flagged,
        sources=sources,
        rule_ids=sorted(set(a.rule_ids) | set(b.rule_ids)),
    )


def _scan_result(guard: Guard, ex: ExtractedInvoice, base: InjectionResult) -> ExtractedInvoice:
    texts = {"other_text": ex.other_text}
    texts |= {f"field:{n}": getattr(ex, n).value or "" for n in FIELDS}
    return ex.model_copy(update={"injection": _merge(base, guard.scan(texts))})


def extract_invoice(
    pdf_path: Path | str,
    *,
    client: LLMClient,
    guard: Guard,
    buyer_gstin: str | None = None,
    escalation_client: LLMClient | None = None,
    image_path: Path | str | None = None,
    pdf_file: str | None = None,
    case_key: str | None = None,
    threshold: float = THRESHOLD,
) -> ExtractedInvoice:
    """Read one invoice. Never raises for a bad model answer, a budget stop or a provider
    failure: those come back as an empty result with check ``extraction_failed``,
    ``budget_stop`` or ``llm_error`` (the case then goes to a person)."""
    pdf_path = Path(pdf_path)
    pdf_file = pdf_file or pdf_path.name
    # 1. the image the model sees
    if image_path is not None:
        data = Path(image_path).read_bytes()
        media = (
            "image/jpeg" if Path(image_path).suffix.lower() in (".jpg", ".jpeg") else "image/png"
        )
        image = ImagePart(data, media)
    else:
        image = ImagePart(render_page_png(pdf_path), "image/png")
    # 2. free scan of the text layer (attackers can hide text that no image shows)
    layer_text = pdf_text_layer(pdf_path) if pdf_path.suffix.lower() == ".pdf" else ""
    base = guard.scan({"text_layer": layer_text}) if layer_text else InjectionResult()

    def read(c: LLMClient, purpose: str, escalated: bool) -> ExtractedInvoice:
        try:
            answer, errors = _ask(c, guard, image, case_key, purpose)
        except BudgetExceeded:
            return _empty(pdf_file, c.model_id, "budget_stop", base)
        except (RecordingMissing, UnknownModel, ConfigError):
            raise  # configuration problems must stop the run, not hide as a bad document
        except Exception:  # provider failure after retries
            return _empty(pdf_file, c.model_id, "llm_error", base)
        if answer is None:
            return _empty(pdf_file, c.model_id, "extraction_failed", base)
        return _build(pdf_file, c.model_id, escalated, answer, buyer_gstin, base)

    first = read(client, "extract", False)
    best = first
    if (
        escalation_client is not None
        and first.min_critical_confidence < threshold
        and "budget_stop" not in first.checks
    ):
        second = read(escalation_client, "extract-escalation", True)
        if second.min_critical_confidence > first.min_critical_confidence:
            best = second
        else:
            best = first.model_copy(update={"escalated": True})
    # 7. scan everything the model copied out of the document
    return _scan_result(guard, best, best.injection)
