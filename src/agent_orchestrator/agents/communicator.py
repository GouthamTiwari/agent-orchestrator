"""Communicator: draft a follow-up e-mail to a supplier. Drafts only; nothing is sent (spec 6.7).

- **Who gets one:** a case whose final action is ``chase_supplier_and_hold_itc`` or
  ``verify_with_supplier``, and only when the policy auto-allowed it or a person approved or
  edited the action to it. A blocked, pending or rejected case never gets a draft.
- **What the model sees:** facts from the business's own books and the GSTR-2B statement
  (supplier name and GSTIN, invoice number and date, amounts). Never text read from the
  invoice image: not ``other_text``, not anything else the extraction copied.
- **Grounding check:** every number, date and invoice id in the draft must come from those
  facts; it must name the invoice, be in the requested language and contain no link or
  address. Anything else, or no model at all, gives the fixed template for that action and
  language (English, Hindi or Telugu), which needs no model.
- **Idempotency key:** business, period, case, final action and supplier GSTIN (spec 9.2).
  The wording is not in the key, so a re-run finds the first draft.

There is no SMTP or network code here, by design: a test greps for it.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_orchestrator.interfaces.protocols import Guard
from agent_orchestrator.llm.budget import BudgetExceeded
from agent_orchestrator.llm.client import ConfigError, LLMClient
from agent_orchestrator.llm.pricing import UnknownModel
from agent_orchestrator.llm.recording import RecordingMissing
from agent_orchestrator.llm.types import Message
from agent_orchestrator.models import (
    ApprovalRecord,
    BookRecord,
    Case,
    Decision,
    DraftEmail,
    PortalRecord,
)
from agent_orchestrator.money import format_inr
from agent_orchestrator.store.idempotency import outbox_key

Language = Literal["en", "hi", "te"]
LANGUAGES: tuple[str, ...] = ("en", "hi", "te")
LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu"}
DRAFT_ACTIONS = ("chase_supplier_and_hold_itc", "verify_with_supplier")
MAX_SCHEMA_RETRIES = 2
MAX_OUTPUT_TOKENS = 700
MONTHS = {
    "en": "January February March April May June July August September October November "
    "December".split(),
    "hi": "जनवरी फ़रवरी मार्च अप्रैल मई जून जुलाई अगस्त सितंबर अक्टूबर नवंबर दिसंबर".split(),
    "te": "జనవరి ఫిబ్రవరి మార్చి ఏప్రిల్ మే జూన్ జూలై ఆగస్టు సెప్టెంబర్ అక్టోబర్ నవంబర్ డిసెంబర్".split(),
}
# A Hindi draft must contain Devanagari and a Telugu one Telugu script.
_SCRIPT = {"hi": re.compile("[ऀ-ॿ]"), "te": re.compile("[ఀ-౿]")}


def _text(package: str, name: str) -> str:
    return resources.files(package).joinpath(name).read_text("utf-8").strip()


SYSTEM_PROMPT = _text("agent_orchestrator.llm.prompts", "email_system.txt")
USER_PROMPT = Template(_text("agent_orchestrator.llm.prompts", "email_user.txt"))


def template_name(action: str, language: str) -> str:
    short = {"chase_supplier_and_hold_itc": "chase", "verify_with_supplier": "verify"}[action]
    return f"email_fallback_{short}_{language}.txt"


class EmailAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=2500)


@dataclass(frozen=True)
class DraftResult:
    draft: DraftEmail | None
    source: Literal["llm", "template", "not_eligible"]
    # why the template was used: "no_model", "grounding_failed", "schema_failed",
    # "budget_stop", "llm_error"; "" for an accepted model draft or no draft
    note: str = ""
    errors: tuple[str, ...] = field(default=(), compare=False)


# --- who gets a draft -----------------------------------------------------------------------------


def final_action(decision: Decision | None, approval: ApprovalRecord | None) -> str | None:
    """The action that actually stands for a case, or None while it waits for a person.

    - an approval a person decided: its final action (approved, edited or rejected);
    - else a decision the policy auto-allowed: its action;
    - a blocked decision: ``escalate_human`` (a person must look; nothing automatic happens);
    - otherwise (approval pending or not yet requested): None.
    """
    if approval is not None and approval.status != "pending":
        return approval.final_action
    if decision is None:
        return None
    if decision.outcome == "auto_allow":
        return decision.action
    if decision.outcome == "block":
        return "escalate_human"
    return None


def eligible(decision: Decision | None, approval: ApprovalRecord | None) -> bool:
    """Spec 6.7: a chase or verify action that the policy allowed or a person approved."""
    if decision is not None and decision.outcome == "block":
        return False  # a blocked case gets nothing automatic, whatever happened later
    allowed = (decision is not None and decision.outcome == "auto_allow") or (
        approval is not None and approval.status in ("approved", "edited")
    )
    return allowed and final_action(decision, approval) in DRAFT_ACTIONS


def load_supplier_languages(business_json: Path | str) -> dict[str, str]:
    """GSTIN -> language from a generated ``business.json`` (fictional supplier profiles)."""
    data = json.loads(Path(business_json).read_text("utf-8"))
    return {
        s["gstin"]: s.get("language", "en")
        for s in data.get("suppliers", [])
        if s.get("language", "en") in LANGUAGES
    }


# --- the facts a draft may use --------------------------------------------------------------------


def _tax(r: BookRecord | PortalRecord) -> int:
    return r.igst_paise + r.cgst_paise + r.sgst_paise


def period_label(period: str, language: str) -> str:
    year, month = period.split("-")
    return f"{MONTHS[language][int(month) - 1]} {year}"


def email_facts(
    case: Case,
    action: str,
    books: Mapping[int, BookRecord],
    portal: Mapping[int, PortalRecord],
    language: str,
) -> dict[str, Any]:
    """Facts from the books and GSTR-2B only (never from the invoice image)."""
    book = books[case.book_lines[0]] if case.book_lines else None
    p2b = portal[case.portal_index] if case.portal_index is not None else None
    main = p2b or book
    if main is None:
        raise ValueError("a case without a book line or a GSTR-2B row cannot be e-mailed")
    facts: dict[str, Any] = {
        "action": action,
        "language": LANGUAGE_NAMES[language],
        "period": case.period,
        "period_label": period_label(case.period, language),
        "supplier_name": (book or main).supplier_name,
        "supplier_gstin": main.supplier_gstin,  # the 2B GSTIN corrects a typo in the books
        "invoice_no": (book or main).invoice_no,
        "invoice_date": (book or main).date.isoformat(),
    }
    if book is not None:
        facts["our_records"] = {
            "taxable_value": format_inr(book.taxable_paise),
            "gst": format_inr(_tax(book)),
            "rate_percent": book.rate,
        }
    if p2b is not None:
        facts["gstr2b"] = {
            "invoice_no": p2b.invoice_no,
            "invoice_date": p2b.date.isoformat(),
            "taxable_value": format_inr(p2b.taxable_paise),
            "gst": format_inr(_tax(p2b)),
            "rate_percent": p2b.rate,
            "reported_in_period": p2b.reported_in_period,
        }
    return facts


# --- grounding ------------------------------------------------------------------------------------

# Names of GST returns contain digits but are not facts ("GSTR-2B", "GSTR-1", "IFF").
_RETURN_NAMES = re.compile(r"(?i)\bGSTR[\s-]?\d[A-Z]?\b|\b2B\b")
# Invoice numbers and GSTINs: ASCII letters and digits together, with "/" or "-" allowed.
_ID = re.compile(
    r"(?<![A-Za-z0-9/-])(?=[A-Za-z0-9/-]*\d)(?=[A-Za-z0-9/-]*[A-Za-z])"
    r"[A-Za-z0-9][A-Za-z0-9/-]{3,}(?![A-Za-z0-9/-])"
)
_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b|\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b")
_NUMBER = re.compile(r"\d+(?:,\d+)*(?:\.\d+)?")
_FORBIDDEN = re.compile(r"(?i)https?://|www\.|@|<\s*script|\bclick here\b")


def ascii_digits(text: str) -> str:
    """Devanagari, Telugu and other decimal digits -> 0-9, so they are checked too."""
    return "".join(
        str(unicodedata.decimal(ch)) if ch.isdecimal() and not ch.isascii() else ch for ch in text
    )


def _decimal(token: str) -> Decimal | None:
    try:
        return Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None


def allowed_values(facts: Any) -> tuple[set[str], set[Decimal], set[dt.date]]:
    """Ids, numbers and dates that appear anywhere in ``facts``."""
    ids: set[str] = set()
    numbers: set[Decimal] = set()
    dates: set[dt.date] = set()

    def walk(v: Any) -> None:
        if isinstance(v, Mapping):
            for x in v.values():
                walk(x)
        elif isinstance(v, list | tuple | set):
            for x in v:
                walk(x)
        elif isinstance(v, bool) or v is None:
            return
        elif isinstance(v, int | float | Decimal):
            numbers.add(Decimal(str(v)))
        else:
            s = ascii_digits(str(v))
            for m in _DATE.finditer(s):
                d = _parse_date(m)
                if d:
                    dates.add(d)
                    numbers.update({Decimal(d.day), Decimal(d.month), Decimal(d.year)})
            if re.fullmatch(r"\d{4}-\d{2}", s):  # a period: its month and year
                y, mo = s.split("-")
                numbers.update({Decimal(int(y)), Decimal(int(mo))})
            for m in _ID.finditer(s):
                ids.add(m.group(0).upper())
            for m in _NUMBER.finditer(_ID.sub(" ", _DATE.sub(" ", s))):
                n = _decimal(m.group(0))
                if n is not None:
                    numbers.add(n)

    walk(facts)
    return ids, numbers, dates


def _parse_date(m: re.Match[str]) -> dt.date | None:
    try:
        if m.group(1):
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return dt.date(int(m.group(6)), int(m.group(5)), int(m.group(4)))  # day first (India)
    except ValueError:
        return None


def ungrounded(text: str, facts: Any) -> list[str]:
    """What in ``text`` does not come from ``facts``: ids, dates and numbers. The messages
    say what kind of thing failed, never the value."""
    ids, numbers, dates = allowed_values(facts)
    s = _RETURN_NAMES.sub(" ", ascii_digits(text))
    problems = []
    for m in _ID.finditer(s):
        if m.group(0).upper().rstrip(".,;:") not in ids:
            problems.append("an id or code that is not in the facts")
    s = _ID.sub(" ", s)
    for m in _DATE.finditer(s):
        if _parse_date(m) not in dates:
            problems.append("a date that is not in the facts")
    s = _DATE.sub(" ", s)
    for m in _NUMBER.finditer(s):
        n = _decimal(m.group(0))
        if n is None or n not in numbers:
            problems.append("a number that is not in the facts")
    return problems


def check_draft(answer: EmailAnswer, facts: Mapping[str, Any], language: str) -> list[str]:
    text = f"{answer.subject}\n{answer.body}"
    problems = ungrounded(text, facts)
    if facts["invoice_no"].upper() not in ascii_digits(text).upper():
        problems.append("the draft does not name the invoice number")
    if _FORBIDDEN.search(text):
        problems.append("the draft contains a link, an address or markup")
    if language in _SCRIPT and not _SCRIPT[language].search(text):
        problems.append(f"the draft is not in {LANGUAGE_NAMES[language]}")
    return list(dict.fromkeys(problems))


# --- the drafts -----------------------------------------------------------------------------------


def template_draft(facts: Mapping[str, Any], action: str, language: str) -> tuple[str, str]:
    """(subject, body) from the fixed template. No model; always grounded by construction."""
    text = _text("agent_orchestrator.templates", template_name(action, language))
    ours, p2b = facts.get("our_records") or {}, facts.get("gstr2b") or {}
    values = {
        "supplier_name": facts["supplier_name"],
        "supplier_gstin": facts["supplier_gstin"],
        "invoice_no": facts["invoice_no"],
        "invoice_date": facts["invoice_date"],
        "period_label": facts["period_label"],
        "taxable": ours.get("taxable_value", p2b.get("taxable_value", "")),
        "tax": ours.get("gst", p2b.get("gst", "")),
        "book_taxable": ours.get("taxable_value", "-"),
        "book_tax": ours.get("gst", "-"),
        "portal_taxable": p2b.get("taxable_value", "-"),
        "portal_tax": p2b.get("gst", "-"),
    }
    filled = Template(text).substitute(values)
    first, _, body = filled.partition("\n")
    subject = first.split(":", 1)[1].strip() if ":" in first else first.strip()
    return subject[:200], body.strip()


def _ask(
    client: LLMClient, guard: Guard, facts: Mapping[str, Any], language: str, case_key: str
) -> tuple[EmailAnswer | None, list[str], str]:
    """Returns (answer, problems, note). A grounding failure is final: no retry, use the
    template (it is cheap and always correct). Bad JSON is re-asked up to 2 times."""
    user = USER_PROMPT.substitute(
        language=LANGUAGE_NAMES[language],
        facts=json.dumps(facts, indent=2, ensure_ascii=False, sort_keys=True),
    )
    messages = [Message.user(user)]
    errors: list[str] = []
    for attempt in range(1 + MAX_SCHEMA_RETRIES):
        result = client.complete(
            "email" if attempt == 0 else "email-fix",
            messages,
            system=SYSTEM_PROMPT,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.2,
            case_key=case_key,
        )
        checked = guard.validate(result.text, EmailAnswer)
        if checked.ok:
            problems = check_draft(checked.value, facts, language)
            return (None, problems, "grounding_failed") if problems else (checked.value, [], "")
        errors = checked.errors
        messages = [
            *messages,
            Message.assistant(result.text[:4000]),
            Message.user(
                f"Your reply did not match the required JSON. Problems: "
                f"{'; '.join(errors[:8])[:600]}. Reply again with the JSON object only."
            ),
        ]
    return None, errors, "schema_failed"


def draft_email(
    case: Case,
    *,
    decision: Decision | None,
    approval: ApprovalRecord | None,
    books: Mapping[int, BookRecord],
    portal: Mapping[int, PortalRecord],
    guard: Guard,
    client: LLMClient | None = None,
    language: str = "en",
    now: dt.datetime | None = None,
) -> DraftResult:
    """Draft one e-mail, or nothing when the case is not eligible. Never raises for a bad
    answer, a budget stop or a provider failure (the template is used); configuration errors
    still stop the run."""
    if not eligible(decision, approval):
        return DraftResult(None, "not_eligible")
    if language not in LANGUAGES:
        language = "en"
    action = final_action(decision, approval)
    assert action in DRAFT_ACTIONS  # eligible() guarantees it
    facts = email_facts(case, action, books, portal, language)
    answer, errors, note = None, [], "no_model"
    if client is not None:
        try:
            answer, errors, note = _ask(client, guard, facts, language, case.case_key)
        except BudgetExceeded:
            note = "budget_stop"
        except (RecordingMissing, UnknownModel, ConfigError):
            raise
        except Exception:  # provider failure after retries
            note = "llm_error"
    if answer is not None:
        subject, body, source = answer.subject.strip(), answer.body.strip(), "llm"
    else:
        (subject, body), source = template_draft(facts, action, language), "template"
    draft = DraftEmail(
        idempotency_key=outbox_key(
            case.business_id, case.period, case.case_key, action, facts["supplier_gstin"]
        ),
        case_key=case.case_key,
        to_supplier_gstin=facts["supplier_gstin"],
        subject=subject,
        body=body,
        language=language,
        created_at=now or dt.datetime.now(dt.UTC),
    )
    return DraftResult(draft, source, note, tuple(errors))
