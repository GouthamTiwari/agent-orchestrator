"""Resolution planner: the LLM proposes one allowed action for a case (spec 6.5).

- Clean matches (``none``, ``rounding_within_tolerance``, ``invoice_no_format_difference``)
  never reach the model: a rule proposes ``none``.
- Otherwise the model sees only facts produced by code: the mismatch type, money in paise,
  match details, flags, extraction check names and the invoice amounts as numbers. No text
  from the document is ever in the prompt: not ``other_text``, not the supplier name, not the
  invoice number. It also sees the allowed actions (from the policy file), the evidence refs
  that exist for the case and the rule passages from research.
- The answer must pick an allowed action, give a reason, cite at least one evidence ref of the
  case and at least one rule passage it was shown. Any problem → re-ask with the problems
  (at most 2 retries) → the fallback proposal ``escalate_human`` with ``proposer="rule"`` and
  flag ``planner_failed``. The policy engine then requires approval (SAFE-PLAN-01).

A proposal is never acted on: the policy engine decides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from importlib import resources
from string import Template
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_orchestrator import money
from agent_orchestrator.agents.research import CLEAN_TYPES, MEANING, ResearchResult, format_passages
from agent_orchestrator.interfaces.protocols import Guard
from agent_orchestrator.llm.budget import BudgetExceeded
from agent_orchestrator.llm.client import ConfigError, LLMClient
from agent_orchestrator.llm.pricing import UnknownModel
from agent_orchestrator.llm.recording import RecordingMissing
from agent_orchestrator.llm.types import Message
from agent_orchestrator.models import Case, ExtractedInvoice, ProposedAction
from agent_orchestrator.policy.engine import EXTRACTION_FAILURES, default_policy
from agent_orchestrator.policy.schema import Policy

MAX_SCHEMA_RETRIES = 2
MAX_OUTPUT_TOKENS = 400
PDF_FIELDS = (
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
MONEY_FIELDS = ("taxable_value", "igst", "cgst", "sgst", "total")

ACTION_MEANING: dict[str, str] = {
    "none": "Nothing to do; the records agree.",
    "chase_supplier_and_hold_itc": "Do not take the credit yet; ask the supplier to report the "
    "invoice in GSTR-1 so it reaches GSTR-2B.",
    "verify_with_supplier": "Ask the supplier which figures are correct before taking credit.",
    "correct_books": "Fix the purchase books (for example a mistyped GSTIN or a duplicate line).",
    "escalate_human": "A person reviews the case; the facts or the law need judgement.",
}


def _prompt(name: str) -> str:
    return (
        resources.files("agent_orchestrator.llm.prompts").joinpath(name).read_text("utf-8").strip()
    )


SYSTEM_PROMPT = _prompt("plan_system.txt")
USER_PROMPT = Template(_prompt("plan_user.txt"))


class PlanAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str
    reason: str = Field(min_length=1, max_length=600)
    evidence_refs: list[str] = Field(min_length=1)
    rule_refs: list[str] = Field(min_length=1)


@dataclass(frozen=True)
class PlanResult:
    proposal: ProposedAction
    fallback: bool = False  # the policy fact ``proposal.fallback``
    # "planner_failed", "no_rule_text", "llm_error", "budget_stop"
    flags: tuple[str, ...] = ()
    attempts: int = 0  # model calls made
    errors: tuple[str, ...] = field(default=(), compare=False)  # field names and rules only


def evidence_refs(case: Case, extraction: ExtractedInvoice | None) -> list[str]:
    """The evidence a proposal may point to: book lines, the GSTR-2B row, read invoice fields."""
    refs = [f"books:line:{n}" for n in case.book_lines]
    if case.portal_index is not None:
        refs.append(f"2b:index:{case.portal_index}")
    if _usable(extraction):
        refs += [f"pdf:field:{name}" for name in PDF_FIELDS if getattr(extraction, name).value]
    return refs


def _usable(extraction: ExtractedInvoice | None) -> bool:
    return extraction is not None and not (EXTRACTION_FAILURES & set(extraction.checks))


def _invoice_numbers(extraction: ExtractedInvoice) -> dict[str, Any]:
    """Amounts as integer paise and the rate as a number. These values were normalised and
    checked by code; no document string passes through."""
    out: dict[str, Any] = {}
    for name in MONEY_FIELDS:
        value = getattr(extraction, name).value
        out[f"{name}_paise"] = money.parse(value) if value else None
    rate = extraction.rate.value
    out["rate_percent"] = float(Decimal(rate)) if rate else None
    return out


def case_facts(
    case: Case, extraction: ExtractedInvoice | None, allowed: list[str], refs: list[str]
) -> dict[str, Any]:
    """Everything the planner is told about the case. Built from typed fields only."""
    facts: dict[str, Any] = {
        "case_key": case.case_key,
        "period": case.period,
        "mismatch_type": case.mismatch_type,
        "mismatch_meaning": MEANING[case.mismatch_type],
        "status": case.status,
        "match_pass": case.match_pass,
        "match_score": case.match_score,
        "differences_books_minus_2b_paise": dict(sorted(case.diffs.items())),
        "itc_at_risk_paise": case.itc_at_risk_paise,
        "flags": sorted(case.flags),
        "has_book_line": bool(case.book_lines),
        "has_2b_record": case.portal_index is not None,
        "has_invoice_pdf": case.pdf_file is not None,
        "allowed_actions": {a: ACTION_MEANING[a] for a in allowed},
        "evidence_refs_available": refs,
    }
    if extraction is None:
        facts["invoice"] = None
    elif not _usable(extraction):
        facts["invoice"] = {"read": False, "checks": sorted(extraction.checks)}
    else:
        facts["invoice"] = {
            "read": True,
            **_invoice_numbers(extraction),
            "min_critical_confidence": extraction.min_critical_confidence,
            "checks": sorted(extraction.checks),
            "injection_flagged": extraction.injection.flagged,
        }
    return facts


def check_plan(
    answer: PlanAnswer, allowed: list[str], refs: list[str], rule_ids: list[str]
) -> list[str]:
    errors = []
    if answer.action not in allowed:
        errors.append("action: not one of the allowed actions")
    known_refs, known_rules = set(refs), set(rule_ids)
    errors += [
        f"evidence_refs[{i}]: not one of the case's evidence refs"
        for i, r in enumerate(answer.evidence_refs)
        if r not in known_refs
    ]
    errors += [
        f"rule_refs[{i}]: not one of the rule passage ids shown"
        for i, r in enumerate(answer.rule_refs)
        if r not in known_rules
    ]
    return errors


def _rule_proposal(case: Case, action: str, reason: str, refs: list[str]) -> ProposedAction:
    return ProposedAction(
        case_key=case.case_key, action=action, reason=reason, evidence_refs=refs, proposer="rule"
    )


def _fallback(case: Case, refs: list[str], flags: tuple[str, ...], attempts: int,
              errors: list[str] | tuple[str, ...] = ()) -> PlanResult:  # fmt: skip
    why = {
        "budget_stop": "The run budget stopped new model work",
        "no_rule_text": "No rule text was found for this mismatch type",
        "llm_error": "The model could not be reached",
    }.get(flags[-1], "The planner's proposal failed validation after retries")
    proposal = _rule_proposal(case, "escalate_human", f"{why}; a person decides.", refs)
    return PlanResult(proposal, True, flags, attempts, tuple(errors))


def plan_case(
    case: Case,
    *,
    research: ResearchResult | None,
    client: LLMClient,
    guard: Guard,
    extraction: ExtractedInvoice | None = None,
    policy: Policy | None = None,
) -> PlanResult:
    """Propose an action for one case. Never raises for a bad answer, a budget stop or a
    provider failure; configuration errors still stop the run."""
    refs = evidence_refs(case, extraction)
    if case.mismatch_type in CLEAN_TYPES:
        reason = "Books, GSTR-2B and the invoice agree; nothing to do (rule, no model call)."
        return PlanResult(_rule_proposal(case, "none", reason, refs))
    policy = policy or default_policy()
    allowed = list(policy.allowed_actions[case.mismatch_type])
    snippets = research.snippets if research is not None else ()
    if not snippets:
        return _fallback(case, refs, ("planner_failed", "no_rule_text"), 0)
    rule_ids = [s.snippet_id for s in snippets]
    summary = research.research.summary if research is not None else ""
    user = USER_PROMPT.substitute(
        facts=json.dumps(case_facts(case, extraction, allowed, refs), indent=2, sort_keys=True),
        rule_ids=", ".join(rule_ids),
        summary=summary or "(no summary available; read the passages)",
        passages=format_passages(snippets),
    )
    messages = [Message.user(user)]
    errors: list[str] = []
    attempts = 0
    try:
        for attempt in range(1 + MAX_SCHEMA_RETRIES):
            attempts += 1
            result = client.complete(
                "plan" if attempt == 0 else "plan-fix",
                messages,
                system=SYSTEM_PROMPT,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.0,
                case_key=case.case_key,
            )
            checked = guard.validate(result.text, PlanAnswer)
            if checked.ok:
                answer: PlanAnswer = checked.value
                errors = check_plan(answer, allowed, refs, rule_ids)
                if not errors:
                    proposal = ProposedAction(
                        case_key=case.case_key,
                        action=answer.action,
                        reason=answer.reason,
                        evidence_refs=list(dict.fromkeys(answer.evidence_refs)),
                        rule_refs=list(dict.fromkeys(answer.rule_refs)),
                        proposer="llm",
                    )
                    return PlanResult(proposal, attempts=attempts)
            else:
                errors = checked.errors
            # The feedback names fields and rules, and repeats our own lists; never values.
            feedback = "; ".join(errors[:8])[:600]
            messages = [
                *messages,
                Message.assistant(result.text[:4000]),
                Message.user(
                    f"Your reply was not accepted. Problems: {feedback}. Allowed actions: "
                    f"{', '.join(allowed)}. Evidence refs: {', '.join(refs) or '(none)'}. "
                    f"Rule passage ids: {', '.join(rule_ids)}. "
                    "Reply again with the corrected JSON object only."
                ),
            ]
    except BudgetExceeded:
        return _fallback(case, refs, ("budget_stop",), attempts, errors)
    except (RecordingMissing, UnknownModel, ConfigError):
        raise
    except Exception:  # provider failure after retries
        return _fallback(case, refs, ("planner_failed", "llm_error"), attempts, errors)
    return _fallback(case, refs, ("planner_failed",), attempts, errors)
