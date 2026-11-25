"""The policy engine: deterministic facts in, a cited ``Decision`` out (docs/SPEC.md 7.2).

1. Build the facts for one case. Money, flags and confidence come from the matcher and the
   extraction checks only; the proposal adds the action it chose, who proposed it, whether
   that action is allowed for the mismatch type, and whether it is the planner's fallback.
2. Evaluate every rule; collect the ones that fire.
3. Outcome = the strictest fired outcome (block > require_approval > auto_allow). If none
   fired, the default (require_approval: fail closed). auto_allow only when an auto_allow rule
   fired and nothing stricter did.

The proposal's reason, evidence and rule references are free text from the model; they are
never facts, so they cannot move the outcome (property-tested).
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from agent_orchestrator.models import Case, Decision, ExtractedInvoice, ProposedAction
from agent_orchestrator.policy.schema import FACTS, All, AnyOf, Condition, Leaf, Not, Policy

RANK = {"auto_allow": 0, "require_approval": 1, "block": 2}
NO_RULE_FIRED = "DEFAULT-NO-RULE"
EXTRACTION_FAILURES = {"extraction_failed", "llm_error", "budget_stop"}


class PolicyError(ValueError):
    """The policy file is invalid; nothing can be decided with it."""


def load_policy(path: Path | str | None = None) -> Policy:
    if path is None:
        text = (
            resources.files("agent_orchestrator.policy.rules")
            .joinpath("policy_v1.yaml")
            .read_text(encoding="utf-8")
        )
    else:
        text = Path(path).read_text(encoding="utf-8")
    try:
        return Policy.model_validate(yaml.safe_load(text))
    except Exception as exc:  # pydantic or YAML error: refuse to run with a broken policy
        raise PolicyError(f"invalid policy file: {exc}") from exc


@lru_cache(maxsize=1)
def default_policy() -> Policy:
    return load_policy()


def policy_version(policy: Policy) -> str:
    return f"policy_v1@{policy.version}"


def build_facts(
    case: Case,
    proposal: ProposedAction,
    policy: Policy,
    extraction: ExtractedInvoice | None = None,
    budget_stopped: bool = False,
    fallback: bool = False,
) -> dict[str, Any]:
    flags = set(case.flags)
    checks = list(extraction.checks) if extraction is not None else []
    failed = bool(EXTRACTION_FAILURES & set(checks))
    injection = "injection" in flags or bool(extraction and extraction.injection.flagged)
    facts = {
        "case.mismatch_type": case.mismatch_type,
        "case.status": case.status,
        "case.match_pass": case.match_pass,
        "case.itc_at_risk_paise": case.itc_at_risk_paise,
        "case.has_book_line": bool(case.book_lines),
        "case.has_pdf": case.pdf_file is not None,
        "case.injection": injection,
        "case.pdf_disagrees": "pdf_disagrees" in flags,
        "case.gstin_invalid": "gstin_invalid" in flags,
        "case.fuzzy_match": "fuzzy_match" in flags,
        "case.ambiguous_match": "ambiguous_match" in flags,
        "case.flags": sorted(flags),
        "proposal.action": proposal.action,
        "proposal.proposer": proposal.proposer,
        "proposal.allowed": proposal.action in policy.allowed_actions.get(case.mismatch_type, []),
        "proposal.fallback": fallback,
        "extraction.present": extraction is not None,
        "extraction.failed": failed,
        # No usable extraction counts as zero confidence (fail closed).
        "extraction.min_confidence": (
            float(extraction.min_critical_confidence)
            if extraction is not None and not failed
            else 0.0
        ),
        "extraction.checks": checks,
        "run.budget_stopped": budget_stopped,
    }
    assert set(facts) == set(FACTS), "facts and schema out of sync"
    return facts


def evaluate(cond: Condition, facts: dict[str, Any]) -> bool:
    if isinstance(cond, All):
        return all(evaluate(c, facts) for c in cond.all)
    if isinstance(cond, AnyOf):
        return any(evaluate(c, facts) for c in cond.any)
    if isinstance(cond, Not):
        return not evaluate(cond.not_, facts)
    assert isinstance(cond, Leaf)
    actual = facts[cond.fact]  # KeyError = a bug: every fact is always present
    v = cond.value
    match cond.op:
        case "eq":
            return actual == v
        case "ne":
            return actual != v
        case "in":
            return actual in v
        case "not_in":
            return actual not in v
        case "lt":
            return actual < v
        case "le":
            return actual <= v
        case "gt":
            return actual > v
        case "ge":
            return actual >= v
        case "contains":
            return v in actual
    raise AssertionError(f"unhandled op {cond.op}")  # pragma: no cover


def decide(
    case: Case,
    proposal: ProposedAction,
    *,
    extraction: ExtractedInvoice | None = None,
    policy: Policy | None = None,
    budget_stopped: bool = False,
    fallback: bool = False,
    now: dt.datetime | None = None,
) -> Decision:
    """Decide one case. Pure: same inputs, same outcome and fired rules."""
    policy = policy or default_policy()
    if proposal.case_key != case.case_key:
        raise ValueError("proposal is for a different case")
    facts = build_facts(case, proposal, policy, extraction, budget_stopped, fallback)
    fired = [r for r in policy.rules if evaluate(r.when, facts)]
    if fired:
        outcome = max((r.outcome for r in fired), key=RANK.__getitem__)
        fired_ids = [r.id for r in fired]
    else:
        outcome = policy.defaults.outcome_if_no_rule_fires
        fired_ids = [NO_RULE_FIRED]
    return Decision(
        case_key=case.case_key,
        outcome=outcome,
        action=proposal.action,
        fired_rules=fired_ids,
        policy_version=policy_version(policy),
        decided_at=now or dt.datetime.now(dt.UTC),
    )
