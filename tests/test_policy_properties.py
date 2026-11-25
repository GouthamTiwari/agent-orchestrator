"""Proof that the model cannot talk its way past the policy (docs/SPEC.md section 7.3).

The oracle below is written separately from the engine, straight from the spec: it says when
an automatic action is acceptable. If the engine ever auto-allows something the oracle would
not, hypothesis finds the smallest such case.
"""

import datetime as dt
import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from agent_orchestrator.models import (
    ACTION_KINDS,
    CASE_STATUSES,
    MISMATCH_TYPES,
    Case,
    ExtractedInvoice,
    FieldValue,
    InjectionResult,
    ProposedAction,
)
from agent_orchestrator.policy.engine import decide, default_policy

POLICY = default_policy()
NOW = dt.datetime(2026, 11, 25, tzinfo=dt.UTC)
KEY = "0123456789abcdef"
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
EXAMPLES = 20_000 if os.environ.get("POLICY_SLOW") else 2_000
FLAGS = [
    "injection",
    "pdf_disagrees",
    "gstin_invalid",
    "fuzzy_match",
    "ambiguous_match",
    "low_confidence",
]
CHECKS = [
    "extraction_failed",
    "llm_error",
    "budget_stop",
    "tax_arithmetic_ok",
    "buyer_gstin_mismatch",
]


def oracle(c: Case, p: ProposedAction, ex, budget_stopped: bool, fallback: bool) -> bool:
    """When may an action happen with no person? (independent of the engine's code)"""
    clean = {"none", "rounding_within_tolerance", "invoice_no_format_difference"}
    right_action = (p.action == "none" and c.mismatch_type in clean) or (
        p.action == "chase_supplier_and_hold_itc" and c.mismatch_type == "missing_in_2b"
    )
    return (
        right_action
        and p.action in POLICY.allowed_actions[c.mismatch_type]
        and c.pdf_file is not None
        and bool(c.book_lines)
        and ex is not None
        and not ex.injection.flagged
        and not {"extraction_failed", "llm_error", "budget_stop"} & set(ex.checks)
        and ex.min_critical_confidence >= POLICY.defaults.min_extraction_confidence
        and c.itc_at_risk_paise <= POLICY.defaults.auto_hold_max_tax_paise
        and not {"injection", "pdf_disagrees", "gstin_invalid", "fuzzy_match", "ambiguous_match"}
        & set(c.flags)
        and c.status != "needs_human"
        and not budget_stopped
        and not fallback
    )


cases = st.builds(
    lambda mt, status, flags, risk, pdf, lines, mp: Case(
        case_key=KEY,
        business_id="B",
        period="2026-10",
        mismatch_type=mt,
        status=status,
        book_lines=lines,
        portal_index=None,
        pdf_file=pdf,
        match_pass=mp,
        match_score=0.0,
        itc_at_risk_paise=risk,
        facts_hash="f" * 16,
        flags=flags,
    ),
    st.sampled_from(MISMATCH_TYPES),
    st.sampled_from(CASE_STATUSES),
    st.lists(st.sampled_from(FLAGS), unique=True, max_size=3),
    st.one_of(st.integers(0, 10_000_000), st.sampled_from([4_999_999, 5_000_000, 5_000_001])),
    st.one_of(st.none(), st.just("invoices/a.pdf")),
    st.sampled_from([[], [1], [1, 2]]),
    st.sampled_from(["exact", "normalised", "no_year", "gstin_repair", "fuzzy", "none"]),
)

extractions = st.one_of(
    st.none(),
    st.builds(
        lambda conf, injected, checks: ExtractedInvoice(
            pdf_file="invoices/a.pdf",
            model_id="m",
            escalated=False,
            **dict.fromkeys(FIELDS, FieldValue(value="x", confidence=conf)),
            min_critical_confidence=conf,
            checks=checks,
            injection=InjectionResult(flagged=injected),
        ),
        st.one_of(st.floats(0, 1), st.sampled_from([0.8499, 0.85, 0.8501])),
        st.booleans(),
        st.lists(st.sampled_from(CHECKS), unique=True, max_size=2),
    ),
)

free_text = st.text(max_size=200)
refs = st.lists(st.text(max_size=30), max_size=4)

proposals = st.builds(
    lambda action, reason, ev, rr, who: ProposedAction(
        case_key=KEY, action=action, reason=reason, evidence_refs=ev, rule_refs=rr, proposer=who
    ),
    st.sampled_from(ACTION_KINDS),
    free_text,
    refs,
    refs,
    st.sampled_from(["rule", "llm"]),
)


@settings(max_examples=EXAMPLES, deadline=None)
@given(cases, proposals, extractions, st.booleans(), st.booleans())
def test_auto_allow_only_when_the_oracle_agrees(c, p, ex, budget, fallback):
    d = decide(c, p, extraction=ex, budget_stopped=budget, fallback=fallback, now=NOW)
    allowed = oracle(c, p, ex, budget, fallback)
    if d.outcome == "auto_allow":
        assert allowed, d.fired_rules
    # The other direction too: the engine is not simply refusing everything.
    if allowed:
        assert d.outcome == "auto_allow", d.fired_rules


@settings(max_examples=EXAMPLES, deadline=None)
@given(cases, proposals, extractions, free_text, refs, refs)
def test_free_text_never_changes_the_decision(c, p, ex, reason, ev, rr):
    other = p.model_copy(update={"reason": reason, "evidence_refs": ev, "rule_refs": rr})
    a = decide(c, p, extraction=ex, now=NOW)
    b = decide(c, other, extraction=ex, now=NOW)
    assert (a.outcome, a.fired_rules) == (b.outcome, b.fired_rules)


@settings(max_examples=500, deadline=None)
@given(st.text(min_size=1, max_size=40).filter(lambda s: s not in ACTION_KINDS))
def test_unknown_action_names_never_reach_the_engine(action):
    with pytest.raises(ValidationError):
        ProposedAction(case_key=KEY, action=action, reason="", proposer="llm")


@settings(max_examples=EXAMPLES, deadline=None)
@given(cases, proposals, extractions)
def test_outcome_is_the_strictest_fired_rule(c, p, ex):
    d = decide(c, p, extraction=ex, now=NOW)
    rank = {"auto_allow": 0, "require_approval": 1, "block": 2}
    by_id = {r.id: r.outcome for r in POLICY.rules}
    fired = [by_id[r] for r in d.fired_rules if r in by_id]
    if fired:
        assert rank[d.outcome] == max(rank[o] for o in fired)
    else:
        assert d.outcome == "require_approval"


@pytest.mark.slow
def test_slow_run_uses_twenty_thousand_examples():
    """Run with: POLICY_SLOW=1 uv run pytest -m slow tests/test_policy_properties.py"""
    assert EXAMPLES == 20_000 or not os.environ.get("POLICY_SLOW")
