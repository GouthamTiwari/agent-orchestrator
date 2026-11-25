import datetime as dt
import json
from pathlib import Path

import pytest
import yaml

from agent_orchestrator.agents.intake import run_intake_dir
from agent_orchestrator.agents.matcher import match
from agent_orchestrator.eval import generate as gen
from agent_orchestrator.models import (
    Case,
    ExtractedInvoice,
    FieldValue,
    InjectionResult,
    ProposedAction,
)
from agent_orchestrator.policy import engine
from agent_orchestrator.policy.engine import NO_RULE_FIRED, PolicyError, decide, load_policy

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed" / "seed_data"
POLICY_TEXT = (
    ROOT / "src" / "agent_orchestrator" / "policy" / "rules" / "policy_v1.yaml"
).read_text(encoding="utf-8")
NOW = dt.datetime(2026, 11, 25, tzinfo=dt.UTC)
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


def extraction(conf=0.95, injected=False, checks=()):
    fv = FieldValue(value="x", confidence=conf)
    return ExtractedInvoice(
        pdf_file="p.pdf",
        model_id="fake",
        escalated=False,
        **dict.fromkeys(FIELDS, fv),
        min_critical_confidence=conf,
        checks=list(checks),
        injection=InjectionResult(flagged=injected),
    )


def case(
    mismatch="none",
    status="matched",
    flags=(),
    at_risk=0,
    pdf="p.pdf",
    lines=(1,),
    match_pass="exact",
):
    return Case(
        case_key="0123456789abcdef",
        business_id="B",
        period="2026-10",
        mismatch_type=mismatch,
        status=status,
        book_lines=list(lines),
        portal_index=0,
        pdf_file=pdf,
        match_pass=match_pass,
        match_score=100.0,
        itc_at_risk_paise=at_risk,
        facts_hash="f" * 16,
        flags=list(flags),
    )


def proposal(action="none", reason="", proposer="rule"):
    return ProposedAction(
        case_key="0123456789abcdef", action=action, reason=reason, proposer=proposer
    )


def outcome(c, p, **kw):
    kw.setdefault("extraction", extraction())
    return decide(c, p, now=NOW, **kw)


# --- loading ----------------------------------------------------------------------------------


def _mutated(tmp_path, fn):
    data = yaml.safe_load(POLICY_TEXT)
    fn(data)
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_policy_loads_with_every_legal_rule_cited():
    p = load_policy()
    assert p.version == "1.0.0" and len(p.rules) == 19
    for r in p.rules:
        if r.kind == "legal":
            assert r.sources and all(
                s.url.startswith("https://") and s.checked_on for s in r.sources
            )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["rules"][0].pop("sources"), "needs at least one source"),
        (lambda d: d["rules"][0]["sources"][0].pop("url"), "url"),
        (lambda d: d["rules"][0]["sources"][0].pop("checked_on"), "checked_on"),
        (lambda d: d["rules"][9].pop("design_ref"), "design_ref"),
        (
            lambda d: d["rules"][2].update(
                when={"fact": "__import__('os')", "op": "eq", "value": 1}
            ),
            "unknown fact",
        ),
        (
            lambda d: d["rules"][2].update(
                when={"fact": "case.injection", "op": "eq", "value": "yes"}
            ),
            "true/false",
        ),
        (
            lambda d: d["rules"][2].update(
                when={"fact": "case.mismatch_type", "op": "eq", "value": "made_up"}
            ),
            "unknown mismatch",
        ),
        (
            lambda d: d["rules"][2].update(when={"fact": "case.status", "op": "lt", "value": 3}),
            "numeric",
        ),
        (
            lambda d: d["rules"][2].update(
                when={"fact": "case.status", "op": "regex", "value": ".*"}
            ),
            "op",
        ),
        (lambda d: d["defaults"].update(outcome_if_no_rule_fires="auto_allow"), "fail closed"),
        (lambda d: d["allowed_actions"].pop("missing_in_2b"), "every mismatch type"),
        (lambda d: d["allowed_actions"].update(none=["pay_supplier"]), "unknown"),
        (lambda d: d["rules"].append(dict(d["rules"][0])), "duplicate rule ids"),
        (lambda d: d["rules"][0].update(unless=["NOPE-01"]), "unknown rules"),
    ],
)
def test_broken_policy_files_refuse_to_load(tmp_path, mutate, message):
    with pytest.raises(PolicyError, match=message):
        load_policy(_mutated(tmp_path, mutate))


def test_amount_rule_matches_the_default_ceiling():
    p = load_policy()
    rule = next(r for r in p.rules if r.id == "BUS-AMT-01")
    assert rule.when.value == p.defaults.auto_hold_max_tax_paise == 5_000_000


# --- the seed month ---------------------------------------------------------------------------


def test_seed_month_decisions():
    r = run_intake_dir(SEED, "2026-10")
    cases = match(r.books, r.portal, "SEED", "2026-10")
    books = {b.line_no: b for b in r.books}
    portal = {p.index: p for p in r.portal}
    labels = [json.loads(x) for x in (SEED / "labels.jsonl").read_text().splitlines()]
    want = {
        **{f"case{i:02d}": "auto_allow" for i in range(1, 11)},
        **{f"case{i:02d}": "require_approval" for i in range(11, 18)},
        "case18": "block",
    }
    for lab in labels:
        found = [
            c
            for c in cases
            if any(books[ln].invoice_no == lab["invoice_no"] for ln in c.book_lines)
            or (
                c.portal_index is not None
                and portal[c.portal_index].invoice_no == lab["invoice_no"]
            )
        ]
        dup = lab["case_type"] == "duplicate_in_books"
        c = next(x for x in found if (x.status == "duplicate") == dup)
        ex = extraction(injected=lab["invoice_no"] == "IMA/2026/0117") if c.pdf_file else None
        p = ProposedAction(
            case_key=c.case_key, action=lab["expected_action"], reason="", proposer="rule"
        )
        d = decide(c, p, extraction=ex, now=NOW)
        assert d.outcome == want[lab["id"]], (lab["id"], d.fired_rules)
    assert d.fired_rules == ["GST-ITC-VAL-01", "SAFE-INJ-01"]  # case18: block wins


# --- the generated dataset --------------------------------------------------------------------


def test_policy_agrees_with_every_generated_label(tmp_path):
    """With the expected action proposed and a clean read of each PDF, the engine's outcome
    is auto_allow exactly when the label says auto_ok, and block exactly for injections."""
    gen.generate(gen.load_config(), out_dir=tmp_path, labels_dir=None, render=False)
    checked = 0
    for md in sorted(tmp_path.glob("B*/2026-*")):
        labels = [json.loads(x) for x in (md / "labels.jsonl").read_text("utf-8").splitlines()]
        r = run_intake_dir(md, md.name)
        cases = match(r.books, r.portal, md.parent.name, md.name)
        by_line = {c.book_lines[0]: c for c in cases if c.book_lines}
        by_portal = {c.portal_index: c for c in cases if not c.book_lines}
        for lab in labels:
            c = (
                by_line[lab["book_lines"][0]]
                if lab["book_lines"]
                else by_portal[lab["portal_index"]]
            )
            ex = extraction(injected=lab["injection"]) if c.pdf_file else None
            p = ProposedAction(
                case_key=c.case_key, action=lab["expected_action"], reason="", proposer="rule"
            )
            d = decide(c, p, extraction=ex, now=NOW)
            assert (d.outcome == "auto_allow") == lab["auto_ok"], (lab["id"], d.fired_rules)
            assert (d.outcome == "block") == lab["injection"], lab["id"]
            checked += 1
    assert checked == 593


# --- precedence and individual rules ----------------------------------------------------------


def test_block_beats_approval_beats_auto():
    d = outcome(case(), proposal(), extraction=extraction(conf=0.5, injected=True))
    assert d.outcome == "block"
    assert {"SAFE-INJ-01", "SAFE-CONF-01", "GST-ITC-OK-01"} <= set(d.fired_rules)
    d = outcome(case(), proposal(), extraction=extraction(conf=0.5))
    assert d.outcome == "require_approval" and "GST-ITC-OK-01" in d.fired_rules


def test_no_rule_fired_fails_closed():
    # escalate_human on a clean match is allowed by nobody: SAFE-ALLOW-01 blocks it, so build a
    # case where no rule applies: an allowed escalation on a missing invoice.
    d = outcome(case("missing_in_2b", "unmatched", at_risk=1000), proposal("escalate_human"))
    assert d.outcome == "require_approval" and d.fired_rules == [NO_RULE_FIRED]


@pytest.mark.parametrize(
    ("kw", "rule", "result"),
    [
        (dict(c=case(), p=proposal("correct_books")), "SAFE-ALLOW-01", "block"),
        (dict(c=case(flags=["pdf_disagrees"]), p=proposal()), "SAFE-PDF-01", "require_approval"),
        (
            dict(
                c=case(
                    flags=["fuzzy_match"],
                    mismatch="invoice_no_format_difference",
                    status="matched_after_normalisation",
                    match_pass="fuzzy",
                ),
                p=proposal(),
            ),
            "SAFE-MATCH-01",
            "require_approval",
        ),
        (dict(c=case(flags=["gstin_invalid"]), p=proposal()), "DQ-GSTIN-01", "require_approval"),
        (
            dict(c=case(pdf=None), p=proposal(), extraction=None),
            "SAFE-NOPDF-01",
            "require_approval",
        ),
        (dict(c=case(), p=proposal(), extraction=None), "SAFE-CONF-01", "require_approval"),
        (
            dict(c=case(), p=proposal(), extraction=extraction(checks=["extraction_failed"])),
            "GST-ITC-DOC-01",
            "require_approval",
        ),
        (dict(c=case(), p=proposal(), budget_stopped=True), "SAFE-BUDGET-01", "require_approval"),
        (
            dict(
                c=case("missing_in_2b", "unmatched", at_risk=5_000_001),
                p=proposal("chase_supplier_and_hold_itc"),
            ),
            "BUS-AMT-01",
            "require_approval",
        ),
        (
            dict(
                c=case("taxable_value_mismatch", "mismatch"),
                p=proposal("escalate_human"),
                fallback=True,
            ),
            "SAFE-PLAN-01",
            "require_approval",
        ),
    ],
)
def test_each_safety_rule(kw, rule, result):
    c, p = kw.pop("c"), kw.pop("p")
    d = outcome(c, p, **kw)
    assert rule in d.fired_rules and d.outcome == result


def test_auto_hold_ceiling_is_inclusive():
    c = case("missing_in_2b", "unmatched", at_risk=5_000_000)
    assert outcome(c, proposal("chase_supplier_and_hold_itc")).outcome == "auto_allow"


def test_decision_records_rules_version_and_is_deterministic():
    d1 = outcome(case(), proposal())
    d2 = outcome(case(), proposal())
    assert d1 == d2 and d1.policy_version == "policy_v1@1.0.0" and d1.action == "none"


def test_proposal_for_another_case_is_refused():
    other = ProposedAction(case_key="fedcba9876543210", action="none", reason="", proposer="rule")
    with pytest.raises(ValueError, match="different case"):
        outcome(case(), other)


def test_facts_cover_the_schema_exactly():
    p = load_policy()
    facts = engine.build_facts(case(), proposal(), p, extraction())
    from agent_orchestrator.policy.schema import FACTS

    assert set(facts) == set(FACTS)
    assert "proposal.reason" not in facts  # free text is never a fact
