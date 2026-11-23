import datetime as dt
import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, ValidationError

from agent_orchestrator import models
from agent_orchestrator.models import (
    ApprovalRecord,
    BookRecord,
    BudgetSnapshot,
    Case,
    Decision,
    DraftEmail,
    ExtractedInvoice,
    FieldValue,
    PortalRecord,
    ProposedAction,
    RunState,
    fields_with_reducers,
    fields_without_reducers,
)

NOW = dt.datetime(2026, 11, 23, 12, 0, tzinfo=dt.UTC)

BOOK = dict(
    line_no=1,
    row_hash="a1b2c3d4e5f6",
    invoice_no="SAM/2026/0100",
    inv_key="SAM|2026|100",
    inv_key_noyear="SAM|100",
    date=dt.date(2026, 10, 10),
    supplier_name="Sample Packaging Co",
    supplier_gstin="36TSDKP2716L1ZA",
    gstin_valid=True,
    taxable_paise=4730000,
    rate=18,
    igst_paise=0,
    cgst_paise=425700,
    sgst_paise=425700,
    pdf_file="invoices/SAM_2026_0100.pdf",
)

CASE = dict(
    case_key="0123456789abcdef",
    business_id="B01",
    period="2026-10",
    mismatch_type="none",
    status="matched",
    book_lines=[1],
    portal_index=0,
    match_pass="exact",
    match_score=100.0,
    facts_hash="f" * 16,
)


def _model_classes():
    return [
        obj
        for name in models.__all__
        if isinstance(obj := getattr(models, name), type) and issubclass(obj, BaseModel)
    ]


def test_every_model_forbids_extra_fields_and_is_frozen():
    classes = _model_classes()
    assert len(classes) == 15
    for cls in classes:
        assert cls.model_config.get("extra") == "forbid", cls.__name__
        assert cls.model_config.get("frozen") is True, cls.__name__


def test_misspelt_field_fails_loudly():
    with pytest.raises(ValidationError, match="extra"):
        BookRecord(**BOOK, supplier_gstn="typo")


def test_records_are_immutable():
    rec = BookRecord(**BOOK)
    with pytest.raises(ValidationError):
        rec.taxable_paise = 1


def test_money_fields_reject_floats():
    with pytest.raises(ValidationError):
        BookRecord(**{**BOOK, "taxable_paise": 4730000.0})
    with pytest.raises(ValidationError):
        BookRecord(**{**BOOK, "cgst_paise": -1})


def test_records_round_trip_through_json():
    rec = BookRecord(**BOOK)
    assert BookRecord.model_validate_json(rec.model_dump_json()) == rec
    case = Case(**CASE, diffs={"taxable": -100})
    assert Case.model_validate_json(case.model_dump_json()) == case


def test_portal_period_format():
    base = {k: v for k, v in BOOK.items() if k not in ("line_no", "row_hash", "gstin_valid")}
    base.pop("pdf_file")
    rec = PortalRecord(**base, index=0, reported_in_period="2026-10")
    assert rec.reported_in_period == "2026-10"
    with pytest.raises(ValidationError):
        PortalRecord(**base, index=0, reported_in_period="2026-13")


def test_vocabularies_are_exactly_the_spec_lists():
    assert models.MISMATCH_TYPES == (
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
    )
    assert models.CASE_STATUSES == (
        "matched",
        "matched_after_normalisation",
        "mismatch",
        "unmatched",
        "duplicate",
        "needs_human",
    )
    assert models.ACTION_KINDS == (
        "none",
        "chase_supplier_and_hold_itc",
        "verify_with_supplier",
        "correct_books",
        "escalate_human",
    )


def test_unknown_action_or_mismatch_is_rejected():
    with pytest.raises(ValidationError):
        Case(**{**CASE, "mismatch_type": "something_new"})
    with pytest.raises(ValidationError):
        ProposedAction(case_key="k", action="pay_supplier", reason="r", proposer="llm")


def test_case_key_must_be_16_hex():
    with pytest.raises(ValidationError):
        Case(**{**CASE, "case_key": "NOT-A-KEY"})


def test_proposal_reason_is_capped():
    ProposedAction(case_key="k", action="none", reason="x" * 600, proposer="rule")
    with pytest.raises(ValidationError):
        ProposedAction(case_key="k", action="none", reason="x" * 601, proposer="rule")


def test_decision_needs_a_rule_and_an_aware_time():
    ok = dict(case_key="k", outcome="auto_allow", action="none", policy_version="v1")
    Decision(**ok, fired_rules=["GST-ITC-OK-01"], decided_at=NOW)
    with pytest.raises(ValidationError):
        Decision(**ok, fired_rules=[], decided_at=NOW)
    with pytest.raises(ValidationError):
        Decision(**ok, fired_rules=["GST-ITC-OK-01"], decided_at=NOW.replace(tzinfo=None))


def _approval(**kw):
    base = dict(
        approval_id="a1",
        idempotency_key="i1",
        case_key="k",
        run_id="r1",
        proposed="verify_with_supplier",
        allowed_options=["verify_with_supplier", "escalate_human"],
    )
    return ApprovalRecord(**{**base, **kw})


def test_approval_states():
    assert _approval().status == "pending"
    done = dict(decided_by="Goutham", decided_at=NOW)
    _approval(status="approved", final_action="verify_with_supplier", **done)
    _approval(status="edited", final_action="escalate_human", reason="needs a call", **done)
    _approval(status="rejected", final_action="escalate_human", reason="wrong match", **done)


@pytest.mark.parametrize(
    "kw",
    [
        dict(final_action="verify_with_supplier"),  # pending with a final action
        dict(status="approved"),  # approved without action/approver/time
        dict(status="approved", final_action="escalate_human"),  # approved but changed
        dict(status="edited", final_action="verify_with_supplier"),  # edited but unchanged
        dict(status="edited", final_action="escalate_human"),  # edited without reason
        dict(status="edited", final_action="correct_books", reason="r"),  # not allowed
        dict(status="rejected", final_action="escalate_human"),  # rejected without reason
        dict(status="rejected", final_action="verify_with_supplier", reason="r"),
        dict(proposed="correct_books"),  # proposed not in allowed options
    ],
)
def test_inconsistent_approvals_are_rejected(kw):
    if kw.get("status") not in (None, "pending"):
        kw = {"decided_by": "Goutham", "decided_at": NOW, **kw}
    with pytest.raises(ValidationError):
        _approval(**kw)


def test_draft_email_language_is_limited():
    ok = dict(idempotency_key="i", case_key="k", to_supplier_gstin="g", subject="s", body="b")
    DraftEmail(**ok, language="te", created_at=NOW)
    with pytest.raises(ValidationError):
        DraftEmail(**ok, language="fr", created_at=NOW)


def test_extraction_confidence_bounds_and_defaults():
    fv = FieldValue(value="x", confidence=0.9)
    fields = dict.fromkeys(
        [
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
        ],
        fv,
    )
    ex = ExtractedInvoice(
        pdf_file="p.pdf",
        model_id="m",
        escalated=False,
        min_critical_confidence=0.9,
        **fields,
    )
    assert ex.injection.flagged is False and ex.other_text == ""
    with pytest.raises(ValidationError):
        FieldValue(value="x", confidence=1.2)


def test_budget_snapshot_defaults():
    b = BudgetSnapshot(started_at=NOW)
    assert (b.usd_spent, b.tokens, b.stop_new_llm_work) == (0.0, 0, False)


# --- graph state reducers -------------------------------------------------------------


def test_branch_written_fields_have_add_reducers():
    reducers = fields_with_reducers()
    assert set(reducers) == {
        "extractions",
        "research",
        "proposals",
        "decisions",
        "drafts",
        "errors",
    }
    assert all(r is operator.add for r in reducers.values())


def test_fields_without_reducers_lists_the_rest():
    assert fields_without_reducers() == sorted(
        [
            "run_id",
            "business_id",
            "period",
            "input_hash",
            "policy_version",
            "books",
            "portal",
            "dq_issues",
            "cases",
            "approvals_resolved",
            "budget",
        ]
    )
    assert set(fields_without_reducers()) | set(fields_with_reducers()) == set(
        RunState.__annotations__
    )


def test_helper_spots_a_missing_reducer_in_any_state_type():
    class Bad(TypedDict):
        ok: Annotated[list[int], operator.add]
        lost: list[int]

    assert fields_without_reducers(Bad) == ["lost"]


def _fan_out_graph(state_type):
    """START -> one branch per item (parallel, via Send) -> END. Each branch writes one item."""

    def fan_out(state):
        return [Send("branch", {"item": i}) for i in range(5)]

    def branch(payload):
        return {"out": [payload["item"]]}

    g = StateGraph(state_type)
    g.add_node("branch", branch)
    g.add_conditional_edges(START, fan_out, ["branch"])
    g.add_edge("branch", END)
    return g.compile()


def test_reducer_keeps_every_parallel_write():
    class WithReducer(TypedDict):
        out: Annotated[list[int], operator.add]

    result = _fan_out_graph(WithReducer).invoke({"out": []})
    assert sorted(result["out"]) == [0, 1, 2, 3, 4]


def test_without_reducer_parallel_writes_are_refused():
    class NoReducer(TypedDict):
        out: list[int]

    # LangGraph refuses several writes to a plain field in one step. This is the lost-update
    # bug the reducers prevent, and why fields_without_reducers exists.
    with pytest.raises(Exception, match="(?i)multiple|concurrent|one value"):
        _fan_out_graph(NoReducer).invoke({"out": []})
