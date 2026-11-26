import csv
import datetime as dt
import json
from decimal import Decimal
from pathlib import Path

import pytest

from agent_orchestrator.agents.communicator import draft_email
from agent_orchestrator.agents.intake import run_intake_dir
from agent_orchestrator.agents.matcher import match
from agent_orchestrator.agents.reporter import (
    build_report,
    check_summary,
    record_in_store,
    render_markdown,
    summarise,
    summary_input,
    write_report,
)
from agent_orchestrator.interfaces.stubs import RegexGuard
from agent_orchestrator.llm.budget import RunBudget
from agent_orchestrator.llm.client import LLMClient
from agent_orchestrator.llm.fake import FakeBackend
from agent_orchestrator.models import (
    ApprovalRecord,
    BudgetSnapshot,
    ExtractedInvoice,
    FieldValue,
    InjectionResult,
    ProposedAction,
)
from agent_orchestrator.policy.engine import decide, default_policy
from agent_orchestrator.store.repo import Repo
from agent_orchestrator.tracing import Tracer

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed" / "seed_data"
NOW = dt.datetime(2026, 11, 26, tzinfo=dt.UTC)
NAMES = (
    "supplier_name supplier_gstin buyer_gstin invoice_no invoice_date taxable_value rate igst "
    "cgst sgst total"
).split()

# The seed month by hand, from purchase_register.csv and labels.jsonl (GST = IGST+CGST+SGST):
#   safe (final action none): lines 1-8 and line 15 (the first FIC/2026/0114)
#     8,514.00 + 2,467.50 + 5,013.00 + 2,259.00 + 4,295.00 + 3,017.50 + 7,915.00 + 2,592.00
#     + 25,911.00 = 61,984.00
#   held (chase and hold, missing in 2B): SPE/0108 40,032.00 + EXA/0109 8,262.50 = 48,294.50
#   at risk (waiting for a person or blocked): SAM/0110 9,290.00 + DEM/0111 265.00
#     + TES/0112 9,387.00 + PLA/0113 7,622.50 + FIC/0114 duplicate 25,911.00
#     + MOC/0115 9,042.50 + IMA/0117 (blocked) 34,533.00 = 96,051.00
#   DUM/2026/0116 is in GSTR-2B only: nothing claimed, no bucket.
HAND = {
    "total": 19,
    "by_status": {"duplicate": 1, "matched": 8, "matched_after_normalisation": 1,
                  "mismatch": 6, "unmatched": 3},
    "by_outcome": {"auto_allow": 11, "block": 1, "require_approval": 7},
    "by_final_action": {"chase_supplier_and_hold_itc": 2, "escalate_human": 1, "none": 9,
                        "pending": 7},
    "safe": 6198400, "held": 4829450, "at_risk": 9605100,
    "safe_inr": "₹61,984.00", "held_inr": "₹48,294.50", "at_risk_inr": "₹96,051.00",
    "pending": ["SAM/2026/0110", "DEM/2026/0111", "TES/2026/0112", "PLA/2026/0113",
                "FIC/2026/0114", "MOC/2026/0115", "DUM/2026/0116"],
    "auto": ["SAM/2026/0100", "DEM/2026/0101", "TES/2026/0102", "PLA/2026/0103",
             "FIC/2026/0104", "MOC/2026/0105", "DUM/2026/0106", "IMA/2026/0107",
             "SPE/2026/0108", "EXA/2026/0109", "FIC/2026/0114"],
}  # fmt: skip


def read(conf=0.95, injected=False):
    fv = FieldValue(value="x", confidence=conf)
    return ExtractedInvoice(
        pdf_file="p.pdf", model_id="fake", escalated=False, **dict.fromkeys(NAMES, fv),
        min_critical_confidence=conf, injection=InjectionResult(flagged=injected),
    )  # fmt: skip


@pytest.fixture(scope="module")
def month():
    """The seed month decided with the labelled actions, approvals requested, drafts made."""
    r = run_intake_dir(SEED, "2026-10")
    cases = match(r.books, r.portal, "SEED", "2026-10")
    books = {b.line_no: b for b in r.books}
    portal = {p.index: p for p in r.portal}
    labels = {
        json.loads(x)["invoice_no"]: json.loads(x)
        for x in (SEED / "labels.jsonl").read_text("utf-8").splitlines()
    }
    policy = default_policy()
    decisions, approvals, drafts = [], [], []
    for c in cases:
        no = (
            books[c.book_lines[0]].invoice_no if c.book_lines else portal[c.portal_index].invoice_no
        )
        action = labels[no]["expected_action"] if c.status != "matched" else "none"
        p = ProposedAction(case_key=c.case_key, action=action, reason="", proposer="rule")
        ex = read(injected=no == "IMA/2026/0117") if c.pdf_file else None
        d = decide(c, p, extraction=ex, now=NOW)
        decisions.append(d)
        if d.outcome == "require_approval":
            approvals.append(
                ApprovalRecord(
                    approval_id=f"a{len(approvals)}",
                    idempotency_key=f"k{len(approvals)}",
                    case_key=c.case_key,
                    run_id="r1",
                    proposed=action,
                    allowed_options=policy.allowed_actions[c.mismatch_type],
                )  # fmt: skip
            )
        out = draft_email(c, decision=d, approval=None, books=books, portal=portal,
                          guard=RegexGuard(), now=NOW)  # fmt: skip
        if out.draft:
            drafts.append(out.draft)
    return r, cases, decisions, approvals, drafts


def report_for(month, approvals=None, **kw):
    r, cases, decisions, pending, drafts = month
    return build_report(
        run_id="r1", business_id="SEED", period="2026-10", cases=cases, decisions=decisions,
        approvals=pending if approvals is None else approvals, books=r.books, portal=r.portal,
        dq_issues=r.dq_issues, drafts=drafts, **kw,
    )  # fmt: skip


# --- the hand-computed seed month -----------------------------------------------------------------


def test_report_totals_equal_the_hand_computed_seed_month(month):
    rep = report_for(month)
    c = rep["cases"]
    assert c["total"] == HAND["total"]
    assert c["by_status"] == HAND["by_status"]
    assert c["by_outcome"] == HAND["by_outcome"]
    assert c["by_final_action"] == HAND["by_final_action"]
    itc = rep["itc"]
    assert (itc["safe"]["paise"], itc["held"]["paise"], itc["at_risk"]["paise"]) == (
        HAND["safe"], HAND["held"], HAND["at_risk"],
    )  # fmt: skip
    assert (itc["safe"]["inr"], itc["held"]["inr"], itc["at_risk"]["inr"]) == (
        HAND["safe_inr"], HAND["held_inr"], HAND["at_risk_inr"],
    )  # fmt: skip
    assert [x["invoice_no"] for x in rep["approvals"]["pending_list"]] == HAND["pending"]
    assert rep["approvals"]["pending"] == 7 and rep["approvals"]["not_yet_requested"] == 0
    assert [x["invoice_no"] for x in rep["auto_actioned"]] == HAND["auto"]
    assert [b["invoice_no"] for b in rep["blocked"]] == ["IMA/2026/0117"]
    assert rep["blocked"][0]["fired_rules"] == ["GST-ITC-VAL-01", "SAFE-INJ-01"]
    assert "injection" in rep["blocked"][0]["reasons"][1].lower()
    assert [i["row_ref"] for i in rep["data_quality"]] == [
        "books:line:13", "books:line:16", "2b:index:9",
    ]  # fmt: skip
    assert rep["drafts"] == {"count": 2, "by_language": {"en": 2}}


def test_the_three_buckets_add_up_to_all_gst_in_the_books():
    """Independent of the code: read the CSV directly."""
    total = Decimal(0)
    with open(SEED / "purchase_register.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += Decimal(row["igst"]) + Decimal(row["cgst"]) + Decimal(row["sgst"])
    assert int(total * 100) == HAND["safe"] + HAND["held"] + HAND["at_risk"]


def test_every_number_in_the_report_traces_to_a_case(month):
    rep = report_for(month)
    rows = rep["cases_detail"]
    for name in ("safe", "held", "at_risk"):
        assert rep["itc"][name]["paise"] == sum(
            r["itc_paise"] for r in rows if r["itc_bucket"] == name
        )
    assert sum(rep["cases"]["by_final_action"].values()) == len(rows) == rep["cases"]["total"]


def test_decided_approvals_move_the_final_actions(month):
    r, cases, decisions, pending, drafts = month
    by_no = {x["invoice_no"]: x["case_key"] for x in report_for(month)["cases_detail"]}

    def decided(a, status, final, reason="checked"):
        return a.model_copy(update={
            "status": status, "final_action": final, "decided_by": "goutham", "decided_at": NOW,
            "reason": reason if status != "approved" else None,
        })  # fmt: skip

    key = {a.case_key: a for a in pending}
    changed = [
        decided(key[by_no["SAM/2026/0110"]], "approved", "verify_with_supplier"),
        decided(key[by_no["TES/2026/0112"]], "edited", "escalate_human"),
        decided(key[by_no["DEM/2026/0111"]], "rejected", "escalate_human"),
    ]
    done = {a.case_key for a in changed}
    rep = report_for(month, approvals=[*changed, *(a for a in pending if a.case_key not in done)])
    ap = rep["approvals"]
    assert (ap["pending"], ap["approved"], ap["edited"], ap["rejected"]) == (4, 1, 1, 1)
    assert rep["cases"]["by_final_action"] == {
        "chase_supplier_and_hold_itc": 2, "escalate_human": 3, "none": 9, "pending": 4,
        "verify_with_supplier": 1,
    }  # fmt: skip
    assert rep["itc"]["at_risk"]["paise"] == HAND["at_risk"]  # nothing became safe


def test_the_report_is_deterministic(month):
    assert report_for(month) == report_for(month)


def test_budget_and_cost_are_reported(month):
    snap = BudgetSnapshot(usd_spent=0.0123, tokens=4567, started_at=NOW, stop_new_llm_work=True,
                          reason="usd limit 90% reached")  # fmt: skip
    cost = {"planner": {"spans": 3, "usd": 0.004}, "extraction": {"spans": 17, "usd": 0.008}}
    rep = report_for(month, budget=snap, cost_by_agent=cost)
    assert rep["budget"] == {"usd_spent": 0.0123, "tokens": 4567, "stopped": True,
                             "stop_reason": "usd limit 90% reached"}  # fmt: skip
    assert rep["cost"]["usd_total"] == 0.012 and list(rep["cost"]["by_agent"]) == [
        "extraction", "planner",
    ]  # fmt: skip
    md = render_markdown(rep)
    assert "stopped: usd limit 90% reached" in md and "| planner | 3 | 0.0040 |" in md


def test_the_markdown_shows_the_totals(month):
    md = render_markdown(report_for(month))
    for text in (HAND["safe_inr"], HAND["held_inr"], HAND["at_risk_inr"], "## Blocked (1)",
                 "**IMA/2026/0117**", "SAFE-INJ-01", "Pending 7", "(none sent)",
                 "## Data quality (3)", "## Done automatically (11)"):  # fmt: skip
        assert text in md, text


# --- the summary ----------------------------------------------------------------------------------

GOOD = (
    "In 2026-10 there were 19 cases, and 11 were closed automatically. Credit of ₹61,984.00 is "
    "safe to claim, ₹48,294.50 is held and ₹96,051.00 is at risk. 7 approvals are pending and "
    "1 case was blocked."
)


def summary_client(text):
    backend = FakeBackend([json.dumps({"summary": text})])
    return LLMClient(backend, mode="fake"), backend


def test_a_grounded_summary_is_kept(month):
    rep = report_for(month)
    c, backend = summary_client(GOOD)
    out = summarise(rep, client=c, guard=RegexGuard())
    assert out.summary == GOOD and out.note == ""
    prompt = backend.calls[0][1][0].parts[0]
    assert "₹61,984.00" in prompt and "case_key" not in prompt and "IMA/2026" not in prompt


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (GOOD.replace("₹48,294.50", "₹48,300.00"), "a number that is not in the facts"),
        (GOOD + " That is 34% of the credit.", "a number that is not in the facts"),
        ("Nineteen cases. Most closed.", "3 to 5 sentences"),
        (GOOD + " A. B. C.", "3 to 5 sentences"),
    ],
)
def test_a_summary_that_fails_the_number_check_is_left_out(month, text, problem):
    c, _ = summary_client(text)
    out = summarise(report_for(month), client=c, guard=RegexGuard())
    assert out.summary is None and out.note == "number_check_failed"
    assert problem in " ".join(out.problems)


def test_summary_failures_never_stop_the_report(month):
    rep = report_for(month)
    stopped = RunBudget(usd=1.0)
    stopped.record(0.95, 0)
    c = LLMClient(FakeBackend(["{}"]), mode="fake", run_budget=stopped)
    assert summarise(rep, client=c, guard=RegexGuard()).note == "budget_stop"
    c = LLMClient(FakeBackend([RuntimeError("down")]), mode="fake")
    assert summarise(rep, client=c, guard=RegexGuard()).note == "llm_error"
    c = LLMClient(FakeBackend(["not json"]), mode="fake")
    assert summarise(rep, client=c, guard=RegexGuard()).note == "schema_failed"


def test_the_summary_input_has_totals_only(month):
    totals = summary_input(report_for(month))
    assert set(totals) == {
        "period", "cases", "itc", "approvals", "auto_actioned", "blocked",
        "data_quality_issues", "drafts", "budget_stopped",
    }  # fmt: skip
    assert check_summary(GOOD, totals) == []


# --- files and the runs row -----------------------------------------------------------------------


def test_write_report_files_and_update_the_runs_row(month, tmp_path):
    rep = report_for(month, budget=BudgetSnapshot(usd_spent=0.01, tokens=99, started_at=NOW))
    c, _ = summary_client(GOOD)
    summary = summarise(rep, client=c, guard=RegexGuard())
    tracer = Tracer("r1")
    with tracer.span("reporter", "report"):
        pass
    paths = write_report(tmp_path / "runs" / "r1", rep, summary=summary,
                         trace=tracer.to_dict(), generated_at=NOW)  # fmt: skip
    data = json.loads(paths["json"].read_text("utf-8"))
    assert data["summary"] == GOOD and data["generated_at"] == NOW.isoformat()
    assert data["itc"]["safe"]["paise"] == HAND["safe"]
    assert GOOD in paths["md"].read_text("utf-8")
    assert json.loads(paths["trace"].read_text("utf-8"))["by_agent"]["reporter"]["spans"] == 1

    repo = Repo.open(tmp_path / "s.sqlite")
    repo.create_run("r1", "SEED", "2026-10", "h", "p")
    record_in_store(repo, rep, status="paused")
    row = repo.get_run("r1")
    assert (row["status"], row["usd_spent"], row["tokens"]) == ("paused", 0.01, 99)
    repo.close()


def test_runs_output_is_gitignored():
    assert "runs/" in (ROOT / ".gitignore").read_text("utf-8").splitlines()
