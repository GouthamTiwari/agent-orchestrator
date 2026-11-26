import datetime as dt
import json
import re
from pathlib import Path

import pytest

from agent_orchestrator.agents import communicator as cm
from agent_orchestrator.agents.communicator import (
    DRAFT_ACTIONS,
    LANGUAGES,
    EmailAnswer,
    check_draft,
    draft_email,
    eligible,
    email_facts,
    final_action,
    load_supplier_languages,
    template_draft,
    ungrounded,
)
from agent_orchestrator.agents.intake import run_intake_dir
from agent_orchestrator.agents.matcher import match
from agent_orchestrator.interfaces.stubs import RegexGuard
from agent_orchestrator.llm.budget import RunBudget
from agent_orchestrator.llm.client import LLMClient
from agent_orchestrator.llm.fake import FakeBackend
from agent_orchestrator.llm.recording import Recorder, RecordingMissing
from agent_orchestrator.models import ApprovalRecord, Decision
from agent_orchestrator.store.idempotency import outbox_key
from agent_orchestrator.store.repo import Repo

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed" / "seed_data"
NOW = dt.datetime(2026, 11, 26, tzinfo=dt.UTC)
CHASE, VERIFY = DRAFT_ACTIONS


@pytest.fixture(scope="module")
def seed():
    r = run_intake_dir(SEED, "2026-10")
    cases = match(r.books, r.portal, "SEED", "2026-10")
    books = {b.line_no: b for b in r.books}
    portal = {p.index: p for p in r.portal}

    def by_invoice(no, status=None):
        return next(
            c
            for c in cases
            if (c.book_lines and books[c.book_lines[0]].invoice_no == no)
            or (c.portal_index is not None and portal[c.portal_index].invoice_no == no)
            if status is None or c.status == status
        )

    return cases, books, portal, by_invoice


def decision(case, outcome="auto_allow", action=CHASE):
    return Decision(
        case_key=case.case_key, outcome=outcome, action=action, fired_rules=["X-01"],
        policy_version="policy_v1@1.0.0", decided_at=NOW,
    )  # fmt: skip


def approval(case, status="pending", proposed=VERIFY, final=None, options=None):
    decided = status != "pending"
    return ApprovalRecord(
        approval_id="a1", idempotency_key="k1", case_key=case.case_key, run_id="r1",
        proposed=proposed, allowed_options=options or [VERIFY, "escalate_human"], status=status,
        final_action=(final or proposed) if decided else None,
        reason="checked" if status in ("edited", "rejected") else None,
        decided_by="goutham" if decided else None, decided_at=NOW if decided else None,
    )  # fmt: skip


def client(*script):
    backend = FakeBackend(list(script))
    return LLMClient(backend, mode="fake"), backend


def draft(seed, invoice="SPE/2026/0108", c=None, d=None, a=None, language="en"):
    cases, books, portal, by_invoice = seed
    case = by_invoice(invoice)
    d = d if d is not None else decision(case)
    return draft_email(
        case, decision=d, approval=a, books=books, portal=portal, guard=RegexGuard(),
        client=c, language=language, now=NOW,
    )  # fmt: skip


# --- who gets a draft -----------------------------------------------------------------------------


def test_final_action_and_eligibility(seed):
    case = seed[3]("SAM/2026/0110")
    pending = approval(case)
    approved = approval(case, "approved")
    edited = approval(case, "edited", proposed="escalate_human", final=VERIFY)
    edited_away = approval(case, "edited", final="escalate_human")
    rejected = approval(case, "rejected", final="escalate_human")
    need = decision(case, "require_approval", VERIFY)
    table = [
        (decision(case, "auto_allow", CHASE), None, CHASE, True),
        (decision(case, "auto_allow", "none"), None, "none", False),
        (need, None, None, False),
        (need, pending, None, False),
        (need, approved, VERIFY, True),
        (need, edited, VERIFY, True),
        (need, edited_away, "escalate_human", False),
        (need, rejected, "escalate_human", False),
        (decision(case, "block", "none"), None, "escalate_human", False),
        (decision(case, "block", VERIFY), approved, VERIFY, False),  # blocked stays blocked
        (None, None, None, False),
    ]
    for d, a, action, ok in table:
        assert final_action(d, a) == action, (d and d.outcome, a and a.status)
        assert eligible(d, a) is ok, (d and d.outcome, a and a.status)


def test_not_eligible_means_no_draft_and_no_model_call(seed):
    c, backend = client("never")
    case = seed[3]("SAM/2026/0110")
    out = draft(seed, "SAM/2026/0110", c=c, d=decision(case, "require_approval", VERIFY))
    assert out.draft is None and out.source == "not_eligible" and backend.call_count == 0


# --- templates ------------------------------------------------------------------------------------


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize(("invoice", "action"), [("SPE/2026/0108", CHASE),
                                                 ("SAM/2026/0110", VERIFY)])  # fmt: skip
def test_every_template_is_filled_and_grounded(seed, invoice, action, language):
    cases, books, portal, by_invoice = seed
    facts = email_facts(by_invoice(invoice), action, books, portal, language)
    subject, body = template_draft(facts, action, language)
    assert "$" not in subject + body and invoice in subject and invoice in body
    assert check_draft(EmailAnswer(subject=subject, body=body), facts, language) == []


def test_the_chase_template_states_the_book_figures(seed):
    out = draft(seed)
    assert out.source == "template" and out.note == "no_model"
    body = out.draft.body
    assert "SPE/2026/0108" in body and "2026-10-18" in body and "36UVXKJ7362F1ZQ" in body
    assert "₹2,22,400.00" in body and "₹40,032.00" in body and "October 2026" in body


def test_the_verify_template_shows_both_sets_of_figures(seed):
    case = seed[3]("SAM/2026/0110")
    a = approval(case, "approved")
    out = draft(seed, "SAM/2026/0110", d=decision(case, "require_approval", VERIFY), a=a)
    body = out.draft.body
    assert "₹1,85,800.00" in body and "₹1,95,800.00" in body  # books vs GSTR-2B
    assert "₹9,290.00" in body and "₹9,790.00" in body


def test_hindi_and_telugu_templates_use_their_script(seed):
    hi = draft(seed, language="hi").draft
    te = draft(seed, language="te").draft
    assert re.search("[ऀ-ॿ]", hi.body) and hi.language == "hi"
    assert re.search("[ఀ-౿]", te.body) and te.language == "te"
    assert "अक्टूबर 2026" in hi.body and "అక్టోబర్ 2026" in te.body


def test_an_unknown_language_falls_back_to_english(seed):
    assert draft(seed, language="fr").draft.language == "en"


# --- model drafts ---------------------------------------------------------------------------------


def email(body, subject="Invoice not in our GSTR-2B"):
    return json.dumps({"subject": subject, "body": body})


GOOD = (
    "Dear Specimen Textiles,\n\nYour invoice SPE/2026/0108 dated 2026-10-18 (GSTIN "
    "36UVXKJ7362F1ZQ), taxable value ₹2,22,400.00 and GST ₹40,032.00, is not in our GSTR-2B "
    "for October 2026. Please report it in your GSTR-1. We are holding the credit until then."
    "\n\nRegards,\nAccounts team"
)


def test_a_grounded_model_draft_is_used(seed):
    c, backend = client(email(GOOD))
    out = draft(seed, c=c)
    assert out.source == "llm" and out.note == "" and out.draft.body == GOOD.strip()
    assert backend.call_count == 1
    system, messages = backend.calls[0]
    assert system == cm.SYSTEM_PROMPT and "SPE/2026/0108" in messages[0].parts[0]


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        (GOOD + " Please reply within 7 days.", "a number that is not in the facts"),
        (GOOD.replace("₹40,032.00", "₹40,302.00"), "a number that is not in the facts"),
        (GOOD.replace("2026-10-18", "2026-10-19"), "a date that is not in the facts"),
        (GOOD + " See also invoice SPE/2026/0999.", "an id or code that is not in the facts"),
        (GOOD.replace("SPE/2026/0108", "your invoice"), "does not name the invoice number"),
        (GOOD + " Pay at https://pay.example.invalid", "a link, an address or markup"),
        (GOOD + " Write to accounts@example.invalid", "a link, an address or markup"),
        (GOOD + " Under section 16(4) the credit lapses.", "a number that is not in the facts"),
    ],
)
def test_an_ungrounded_model_draft_falls_back_to_the_template(seed, body, problem):
    c, backend = client(email(body))
    out = draft(seed, c=c)
    assert out.source == "template" and out.note == "grounding_failed"
    assert problem in " ".join(out.errors) and backend.call_count == 1  # no retry
    assert (
        out.draft.body
        == template_draft(
            email_facts(seed[3]("SPE/2026/0108"), CHASE, seed[1], seed[2], "en"), CHASE, "en"
        )[1]
    )


def test_a_draft_in_the_wrong_language_falls_back(seed):
    c, _ = client(email(GOOD))
    out = draft(seed, c=c, language="hi")
    assert out.source == "template" and "not in Hindi" in " ".join(out.errors)
    assert re.search("[ऀ-ॿ]", out.draft.body)


def test_native_digits_are_checked_too(seed):
    facts = email_facts(seed[3]("SPE/2026/0108"), CHASE, seed[1], seed[2], "hi")
    assert ungrounded("जीएसटी ₹४०,०३२.०० है", facts) == []
    assert ungrounded("जीएसटी ₹४०,०३३.०० है", facts) == ["a number that is not in the facts"]


def test_bad_json_is_retried_then_the_template_is_used(seed):
    c, backend = client("Dear supplier, please file.")
    out = draft(seed, c=c)
    assert backend.call_count == 3 and out.source == "template" and out.note == "schema_failed"


def test_budget_stop_and_provider_errors_use_the_template(seed):
    stopped = RunBudget(usd=1.0)
    stopped.record(0.95, 0)
    c = LLMClient(FakeBackend([email(GOOD)]), mode="fake", run_budget=stopped)
    assert draft(seed, c=c).note == "budget_stop"
    c, _ = client(RuntimeError("bedrock down"))
    out = draft(seed, c=c)
    assert out.note == "llm_error" and out.draft is not None


def test_configuration_errors_stop_the_run(seed, tmp_path):
    c = LLMClient(FakeBackend(["{}"]), mode="replay", recorder=Recorder(tmp_path / "r.jsonl"))
    with pytest.raises(RecordingMissing):
        draft(seed, c=c)


# --- facts, keys, the outbox ----------------------------------------------------------------------


def test_the_facts_come_from_books_and_gstr2b_only(seed):
    """The communicator takes no extraction at all, so text read from an invoice image
    (other_text, the planted note on IMA/2026/0117) cannot reach a draft or its prompt."""
    import inspect

    assert "extraction" not in inspect.signature(draft_email).parameters
    cases, books, portal, by_invoice = seed
    facts = email_facts(by_invoice("IMA/2026/0117"), VERIFY, books, portal, "en")
    text = json.dumps(facts, ensure_ascii=False)
    assert "pre-verified" not in text and "Note to AI" not in text
    assert set(facts) == {
        "action", "language", "period", "period_label", "supplier_name", "supplier_gstin",
        "invoice_no", "invoice_date", "our_records", "gstr2b",
    }  # fmt: skip


def test_the_gstr2b_gstin_corrects_a_typo_in_the_books(seed):
    cases, books, portal, by_invoice = seed
    facts = email_facts(by_invoice("TES/2026/0112"), VERIFY, books, portal, "en")
    assert (
        facts["supplier_gstin"] == portal[by_invoice("TES/2026/0112").portal_index].supplier_gstin
    )
    assert (
        facts["supplier_gstin"] != books[by_invoice("TES/2026/0112").book_lines[0]].supplier_gstin
    )


def test_the_idempotency_key_ignores_the_wording(seed, tmp_path):
    case = seed[3]("SPE/2026/0108")
    first = draft(seed, c=client(email(GOOD))[0]).draft
    second = draft(seed).draft  # template wording this time
    assert first.body != second.body
    assert (
        first.idempotency_key
        == second.idempotency_key
        == outbox_key("SEED", "2026-10", case.case_key, CHASE, "36UVXKJ7362F1ZQ")
    )
    repo = Repo.open(tmp_path / "o.sqlite")
    repo.create_run("r1", "SEED", "2026-10", "h", "p")
    stored, created = repo.insert_outbox(first, "r1")
    again, created2 = repo.insert_outbox(second, "r1")
    assert created and not created2 and again.body == first.body
    assert len(repo.list_outbox("r1")) == 1
    repo.close()


def clean_read():
    from agent_orchestrator.models import ExtractedInvoice, FieldValue

    fv = FieldValue(value="x", confidence=0.95)
    names = ("supplier_name supplier_gstin buyer_gstin invoice_no invoice_date taxable_value "
             "rate igst cgst sgst total").split()  # fmt: skip
    return ExtractedInvoice(
        pdf_file="p.pdf", model_id="fake", escalated=False, **dict.fromkeys(names, fv),
        min_critical_confidence=0.95,
    )  # fmt: skip


def test_the_seed_month_drafts_two_chase_emails(seed):
    cases, books, portal, _ = seed
    from agent_orchestrator.models import ProposedAction
    from agent_orchestrator.policy.engine import decide

    drafted = []
    for c in cases:
        action = CHASE if c.mismatch_type == "missing_in_2b" else "none"
        if c.mismatch_type not in ("missing_in_2b", "none"):
            continue
        p = ProposedAction(case_key=c.case_key, action=action, reason="", proposer="rule")
        d = decide(c, p, extraction=clean_read(), now=NOW)
        out = draft_email(c, decision=d, approval=None, books=books, portal=portal,
                          guard=RegexGuard(), now=NOW)  # fmt: skip
        if out.draft:
            drafted.append(out.draft)
    assert [x.to_supplier_gstin for x in drafted] == ["36UVXKJ7362F1ZQ", "24GDDPG1561H1ZU"]


def test_supplier_languages_from_a_business_profile(tmp_path):
    path = tmp_path / "business.json"
    path.write_text(json.dumps({"suppliers": [
        {"gstin": "A", "language": "hi"}, {"gstin": "B", "language": "te"},
        {"gstin": "C"}, {"gstin": "D", "language": "fr"},
    ]}), "utf-8")  # fmt: skip
    assert load_supplier_languages(path) == {"A": "hi", "B": "te", "C": "en"}


def test_no_code_in_the_repo_can_send_mail():
    pattern = re.compile(r"smtplib|sendmail|requests\.post|SMTP\(|send_message\(")
    hits = [
        str(p.relative_to(ROOT))
        for p in (ROOT / "src").rglob("*.py")
        if pattern.search(p.read_text("utf-8"))
    ]
    assert hits == []
