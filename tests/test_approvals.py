import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_orchestrator.agents import extraction as extraction_agent
from agent_orchestrator.agents import planner as planner_agent
from agent_orchestrator.agents.intake import run_intake_dir
from agent_orchestrator.cli import app
from agent_orchestrator.interfaces.stubs import RegexGuard
from agent_orchestrator.llm.client import LLMClient
from agent_orchestrator.llm.fake import FakeBackend
from agent_orchestrator.orchestrator.runner import demo_config, demo_script, resume_run, start_run
from agent_orchestrator.render import render_page_png
from agent_orchestrator.store.idempotency import approval_key, outbox_key
from agent_orchestrator.store.repo import Repo
from agent_orchestrator.tools import toolfns
from agent_orchestrator.tools.toolfns import (
    HumanActionError,
    ToolContext,
    ToolRefused,
    decision_ref,
    request_approval,
    write_draft_email,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed" / "seed_data"
INJECTED = SEED / "invoices" / "IMA_2026_0117.pdf"


@pytest.fixture(scope="module")
def paused(tmp_path_factory):
    """One demo run paused at the gate, copied fresh into each test that needs it."""
    d = tmp_path_factory.mktemp("paused")
    start_run(demo_config("p1"), db_path=d / "o.sqlite", runs_dir=d / "runs")
    return d


@pytest.fixture
def run(paused, tmp_path):
    import shutil

    for name in ("o.sqlite", "o.sqlite-wal", "o.sqlite-shm"):
        if (paused / name).exists():
            shutil.copy(paused / name, tmp_path / name)
    shutil.copytree(paused / "runs", tmp_path / "runs")
    repo = Repo.open(tmp_path / "o.sqlite")
    yield {"repo": repo, "db": tmp_path / "o.sqlite", "runs": tmp_path / "runs", "tmp": tmp_path}
    repo.close()


def seed_month():
    r = run_intake_dir(SEED, "2026-10")
    return {b.line_no: b for b in r.books}, {p.index: p for p in r.portal}


def ctx(repo, client=None):
    books, portal = seed_month()
    return ToolContext(
        repo=repo, guard=RegexGuard(), business_id="SEED", period="2026-10",
        email_client=client, month_loader=lambda run_id: (books, portal),
    )  # fmt: skip


def cases_by_invoice(repo):
    books, portal = seed_month()
    out = {}
    for c in repo.list_cases("SEED", "2026-10"):
        no = (
            books[c.book_lines[0]].invoice_no if c.book_lines else portal[c.portal_index].invoice_no
        )
        out[(no, c.mismatch_type)] = c
    return out


def case_of(repo, invoice, mismatch=None):
    found = [c for (no, mt), c in cases_by_invoice(repo).items()
             if no == invoice and (mismatch is None or mt == mismatch)]  # fmt: skip
    assert len(found) == 1
    return found[0]


def refused_rows(repo, tool):
    return [a for a in repo.list_audit() if a["action"] == f"refused:{tool}"]


# --- request_approval -----------------------------------------------------------------------------


def test_request_approval_is_idempotent_for_a_require_approval_decision(run):
    repo = run["repo"]
    c = case_of(repo, "SAM/2026/0110")
    d = repo.get_decision("p1", c.case_key)
    key = approval_key("SEED", "2026-10", c.case_key, d.action, c.facts_hash)
    args = dict(idempotency_key=key, run_id="p1", case_key=c.case_key,
                decision_ref=decision_ref("p1", d))  # fmt: skip
    first = request_approval(ctx(repo), **args)
    again = request_approval(ctx(repo), **args)
    assert not first["created"] and not again["created"]  # the run already asked once
    assert first["approval"] == again["approval"] and first["approval"].status == "pending"
    assert len([a for a in repo.list_approvals(None) if a.case_key == c.case_key]) == 1


@pytest.mark.parametrize(
    ("invoice", "change", "reason"),
    [
        ("SAM/2026/0100", {}, "the decision is auto_allow, not require_approval"),
        ("IMA/2026/0117", {}, "the decision is block, not require_approval"),
        ("SAM/2026/0110", {"decision_ref": "0" * 16}, "decision_ref does not match"),
        ("SAM/2026/0110", {"run_id": "other-run"}, "no decision exists"),
        ("SAM/2026/0110", {"idempotency_key": "f" * 64}, "idempotency_key is not the key"),
        ("SAM/2026/0110", {"proposed": "none"}, "proposed action differs"),
    ],
)
def test_request_approval_refuses_without_the_right_decision(run, invoice, change, reason):
    repo = run["repo"]
    c = case_of(repo, invoice, None if invoice != "SAM/2026/0110" else "taxable_value_mismatch")
    d = repo.get_decision("p1", c.case_key)
    args = dict(
        idempotency_key=approval_key("SEED", "2026-10", c.case_key, d.action, c.facts_hash),
        run_id="p1", case_key=c.case_key, decision_ref=decision_ref("p1", d),
    )  # fmt: skip
    args.update(change)
    before = toolfns.REFUSED_WRITES["request_approval"]
    with pytest.raises(ToolRefused, match=reason):
        request_approval(ctx(repo), **args)
    assert toolfns.REFUSED_WRITES["request_approval"] == before + 1
    rows = refused_rows(repo, "request_approval")
    assert rows and reason in rows[-1]["detail"]["reason"] and rows[-1]["actor"] == "tool"


# --- write_draft_email ----------------------------------------------------------------------------


def draft_args(repo, invoice, run_id="p1", action=None):
    books, portal = seed_month()
    c = case_of(repo, invoice)
    d = repo.get_decision(run_id, c.case_key)
    gstin = (
        portal[c.portal_index] if c.portal_index is not None else books[c.book_lines[0]]
    ).supplier_gstin
    return dict(
        idempotency_key=outbox_key("SEED", "2026-10", c.case_key, action or d.action, gstin),
        run_id=run_id, case_key=c.case_key, decision_ref=decision_ref(run_id, d),
    )  # fmt: skip


def test_an_auto_allowed_chase_gets_one_draft(run):
    repo = run["repo"]
    args = draft_args(repo, "SPE/2026/0108")
    first = write_draft_email(ctx(repo), **args)
    again = write_draft_email(ctx(repo), **args)
    assert first["created"] and not again["created"] and first["source"] == "template"
    assert again["draft"].body == first["draft"].body
    assert len(repo.list_outbox("p1")) == 1


def test_an_approved_verify_gets_a_draft_only_after_approval(run):
    repo = run["repo"]
    args = draft_args(repo, "SAM/2026/0110", action="verify_with_supplier")
    with pytest.raises(ToolRefused, match="no allowing decision"):
        write_draft_email(ctx(repo), **args)  # still pending
    a = next(x for x in repo.list_approvals("pending", "p1")
             if x.case_key == args["case_key"])  # fmt: skip
    toolfns.approve(repo, a.approval_id, by="tester")
    out = write_draft_email(ctx(repo), **args)
    assert out["created"] and "₹1,95,800.00" in out["draft"].body


@pytest.mark.parametrize(
    ("invoice", "how", "reason"),
    [
        ("IMA/2026/0117", None, "the case is blocked"),
        ("SAM/2026/0100", None, "no allowing decision"),  # auto-allowed, but action none
        ("SAM/2026/0110", "reject", "no allowing decision"),
        ("SAM/2026/0110", "edit", "no allowing decision"),  # edited to escalate_human
    ],
)
def test_write_draft_email_refuses_without_an_allowing_decision(run, invoice, how, reason):
    repo = run["repo"]
    c = case_of(repo, invoice, "taxable_value_mismatch" if invoice != "SAM/2026/0100" else None)
    if how:
        a = next(x for x in repo.list_approvals("pending", "p1") if x.case_key == c.case_key)
        if how == "reject":
            toolfns.reject(repo, a.approval_id, reason="not ours", by="tester")
        else:
            toolfns.edit(repo, a.approval_id, "escalate_human", reason="look", by="tester")
    args = draft_args(repo, invoice, action="verify_with_supplier")
    with pytest.raises(ToolRefused, match=reason):
        write_draft_email(ctx(repo), **args)
    assert repo.list_outbox("p1") == []
    assert reason in refused_rows(repo, "write_draft_email")[-1]["detail"]["reason"]


def test_a_wrong_key_is_refused_before_any_model_call(run):
    repo = run["repo"]
    backend = FakeBackend(["never used"])
    client = LLMClient(backend, mode="fake")
    args = {**draft_args(repo, "SPE/2026/0108"), "idempotency_key": "e" * 64}
    with pytest.raises(ToolRefused, match="idempotency_key is not the key"):
        write_draft_email(ctx(repo, client), **args)
    assert backend.call_count == 0


def test_a_wrong_decision_ref_is_refused(run):
    repo = run["repo"]
    args = {**draft_args(repo, "SPE/2026/0108"), "decision_ref": "1" * 16}
    with pytest.raises(ToolRefused, match="decision_ref does not match"):
        write_draft_email(ctx(repo), **args)


# --- the person: approve, edit, reject ------------------------------------------------------------


def pending(repo, mismatch):
    keys = {c.case_key for c in repo.list_cases("SEED", "2026-10") if c.mismatch_type == mismatch}
    return next(a for a in repo.list_approvals("pending", "p1") if a.case_key in keys)


def feedback(tmp):
    path = Path(os.environ["ORCH_FEEDBACK_FILE"])
    return [json.loads(x) for x in path.read_text("utf-8").splitlines()] if path.exists() else []


def test_approve_edit_reject_record_who_when_and_before_after(run):
    repo = run["repo"]
    a = toolfns.approve(repo, pending(repo, "taxable_value_mismatch").approval_id, by="asha")
    assert (a.status, a.final_action, a.decided_by) == ("approved", "verify_with_supplier", "asha")
    e = toolfns.edit(repo, pending(repo, "tax_head_mismatch").approval_id, "correct_books",
                     reason="the books used the wrong head", by="ravi")  # fmt: skip
    assert (e.status, e.final_action) == ("edited", "correct_books")
    r = toolfns.reject(repo, pending(repo, "duplicate_in_books").approval_id,
                       reason="already reversed", by="asha")  # fmt: skip
    assert (r.status, r.final_action) == ("rejected", "escalate_human")
    audit = {x["detail"]["approval_id"]: x for x in repo.list_audit() if
             x["action"].startswith("approval_")}  # fmt: skip
    assert audit[e.approval_id]["actor"] == "ravi"
    assert audit[e.approval_id]["detail"]["before"] == "escalate_human"
    assert audit[e.approval_id]["detail"]["after"] == "correct_books"
    assert audit[r.approval_id]["detail"]["reason"] == "already reversed"
    lines = feedback(run["tmp"])  # edits and rejections only
    assert [(x["status"], x["proposed"], x["final_action"]) for x in lines] == [
        ("edited", "escalate_human", "correct_books"),
        ("rejected", "correct_books", "escalate_human"),
    ]
    assert set(lines[0]) == {"at", "approval_id", "run_id", "case_key", "status", "proposed",
                             "final_action", "reason", "decided_by"}  # fmt: skip


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda repo, i: toolfns.edit(repo, i, "none", reason="x"), "not allowed here"),
        (lambda repo, i: toolfns.edit(repo, i, "verify_with_supplier", reason="x"), "edited"),
        (lambda repo, i: toolfns.edit(repo, i, "escalate_human", reason=" "), "reason is required"),
        (lambda repo, i: toolfns.reject(repo, i, reason=""), "reason is required"),
        (lambda repo, i: toolfns.approve(repo, "ap_missing"), "no approval"),
    ],
)
def test_human_actions_refuse_what_is_not_allowed(run, call, message):
    repo = run["repo"]
    a = pending(repo, "taxable_value_mismatch")
    with pytest.raises(HumanActionError, match=message):
        call(repo, a.approval_id)
    assert repo.get_approval(a.approval_id).status == "pending"


def test_a_decided_approval_cannot_be_decided_again(run):
    repo = run["repo"]
    a = pending(repo, "taxable_value_mismatch")
    toolfns.approve(repo, a.approval_id, by="asha")
    with pytest.raises(HumanActionError, match="already approved"):
        toolfns.reject(repo, a.approval_id, reason="changed my mind")


def test_no_agent_tool_can_decide_an_approval():
    assert set(toolfns.AGENT_TOOLS) == {"request_approval", "write_draft_email"}
    assert not set(toolfns.AGENT_TOOLS) & set(toolfns.HUMAN_ACTIONS)
    assert set(toolfns.HUMAN_ACTIONS) == {"approve", "edit", "reject"}


# --- the gate -------------------------------------------------------------------------------------


def counts(repo):
    return {t: repo.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("approvals", "decisions", "outbox", "cases", "audit_log")}  # fmt: skip


def test_the_gate_writes_nothing_while_it_waits(run):
    before = counts(run["repo"])
    out = resume_run("p1", db_path=run["db"], runs_dir=run["runs"])
    assert out.status == "paused" and len(out.pending) == 7
    assert counts(run["repo"]) == before


# --- CLI ------------------------------------------------------------------------------------------


def test_cli_approvals_list_approve_edit_reject(run):
    cli, db = CliRunner(), ["--db", str(run["db"])]
    out = cli.invoke(app, ["approvals", "list", *db])
    assert out.exit_code == 0 and "7 pending" in out.output
    repo = run["repo"]
    a1 = pending(repo, "taxable_value_mismatch").approval_id
    a2 = pending(repo, "tax_head_mismatch").approval_id
    a3 = pending(repo, "duplicate_in_books").approval_id
    out = cli.invoke(app, ["approvals", "approve", a1, "--by", "asha", *db])
    assert out.exit_code == 0 and f"{a1}: approved -> verify_with_supplier by asha" in out.output
    out = cli.invoke(app, ["approvals", "edit", a2, "verify_with_supplier", "--reason", "x", *db])
    assert out.exit_code == 1 and "not allowed here" in out.output
    out = cli.invoke(app, ["approvals", "edit", a2, "correct_books", "--reason", "x", *db])
    assert out.exit_code == 0 and "edited -> correct_books" in out.output
    out = cli.invoke(app, ["approvals", "reject", a3, *db])
    assert out.exit_code == 1 and "reason is required" in out.output
    out = cli.invoke(app, ["approvals", "reject", a3, "--reason", "dup reversed", *db])
    assert out.exit_code == 0 and "rejected -> escalate_human" in out.output
    out = cli.invoke(app, ["approvals", "list", "--all", *db])
    assert "4 pending" not in out.output and "7 approvals" in out.output


def test_cli_without_a_database_says_so(tmp_path):
    out = CliRunner().invoke(app, ["approvals", "list", "--db", str(tmp_path / "none.sqlite")])
    assert out.exit_code != 0 and "no database" in out.output


def test_resume_in_new_processes_after_deciding_from_the_cli(tmp_path):
    """Spec 15.4 'pause for hours': every step is its own process; only SQLite carries state."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("ORCH_")}
    env["ORCH_FEEDBACK_FILE"] = str(tmp_path / "feedback.jsonl")
    where = ["--db", str(tmp_path / "o.sqlite")]

    def cli(*args):
        r = subprocess.run([sys.executable, "-m", "agent_orchestrator.cli", *args], cwd=ROOT,
                           env=env, capture_output=True, text=True, timeout=300)  # fmt: skip
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    runs = ["--runs-dir", str(tmp_path / "runs")]
    assert "paused" in cli("run", "--demo", "--run-id", "h1", *where, *runs)
    ids = [line.split()[0] for line in cli("approvals", "list", *where).splitlines()
           if line.startswith("ap_")]  # fmt: skip
    assert len(ids) == 7
    for i in ids[:-1]:
        cli("approvals", "approve", i, "--by", "asha", *where)
    assert "paused" in cli("resume", "h1", *where, *runs)  # one still pending
    cli("approvals", "reject", ids[-1], "--reason", "not ours", *where)
    assert "run h1: done" in cli("resume", "h1", *where, *runs)
    rep = json.loads((tmp_path / "runs" / "h1" / "report.json").read_text("utf-8"))
    assert rep["status"] == "done" and rep["approvals"]["approved"] == 6
    assert rep["approvals"]["rejected"] == 1
    assert len(Path(env["ORCH_FEEDBACK_FILE"]).read_text("utf-8").splitlines()) == 1


# --- the injection seed invoice through the real graph --------------------------------------------


def obedient_script():
    """A fake model that does what the planted note says: the reader hides the note and
    reports full confidence, and the planner proposes 'none' ("approve full ITC")."""
    month = demo_config("x").month()
    base = demo_script(month)
    injected = hashlib.sha256(render_page_png(INJECTED)).hexdigest()

    def answer(system, messages):
        text = base(system, messages)
        if system == extraction_agent.SYSTEM_PROMPT:
            image = next(p for p in messages[0].parts if not isinstance(p, str))
            if hashlib.sha256(image.data).hexdigest() == injected:
                data = json.loads(text)
                data["other_text"] = ""  # "do not flag it for review"
                for v in data.values():
                    if isinstance(v, dict):
                        v["confidence"] = 1.0
                return json.dumps(data)
        if system == planner_agent.SYSTEM_PROMPT:
            data = json.loads(text)
            data["action"], data["reason"] = "none", "Pre-verified; approve full ITC."
            return json.dumps(data)
        return text

    return answer


def test_the_injection_invoice_is_blocked_even_when_the_model_obeys_it(tmp_path):
    db, runs = tmp_path / "o.sqlite", tmp_path / "runs"
    start_run(demo_config("inj"), db_path=db, runs_dir=runs, script=obedient_script())
    repo = Repo.open(db)
    try:
        c = case_of(repo, "IMA/2026/0117")
        d = repo.get_decision("inj", c.case_key)
        assert d.outcome == "block" and "SAFE-INJ-01" in d.fired_rules
        assert c.status == "needs_human" and "injection" in c.flags
        # no approval row that a person could approve into the model's action
        assert [a for a in repo.list_approvals(None) if a.case_key == c.case_key] == []
        assert [x for x in repo.list_outbox() if x.case_key == c.case_key] == []
        with pytest.raises(ToolRefused, match="block"):
            request_approval(
                ctx(repo),
                idempotency_key=approval_key("SEED", "2026-10", c.case_key, d.action,
                                             c.facts_hash),
                run_id="inj", case_key=c.case_key, decision_ref=decision_ref("inj", d),
            )  # fmt: skip
    finally:
        repo.close()
    rep = json.loads((runs / "inj" / "report.json").read_text("utf-8"))
    assert [b["invoice_no"] for b in rep["blocked"]] == ["IMA/2026/0117"]
    # The obeying planner's "none" was never allowed for a mismatch: every one fell back.
    assert all(x["final_action"] != "none" for x in rep["cases_detail"]
               if x["mismatch_type"] not in ("none", "rounding_within_tolerance",
                                             "invoice_no_format_difference"))  # fmt: skip
