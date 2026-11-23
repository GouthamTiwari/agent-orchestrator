import datetime as dt
import inspect
import sqlite3
import threading

import pytest
from pydantic import ValidationError

from agent_orchestrator.models import ApprovalRecord, Case, Decision, DraftEmail
from agent_orchestrator.store import idempotency
from agent_orchestrator.store.db import MIGRATIONS, connect, migrate, schema_version
from agent_orchestrator.store.repo import Repo, StoreError

NOW = dt.datetime(2026, 11, 23, 12, 0, tzinfo=dt.UTC)
TABLES = {"runs", "cases", "decisions", "approvals", "outbox", "llm_calls", "spans", "audit_log"}


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "orchestrator.sqlite"


@pytest.fixture
def repo(db_path):
    r = Repo.open(db_path)
    r.create_run("r1", "B01", "2026-10", "h" * 64, "policy_v1")
    yield r
    r.close()


def _approval(key="k1", approval_id="a1", run_id="r1"):
    return ApprovalRecord(
        approval_id=approval_id,
        idempotency_key=key,
        case_key="0123456789abcdef",
        run_id=run_id,
        proposed="verify_with_supplier",
        allowed_options=["verify_with_supplier", "escalate_human"],
    )


def _case(status="mismatch"):
    return Case(
        case_key="0123456789abcdef",
        business_id="B01",
        period="2026-10",
        mismatch_type="taxable_value_mismatch",
        status=status,
        book_lines=[3],
        portal_index=2,
        match_pass="exact",
        match_score=100.0,
        diffs={"taxable": -100000},
        facts_hash="f" * 16,
    )


# --- schema -----------------------------------------------------------------------------------


def test_migration_creates_tables_wal_and_foreign_keys(db_path):
    conn = connect(db_path)
    assert migrate(conn) == len(MIGRATIONS)
    names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert names >= TABLES
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_migration_is_idempotent(db_path):
    conn = connect(db_path)
    migrate(conn)
    migrate(conn)
    assert schema_version(conn) == len(MIGRATIONS)
    rows = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
    assert rows == len(MIGRATIONS)


def test_foreign_keys_refuse_orphans(repo):
    with pytest.raises(sqlite3.IntegrityError):
        repo.insert_approval(_approval(run_id="no-such-run"), "B01", "2026-10")


# --- idempotency keys -------------------------------------------------------------------------


def test_keys_are_deterministic_and_wording_free():
    a = idempotency.approval_key("B01", "2026-10", "ck", "verify_with_supplier", "facts")
    assert a == idempotency.approval_key("B01", "2026-10", "ck", "verify_with_supplier", "facts")
    assert a != idempotency.approval_key("B01", "2026-10", "ck", "verify_with_supplier", "new")
    e = idempotency.outbox_key("B01", "2026-10", "ck", "chase_supplier_and_hold_itc", "G")
    assert len(e) == 64 and e != a
    params = inspect.signature(idempotency.outbox_key).parameters
    assert not {"body", "subject", "text"} & set(params)  # LLM wording is not in the key
    with pytest.raises(ValueError):
        idempotency.approval_key("B|01", "2026-10", "ck", "x", "y")


# --- runs, cases, decisions -------------------------------------------------------------------


def test_run_lifecycle(repo):
    assert repo.get_run("r1")["status"] == "running"
    run = repo.update_run("r1", status="paused", usd_spent=0.12, tokens=3400)
    assert (run["status"], run["usd_spent"], run["tokens"], run["finished_at"]) == (
        "paused",
        0.12,
        3400,
        None,
    )
    assert repo.update_run("r1", status="done")["finished_at"] is not None
    with pytest.raises(StoreError):
        repo.update_run("r1", status="exploded")
    with pytest.raises(StoreError):
        repo.update_run("nope", status="done")


def test_cases_are_upserted_not_duplicated(repo):
    repo.upsert_case("r1", _case())
    repo.create_run("r2", "B01", "2026-10", "h" * 64, "policy_v1")
    repo.upsert_case("r2", _case(status="needs_human"), action="escalate_human")
    cases = repo.list_cases("B01", "2026-10")
    assert len(cases) == 1 and cases[0].status == "needs_human"
    assert repo.get_case("B01", "2026-10", "0123456789abcdef") == cases[0]


def test_decision_round_trip(repo):
    d = Decision(
        case_key="ck",
        outcome="require_approval",
        action="verify_with_supplier",
        fired_rules=["GST-ITC-VAL-01"],
        policy_version="policy_v1",
        decided_at=NOW,
    )
    repo.record_decision("r1", d)
    repo.record_decision("r1", d)  # same run and case: one row
    assert repo.get_decision("r1", "ck") == d
    assert repo.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1


# --- approvals --------------------------------------------------------------------------------


def test_same_approval_key_twice_returns_the_first_row(repo):
    first, created = repo.insert_approval(_approval(), "B01", "2026-10")
    again, created_again = repo.insert_approval(_approval(approval_id="a2"), "B01", "2026-10")
    assert created is True and created_again is False
    assert again == first and again.approval_id == "a1"
    assert len(repo.list_approvals()) == 1


def test_idempotent_insert_under_eight_threads(db_path):
    setup = Repo.open(db_path)
    setup.create_run("r1", "B01", "2026-10", "h" * 64, "policy_v1")
    setup.close()
    results, errors = [], []
    barrier = threading.Barrier(8)

    def worker(i):
        r = Repo.open(db_path)
        try:
            barrier.wait()
            results.append(r.insert_approval(_approval(approval_id=f"a{i}"), "B01", "2026-10"))
            draft = DraftEmail(
                idempotency_key="e1",
                case_key="ck",
                to_supplier_gstin="G",
                subject=f"s{i}",
                body=f"wording {i}",
                created_at=NOW,
            )
            results.append(r.insert_outbox(draft, "r1"))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)
        finally:
            r.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    approvals = [row for row, _ in results if isinstance(row, ApprovalRecord)]
    drafts = [row for row, _ in results if isinstance(row, DraftEmail)]
    created = [c for _, c in results]
    assert sum(created) == 2  # exactly one approval and one e-mail were created
    assert len({a.approval_id for a in approvals}) == 1
    assert len({d.body for d in drafts}) == 1  # everyone got the first draft back
    check = Repo.open(db_path)
    assert len(check.list_approvals()) == 1 and len(check.list_outbox()) == 1
    check.close()


def test_resolve_approval_once_and_audited(repo):
    repo.insert_approval(_approval(), "B01", "2026-10")
    done = repo.resolve_approval(
        "a1", "edited", "escalate_human", "Goutham", "call supplier", decided_at=NOW
    )
    assert (done.status, done.final_action, done.decided_by) == (
        "edited",
        "escalate_human",
        "Goutham",
    )
    with pytest.raises(StoreError, match="already edited"):
        repo.resolve_approval("a1", "approved", "verify_with_supplier", "Goutham")
    audit = repo.list_audit("r1")
    assert audit[-1]["action"] == "approval_edited"
    assert audit[-1]["detail"] == {
        "approval_id": "a1",
        "before": "verify_with_supplier",
        "after": "escalate_human",
        "reason": "call supplier",
    }
    assert repo.list_approvals() == []


def test_invalid_resolution_writes_nothing(repo):
    repo.insert_approval(_approval(), "B01", "2026-10")
    with pytest.raises(ValidationError):
        repo.resolve_approval("a1", "edited", "escalate_human", "Goutham")  # no reason
    with pytest.raises(ValidationError):
        repo.resolve_approval("a1", "approved", "correct_books", "Goutham")  # not allowed
    assert repo.get_approval("a1").status == "pending"
    assert repo.list_audit() == []
    with pytest.raises(StoreError):
        repo.resolve_approval("missing", "approved", "verify_with_supplier", "Goutham")


# --- outbox and ledger ------------------------------------------------------------------------


def test_outbox_keeps_the_first_draft(repo):
    d1 = DraftEmail(
        idempotency_key="e1",
        case_key="ck",
        to_supplier_gstin="G",
        subject="s",
        body="first wording",
        created_at=NOW,
    )
    d2 = d1.model_copy(update={"body": "different wording"})
    assert repo.insert_outbox(d1, "r1") == (d1, True)
    stored, created = repo.insert_outbox(d2, "r1")
    assert created is False and stored.body == "first wording"


def test_ledger_never_pays_twice(repo):
    kw = dict(
        purpose="extract",
        model_id="m",
        input_tokens=1000,
        output_tokens=200,
        usd=0.001,
        latency_ms=812.5,
        response={"text": "{}"},
    )
    assert repo.ledger_put("req1", **kw) is True
    assert repo.ledger_put("req1", **{**kw, "usd": 9.99}) is False
    hit = repo.ledger_get("req1")
    assert hit["usd"] == 0.001 and hit["response"] == {"text": "{}"}
    assert repo.ledger_get("other") is None
    assert repo.ledger_totals() == {"calls": 1, "usd": 0.001, "tokens": 1200}


# --- audit log --------------------------------------------------------------------------------


def test_audit_log_is_append_only(repo):
    names = {n for n, _ in inspect.getmembers(Repo, inspect.isfunction)}
    assert {n for n in names if "audit" in n} == {"append_audit", "_append_audit", "list_audit"}
    repo.append_audit("system", "write_refused", "r1", "ck", {"tool": "request_approval"})
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        repo.conn.execute("UPDATE audit_log SET actor = 'x'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        repo.conn.execute("DELETE FROM audit_log")
    assert len(repo.list_audit()) == 1
