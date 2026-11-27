"""Spec 15.4: running the same month again (a new run id, the same inputs) creates no
duplicates. Cases are upserted by case key; approvals and drafts use deterministic keys, and an
approval a person already decided for the same facts is reused, not asked again."""

import json
from pathlib import Path

from agent_orchestrator.orchestrator.runner import demo_config, resume_run, start_run
from agent_orchestrator.store.repo import Repo
from agent_orchestrator.tools import toolfns


def snapshot(db: Path):
    repo = Repo.open(db)
    try:
        q = repo.conn.execute
        return {
            "cases": sorted(tuple(r) for r in q(
                "SELECT case_key, mismatch_type, status, action, facts_hash, case_json FROM cases"
            )),
            "approvals": sorted(tuple(r) for r in q(
                "SELECT approval_id, idempotency_key, case_key, proposed, status, final_action,"
                " decided_by FROM approvals"
            )),
            "outbox": sorted(tuple(r) for r in q(
                "SELECT idempotency_key, case_key, to_supplier_gstin, subject, body FROM outbox"
            )),
            "approval_keys": q(
                "SELECT COUNT(DISTINCT idempotency_key) FROM approvals"
            ).fetchone()[0],
            "outbox_keys": q("SELECT COUNT(DISTINCT idempotency_key) FROM outbox").fetchone()[0],
        }  # fmt: skip
    finally:
        repo.close()


def approve_all(db: Path) -> None:
    repo = Repo.open(db)
    try:
        for a in repo.list_approvals("pending"):
            toolfns.approve(repo, a.approval_id, by="asha")
    finally:
        repo.close()


def test_rerunning_a_finished_month_creates_no_duplicates(tmp_path):
    where = {"db_path": tmp_path / "o.sqlite", "runs_dir": tmp_path / "runs"}
    start_run(demo_config("first"), **where)
    approve_all(where["db_path"])
    assert resume_run("first", **where).status == "done"
    after_first = snapshot(where["db_path"])
    assert len(after_first["approvals"]) == after_first["approval_keys"] == 7
    assert len(after_first["outbox"]) == after_first["outbox_keys"] == 4

    second = start_run(demo_config("second"), **where)
    # Every approval was already decided for these exact facts: nothing to ask, so no pause.
    assert second.status == "done" and second.pending == []
    after_second = snapshot(where["db_path"])
    assert after_second == after_first  # one approval per key, one draft per key, same cases

    rep = json.loads((where["runs_dir"] / "second" / "report.json").read_text("utf-8"))
    assert rep["approvals"]["approved"] == 7 and rep["approvals"]["pending"] == 0
    assert rep["drafts"]["count"] == 4  # the first run's drafts, found again
    first_rep = json.loads((where["runs_dir"] / "first" / "report.json").read_text("utf-8"))
    assert rep["cases"]["by_final_action"] == first_rep["cases"]["by_final_action"]


def test_rerunning_while_approvals_are_pending_asks_the_same_questions(tmp_path):
    where = {"db_path": tmp_path / "o.sqlite", "runs_dir": tmp_path / "runs"}
    first = start_run(demo_config("a"), **where)
    second = start_run(demo_config("b"), **where)
    assert second.status == "paused" and sorted(second.pending) == sorted(first.pending)
    assert snapshot(where["db_path"])["approval_keys"] == 7
    approve_all(where["db_path"])  # one decision answers both runs
    assert resume_run("a", **where).status == "done"
    assert resume_run("b", **where).status == "done"
    snap = snapshot(where["db_path"])
    assert len(snap["outbox"]) == snap["outbox_keys"] == 4
