import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_orchestrator.cli import app
from agent_orchestrator.models import fields_with_reducers
from agent_orchestrator.orchestrator import runner
from agent_orchestrator.orchestrator.graph import checkpointer, compile_graph
from agent_orchestrator.orchestrator.runner import (
    ThreadRepo,
    build_deps,
    demo_config,
    resume_run,
    start_run,
)
from agent_orchestrator.orchestrator.state import BRANCH_WRITES, GraphState, initial_state
from agent_orchestrator.render import render_page_png
from agent_orchestrator.store.repo import Repo

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "agent_orchestrator"
INV = ROOT / "data" / "seed" / "seed_data" / "invoices"


@pytest.fixture
def where(tmp_path):
    return {"db_path": tmp_path / "o.sqlite", "runs_dir": tmp_path / "runs"}


def decide_all(db, run_id, how=None):
    """Resolve every pending approval of a run; ``how`` maps case_key -> (status, action)."""
    repo = Repo.open(db)
    try:
        for a in repo.list_approvals("pending", run_id):
            status, action = (how or {}).get(a.case_key, ("approved", a.proposed))
            reason = "checked by a person" if status != "approved" else None
            repo.resolve_approval(a.approval_id, status, action, "tester", reason=reason)
    finally:
        repo.close()


def report(where, run_id):
    return json.loads((where["runs_dir"] / run_id / "report.json").read_text("utf-8"))


# --- structure ------------------------------------------------------------------------------------


def test_only_the_graph_module_imports_langgraph():
    importers = sorted(
        str(p.relative_to(SRC)).replace("\\", "/")
        for p in SRC.rglob("*.py")
        if re.search(r"^\s*(from|import) langgraph", p.read_text("utf-8"), re.M)
    )
    assert importers == ["orchestrator/graph.py"]
    for p in (SRC / "agents").glob("*.py"):
        assert "langgraph" not in p.read_text("utf-8"), p.name


def test_every_field_a_branch_writes_has_a_reducer():
    reducers = set(fields_with_reducers(GraphState))
    for node, fields in BRANCH_WRITES.items():
        assert fields <= reducers, (node, fields - reducers)


def test_branches_write_only_reducer_fields_in_a_real_run(where):
    """Watch every update the parallel branches actually return."""
    repo = ThreadRepo(where["db_path"])
    saver, conn = checkpointer(where["db_path"])
    try:
        deps = build_deps(demo_config("watch2"), repo, where["runs_dir"], where["db_path"])
        repo.create_run("watch2", "SEED", "2026-10", "h", "p")
        graph = compile_graph(deps, saver)
        seen: dict[str, set[str]] = {}
        config = {"configurable": {"thread_id": "watch2"}, "max_concurrency": 2}
        for chunk in graph.stream(initial_state(run_id="watch2"), config, stream_mode="updates",
                                  durability="sync"):  # fmt: skip
            for node, update in chunk.items():
                if isinstance(update, dict):
                    seen.setdefault(node, set()).update(update)
    finally:
        conn.close()
        repo.close()
    reducers = set(fields_with_reducers(GraphState))
    for node in ("extract_one", "research_and_plan"):
        assert seen[node] and seen[node] <= BRANCH_WRITES[node] <= reducers, (node, seen[node])


# --- a run: pause, the gate, resume ---------------------------------------------------------------


def test_the_demo_pauses_at_the_approval_gate(where):
    r = start_run(demo_config("d1"), **where)
    assert r.status == "paused" and len(r.pending) == 7
    rep = report(where, "d1")  # the partial report exists while waiting
    assert rep["status"] == "paused" and rep["approvals"]["pending"] == 7
    assert rep["cases"]["by_outcome"] == {"auto_allow": 11, "block": 1, "require_approval": 7}
    assert [b["invoice_no"] for b in rep["blocked"]] == ["IMA/2026/0117"]
    repo = Repo.open(where["db_path"])
    try:
        assert repo.get_run("d1")["status"] == "paused"
        assert repo.list_outbox("d1") == []  # drafts come after the gate
        assert len(repo.list_cases("SEED", "2026-10")) == 19
    finally:
        repo.close()


def test_the_checkpoint_is_sqlite_keyed_by_run_id(where):
    start_run(demo_config("d2"), **where)
    saver, conn = checkpointer(where["db_path"])
    try:
        snap = saver.get_tuple({"configurable": {"thread_id": "d2"}})
        assert snap is not None and snap.config["configurable"]["thread_id"] == "d2"
        assert type(saver).__name__ == "SqliteSaver"
    finally:
        conn.close()


def test_runs_use_sync_durability(where, monkeypatch):
    calls = []
    real = runner.compile_graph

    def spy(deps, saver):
        graph = real(deps, saver)
        invoke = graph.invoke

        def recorded(*a, **k):
            calls.append(k.get("durability"))
            return invoke(*a, **k)

        graph.invoke = recorded
        return graph

    monkeypatch.setattr(runner, "compile_graph", spy)
    start_run(demo_config("d3"), **where)
    assert calls == ["sync"]


def test_the_gate_waits_until_every_approval_is_decided(where):
    r = start_run(demo_config("d4"), **where)
    again = resume_run("d4", **where)  # nothing decided: interrupt again, same list
    assert again.status == "paused" and again.pending == r.pending
    repo = Repo.open(where["db_path"])
    first = repo.list_approvals("pending", "d4")[0]
    repo.resolve_approval(first.approval_id, "approved", first.proposed, "tester")
    repo.close()
    partly = resume_run("d4", **where)
    assert partly.status == "paused" and len(partly.pending) == 6
    assert first.approval_id not in partly.pending


def test_resume_finishes_the_run_with_drafts_and_a_report(where):
    start_run(demo_config("d5"), **where)
    decide_all(where["db_path"], "d5")
    done = resume_run("d5", **where)
    assert done.status == "done" and done.pending == []
    rep = report(where, "d5")
    assert rep["status"] == "done"
    assert rep["cases"]["by_final_action"] == {
        "chase_supplier_and_hold_itc": 2, "correct_books": 2, "escalate_human": 4, "none": 9,
        "verify_with_supplier": 2,
    }  # fmt: skip
    assert rep["drafts"]["count"] == 4  # 2 auto chase + 2 approved verify
    assert set(rep["cost"]["by_agent"]) >= {"extraction", "research", "planner", "policy"}
    repo = Repo.open(where["db_path"])
    try:
        assert repo.get_run("d5")["status"] == "done"
        assert len(repo.list_outbox("d5")) == 4
    finally:
        repo.close()
    assert (where["runs_dir"] / "d5" / "trace.json").is_file()
    with pytest.raises(ValueError, match="nothing left to resume"):
        resume_run("d5", **where)


def test_edits_and_rejections_change_the_outcome(where):
    start_run(demo_config("d6"), **where)
    repo = Repo.open(where["db_path"])
    pending = repo.list_approvals("pending", "d6")
    repo.close()
    verify = [a for a in pending if a.proposed == "verify_with_supplier"]
    how = {
        verify[0].case_key: ("rejected", "escalate_human"),
        verify[1].case_key: ("edited", "escalate_human"),
    }
    decide_all(where["db_path"], "d6", how)
    resume_run("d6", **where)
    rep = report(where, "d6")
    assert rep["approvals"]["rejected"] == 1 and rep["approvals"]["edited"] == 1
    assert "verify_with_supplier" not in rep["cases"]["by_final_action"]
    assert rep["drafts"]["count"] == 2  # only the two automatic chase e-mails


# --- budget ---------------------------------------------------------------------------------------


def test_a_tiny_budget_stops_new_llm_work_and_says_which_limit(where):
    r = start_run(demo_config("b1", budget_tokens=6000), **where)
    rep = report(where, "b1")
    assert rep["budget"]["stopped"] and rep["budget"]["stop_reason"].startswith("tokens at")
    stopped = [c for c in rep["cases_detail"] if c["status"] == "needs_human"]
    assert stopped and all("SAFE-BUDGET-01" in c["fired_rules"] for c in stopped)
    assert all(c["outcome"] != "auto_allow" for c in stopped)
    repo = Repo.open(where["db_path"])
    try:
        cases = {c.case_key: c for c in repo.list_cases("SEED", "2026-10")}
        assert all("budget_stop" in cases[c["case_key"]].flags for c in stopped)
        used = repo.conn.execute(
            "SELECT SUM(input_tokens + output_tokens) FROM llm_calls WHERE run_id='b1'"
        ).fetchone()[0]
    finally:
        repo.close()
    assert used < 6000  # no call crossed the limit
    assert r.status == "paused"  # the deterministic steps and the gate still ran
    assert "stopped: tokens at" in (where["runs_dir"] / "b1" / "report.md").read_text("utf-8")


# --- CLI ------------------------------------------------------------------------------------------


def test_cli_run_demo_then_resume(where):
    cli = CliRunner()
    args = ["--db", str(where["db_path"]), "--runs-dir", str(where["runs_dir"])]
    out = cli.invoke(app, ["run", "--demo", "--run-id", "c1", *args])
    assert out.exit_code == 0, out.output
    assert "run c1: paused" in out.output and "7 approvals waiting" in out.output
    decide_all(where["db_path"], "c1")
    out = cli.invoke(app, ["resume", "c1", *args])
    assert out.exit_code == 0 and "run c1: done" in out.output
    out = cli.invoke(app, ["resume", "c1", *args])
    assert out.exit_code == 1 and "nothing left to resume" in out.output


def test_cli_live_run_asks_first(where, tmp_path):
    month = ROOT / "data" / "seed" / "seed_data"
    out = CliRunner().invoke(
        app,
        ["run", "--data", str(month), "--period", "2026-10", "--live", "--db",
         str(where["db_path"]), "--runs-dir", str(where["runs_dir"])],
        input="n\n",
    )  # fmt: skip
    assert out.exit_code == 1 and "LIVE run" in out.output
    assert not where["db_path"].exists()  # nothing started


def test_cli_run_needs_a_month():
    out = CliRunner().invoke(app, ["run"])
    assert out.exit_code != 0


# --- PDFium on one thread (the Task 13 race) ------------------------------------------------------


def test_rendering_from_many_threads_gives_the_same_bytes():
    pdfs = sorted(INV.glob("*.pdf"))[:6]
    ref = {p: hashlib.sha256(render_page_png(p)).hexdigest() for p in pdfs}
    with ThreadPoolExecutor(4) as ex:
        got = list(ex.map(lambda p: (p, hashlib.sha256(render_page_png(p)).hexdigest()),
                          pdfs * 4))  # fmt: skip
    assert all(h == ref[p] for p, h in got)
