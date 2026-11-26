"""Spec 15.4: kill the process mid-run, resume in a new one, and get exactly the result of an
uninterrupted run, without paying twice for any model call that had finished.

Each run is a real subprocess (``python -m agent_orchestrator.cli``). The crash is
``ORCH_CRASH_AFTER_LLM_CALLS=N``: ``os._exit(1)`` right after the N-th paid call, with no
cleanup. The demo model appends one line per paid call to ``ORCH_FAKE_CALL_LOG``, which counts
calls across both processes. The seed month makes 35 calls: 17 invoice reads, then 8 rule
summaries and 10 plans. N=5 crashes while invoices are being read in parallel; N=24 while
cases are being planned in parallel.

A call still in flight in the other parallel branch when the process dies is lost with it and
made again after resume (at most one, with concurrency 2). That is the real cost of a crash;
the test measures it instead of hiding it.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_orchestrator.store.repo import Repo

ROOT = Path(__file__).resolve().parents[1]


def cli(args, env_extra, timeout=300):
    inherited = {k: v for k, v in os.environ.items() if not k.startswith("ORCH_")}
    env = {**inherited, **env_extra}
    return subprocess.run(
        [sys.executable, "-m", "agent_orchestrator.cli", *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout,
    )  # fmt: skip


def outcome(db: Path, run_id: str):
    """Everything a run decided, without timestamps or run ids."""
    repo = Repo.open(db)
    try:
        cases = sorted(
            (c.case_key, c.status, c.mismatch_type, tuple(c.flags), c.facts_hash)
            for c in repo.list_cases("SEED", "2026-10")
        )
        decisions = sorted(
            (r["case_key"], r["outcome"], r["action"], r["fired_rules"])
            for r in repo.conn.execute("SELECT * FROM decisions WHERE run_id = ?", (run_id,))
        )
        approvals = sorted((a.case_key, a.proposed, a.status) for a in repo.list_approvals(None))
        status = repo.get_run(run_id)["status"]
        paid = repo.conn.execute(
            "SELECT COUNT(*) FROM llm_calls WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
    finally:
        repo.close()
    return {"cases": cases, "decisions": decisions, "approvals": approvals, "status": status,
            "ledger_rows": paid}  # fmt: skip


def calls(log: Path) -> int:
    return len(log.read_text("utf-8").splitlines()) if log.exists() else 0


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    d = tmp_path_factory.mktemp("base")
    log = d / "calls.log"
    args = ["run", "--demo", "--run-id", "base", "--db", str(d / "o.sqlite"),
            "--runs-dir", str(d / "runs")]  # fmt: skip
    r = cli(args, {"ORCH_FAKE_CALL_LOG": str(log)})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "run base: paused" in r.stdout
    return outcome(d / "o.sqlite", "base"), calls(log)


@pytest.mark.parametrize("crash_after", [5, 24])
def test_crash_then_resume_equals_an_uninterrupted_run(tmp_path, uninterrupted, crash_after):
    base, base_calls = uninterrupted
    assert base_calls == 35
    db, runs, log = tmp_path / "o.sqlite", tmp_path / "runs", tmp_path / "calls.log"
    where = ["--db", str(db), "--runs-dir", str(runs)]
    env = {"ORCH_FAKE_CALL_LOG": str(log)}

    crashed = cli(["run", "--demo", "--run-id", "base", *where],
                  {**env, "ORCH_CRASH_AFTER_LLM_CALLS": str(crash_after)})  # fmt: skip
    assert crashed.returncode == 1, crashed.stdout + crashed.stderr
    assert "run base:" not in crashed.stdout  # it died before finishing
    at_crash = calls(log)
    # Two branches run in parallel: the other one's call may already be with the model when
    # the process dies. That call is paid, its answer is lost with the process, and it is made
    # again after resume. At most one per extra parallel branch (concurrency 2, so one).
    in_flight = at_crash - crash_after
    assert in_flight in (0, 1)

    resumed = cli(["resume", "base", *where], env)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "run base: paused" in resumed.stdout

    after = outcome(db, "base")
    assert after == base  # same cases, decisions, approvals, status and ledger rows
    # Every call that finished before the crash was answered from the ledger after it:
    # the only extra calls are the ones that were in flight.
    assert calls(log) == base_calls + in_flight
