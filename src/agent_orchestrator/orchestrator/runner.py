"""Start, pause and resume runs (spec 8.2). Kill-safe: everything needed to continue is in
SQLite (the checkpoint, the store, the paid-call ledger) plus ``runs/<run_id>/run.json``.

- ``start_run`` creates the run, invokes the graph and stops at the approval gate (``paused``)
  or the end (``done``). While paused, a partial report is written.
- ``resume_run`` rebuilds everything in a new process from ``run.json`` and continues from the
  last checkpoint. Finished nodes and finished parallel branches are not re-run; any model call
  that is repeated anyway is answered from the paid-call ledger for free.
- ``ORCH_CRASH_AFTER_LLM_CALLS=N`` (tests only) kills the process with ``os._exit(1)`` right
  after the N-th paid model call, once that call is stored in the ledger.

The demo (``orchestrator run --demo``) uses a scripted fake model: it "reads" each invoice by
returning the values of its book line, plus the PDF's own text as ``other_text`` (so the
planted note on the injection invoice is seen and flagged), cites the first passage it is
shown and proposes the first allowed action. No network, no cost.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_orchestrator.agents import extraction as extraction_agent
from agent_orchestrator.agents import planner as planner_agent
from agent_orchestrator.agents import research as research_agent
from agent_orchestrator.agents.intake import compute_input_hash, read_books
from agent_orchestrator.interfaces.adapters import load_guard, load_retriever
from agent_orchestrator.interfaces.stubs import SqliteCache
from agent_orchestrator.llm.budget import RunBudget, SpendLedger
from agent_orchestrator.llm.client import LLMClient, make_backend
from agent_orchestrator.llm.fake import FakeBackend
from agent_orchestrator.llm.ledger import PaidCallCache
from agent_orchestrator.llm.retry import Pacer
from agent_orchestrator.orchestrator.graph import (
    Deps,
    MonthInput,
    checkpointer,
    compile_graph,
    resume_command,
    write_run_report,
)
from agent_orchestrator.orchestrator.state import initial_state
from agent_orchestrator.policy.engine import default_policy, policy_version
from agent_orchestrator.render import pdf_text_layer, render_page_png
from agent_orchestrator.store.db import connect
from agent_orchestrator.store.repo import Repo
from agent_orchestrator.tracing import Tracer

ROOT = Path(__file__).resolve().parents[3]
SEED_DIR = ROOT / "data" / "seed" / "seed_data"
DEFAULT_DB = ROOT / "data" / "orchestrator.sqlite"
DEFAULT_RUNS = ROOT / "runs"
CRASH_ENV = "ORCH_CRASH_AFTER_LLM_CALLS"
CALL_LOG_ENV = "ORCH_FAKE_CALL_LOG"  # tests: the demo model appends one line per paid call
RESUME = object()  # _drive: continue from the checkpoint instead of starting


# --- the store, one connection per thread ---------------------------------------------------------


class ThreadRepo:
    """A ``Repo`` per thread on one SQLite file: graph branches run in worker threads, and a
    sqlite3 connection must not be shared between threads."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        Repo.open(self.path).close()  # create or migrate the schema once
        self._local = threading.local()
        self._all: list[Repo] = []
        self._lock = threading.Lock()

    def _repo(self) -> Repo:
        repo = getattr(self._local, "repo", None)
        if repo is None:
            repo = Repo(connect(self.path, check_same_thread=False))
            self._local.repo = repo
            with self._lock:
                self._all.append(repo)
        return repo

    def __getattr__(self, name: str) -> Any:
        return getattr(self._repo(), name)

    def close(self) -> None:
        with self._lock:
            for repo in self._all:
                repo.close()
            self._all.clear()
        self._local = threading.local()


def run_spend(repo: Any, run_id: str) -> tuple[float, int]:
    """USD and tokens this run has paid so far, from the ledger (survives a crash)."""
    row = repo.conn.execute(
        "SELECT COALESCE(SUM(usd), 0), COALESCE(SUM(input_tokens + output_tokens), 0)"
        " FROM llm_calls WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    return float(row[0]), int(row[1])


# --- run settings (saved so a new process can resume) ---------------------------------------------


@dataclass
class RunConfig:
    run_id: str
    business_id: str
    period: str
    books_path: str
    portal_path: str
    invoices_dir: str | None
    mode: str = "fake"  # "fake" (demo), "replay" or "live"
    provider: str = "bedrock"
    use_rag: bool = False
    budget_usd: float = 1.0
    budget_tokens: int = 500_000
    budget_seconds: float = 900.0
    max_concurrency: int = 2
    recording: str | None = None  # replay/live: data/recordings/<name>.jsonl

    def month(self) -> MonthInput:
        return MonthInput(
            business_id=self.business_id,
            period=self.period,
            books_path=Path(self.books_path),
            portal_path=Path(self.portal_path),
            invoices_dir=Path(self.invoices_dir) if self.invoices_dir else None,
        )

    def save(self, runs_dir: Path) -> Path:
        path = runs_dir / self.run_id / "run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", "utf-8", newline="\n")
        return path

    @classmethod
    def load(cls, runs_dir: Path, run_id: str) -> RunConfig:
        path = runs_dir / run_id / "run.json"
        if not path.is_file():
            raise FileNotFoundError(f"no run.json for run {run_id} in {runs_dir}")
        return cls(**json.loads(path.read_text("utf-8")))


def demo_config(run_id: str | None = None, seed_dir: Path = SEED_DIR, **kw: Any) -> RunConfig:
    return RunConfig(
        run_id=run_id or new_run_id("SEED", "2026-10"),
        business_id="SEED",
        period="2026-10",
        books_path=str(seed_dir / "purchase_register.csv"),
        portal_path=str(seed_dir / "gstr2b_like.json"),
        invoices_dir=str(seed_dir / "invoices"),
        **kw,
    )


def new_run_id(business_id: str, period: str) -> str:
    return f"{business_id}-{period}-{uuid.uuid4().hex[:8]}"


# --- the demo model -------------------------------------------------------------------------------

_IDS = re.compile(r"(?:Passage ids you may cite|Rule passage ids you may cite): (.+)")


def demo_script(month: MonthInput) -> Any:
    """A deterministic stand-in for every model the graph calls (see the module docstring)."""
    books, _ = read_books(
        month.books_path, month.invoices_dir.parent if month.invoices_dir else None
    )
    base = month.invoices_dir.parent if month.invoices_dir else None
    answers: dict[str, str] = {}  # sha256 of the rendered page -> extraction answer
    for b in books:
        if not b.pdf_file or base is None or not (base / b.pdf_file).is_file():
            continue
        pdf = base / b.pdf_file
        image = hashlib.sha256(render_page_png(pdf)).hexdigest()
        if image not in answers:
            answers[image] = _extraction_answer(b, pdf_text_layer(pdf))
    log = os.environ.get(CALL_LOG_ENV)
    lock = threading.Lock()
    policy = default_policy()

    def answer(system: str, messages: Any) -> str:
        if log:
            with lock, open(log, "a", encoding="utf-8") as f:
                f.write(system[:20].replace("\n", " ") + "\n")
        user = messages[0].parts
        if system == extraction_agent.SYSTEM_PROMPT:
            image = next(p for p in user if not isinstance(p, str))
            return answers[hashlib.sha256(image.data).hexdigest()]
        text = next(p for p in user if isinstance(p, str))
        ids = [x.strip() for x in _IDS.search(text).group(1).split(",")]
        if system == research_agent.SYSTEM_PROMPT:
            return json.dumps({"summary": f"See the passage [{ids[0]}].", "snippet_ids": ids[:1]})
        if system == planner_agent.SYSTEM_PROMPT:
            facts = json.loads(text.split("produced by code):", 1)[1].split("Rule passage ids")[0])
            return json.dumps({
                # The policy file lists each type's actions in order of preference; the
                # facts JSON has sorted keys, so its own order would be alphabetical.
                "action": policy.allowed_actions[facts["mismatch_type"]][0],
                "reason": "Demo answer: the policy's first allowed action for this type.",
                "evidence_refs": facts["evidence_refs_available"][:1] or ["none"],
                "rule_refs": ids[:1],
            })  # fmt: skip
        raise ValueError("the demo model has no answer for this prompt")

    return answer


def _extraction_answer(b: Any, text_layer: str) -> str:
    def f(v: Any) -> dict[str, Any]:
        return {"value": v, "confidence": 0.97}

    def rs(paise: int) -> str | None:
        return f"Rs {paise / 100:,.2f}" if paise else None

    total = b.taxable_paise + b.igst_paise + b.cgst_paise + b.sgst_paise
    return json.dumps({
        "supplier_name": f(b.supplier_name), "supplier_gstin": f(b.supplier_gstin),
        "buyer_gstin": f(None), "invoice_no": f(b.invoice_no),
        "invoice_date": f(b.date.isoformat()), "taxable_value": f(rs(b.taxable_paise)),
        "rate": f(f"{b.rate}%"), "igst": f(rs(b.igst_paise)), "cgst": f(rs(b.cgst_paise)),
        "sgst": f(rs(b.sgst_paise)), "total": f(rs(total)), "other_text": text_layer,
    })  # fmt: skip


# --- the crash switch (tests) ---------------------------------------------------------------------


class _CrashCounter:
    def __init__(self, after: int) -> None:
        self.after, self.count, self.lock = after, 0, threading.Lock()

    def hit(self) -> None:
        with self.lock:
            self.count += 1
            if self.count >= self.after:
                os._exit(1)  # no cleanup, no flush: as close to pulling the plug as Python gets


class _CrashingClient:
    """Same surface as LLMClient. Counts paid calls only (ledger and replay answers are free)."""

    def __init__(self, inner: LLMClient, counter: _CrashCounter) -> None:
        self.inner, self.counter = inner, counter

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    def complete(self, *args: Any, **kwargs: Any):
        result = self.inner.complete(*args, **kwargs)
        if result.source in ("fake", "live"):
            self.counter.hit()
        return result


# --- building and running -------------------------------------------------------------------------


@dataclass
class RunResult:
    run_id: str
    status: str  # "paused" or "done"
    pending: list[str]
    report: Path


def _clients(cfg: RunConfig, repo: Any, budget: RunBudget, tracer: Tracer) -> dict[str, Any]:
    cache = PaidCallCache(repo)
    common = dict(run_budget=budget, cache=cache, tracer=tracer, run_id=cfg.run_id)
    if cfg.mode == "fake":
        backend = FakeBackend(demo_script(cfg.month()), model_id="fake")
        one = LLMClient(backend, mode="fake", **common)
        clients = {"extract": one, "text": one, "escalation": None}
    else:
        from agent_orchestrator.llm.recording import Recorder

        recorder = Recorder(ROOT / "data" / "recordings" / f"{cfg.recording or cfg.run_id}.jsonl")
        spend = SpendLedger(ROOT / ".spend" / "ledger.jsonl", caps={
            "bedrock": float(os.environ.get("LIVE_BUDGET_USD", "5")),
            "anthropic": float(os.environ.get("ANTHROPIC_BUDGET_USD", "3")),
        })  # fmt: skip
        pacer = Pacer(float(os.environ.get("LLM_REQUESTS_PER_SECOND", "2")))
        extra = dict(spend=spend, recorder=recorder, pacer=pacer, mode=cfg.mode)

        def client(env: str, default: str) -> LLMClient:
            model = (
                os.environ.get("ANTHROPIC_MODEL_ID", "claude-haiku-5-5")
                if cfg.provider == "anthropic"
                else os.environ.get(env, default)
            )
            return LLMClient(make_backend(cfg.provider, model), **common, **extra)

        lite = "global.amazon.nova-2-lite-v1:0"
        clients = {
            "extract": client("BEDROCK_VISION_MODEL_ID", lite),
            "text": client("BEDROCK_TEXT_MODEL_ID", lite),
            "escalation": None
            if cfg.provider == "anthropic"
            else client("BEDROCK_VISION_ESCALATION_MODEL_ID", "apac.amazon.nova-pro-v1:0"),
        }
    crash = os.environ.get(CRASH_ENV, "").strip()
    if crash:
        counter = _CrashCounter(int(crash))
        clients = {k: _CrashingClient(v, counter) if v else None for k, v in clients.items()}
    return clients


def build_deps(cfg: RunConfig, repo: Any, runs_dir: Path, db_path: Path) -> Deps:
    budget = RunBudget(usd=cfg.budget_usd, tokens=cfg.budget_tokens, seconds=cfg.budget_seconds)
    usd, tokens = run_spend(repo, cfg.run_id)
    if usd or tokens:
        budget.record(usd, tokens)  # a resumed run continues the same budget
    tracer = Tracer(cfg.run_id, repo, span_prefix=f"p{uuid.uuid4().hex[:6]}-")
    clients = _clients(cfg, repo, budget, tracer)
    guard, _ = load_guard()
    retriever, _ = load_retriever(prefer_adapter=cfg.use_rag)
    return Deps(
        run_id=cfg.run_id, month=cfg.month(), repo=repo, guard=guard, retriever=retriever,
        cache=SqliteCache(db_path), extract_client=clients["extract"],
        text_client=clients["text"], escalation_client=clients["escalation"], budget=budget,
        tracer=tracer, policy=default_policy(), runs_dir=runs_dir,
    )  # fmt: skip


def _graph_config(cfg: RunConfig) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": cfg.run_id},  # one checkpoint thread per run
        "max_concurrency": cfg.max_concurrency,
        "recursion_limit": 100,
    }


def _drive(cfg: RunConfig, db_path: Path, runs_dir: Path, graph_input: Any) -> RunResult:
    repo = ThreadRepo(db_path)
    saver, saver_conn = checkpointer(db_path)
    try:
        deps = build_deps(cfg, repo, runs_dir, db_path)
        app = compile_graph(deps, saver)
        config = _graph_config(cfg)
        if graph_input is RESUME:
            snap = app.get_state(config)
            if not snap.next:
                raise ValueError(f"run {cfg.run_id} has nothing left to resume")
            # At the gate: "go" (the gate re-reads the database). After a crash: no input,
            # so the graph continues from its last checkpoint.
            graph_input = resume_command() if snap.interrupts else None
        try:
            out = app.invoke(graph_input, config, durability="sync")
        except BaseException:
            repo.update_run(cfg.run_id, status="failed")
            raise
        interrupts = out.get("__interrupt__") or []
        if interrupts:
            pending = list(interrupts[0].value.get("pending", []))
            state = app.get_state(config).values
            paths = write_run_report(deps, state, status="paused")  # the partial report
            return RunResult(cfg.run_id, "paused", pending, paths["md"])
        return RunResult(cfg.run_id, "done", [], runs_dir / cfg.run_id / "report.md")
    finally:
        saver_conn.close()
        repo.close()


def start_run(cfg: RunConfig, *, db_path: Path = DEFAULT_DB, runs_dir: Path = DEFAULT_RUNS):
    db_path, runs_dir = Path(db_path), Path(runs_dir)
    cfg.save(runs_dir)
    repo = ThreadRepo(db_path)
    try:
        m = cfg.month()
        repo.create_run(
            cfg.run_id, cfg.business_id, cfg.period,
            compute_input_hash(m.books_path, m.portal_path, m.invoices_dir),
            policy_version(default_policy()),
        )  # fmt: skip
    finally:
        repo.close()
    return _drive(cfg, db_path, runs_dir, initial_state(
        run_id=cfg.run_id, business_id=cfg.business_id, period=cfg.period,
        policy_version=policy_version(default_policy()),
    ))  # fmt: skip


def resume_run(run_id: str, *, db_path: Path = DEFAULT_DB, runs_dir: Path = DEFAULT_RUNS):
    """Continue a paused or crashed run in this process.

    After a crash, the graph continues from its last checkpoint (``None`` input). At the
    approval gate, the resume value only says "go"; the gate itself checks the database.
    """
    db_path, runs_dir = Path(db_path), Path(runs_dir)
    cfg = RunConfig.load(runs_dir, run_id)
    repo = ThreadRepo(db_path)
    try:
        repo.update_run(run_id, status="running")
    finally:
        repo.close()
    return _drive(cfg, db_path, runs_dir, RESUME)
