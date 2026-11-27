"""The orchestrator graph (spec 8.1). The only module that imports LangGraph.

```
START → intake → fanout_extract ─Send(per PDF)→ extract_one ─→ join_extract
join_extract → match → route_cases
route_cases: clean matches → rule_propose (no LLM) → decide
             others        ─Send(per case)→ research_and_plan → decide
decide → approval_gate if any case needs a person, else communicate
approval_gate (interrupt) → apply_approvals → communicate → report → END
```

Nodes only call ``agents/*`` functions, the policy engine and the store; the decisions are
made there. Two rules keep a resumed run correct:

- **Parallel branches** (``extract_one``, ``research_and_plan``) return only fields with a
  reducer, so no branch's output is lost (``state.BRANCH_WRITES``, tested).
- **The approval gate writes nothing before ``interrupt()``**: on resume LangGraph re-runs the
  node from the top. The database is the source of truth: the resume payload only says "go",
  and the gate interrupts again while any approval is still pending.
"""

from __future__ import annotations

import sqlite3
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from pydantic import BaseModel

from agent_orchestrator.agents.communicator import eligible
from agent_orchestrator.agents.extraction import extract_invoice
from agent_orchestrator.agents.intake import run_intake
from agent_orchestrator.agents.matcher import match
from agent_orchestrator.agents.planner import plan_case
from agent_orchestrator.agents.reporter import (
    build_report,
    record_in_store,
    summarise,
    write_report,
)
from agent_orchestrator.agents.research import CLEAN_TYPES, research_rules
from agent_orchestrator.interfaces.protocols import Cache, Guard, RuleRetriever
from agent_orchestrator.llm.budget import RunBudget
from agent_orchestrator.models import ApprovalRecord, Case, Decision, ProposedAction
from agent_orchestrator.orchestrator.state import (
    CHECKPOINT_TYPES,
    ExtractTask,
    GraphState,
    PlanNote,
    PlanTask,
)
from agent_orchestrator.policy.engine import decide
from agent_orchestrator.policy.schema import Policy
from agent_orchestrator.store.idempotency import approval_key, outbox_key
from agent_orchestrator.tools import toolfns
from agent_orchestrator.tracing import Tracer

# LLM span purposes -> the agent they belong to, for the cost-per-agent table.
PURPOSE_AGENT = {
    "extract": "extraction",
    "research": "research",
    "plan": "planner",
    "email": "communicator",
    "summary": "reporter",
}


class ApprovalResume(BaseModel):
    """What a resume sends to the gate. It carries no decisions: those are in the database."""

    go: bool = True


@dataclass(frozen=True)
class MonthInput:
    business_id: str
    period: str
    books_path: Path
    portal_path: Path
    invoices_dir: Path | None = None
    expected_buyer_gstin: str | None = None


@dataclass
class Deps:
    """Everything the nodes use. Rebuilt in each process; never stored in the checkpoint."""

    run_id: str
    month: MonthInput
    repo: Any  # store.repo.Repo, or the runner's per-thread wrapper
    guard: Guard
    retriever: RuleRetriever
    cache: Cache
    extract_client: Any  # LLMClient (or a wrapper with the same complete/model_id)
    text_client: Any
    budget: RunBudget
    tracer: Tracer
    policy: Policy
    runs_dir: Path
    escalation_client: Any = None
    email_client: Any = None
    summary_client: Any = None
    languages: dict[str, str] = field(default_factory=dict)
    # One lock per mismatch type: two branches never research the same type at once, so the
    # second one finds the first one's cached summary (and the call count is deterministic).
    research_locks: dict[str, threading.Lock] = field(
        default_factory=lambda: defaultdict(threading.Lock)
    )

    @property
    def pdf_base(self) -> Path | None:
        """The books' ``pdf_file`` paths ("invoices/X.pdf") are relative to this folder."""
        return self.month.invoices_dir.parent if self.month.invoices_dir is not None else None


def _by_pdf(state: GraphState) -> dict[str, Any]:
    return {e.pdf_file: e for e in state.get("extractions", [])}


def _last_by_case(items: list[Any]) -> dict[str, Any]:
    return {x.case_key: x for x in items}


def build_graph(deps: Deps) -> StateGraph:
    t = deps.tracer

    # --- intake and extraction --------------------------------------------------------------

    def intake(state: GraphState) -> dict[str, Any]:
        m = deps.month
        with t.span("intake") as s:
            r = run_intake(m.books_path, m.portal_path, m.period, invoices_dir=m.invoices_dir,
                           expected_buyer_gstin=m.expected_buyer_gstin)  # fmt: skip
            s.set_output(f"{len(r.books)} book lines, {len(r.portal)} 2B rows, "
                         f"{len(r.dq_issues)} data-quality issues")  # fmt: skip
        return {
            "books": r.books,
            "portal": r.portal,
            "dq_issues": r.dq_issues,
            "input_hash": r.input_hash,
            "buyer_gstin": r.buyer_gstin,
        }

    def fanout_extract(state: GraphState) -> dict[str, Any]:
        return {}

    def extract_sends(state: GraphState) -> list[Send] | str:
        base = deps.pdf_base
        if base is None:
            return "join_extract"
        pdfs = list(dict.fromkeys(b.pdf_file for b in state["books"] if b.pdf_file))
        sends = [
            Send("extract_one", ExtractTask(pdf_file=p, buyer_gstin=state["buyer_gstin"]))
            for p in pdfs
            if (base / p).is_file()
        ]
        return sends or "join_extract"

    def extract_one(task: ExtractTask) -> dict[str, Any]:
        with t.span("extraction", task["pdf_file"]) as s:
            ex = extract_invoice(
                deps.pdf_base / task["pdf_file"],
                client=deps.extract_client,
                guard=deps.guard,
                buyer_gstin=task["buyer_gstin"],
                escalation_client=deps.escalation_client,
                pdf_file=task["pdf_file"],
            )
            s.set_output(f"min confidence {ex.min_critical_confidence:.2f}; "
                         f"injection {ex.injection.flagged}; checks {len(ex.checks)}")  # fmt: skip
        return {"extractions": [ex]}

    def join_extract(state: GraphState) -> dict[str, Any]:
        return {}

    # --- matching and planning --------------------------------------------------------------

    def match_node(state: GraphState) -> dict[str, Any]:
        m = deps.month
        with t.span("matcher") as s:
            cases = match(state["books"], state["portal"], m.business_id, m.period, _by_pdf(state))
            for c in cases:
                deps.repo.upsert_case(deps.run_id, c)
            s.set_output(f"{len(cases)} cases")
        return {"cases": cases}

    def route_cases(state: GraphState) -> dict[str, Any]:
        return {}

    def plan_sends(state: GraphState) -> list[Send | str]:
        exs = _by_pdf(state)
        sends: list[Send | str] = [
            Send("research_and_plan", PlanTask(case=c, extraction=exs.get(c.pdf_file or "")))
            for c in state["cases"]
            if c.mismatch_type not in CLEAN_TYPES
        ]
        return [*sends, "rule_propose"]

    def research_and_plan(task: PlanTask) -> dict[str, Any]:
        c = task["case"]
        with deps.research_locks[c.mismatch_type], t.span("research", case_key=c.case_key) as s:
            rr = research_rules(c.mismatch_type, retriever=deps.retriever, cache=deps.cache,
                                client=deps.text_client, guard=deps.guard)  # fmt: skip
            s.set_output(f"{len(rr.snippets)} snippets from {rr.retriever}; cached "
                         f"{rr.research.cached}; {rr.note or 'ok'}")  # fmt: skip
        with t.span("planner", case_key=c.case_key) as s:
            out = plan_case(c, research=rr, client=deps.text_client, guard=deps.guard,
                            extraction=task["extraction"], policy=deps.policy)  # fmt: skip
            s.set_output(f"{out.proposal.proposer} proposes {out.proposal.action}")
        note = PlanNote(case_key=c.case_key, fallback=out.fallback, flags=list(out.flags))
        return {"research": [rr.research], "proposals": [out.proposal], "plan_notes": [note]}

    def rule_propose(state: GraphState) -> dict[str, Any]:
        exs = _by_pdf(state)
        proposals = [
            plan_case(c, research=None, client=deps.text_client, guard=deps.guard,
                      extraction=exs.get(c.pdf_file or ""), policy=deps.policy).proposal
            for c in state["cases"]
            if c.mismatch_type in CLEAN_TYPES
        ]  # fmt: skip
        return {"proposals": proposals}

    # --- deciding ---------------------------------------------------------------------------

    def decide_node(state: GraphState) -> dict[str, Any]:
        proposals = _last_by_case(state.get("proposals", []))
        notes = _last_by_case(state.get("plan_notes", []))
        exs = _by_pdf(state)
        m = deps.month
        cases, decisions, approval_ids = [], [], []
        with t.span("policy") as s:
            for c in state["cases"]:
                note = notes.get(c.case_key)
                p = proposals.get(c.case_key)
                if p is None:  # cannot happen in a finished graph; fail closed if it does
                    p = ProposedAction(case_key=c.case_key, action="escalate_human",
                                       reason="No proposal was made.", proposer="rule")  # fmt: skip
                    note = PlanNote(case_key=c.case_key, fallback=True, flags=["planner_failed"])
                ex = exs.get(c.pdf_file) if c.pdf_file else None
                stopped = bool(
                    (note and "budget_stop" in note.flags) or (ex and "budget_stop" in ex.checks)
                )
                flags = (
                    set(c.flags)
                    | set(note.flags if note else ())
                    | ({"budget_stop"} if stopped else set())
                    | ({"injection"} if ex is not None and ex.injection.flagged else set())
                )
                c = c.model_copy(update={
                    "flags": sorted(flags), "status": "needs_human" if stopped else c.status,
                })  # fmt: skip
                d = decide(c, p, extraction=ex, policy=deps.policy, budget_stopped=stopped,
                           fallback=bool(note and note.fallback))  # fmt: skip
                if d.outcome == "block":  # nothing automatic; a person must look at it
                    c = c.model_copy(update={"status": "needs_human"})
                deps.repo.record_decision(deps.run_id, d)
                deps.repo.upsert_case(
                    deps.run_id, c, action=d.action if d.outcome == "auto_allow" else None
                )
                if d.outcome == "require_approval":
                    approval_ids.append(_request_approval(c, d, m).approval_id)
                cases.append(c)
                decisions.append(d)
            outcomes = defaultdict(int)
            for d in decisions:
                outcomes[d.outcome] += 1
            s.set_output(", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())))
        return {"cases": cases, "decisions": decisions, "approval_ids": approval_ids}

    def _request_approval(c: Case, d: Decision, m: MonthInput) -> ApprovalRecord:
        """Through the write tool, which checks the stored decision. Idempotent: the same case,
        action and facts find the existing row (spec 9.2), even one a person already decided
        in an earlier run of the same month."""
        out = toolfns.request_approval(
            tool_ctx(),
            idempotency_key=approval_key(m.business_id, m.period, c.case_key, d.action,
                                         c.facts_hash),
            run_id=deps.run_id,
            case_key=c.case_key,
            decision_ref=toolfns.decision_ref(deps.run_id, d),
            allowed_options=list(deps.policy.allowed_actions[c.mismatch_type]),
        )  # fmt: skip
        return out["approval"]

    def tool_ctx() -> toolfns.ToolContext:
        m = deps.month
        return toolfns.ToolContext(repo=deps.repo, guard=deps.guard, business_id=m.business_id,
                                   period=m.period, email_client=deps.email_client)  # fmt: skip

    def after_decide(state: GraphState) -> str:
        # Any case that needs a person goes through the gate, even when its approval was
        # already decided in an earlier run of the month (then the gate passes straight on,
        # and apply_approvals still applies that decision).
        return "approval_gate" if state.get("approval_ids") else "communicate"

    def _pending(state: GraphState) -> list[str]:
        ids = state.get("approval_ids", [])
        return [i for i in ids if deps.repo.get_approval(i).status == "pending"]

    # --- the person, then the output --------------------------------------------------------

    def approval_gate(state: GraphState) -> dict[str, Any]:
        # Nothing is written before interrupt(): this node re-runs from the top on resume.
        waiting = _pending(state)
        while waiting:
            interrupt(
                {"run_id": deps.run_id, "pending": waiting},
                response_schema=ApprovalResume.model_json_schema(),
            )
            waiting = _pending(state)  # the database decides, not the resume payload
        return {
            "approvals_resolved": [deps.repo.get_approval(i) for i in state.get("approval_ids", [])]
        }

    def apply_approvals(state: GraphState) -> dict[str, Any]:
        by_case = {c.case_key: c for c in state["cases"]}
        with t.span("approvals") as s:
            for a in state.get("approvals_resolved", []):
                if a.case_key in by_case:
                    deps.repo.upsert_case(deps.run_id, by_case[a.case_key], action=a.final_action)
            s.set_output(f"{len(state.get('approvals_resolved', []))} approvals applied")
        return {}

    def communicate(state: GraphState) -> dict[str, Any]:
        books = {b.line_no: b for b in state["books"]}
        portal = {p.index: p for p in state["portal"]}
        decisions = _last_by_case(state.get("decisions", []))
        approvals = _last_by_case(state.get("approvals_resolved", []))
        drafts = []
        ctx = tool_ctx()
        with t.span("communicator") as s:
            for c in state["cases"]:
                d, a = decisions.get(c.case_key), approvals.get(c.case_key)
                if d is None or not eligible(d, a):
                    continue  # not asking the tool spares a refusal row for every other case
                side = portal.get(c.portal_index) if c.portal_index is not None else None
                side = side or books[c.book_lines[0]]
                action = a.final_action if a is not None and a.status != "pending" else d.action
                out = toolfns.write_draft_email(
                    ctx,
                    idempotency_key=outbox_key(c.business_id, c.period, c.case_key, action,
                                               side.supplier_gstin),
                    run_id=deps.run_id,
                    case_key=c.case_key,
                    decision_ref=toolfns.decision_ref(deps.run_id, d),
                    language=deps.languages.get(side.supplier_gstin, "en"),
                    books=books,
                    portal=portal,
                )  # fmt: skip
                drafts.append(out["draft"])
            s.set_output(f"{len(drafts)} drafts (none sent)")
        return {"drafts": drafts}

    def report(state: GraphState) -> dict[str, Any]:
        with t.span("reporter"):
            write_run_report(deps, state, status="done")
        return {"budget": deps.budget.snapshot()}

    g = StateGraph(GraphState)
    for name, fn in [
        ("intake", intake),
        ("fanout_extract", fanout_extract),
        ("extract_one", extract_one),
        ("join_extract", join_extract),
        ("match", match_node),
        ("route_cases", route_cases),
        ("research_and_plan", research_and_plan),
        ("rule_propose", rule_propose),
        ("decide", decide_node),
        ("approval_gate", approval_gate),
        ("apply_approvals", apply_approvals),
        ("communicate", communicate),
        ("report", report),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    g.add_edge("intake", "fanout_extract")
    g.add_conditional_edges("fanout_extract", extract_sends, ["extract_one", "join_extract"])
    g.add_edge("extract_one", "join_extract")
    g.add_edge("join_extract", "match")
    g.add_edge("match", "route_cases")
    g.add_conditional_edges("route_cases", plan_sends, ["research_and_plan", "rule_propose"])
    g.add_edge("research_and_plan", "decide")
    g.add_edge("rule_propose", "decide")
    g.add_conditional_edges("decide", after_decide, ["approval_gate", "communicate"])
    g.add_edge("approval_gate", "apply_approvals")
    g.add_edge("apply_approvals", "communicate")
    g.add_edge("communicate", "report")
    g.add_edge("report", END)
    return g


def checkpointer(db_path: Path | str) -> tuple[SqliteSaver, sqlite3.Connection]:
    """LangGraph's checkpoint tables in the same SQLite file as the store (spec 8.2)."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
    return SqliteSaver(conn, serde=serde), conn


def compile_graph(deps: Deps, saver: SqliteSaver):
    return build_graph(deps).compile(checkpointer=saver)


def resume_command() -> Command:
    """The value that wakes the approval gate. It only says "go"; decisions are in SQLite."""
    return Command(resume=ApprovalResume().model_dump())


# --- reports (also written by the runner when the run pauses) -------------------------------------


def cost_by_agent(spans: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Spans of every process of this run (from the store), summed per agent. Model calls are
    counted under the agent whose purpose they served."""
    out: dict[str, dict[str, float]] = defaultdict(
        lambda: {"spans": 0, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}
    )
    for s in spans:
        if s["agent"] == "llm":
            agent = PURPOSE_AGENT.get(s["name"].split("-")[0], s["name"])
            row = out[agent]
            row["llm_calls"] += 1
            row["input_tokens"] += s["input_tokens"]
            row["output_tokens"] += s["output_tokens"]
            row["usd"] += s["usd"]
        else:
            out[s["agent"]]["spans"] += 1
    return {k: {**v, "usd": round(v["usd"], 6)} for k, v in sorted(out.items())}


def write_run_report(deps: Deps, state: GraphState, *, status: str) -> dict[str, Path]:
    """Build and write the report for the state as it is now (partial while paused)."""
    m = deps.month
    spans = deps.repo.list_spans(deps.run_id)
    approvals = [deps.repo.get_approval(i) for i in state.get("approval_ids", [])]
    budget = deps.budget.snapshot()
    rep = build_report(
        run_id=deps.run_id, business_id=m.business_id, period=m.period,
        cases=state.get("cases", []), decisions=state.get("decisions", []),
        approvals=approvals, books=state.get("books", []), portal=state.get("portal", []),
        dq_issues=state.get("dq_issues", []), drafts=state.get("drafts", []), budget=budget,
        cost_by_agent=cost_by_agent(spans), policy=deps.policy,
    )  # fmt: skip
    rep["status"] = status
    summary = None
    if deps.summary_client is not None and status == "done":
        summary = summarise(rep, client=deps.summary_client, guard=deps.guard)
    trace = {"run_id": deps.run_id, "spans": spans, "by_agent": rep["cost"]["by_agent"]}
    paths = write_report(deps.runs_dir / deps.run_id, rep, summary=summary, trace=trace)
    record_in_store(deps.repo, rep, status=status)
    return paths
