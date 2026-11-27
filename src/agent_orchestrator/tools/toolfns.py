"""The tool functions (spec 10), shared by the graph, the CLI, the web app and the MCP server,
so every rule is enforced in one place.

**Write tools** (an agent may call them):
- ``request_approval``: refuses unless the stored ``Decision`` for that case has outcome
  ``require_approval``.
- ``write_draft_email``: refuses unless the decision is ``auto_allow`` or a person approved
  or edited the case to a chase/verify action. The draft is built here from the run's own
  books and GSTR-2B, never from text the caller passes in.

Both check that the caller names the exact decision (``decision_ref``) and the deterministic
idempotency key (spec 9.2). A second identical call returns the first row (``created: False``).
A refused call writes an ``audit_log`` row and counts in ``REFUSED_WRITES``.

**Human actions** (``approve``, ``edit``, ``reject``) are for the CLI and the web UI only, as
the person. No tool approves anything. Each is audited (who, when, before -> after, reason),
and edits and rejections are appended to ``eval/human_feedback.jsonl`` (gitignored).
"""

from __future__ import annotations

import datetime as dt
import getpass
import hashlib
import json
import os
import threading
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_orchestrator.agents.communicator import DRAFT_ACTIONS, draft_email, eligible
from agent_orchestrator.models import ApprovalRecord, BookRecord, Decision, PortalRecord
from agent_orchestrator.store.idempotency import approval_key, outbox_key

ROOT = Path(__file__).resolve().parents[3]
FEEDBACK_FILE = ROOT / "eval" / "human_feedback.jsonl"
FEEDBACK_ENV = "ORCH_FEEDBACK_FILE"  # tests point it elsewhere
REFUSED_WRITES: Counter[str] = Counter()  # tool name -> refused calls (exported in Task 16)
_feedback_lock = threading.Lock()


class ToolRefused(PermissionError):
    """A write tool refused the call. The reason names the rule, never document text."""

    def __init__(self, tool: str, reason: str) -> None:
        super().__init__(f"{tool} refused: {reason}")
        self.tool, self.reason = tool, reason


class HumanActionError(ValueError):
    """An approve/edit/reject that is not allowed (wrong state, action or missing reason)."""


@dataclass
class ToolContext:
    repo: Any  # store.repo.Repo or the runner's ThreadRepo
    guard: Any
    business_id: str
    period: str
    email_client: Any = None
    # run_id -> (books by line, portal by index); used when the caller has no state at hand
    month_loader: (
        Callable[[str], tuple[Mapping[int, BookRecord], Mapping[int, PortalRecord]]] | None
    ) = None


def decision_ref(run_id: str, d: Decision) -> str:
    """Names one exact decision: the run, the case, the verdict and the rules behind it."""
    text = "|".join([run_id, d.case_key, d.outcome, d.action, ",".join(d.fired_rules),
                     d.policy_version])  # fmt: skip
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _refuse(ctx: ToolContext, tool: str, run_id: str, case_key: str, reason: str) -> None:
    REFUSED_WRITES[tool] += 1
    ctx.repo.append_audit("tool", f"refused:{tool}", run_id, case_key, {"reason": reason})
    raise ToolRefused(tool, reason)


def _decision(ctx: ToolContext, tool: str, run_id: str, case_key: str, ref: str) -> Decision:
    d = ctx.repo.get_decision(run_id, case_key)
    if d is None:
        _refuse(ctx, tool, run_id, case_key, "no decision exists for this case in this run")
    if decision_ref(run_id, d) != ref:
        _refuse(ctx, tool, run_id, case_key, "decision_ref does not match the stored decision")
    return d


def latest_approval(ctx: ToolContext, case_key: str) -> ApprovalRecord | None:
    """The case's approval for its current facts, from this run or an earlier run of the
    same month (a re-run reuses a decided approval instead of asking again)."""
    case = ctx.repo.get_case(ctx.business_id, ctx.period, case_key)
    rows = ctx.repo.conn.execute(
        "SELECT approval_id FROM approvals WHERE business_id=? AND period=? AND case_key=?"
        " ORDER BY created_at, approval_id",
        (ctx.business_id, ctx.period, case_key),
    ).fetchall()
    records = [ctx.repo.get_approval(r[0]) for r in rows]
    if case is not None:  # only the approval asked about these exact facts counts
        records = [
            a for a in records
            if a.idempotency_key == approval_key(ctx.business_id, ctx.period, case_key,
                                                 a.proposed, case.facts_hash)
        ]  # fmt: skip
    decided = [a for a in records if a.status != "pending"]
    return (decided or records or [None])[-1]


# --- write tools ----------------------------------------------------------------------------------


def request_approval(
    ctx: ToolContext,
    *,
    idempotency_key: str,
    run_id: str,
    case_key: str,
    decision_ref: str,
    proposed: str | None = None,
    allowed_options: list[str] | None = None,
) -> dict[str, Any]:
    """Ask a person to decide a case. Returns ``{"created": bool, "approval": ApprovalRecord}``."""
    tool = "request_approval"
    d = _decision(ctx, tool, run_id, case_key, decision_ref)
    if d.outcome != "require_approval":
        _refuse(ctx, tool, run_id, case_key, f"the decision is {d.outcome}, not require_approval")
    case = ctx.repo.get_case(ctx.business_id, ctx.period, case_key)
    if case is None:
        _refuse(ctx, tool, run_id, case_key, "unknown case")
    proposed = proposed or d.action
    if proposed != d.action:
        _refuse(ctx, tool, run_id, case_key, "proposed action differs from the decision's action")
    key = approval_key(ctx.business_id, ctx.period, case_key, proposed, case.facts_hash)
    if idempotency_key != key:
        _refuse(ctx, tool, run_id, case_key, "idempotency_key is not the key for this case")
    options = list(dict.fromkeys([proposed, *(allowed_options or []), "escalate_human"]))
    record = ApprovalRecord(approval_id=f"ap_{key[:16]}", idempotency_key=key, case_key=case_key,
                            run_id=run_id, proposed=proposed, allowed_options=options)  # fmt: skip
    stored, created = ctx.repo.insert_approval(record, ctx.business_id, ctx.period)
    return {"created": created, "approval": stored}


def write_draft_email(
    ctx: ToolContext,
    *,
    idempotency_key: str,
    run_id: str,
    case_key: str,
    decision_ref: str,
    language: str = "en",
    books: Mapping[int, BookRecord] | None = None,
    portal: Mapping[int, PortalRecord] | None = None,
) -> dict[str, Any]:
    """Draft (never send) the supplier e-mail for a case that may get one.
    Returns ``{"created": bool, "draft": DraftEmail, "source": "llm"|"template"}``."""
    tool = "write_draft_email"
    d = _decision(ctx, tool, run_id, case_key, decision_ref)
    approval = latest_approval(ctx, case_key)
    if not eligible(d, approval):
        why = (
            "the case is blocked"
            if d.outcome == "block"
            else "no allowing decision or approved chase/verify action for this case"
        )
        _refuse(ctx, tool, run_id, case_key, why)
    case = ctx.repo.get_case(ctx.business_id, ctx.period, case_key)
    if books is None or portal is None:
        if ctx.month_loader is None:
            _refuse(ctx, tool, run_id, case_key, "the run's books and GSTR-2B are not available")
        books, portal = ctx.month_loader(run_id)
    action = approval.final_action if approval and approval.status != "pending" else d.action
    assert action in DRAFT_ACTIONS  # eligible() guarantees it
    # The e-mail goes to the GSTIN in GSTR-2B when there is one (it corrects a books typo).
    side = portal.get(case.portal_index) if case.portal_index is not None else None
    gstin = (side or books[case.book_lines[0]]).supplier_gstin
    # Checked before drafting, so a wrong call never costs a model call.
    if idempotency_key != outbox_key(ctx.business_id, ctx.period, case_key, action, gstin):
        _refuse(ctx, tool, run_id, case_key, "idempotency_key is not the key for this e-mail")
    result = draft_email(case, decision=d, approval=approval, books=books, portal=portal,
                         guard=ctx.guard, client=ctx.email_client, language=language)  # fmt: skip
    assert result.draft is not None and result.draft.to_supplier_gstin == gstin
    stored, created = ctx.repo.insert_outbox(result.draft, run_id)
    return {"created": created, "draft": stored, "source": result.source}


# --- human actions (CLI and web UI only) ----------------------------------------------------------


def approver_name() -> str:
    name = os.environ.get("APPROVER_NAME", "").strip()
    if name:
        return name
    try:
        return getpass.getuser()
    except Exception:  # no login name available (some service accounts)
        return "unknown"


def _resolve(repo: Any, approval_id: str, status: str, final_action: str, by: str | None,
             reason: str | None, feedback_file: Path | None) -> ApprovalRecord:  # fmt: skip
    current = repo.get_approval(approval_id)
    if current is None:
        raise HumanActionError(f"no approval {approval_id}")
    if current.status != "pending":
        raise HumanActionError(f"approval {approval_id} is already {current.status}")
    if final_action not in current.allowed_options:
        raise HumanActionError(
            f"{final_action} is not allowed here; choose one of: "
            + ", ".join(current.allowed_options)
        )
    if status in ("edited", "rejected") and not (reason or "").strip():
        raise HumanActionError(f"a reason is required to {status[:-2]}")
    try:
        done = repo.resolve_approval(approval_id, status, final_action, by or approver_name(),
                                     reason=(reason or None))  # fmt: skip
    except ValueError as exc:  # the record's own rules, e.g. an edit to the same action
        raise HumanActionError(str(exc)) from exc
    if status in ("edited", "rejected"):
        _append_feedback(feedback_file or feedback_path(), current, done)
    return done


def feedback_path() -> Path:
    override = os.environ.get(FEEDBACK_ENV, "").strip()
    return Path(override) if override else FEEDBACK_FILE


def _append_feedback(path: Path, before: ApprovalRecord, after: ApprovalRecord) -> None:
    """One JSON line per edit or rejection: what the system proposed and what the person
    chose, for later evaluation. Ids and action names only, no document text."""
    line = {
        "at": after.decided_at.isoformat()
        if after.decided_at
        else dt.datetime.now(dt.UTC).isoformat(),
        "approval_id": after.approval_id,
        "run_id": after.run_id,
        "case_key": after.case_key,
        "status": after.status,
        "proposed": before.proposed,
        "final_action": after.final_action,
        "reason": after.reason,
        "decided_by": after.decided_by,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with _feedback_lock, open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def approve(repo: Any, approval_id: str, *, by: str | None = None,
            feedback_file: Path | None = None) -> ApprovalRecord:  # fmt: skip
    """Accept the proposed action unchanged."""
    current = repo.get_approval(approval_id)
    if current is None:
        raise HumanActionError(f"no approval {approval_id}")
    return _resolve(repo, approval_id, "approved", current.proposed, by, None, feedback_file)


def edit(repo: Any, approval_id: str, action: str, *, reason: str, by: str | None = None,
         feedback_file: Path | None = None) -> ApprovalRecord:  # fmt: skip
    """Choose a different allowed action (reason required)."""
    return _resolve(repo, approval_id, "edited", action, by, reason, feedback_file)


def reject(repo: Any, approval_id: str, *, reason: str, by: str | None = None,
           feedback_file: Path | None = None) -> ApprovalRecord:  # fmt: skip
    """Say no: the case stays with a person (final action ``escalate_human``)."""
    return _resolve(repo, approval_id, "rejected", "escalate_human", by, reason, feedback_file)


# What an agent (the graph, the MCP server) may call, and what only a person may do.
AGENT_TOOLS: dict[str, Callable[..., Any]] = {
    "request_approval": request_approval,
    "write_draft_email": write_draft_email,
}
HUMAN_ACTIONS: dict[str, Callable[..., Any]] = {"approve": approve, "edit": edit, "reject": reject}
