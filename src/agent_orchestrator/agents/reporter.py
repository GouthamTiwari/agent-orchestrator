"""Reporter: the month's numbers, every one traceable to a case (spec 6.8).

``build_report`` is deterministic: the same cases, decisions and approvals give the same JSON.
Money stays in integer paise; rupee strings are only for display.

ITC buckets. Each case with a book line puts the GST on that line (IGST + CGST + SGST) into
one bucket, chosen by the action that stands (see ``communicator.final_action``):
- **safe:** final action ``none``: books, GSTR-2B and the invoice agree; it may be claimed.
- **held:** final action ``chase_supplier_and_hold_itc``: not claimed until the invoice
  reaches GSTR-2B.
- **at risk:** everything else: waiting for a person, escalated, blocked, or with a
  correction or a supplier check still open. A clean match that is not closed (say, an
  injected invoice) is at risk too.
A case with no book line (in GSTR-2B only) claims nothing, so it is in no bucket.

The optional summary (3 to 5 sentences) is written by a model from the totals JSON only. Any
number in it that is not in that JSON, or a wrong length, and the summary is left out.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_orchestrator.agents.communicator import final_action, ungrounded
from agent_orchestrator.interfaces.protocols import Guard
from agent_orchestrator.llm.budget import BudgetExceeded
from agent_orchestrator.llm.client import ConfigError, LLMClient
from agent_orchestrator.llm.pricing import UnknownModel
from agent_orchestrator.llm.recording import RecordingMissing
from agent_orchestrator.llm.types import Message
from agent_orchestrator.models import (
    ApprovalRecord,
    BookRecord,
    BudgetSnapshot,
    Case,
    Decision,
    DQIssue,
    DraftEmail,
    PortalRecord,
)
from agent_orchestrator.money import format_inr
from agent_orchestrator.policy.engine import default_policy
from agent_orchestrator.policy.schema import Policy

PENDING = "pending"
MAX_OUTPUT_TOKENS = 400


def _prompt(name: str) -> str:
    return (
        resources.files("agent_orchestrator.llm.prompts").joinpath(name).read_text("utf-8").strip()
    )


SYSTEM_PROMPT = _prompt("summary_system.txt")
USER_PROMPT = Template(_prompt("summary_user.txt"))


def _tax(b: BookRecord) -> int:
    return b.igst_paise + b.cgst_paise + b.sgst_paise


def _money(paise: int) -> dict[str, Any]:
    return {"paise": paise, "inr": format_inr(paise)}


def _invoice_no(case: Case, books: Mapping[int, BookRecord], portal: Mapping[int, PortalRecord]):
    if case.book_lines:
        return books[case.book_lines[0]].invoice_no
    if case.portal_index is not None:
        return portal[case.portal_index].invoice_no
    return None


def bucket(case: Case, action: str | None) -> str:
    """Which ITC bucket a case is in: "safe", "held", "at_risk" or "none" (no book line)."""
    if not case.book_lines:
        return "none"  # nothing was claimed in the books, so no credit is at stake
    if action == "none":
        return "safe"
    if action == "chase_supplier_and_hold_itc":
        return "held"
    return "at_risk"


def build_report(
    *,
    run_id: str,
    business_id: str,
    period: str,
    cases: Sequence[Case],
    decisions: Iterable[Decision],
    approvals: Iterable[ApprovalRecord] = (),
    books: Iterable[BookRecord] = (),
    portal: Iterable[PortalRecord] = (),
    dq_issues: Iterable[DQIssue] = (),
    drafts: Iterable[DraftEmail] = (),
    budget: BudgetSnapshot | None = None,
    budget_stop_reason: str | None = None,
    cost_by_agent: Mapping[str, Mapping[str, float]] | None = None,
    policy: Policy | None = None,
) -> dict[str, Any]:
    policy = policy or default_policy()
    describe = {r.id: " ".join(r.description.split()) for r in policy.rules}
    books_by_line = {b.line_no: b for b in books}
    portal_by_index = {p.index: p for p in portal}
    decision_of = {d.case_key: d for d in decisions}
    approval_of: dict[str, ApprovalRecord] = {}
    for a in approvals:  # the latest decided record wins over a pending one
        old = approval_of.get(a.case_key)
        if old is None or old.status == "pending":
            approval_of[a.case_key] = a
    drafts = list(drafts)

    rows = []
    for c in sorted(cases, key=lambda x: (x.book_lines[:1] or [10**9], x.case_key)):
        d, a = decision_of.get(c.case_key), approval_of.get(c.case_key)
        action = final_action(d, a)
        b = bucket(c, action)
        amount = _tax(books_by_line[c.book_lines[0]]) if c.book_lines else 0
        rows.append(
            {
                "case_key": c.case_key,
                "invoice_no": _invoice_no(c, books_by_line, portal_by_index),
                "mismatch_type": c.mismatch_type,
                "status": c.status,
                "outcome": d.outcome if d else None,
                "final_action": action or PENDING,
                "approval": a.status if a else None,
                "fired_rules": list(d.fired_rules) if d else [],
                "itc_bucket": b,
                "itc_paise": amount,
            }
        )

    def total(name: str) -> int:
        return sum(r["itc_paise"] for r in rows if r["itc_bucket"] == name)

    auto = [r for r in rows if r["outcome"] == "auto_allow" and r["approval"] is None]
    blocked = [
        {**_pick(r, "case_key", "invoice_no", "mismatch_type", "fired_rules"),
         "reasons": [describe.get(x, x) for x in r["fired_rules"]]}
        for r in rows if r["outcome"] == "block"
    ]  # fmt: skip
    approval_status = Counter(a.status for a in approval_of.values())
    pending = [
        _pick(r, "case_key", "invoice_no", "mismatch_type", "fired_rules")
        for r in rows
        if r["approval"] == "pending"
    ]
    undecided = [r for r in rows if r["outcome"] == "require_approval" and r["approval"] is None]
    agents = {k: dict(v) for k, v in sorted((cost_by_agent or {}).items())}
    return {
        "run_id": run_id,
        "business_id": business_id,
        "period": period,
        "policy_version": next(iter(decision_of.values())).policy_version if decision_of else None,
        "cases": {
            "total": len(rows),
            "by_status": dict(sorted(Counter(r["status"] for r in rows).items())),
            "by_mismatch_type": dict(sorted(Counter(r["mismatch_type"] for r in rows).items())),
            "by_outcome": dict(sorted(Counter(r["outcome"] or PENDING for r in rows).items())),
            "by_final_action": dict(sorted(Counter(r["final_action"] for r in rows).items())),
        },
        "itc": {
            "safe": _money(total("safe")),
            "held": _money(total("held")),
            "at_risk": _money(total("at_risk")),
        },
        "auto_actioned": [
            _pick(r, "case_key", "invoice_no", "mismatch_type", "final_action", "fired_rules")
            for r in auto
        ],
        "approvals": {
            "pending": approval_status.get("pending", 0),
            "approved": approval_status.get("approved", 0),
            "edited": approval_status.get("edited", 0),
            "rejected": approval_status.get("rejected", 0),
            "not_yet_requested": len(undecided),
            "pending_list": pending,
        },
        "blocked": blocked,
        "data_quality": [i.model_dump(mode="json") for i in dq_issues],
        "drafts": {
            "count": len(drafts),
            "by_language": dict(sorted(Counter(x.language for x in drafts).items())),
        },
        "budget": {
            "usd_spent": round(budget.usd_spent, 6) if budget else 0.0,
            "tokens": budget.tokens if budget else 0,
            "stopped": bool(budget and budget.stop_new_llm_work),
            "stop_reason": budget_stop_reason or (budget.reason if budget else None),
        },
        "cost": {
            "usd_total": round(sum(v.get("usd", 0.0) for v in agents.values()), 6),
            "by_agent": agents,
        },
        "cases_detail": rows,
    }


def _pick(row: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {k: row[k] for k in keys}


# --- the optional summary -------------------------------------------------------------------------


class SummaryAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1, max_length=1200)


def summary_input(report: Mapping[str, Any]) -> dict[str, Any]:
    """The totals the summary may use: counts and rupee amounts, no case lists or ids."""
    return {
        "period": report["period"],
        "cases": report["cases"],
        "itc": {k: v["inr"] for k, v in report["itc"].items()},
        "approvals": {k: v for k, v in report["approvals"].items() if k != "pending_list"},
        "auto_actioned": len(report["auto_actioned"]),
        "blocked": len(report["blocked"]),
        "data_quality_issues": len(report["data_quality"]),
        "drafts": report["drafts"]["count"],
        "budget_stopped": report["budget"]["stopped"],
    }


_SENTENCE_END = re.compile(r"[.!?।](?:\s|$)")


def check_summary(text: str, totals: Mapping[str, Any]) -> list[str]:
    problems = ungrounded(text, totals)
    sentences = len(_SENTENCE_END.findall(text.strip() + " "))
    if not 3 <= sentences <= 5:
        problems.append("the summary must have 3 to 5 sentences")
    return list(dict.fromkeys(problems))


@dataclass(frozen=True)
class SummaryResult:
    summary: str | None
    note: str = ""  # "", "number_check_failed", "schema_failed", "budget_stop", "llm_error"
    problems: tuple[str, ...] = ()


def summarise(report: Mapping[str, Any], *, client: LLMClient, guard: Guard) -> SummaryResult:
    """One call, no retry: a summary that fails the number check is simply left out."""
    totals = summary_input(report)
    user = USER_PROMPT.substitute(totals=json.dumps(totals, indent=2, ensure_ascii=False))
    try:
        result = client.complete(
            "summary",
            [Message.user(user)],
            system=SYSTEM_PROMPT,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.0,
        )
    except BudgetExceeded:
        return SummaryResult(None, "budget_stop")
    except (RecordingMissing, UnknownModel, ConfigError):
        raise
    except Exception:
        return SummaryResult(None, "llm_error")
    checked = guard.validate(result.text, SummaryAnswer)
    if not checked.ok:
        return SummaryResult(None, "schema_failed", tuple(checked.errors))
    text = checked.value.summary.strip()
    problems = check_summary(text, totals)
    if problems:
        return SummaryResult(None, "number_check_failed", tuple(problems))
    return SummaryResult(text)


# --- output files ---------------------------------------------------------------------------------


def _table(rows: Sequence[Sequence[Any]], head: Sequence[str]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return out


def render_markdown(report: Mapping[str, Any], summary: str | None = None) -> str:
    c, itc, ap = report["cases"], report["itc"], report["approvals"]
    lines = [
        f"# Reconciliation report: {report['business_id']} {report['period']}",
        "",
        f"Run `{report['run_id']}` · policy `{report['policy_version']}`",
        "",
    ]
    if summary:
        lines += ["## Summary", "", summary, ""]
    lines += ["## Input tax credit", ""]
    lines += _table(
        [
            ["Safe to claim", itc["safe"]["inr"]],
            ["Held (waiting for GSTR-2B)", itc["held"]["inr"]],
            ["At risk (open or blocked)", itc["at_risk"]["inr"]],
        ],
        ["Bucket", "Amount"],
    )
    lines += ["", f"## Cases ({c['total']})", ""]
    lines += _table(sorted(c["by_status"].items()), ["Status", "Cases"])
    lines += [""]
    lines += _table(sorted(c["by_final_action"].items()), ["Final action", "Cases"])
    lines += ["", f"## Done automatically ({len(report['auto_actioned'])})", ""]
    lines += _table(
        [[r["invoice_no"], r["mismatch_type"], r["final_action"], ", ".join(r["fired_rules"])]
         for r in report["auto_actioned"]],
        ["Invoice", "Mismatch", "Action", "Rules"],
    )  # fmt: skip
    lines += [
        "",
        "## Approvals",
        "",
        f"Pending {ap['pending']} · approved {ap['approved']} · edited {ap['edited']} · "
        f"rejected {ap['rejected']} · not yet requested {ap['not_yet_requested']}",
        "",
        f"## Blocked ({len(report['blocked'])})",
        "",
    ]
    for b in report["blocked"]:
        lines.append(f"- **{b['invoice_no']}** ({b['mismatch_type']}):")
        lines += [f"  - {rid}: {why}" for rid, why in zip(b["fired_rules"], b["reasons"],
                                                         strict=True)]  # fmt: skip
    lines += ["", f"## Data quality ({len(report['data_quality'])})", ""]
    lines += [f"- `{i['row_ref']}` {i['code']}: {i['detail']}" for i in report["data_quality"]]
    bud = report["budget"]
    lines += [
        "",
        "## Cost",
        "",
        f"Drafted e-mails: {report['drafts']['count']} (none sent). "
        f"Budget used: USD {bud['usd_spent']:.4f}, {bud['tokens']} tokens"
        + (f"; **stopped: {bud['stop_reason']}**" if bud["stopped"] else "")
        + ".",
        "",
    ]
    lines += _table(
        [[k, int(v.get("spans", 0)), f"{v.get('usd', 0.0):.4f}"]
         for k, v in report["cost"]["by_agent"].items()],
        ["Agent", "Spans", "USD"],
    )  # fmt: skip
    return "\n".join(lines).rstrip() + "\n"


def write_report(
    run_dir: Path | str,
    report: Mapping[str, Any],
    *,
    summary: SummaryResult | None = None,
    trace: Mapping[str, Any] | None = None,
    generated_at: dt.datetime | None = None,
) -> dict[str, Path]:
    """Write ``report.json``, ``report.md`` and (when given) ``trace.json`` into ``run_dir``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        **report,
        "summary": summary.summary if summary else None,
        "summary_note": summary.note if summary else "not_requested",
        "generated_at": (generated_at or dt.datetime.now(dt.UTC)).isoformat(),
    }
    paths = {"json": run_dir / "report.json", "md": run_dir / "report.md"}
    paths["json"].write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8", newline="\n"
    )
    paths["md"].write_text(
        render_markdown(report, summary.summary if summary else None), "utf-8", newline="\n"
    )
    if trace is not None:
        paths["trace"] = run_dir / "trace.json"
        paths["trace"].write_text(
            json.dumps(trace, indent=2, ensure_ascii=False) + "\n", "utf-8", newline="\n"
        )
    return paths


def record_in_store(repo: Any, report: Mapping[str, Any], status: str | None = None) -> None:
    """Update the ``runs`` row with the spend the report states (and the run status if given)."""
    repo.update_run(
        report["run_id"],
        status=status,
        usd_spent=report["budget"]["usd_spent"],
        tokens=report["budget"]["tokens"],
    )
