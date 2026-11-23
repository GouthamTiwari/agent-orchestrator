"""The graph state shared by every node, plus the helper that guards against lost updates.

When the graph fans out (one branch per invoice), several branches finish in the same step and
each returns its own update. For a plain field LangGraph keeps only one of those writes, so the
others are silently lost. A field declared as ``Annotated[list[X], operator.add]`` has a
*reducer*: LangGraph concatenates the branch outputs instead. Every field a parallel branch
writes must therefore have a reducer; `fields_without_reducers` lets a test check that.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict, get_type_hints

from pydantic import AwareDatetime, Field

from agent_orchestrator.models.actions import ApprovalRecord, Decision, DraftEmail, ProposedAction
from agent_orchestrator.models.base import Frozen
from agent_orchestrator.models.cases import Case
from agent_orchestrator.models.extraction import ExtractedInvoice
from agent_orchestrator.models.records import BookRecord, DQIssue, PortalRecord
from agent_orchestrator.models.rules import RuleResearch


class BudgetSnapshot(Frozen):
    """Spend so far in this run. At 90% of any limit, no new LLM work starts."""

    usd_spent: float = Field(default=0.0, ge=0.0)
    tokens: int = Field(default=0, ge=0)
    started_at: AwareDatetime
    stop_new_llm_work: bool = False
    reason: str | None = None


class RunError(Frozen):
    """An error recorded during a run. Errors are data: the run continues where it safely can."""

    node: str
    code: str
    message: str
    case_key: str | None = None
    at: AwareDatetime


class RunState(TypedDict):
    run_id: str
    business_id: str
    period: str
    input_hash: str
    policy_version: str
    books: list[BookRecord]
    portal: list[PortalRecord]
    dq_issues: list[DQIssue]
    extractions: Annotated[list[ExtractedInvoice], operator.add]  # written by parallel branches
    cases: list[Case]
    research: Annotated[list[RuleResearch], operator.add]
    proposals: Annotated[list[ProposedAction], operator.add]
    decisions: Annotated[list[Decision], operator.add]
    approvals_resolved: list[ApprovalRecord]
    drafts: Annotated[list[DraftEmail], operator.add]
    budget: BudgetSnapshot
    errors: Annotated[list[RunError], operator.add]


def _reducer(hint: Any) -> Any | None:
    """The reducer function of an ``Annotated[..., reducer]`` hint, or None."""
    for meta in getattr(hint, "__metadata__", ()):
        if callable(meta):
            return meta
    return None


def fields_with_reducers(state_type: type = RunState) -> dict[str, Any]:
    """Map of field name -> reducer for every field that merges parallel writes."""
    hints = get_type_hints(state_type, include_extras=True)
    return {name: r for name, hint in hints.items() if (r := _reducer(hint)) is not None}


def fields_without_reducers(state_type: type = RunState) -> list[str]:
    """Fields where a second write in the same step would overwrite the first.

    The graph test (Task 13) asserts that no parallel branch writes to any of these.
    """
    hints = get_type_hints(state_type, include_extras=True)
    return sorted(name for name, hint in hints.items() if _reducer(hint) is None)
