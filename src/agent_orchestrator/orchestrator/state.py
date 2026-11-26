"""The graph's state: ``RunState`` (spec 5) plus the few fields only the graph needs.

Fields written by parallel ``Send`` branches carry a reducer (``operator.add``), so the
branches' outputs are concatenated instead of one overwriting another. ``test_graph.py``
checks every branch writes only such fields.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from pydantic import Field

from agent_orchestrator.models import Case, ExtractedInvoice, RunState
from agent_orchestrator.models.base import Frozen


class PlanNote(Frozen):
    """What the planner said about its own proposal, for the policy facts and the report."""

    case_key: str
    fallback: bool = False
    flags: list[str] = Field(default_factory=list)  # "planner_failed", "budget_stop", ...


class GraphState(RunState, total=False):
    buyer_gstin: str
    plan_notes: Annotated[list[PlanNote], operator.add]
    approval_ids: list[str]  # written once, by ``decide``; the gate reads their status


class ExtractTask(TypedDict):
    """The payload of one ``extract_one`` branch."""

    pdf_file: str  # as written in the books, e.g. "invoices/X.pdf"
    buyer_gstin: str


class PlanTask(TypedDict):
    """The payload of one ``research_and_plan`` branch."""

    case: Case
    extraction: ExtractedInvoice | None


# The model classes the checkpoint may contain. LangGraph's serializer deserializes only
# registered classes (anything else is a warning today and an error in a later version), so
# the list is explicit: a checkpoint cannot make this process build an arbitrary object.
CHECKPOINT_TYPES: list[tuple[str, str]] = [
    ("agent_orchestrator.models.records", "BookRecord"),
    ("agent_orchestrator.models.records", "PortalRecord"),
    ("agent_orchestrator.models.records", "DQIssue"),
    ("agent_orchestrator.models.extraction", "ExtractedInvoice"),
    ("agent_orchestrator.models.extraction", "FieldValue"),
    ("agent_orchestrator.models.extraction", "InjectionResult"),
    ("agent_orchestrator.models.cases", "Case"),
    ("agent_orchestrator.models.rules", "RuleResearch"),
    ("agent_orchestrator.models.rules", "RuleSnippet"),
    ("agent_orchestrator.models.actions", "ProposedAction"),
    ("agent_orchestrator.models.actions", "Decision"),
    ("agent_orchestrator.models.actions", "ApprovalRecord"),
    ("agent_orchestrator.models.actions", "DraftEmail"),
    ("agent_orchestrator.models.state", "BudgetSnapshot"),
    ("agent_orchestrator.models.state", "RunError"),
    ("agent_orchestrator.orchestrator.state", "PlanNote"),
]

# Fields each kind of parallel branch may return. All must have reducers.
BRANCH_WRITES: dict[str, set[str]] = {
    "extract_one": {"extractions", "errors"},
    "research_and_plan": {"proposals", "research", "plan_notes", "errors"},
}


def initial_state(**values: Any) -> dict[str, Any]:
    """Empty lists for every list field, so nodes never test for a missing key."""
    base: dict[str, Any] = {
        "books": [], "portal": [], "dq_issues": [], "extractions": [], "cases": [],
        "research": [], "proposals": [], "decisions": [], "approvals_resolved": [],
        "drafts": [], "errors": [], "plan_notes": [], "approval_ids": [],
    }  # fmt: skip
    return base | values
