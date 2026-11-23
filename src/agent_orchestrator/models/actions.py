"""Proposals, decisions, approvals and draft e-mails (spec sections 5, 7 and 11)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from agent_orchestrator.models.base import Frozen
from agent_orchestrator.models.cases import ActionKind

Outcome = Literal["auto_allow", "require_approval", "block"]
ApprovalStatus = Literal["pending", "approved", "edited", "rejected"]


class ProposedAction(Frozen):
    """What the rule table or the LLM planner suggests. A proposal is never acted on directly:
    the policy engine turns it into a `Decision`."""

    case_key: str
    action: ActionKind
    reason: str = Field(max_length=600)
    evidence_refs: list[str] = Field(default_factory=list)  # "books:line:12", "2b:index:5"
    rule_refs: list[str] = Field(default_factory=list)  # must be snippet ids the planner was given
    proposer: Literal["rule", "llm"]


class Decision(Frozen):
    """The policy engine's verdict for one case. Only code writes these, never the LLM."""

    case_key: str
    outcome: Outcome
    action: ActionKind
    fired_rules: list[str] = Field(min_length=1)  # every decision names at least one rule
    policy_version: str
    decided_at: AwareDatetime


class ApprovalRecord(Frozen):
    approval_id: str
    idempotency_key: str
    case_key: str
    run_id: str
    proposed: ActionKind
    allowed_options: list[ActionKind] = Field(min_length=1)
    status: ApprovalStatus = "pending"
    final_action: ActionKind | None = None
    reason: str | None = None
    decided_by: str | None = None
    decided_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.proposed not in self.allowed_options:
            raise ValueError("the proposed action must be one of the allowed options")
        decided = (self.final_action, self.decided_by, self.decided_at)
        if self.status == "pending":
            if any(v is not None for v in decided):
                raise ValueError("a pending approval has no final action, approver or time")
            return self
        if any(v is None for v in decided):
            raise ValueError(f"a {self.status} approval needs final_action, decided_by, decided_at")
        if self.final_action not in self.allowed_options:
            raise ValueError("final_action must be one of the allowed options")
        if self.status == "approved" and self.final_action != self.proposed:
            raise ValueError("approved means the proposed action unchanged; use 'edited'")
        if self.status == "edited" and self.final_action == self.proposed:
            raise ValueError("edited means a different action from the proposed one")
        if self.status in ("edited", "rejected") and not (self.reason or "").strip():
            raise ValueError(f"a {self.status} approval needs a reason")
        if self.status == "rejected" and self.final_action != "escalate_human":
            raise ValueError("a rejected approval leaves the case with a person (escalate_human)")
        return self


class DraftEmail(Frozen):
    """A follow-up e-mail to a supplier. Drafted only; nothing in this project sends mail."""

    idempotency_key: str
    case_key: str
    to_supplier_gstin: str
    subject: str = Field(max_length=200)
    body: str
    language: Literal["en", "hi", "te"] = "en"
    created_at: AwareDatetime
