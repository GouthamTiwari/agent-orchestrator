"""Typed reads and writes for the store. All SQL lives here.

Write rules:
- Approvals and outbox rows are created with ``INSERT ... ON CONFLICT DO NOTHING`` on their
  idempotency key; the caller gets the stored row back plus ``created`` (True only for the
  first insert). Safe under concurrent writers.
- The audit log can only be appended to: this module has no function that updates or deletes
  it, and database triggers refuse it anyway.
- An approval can be resolved once, from ``pending`` only.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any

from agent_orchestrator.models import ApprovalRecord, Case, Decision, DraftEmail
from agent_orchestrator.store.db import connect, migrate

RUN_STATUSES = ("running", "paused", "done", "failed")


class StoreError(RuntimeError):
    """A write was refused (unknown row, wrong state)."""


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds")


def _ts(value: dt.datetime | None) -> str | None:
    return value.astimezone(dt.UTC).isoformat(timespec="microseconds") if value else None


class Repo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    @classmethod
    def open(cls, path: str | Path) -> Repo:
        conn = connect(path)
        migrate(conn)
        return cls(conn)

    def close(self) -> None:
        self.conn.close()

    def _tx(self):
        return _Transaction(self.conn)

    # --- runs ---------------------------------------------------------------------------------

    def create_run(
        self, run_id: str, business_id: str, period: str, input_hash: str, policy_version: str
    ) -> dict[str, Any]:
        now = _now()
        with self._tx():
            self.conn.execute(
                "INSERT INTO runs (run_id, business_id, period, input_hash, policy_version,"
                " status, started_at, updated_at) VALUES (?,?,?,?,?,'running',?,?)",
                (run_id, business_id, period, input_hash, policy_version, now, now),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise StoreError(f"unknown run {run_id}")
        return dict(row)

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        usd_spent: float | None = None,
        tokens: int | None = None,
    ) -> dict[str, Any]:
        if status is not None and status not in RUN_STATUSES:
            raise StoreError(f"bad run status {status!r}")
        now = _now()
        with self._tx():
            cur = self.conn.execute(
                "UPDATE runs SET status = COALESCE(?, status), usd_spent = COALESCE(?, usd_spent),"
                " tokens = COALESCE(?, tokens), updated_at = ?,"
                " finished_at = CASE WHEN ? IN ('done','failed') THEN ? ELSE finished_at END"
                " WHERE run_id = ?",
                (status, usd_spent, tokens, now, status, now, run_id),
            )
            if cur.rowcount != 1:
                raise StoreError(f"unknown run {run_id}")
        return self.get_run(run_id)

    # --- cases and decisions ------------------------------------------------------------------

    def upsert_case(self, run_id: str, case: Case, action: str | None = None) -> None:
        """Latest status per (business, period, case_key): a re-run updates, never duplicates."""
        with self._tx():
            self.conn.execute(
                "INSERT INTO cases (business_id, period, case_key, run_id, mismatch_type, status,"
                " action, facts_hash, case_json, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT (business_id, period, case_key) DO UPDATE SET"
                " run_id = excluded.run_id, mismatch_type = excluded.mismatch_type,"
                " status = excluded.status, action = excluded.action,"
                " facts_hash = excluded.facts_hash, case_json = excluded.case_json,"
                " updated_at = excluded.updated_at",
                (
                    case.business_id,
                    case.period,
                    case.case_key,
                    run_id,
                    case.mismatch_type,
                    case.status,
                    action,
                    case.facts_hash,
                    case.model_dump_json(),
                    _now(),
                ),
            )

    def get_case(self, business_id: str, period: str, case_key: str) -> Case | None:
        row = self.conn.execute(
            "SELECT case_json FROM cases WHERE business_id=? AND period=? AND case_key=?",
            (business_id, period, case_key),
        ).fetchone()
        return Case.model_validate_json(row["case_json"]) if row else None

    def list_cases(self, business_id: str, period: str) -> list[Case]:
        rows = self.conn.execute(
            "SELECT case_json FROM cases WHERE business_id=? AND period=? ORDER BY case_key",
            (business_id, period),
        ).fetchall()
        return [Case.model_validate_json(r["case_json"]) for r in rows]

    def record_decision(self, run_id: str, decision: Decision) -> None:
        with self._tx():
            self.conn.execute(
                "INSERT INTO decisions (run_id, case_key, outcome, action, fired_rules,"
                " policy_version, decided_at) VALUES (?,?,?,?,?,?,?)"
                " ON CONFLICT (run_id, case_key) DO UPDATE SET outcome = excluded.outcome,"
                " action = excluded.action, fired_rules = excluded.fired_rules,"
                " policy_version = excluded.policy_version, decided_at = excluded.decided_at",
                (
                    run_id,
                    decision.case_key,
                    decision.outcome,
                    decision.action,
                    json.dumps(decision.fired_rules),
                    decision.policy_version,
                    _ts(decision.decided_at),
                ),
            )

    def get_decision(self, run_id: str, case_key: str) -> Decision | None:
        row = self.conn.execute(
            "SELECT * FROM decisions WHERE run_id=? AND case_key=?", (run_id, case_key)
        ).fetchone()
        if row is None:
            return None
        return Decision(
            case_key=row["case_key"],
            outcome=row["outcome"],
            action=row["action"],
            fired_rules=json.loads(row["fired_rules"]),
            policy_version=row["policy_version"],
            decided_at=dt.datetime.fromisoformat(row["decided_at"]),
        )

    # --- approvals ----------------------------------------------------------------------------

    def insert_approval(
        self, approval: ApprovalRecord, business_id: str, period: str
    ) -> tuple[ApprovalRecord, bool]:
        """Create a pending approval once. Returns (stored row, created)."""
        if approval.status != "pending":
            raise StoreError("new approvals must be pending")
        with self._tx():
            cur = self.conn.execute(
                "INSERT INTO approvals (approval_id, idempotency_key, run_id, business_id, period,"
                " case_key, proposed, allowed_options, status, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,'pending',?)"
                " ON CONFLICT (idempotency_key) DO NOTHING",
                (
                    approval.approval_id,
                    approval.idempotency_key,
                    approval.run_id,
                    business_id,
                    period,
                    approval.case_key,
                    approval.proposed,
                    json.dumps(approval.allowed_options),
                    _now(),
                ),
            )
            created = cur.rowcount == 1
        stored = self.get_approval_by_key(approval.idempotency_key)
        assert stored is not None
        return stored, created

    def get_approval(self, approval_id: str) -> ApprovalRecord | None:
        row = self.conn.execute(
            "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)
        ).fetchone()
        return _approval(row) if row else None

    def get_approval_by_key(self, idempotency_key: str) -> ApprovalRecord | None:
        row = self.conn.execute(
            "SELECT * FROM approvals WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        return _approval(row) if row else None

    def list_approvals(self, status: str | None = "pending", run_id: str | None = None):
        sql, args = "SELECT * FROM approvals WHERE 1=1", []
        if status:
            sql += " AND status = ?"
            args.append(status)
        if run_id:
            sql += " AND run_id = ?"
            args.append(run_id)
        rows = self.conn.execute(sql + " ORDER BY created_at, approval_id", args).fetchall()
        return [_approval(r) for r in rows]

    def resolve_approval(
        self,
        approval_id: str,
        status: str,
        final_action: str,
        decided_by: str,
        reason: str | None = None,
        decided_at: dt.datetime | None = None,
    ) -> ApprovalRecord:
        """Record a person's decision. Allowed once, from pending only; audited."""
        current = self.get_approval(approval_id)
        if current is None:
            raise StoreError(f"unknown approval {approval_id}")
        if current.status != "pending":
            raise StoreError(f"approval {approval_id} is already {current.status}")
        decided_at = decided_at or dt.datetime.now(dt.UTC)
        # Validate with the model's own rules before touching the database.
        resolved = current.model_copy(
            update=dict(
                status=status,
                final_action=final_action,
                reason=reason,
                decided_by=decided_by,
                decided_at=decided_at,
            )
        )
        ApprovalRecord.model_validate(resolved.model_dump())
        with self._tx():
            cur = self.conn.execute(
                "UPDATE approvals SET status=?, final_action=?, reason=?, decided_by=?,"
                " decided_at=? WHERE approval_id=? AND status='pending'",
                (status, final_action, reason, decided_by, _ts(decided_at), approval_id),
            )
            if cur.rowcount != 1:
                raise StoreError(f"approval {approval_id} changed concurrently")
            self._append_audit(
                decided_by,
                f"approval_{status}",
                current.run_id,
                current.case_key,
                {
                    "approval_id": approval_id,
                    "before": current.proposed,
                    "after": final_action,
                    "reason": reason,
                },
            )
        return self.get_approval(approval_id)

    # --- outbox -------------------------------------------------------------------------------

    def insert_outbox(self, draft: DraftEmail, run_id: str) -> tuple[DraftEmail, bool]:
        """Store a drafted e-mail once per idempotency key. Nothing is ever sent."""
        with self._tx():
            cur = self.conn.execute(
                "INSERT INTO outbox (idempotency_key, run_id, case_key, to_supplier_gstin,"
                " subject, body, language, created_at) VALUES (?,?,?,?,?,?,?,?)"
                " ON CONFLICT (idempotency_key) DO NOTHING",
                (
                    draft.idempotency_key,
                    run_id,
                    draft.case_key,
                    draft.to_supplier_gstin,
                    draft.subject,
                    draft.body,
                    draft.language,
                    _ts(draft.created_at),
                ),
            )
            created = cur.rowcount == 1
        row = self.conn.execute(
            "SELECT * FROM outbox WHERE idempotency_key = ?", (draft.idempotency_key,)
        ).fetchone()
        return _draft(row), created

    def list_outbox(self, run_id: str | None = None) -> list[DraftEmail]:
        sql = "SELECT * FROM outbox" + (" WHERE run_id = ?" if run_id else "") + " ORDER BY id"
        rows = self.conn.execute(sql, (run_id,) if run_id else ()).fetchall()
        return [_draft(r) for r in rows]

    # --- paid-call ledger ---------------------------------------------------------------------

    def ledger_get(self, request_hash: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM llm_calls WHERE request_hash = ?", (request_hash,)
        ).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["response"] = json.loads(out.pop("response_json"))
        return out

    def ledger_put(
        self,
        request_hash: str,
        *,
        purpose: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        usd: float,
        latency_ms: float,
        response: Any,
        run_id: str | None = None,
    ) -> bool:
        """Remember a paid call by its request hash. Returns True if this is the first copy."""
        with self._tx():
            cur = self.conn.execute(
                "INSERT INTO llm_calls (request_hash, run_id, purpose, model_id, input_tokens,"
                " output_tokens, usd, latency_ms, response_json, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT (request_hash) DO NOTHING",
                (
                    request_hash,
                    run_id,
                    purpose,
                    model_id,
                    input_tokens,
                    output_tokens,
                    usd,
                    latency_ms,
                    json.dumps(response, ensure_ascii=False),
                    _now(),
                ),
            )
            return cur.rowcount == 1

    def ledger_totals(self) -> dict[str, float]:
        row = self.conn.execute(
            "SELECT COUNT(*) AS calls, COALESCE(SUM(usd),0) AS usd,"
            " COALESCE(SUM(input_tokens + output_tokens),0) AS tokens FROM llm_calls"
        ).fetchone()
        return {"calls": row["calls"], "usd": row["usd"], "tokens": row["tokens"]}

    # --- spans --------------------------------------------------------------------------------

    def insert_span(self, span: dict[str, Any]) -> None:
        cols = (
            "run_id",
            "span_id",
            "parent_id",
            "agent",
            "name",
            "case_key",
            "input_hash",
            "output_summary",
            "input_tokens",
            "output_tokens",
            "usd",
            "latency_ms",
            "decision",
            "error",
            "started_at",
            "finished_at",
        )
        with self._tx():
            self.conn.execute(
                f"INSERT INTO spans ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                tuple(span.get(c) for c in cols),
            )

    def list_spans(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM spans WHERE run_id = ? ORDER BY started_at, span_id", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- audit log (append-only) --------------------------------------------------------------

    def append_audit(
        self,
        actor: str,
        action: str,
        run_id: str | None = None,
        case_key: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> int:
        with self._tx():
            return self._append_audit(actor, action, run_id, case_key, detail or {})

    def _append_audit(self, actor, action, run_id, case_key, detail) -> int:
        cur = self.conn.execute(
            "INSERT INTO audit_log (at, actor, action, run_id, case_key, detail)"
            " VALUES (?,?,?,?,?,?)",
            (_now(), actor, action, run_id, case_key, json.dumps(detail, ensure_ascii=False)),
        )
        return int(cur.lastrowid)

    def list_audit(self, run_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM audit_log" + (" WHERE run_id = ?" if run_id else "") + " ORDER BY id"
        rows = self.conn.execute(sql, (run_id,) if run_id else ()).fetchall()
        return [{**dict(r), "detail": json.loads(r["detail"])} for r in rows]


class _Transaction:
    """``BEGIN IMMEDIATE`` ... ``COMMIT`` (or ``ROLLBACK`` on error). Nested use is flattened."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.outer = False

    def __enter__(self):
        if not self.conn.in_transaction:
            self.conn.execute("BEGIN IMMEDIATE")
            self.outer = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.outer:
            self.conn.execute("ROLLBACK" if exc_type else "COMMIT")
        return False


def _approval(row: sqlite3.Row) -> ApprovalRecord:
    return ApprovalRecord(
        approval_id=row["approval_id"],
        idempotency_key=row["idempotency_key"],
        case_key=row["case_key"],
        run_id=row["run_id"],
        proposed=row["proposed"],
        allowed_options=json.loads(row["allowed_options"]),
        status=row["status"],
        final_action=row["final_action"],
        reason=row["reason"],
        decided_by=row["decided_by"],
        decided_at=dt.datetime.fromisoformat(row["decided_at"]) if row["decided_at"] else None,
    )


def _draft(row: sqlite3.Row) -> DraftEmail:
    return DraftEmail(
        idempotency_key=row["idempotency_key"],
        case_key=row["case_key"],
        to_supplier_gstin=row["to_supplier_gstin"],
        subject=row["subject"],
        body=row["body"],
        language=row["language"],
        created_at=dt.datetime.fromisoformat(row["created_at"]),
    )
