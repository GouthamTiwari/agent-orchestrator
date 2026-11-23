# 06 · Store and tracing

## What changed
- `store/db.py`: one SQLite file with the eight tables of spec 9.1: `runs`, `cases`, `decisions`, `approvals`, `outbox`, `llm_calls`, `spans`, `audit_log`.
  - Connection settings: WAL journal mode, foreign keys on, a 30 s busy timeout and autocommit mode with explicit `BEGIN IMMEDIATE` transactions.
  - Migrations are numbered and recorded in `schema_version`; `migrate` runs them under a write lock and is a no-op the second time.
  - The audit log has database triggers that refuse UPDATE and DELETE, so append-only does not depend on the code alone.
- `store/idempotency.py`: approval key = sha256(`approval` | business | period | case | proposed action | facts hash); e-mail key = sha256(`email` | business | period | case | final action | supplier GSTIN). The LLM's wording is not in the e-mail key. A `|` inside a part is refused, so two different inputs cannot produce the same joined text.
- `store/repo.py`: all SQL in one place.
  - **Runs:** create, read, update status and spend; `finished_at` is set on done or failed.
  - **Cases:** upserted by (business, period, case_key), so a re-run updates rather than duplicates.
  - **Decisions:** one per (run, case).
  - **Approvals and outbox:** `INSERT ... ON CONFLICT DO NOTHING` on the idempotency key, returning `(stored row, created)`.
  - **Resolving an approval:** only from `pending`, and only after the `ApprovalRecord` model rules pass (reason needed for edit and reject, the action must be allowed). It is written together with its audit entry, which holds who, before, after and reason.
  - **Paid-call ledger:** `ledger_get`, `ledger_put` (the first copy wins) and totals.
  - **Spans:** insert and list. **Audit log:** append and list only.
- `tracing.py`: `Tracer.span(agent, name, case_key=, input_hash=)` context manager.
  - It records tokens, USD, latency (`perf_counter`), output summary, decision and error, and re-raises the error.
  - `parent_id` comes from a context variable, so nested spans and parallel threads each keep the right parent.
  - Text is capped at 500 characters. Per-agent totals; JSON export with LF endings.

## Checks
- `uv run ruff check .`, `uv run ruff format --check .`: pass.
- `uv run pytest`: 183 passed (21 new).
  - **Schema:** tables, WAL and foreign keys; the migration runs twice harmlessly; an orphan approval is refused.
  - **Keys:** deterministic, and wording-free for e-mails.
  - **Runs, cases, decisions:** run life cycle; cases upserted across two runs; decisions round-trip.
  - **Approvals:** the second insert returns the first row; resolve once, with audit; invalid resolutions write nothing.
  - **Outbox and ledger:** the first e-mail draft wins; the ledger never stores a second price for the same request.
  - **Audit log:** append-only, checked through both the repo's functions and the triggers.
  - **Concurrency:** 8 threads, each with its own connection, insert the same approval and the same e-mail at once. Exactly one of each is created and every thread gets that row back. Repeated 15 times with no failure.
  - **Tracing:** usage and latency, errors, nesting, per-thread parents, negative usage refused, totals and JSON export, truncation.
