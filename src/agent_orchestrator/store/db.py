"""SQLite connection and schema (docs/SPEC.md section 9.1).

One local file holds runs, cases, decisions, approvals, the e-mail outbox, the paid-call ledger,
trace spans and the audit log. LangGraph keeps its own checkpoint tables in the same file.

- WAL journal mode: readers (the web UI) never block the writer (a run).
- Foreign keys on: a decision or approval cannot point at a run that does not exist.
- ``busy_timeout``: concurrent writers wait instead of failing at once.
- Migrations are numbered and recorded in ``schema_version``; running ``migrate`` again is a
  no-op, so every process can call it at start-up.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

MIGRATIONS: list[str] = [
    # 1: initial schema
    """
    CREATE TABLE runs (
        run_id          TEXT PRIMARY KEY,
        business_id     TEXT NOT NULL,
        period          TEXT NOT NULL,
        input_hash      TEXT NOT NULL,
        policy_version  TEXT NOT NULL,
        status          TEXT NOT NULL CHECK (status IN ('running','paused','done','failed')),
        usd_spent       REAL NOT NULL DEFAULT 0,
        tokens          INTEGER NOT NULL DEFAULT 0,
        started_at      TEXT NOT NULL,
        updated_at      TEXT NOT NULL,
        finished_at     TEXT
    );

    CREATE TABLE cases (
        business_id     TEXT NOT NULL,
        period          TEXT NOT NULL,
        case_key        TEXT NOT NULL,
        run_id          TEXT NOT NULL REFERENCES runs(run_id),
        mismatch_type   TEXT NOT NULL,
        status          TEXT NOT NULL,
        action          TEXT,
        facts_hash      TEXT NOT NULL,
        case_json       TEXT NOT NULL,
        updated_at      TEXT NOT NULL,
        PRIMARY KEY (business_id, period, case_key)
    );

    CREATE TABLE decisions (
        run_id          TEXT NOT NULL REFERENCES runs(run_id),
        case_key        TEXT NOT NULL,
        outcome         TEXT NOT NULL CHECK (outcome IN ('auto_allow','require_approval','block')),
        action          TEXT NOT NULL,
        fired_rules     TEXT NOT NULL,
        policy_version  TEXT NOT NULL,
        decided_at      TEXT NOT NULL,
        PRIMARY KEY (run_id, case_key)
    );

    CREATE TABLE approvals (
        approval_id     TEXT PRIMARY KEY,
        idempotency_key TEXT NOT NULL UNIQUE,
        run_id          TEXT NOT NULL REFERENCES runs(run_id),
        business_id     TEXT NOT NULL,
        period          TEXT NOT NULL,
        case_key        TEXT NOT NULL,
        proposed        TEXT NOT NULL,
        allowed_options TEXT NOT NULL,
        status          TEXT NOT NULL CHECK (status IN ('pending','approved','edited','rejected')),
        final_action    TEXT,
        reason          TEXT,
        decided_by      TEXT,
        decided_at      TEXT,
        created_at      TEXT NOT NULL
    );
    CREATE INDEX approvals_pending ON approvals(status, created_at);

    CREATE TABLE outbox (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        idempotency_key   TEXT NOT NULL UNIQUE,
        run_id            TEXT NOT NULL REFERENCES runs(run_id),
        case_key          TEXT NOT NULL,
        to_supplier_gstin TEXT NOT NULL,
        subject           TEXT NOT NULL,
        body              TEXT NOT NULL,
        language          TEXT NOT NULL,
        created_at        TEXT NOT NULL
    );

    CREATE TABLE llm_calls (
        request_hash    TEXT PRIMARY KEY,
        run_id          TEXT,
        purpose         TEXT NOT NULL,
        model_id        TEXT NOT NULL,
        input_tokens    INTEGER NOT NULL,
        output_tokens   INTEGER NOT NULL,
        usd             REAL NOT NULL,
        latency_ms      REAL NOT NULL,
        response_json   TEXT NOT NULL,
        created_at      TEXT NOT NULL
    );

    CREATE TABLE spans (
        run_id          TEXT NOT NULL,
        span_id         TEXT NOT NULL,
        parent_id       TEXT,
        agent           TEXT NOT NULL,
        name            TEXT NOT NULL,
        case_key        TEXT,
        input_hash      TEXT,
        output_summary  TEXT,
        input_tokens    INTEGER NOT NULL DEFAULT 0,
        output_tokens   INTEGER NOT NULL DEFAULT 0,
        usd             REAL NOT NULL DEFAULT 0,
        latency_ms      REAL NOT NULL,
        decision        TEXT,
        error           TEXT,
        started_at      TEXT NOT NULL,
        finished_at     TEXT NOT NULL,
        PRIMARY KEY (run_id, span_id)
    );

    CREATE TABLE audit_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        at              TEXT NOT NULL,
        actor           TEXT NOT NULL,
        action          TEXT NOT NULL,
        run_id          TEXT,
        case_key        TEXT,
        detail          TEXT NOT NULL
    );
    -- Append-only, enforced by the database itself, not just by the code.
    CREATE TRIGGER audit_log_no_update BEFORE UPDATE ON audit_log
        BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
    CREATE TRIGGER audit_log_no_delete BEFORE DELETE ON audit_log
        BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
    """,
]


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the database with the project's settings. One connection per thread."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)  # autocommit mode
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    return int(row["v"] or 0)


def migrate(conn: sqlite3.Connection) -> int:
    """Apply any migrations not yet applied, each in its own transaction. Returns the version."""
    conn.execute("BEGIN IMMEDIATE")  # one migrator at a time
    try:
        current = schema_version(conn)
        for version, script in enumerate(MIGRATIONS, start=1):
            if version <= current:
                continue
            for statement in _statements(script):
                conn.execute(statement)
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            current = version
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return current


def _statements(script: str) -> list[str]:
    """Split a migration into statements, keeping trigger bodies (BEGIN ... END;) whole."""
    out, buf = [], []
    for line in script.splitlines():
        if line.strip().startswith("--") or not line.strip():
            continue
        buf.append(line)
        text = "\n".join(buf)
        if sqlite3.complete_statement(text):
            out.append(text.strip())
            buf = []
    if buf and "\n".join(buf).strip():
        raise ValueError(f"incomplete SQL statement in migration: {' '.join(buf)[:80]}")
    return out
