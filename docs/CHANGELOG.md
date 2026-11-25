# Changelog

One line per merged task. Details in `docs/changelogs/`.

- 01 · Repo setup: git identity and authorship hooks, ignores, project skeleton, CI, seed data, official CBIC rule texts.
- 02 · Domain models, money and GSTIN helpers: integer-paise money, GSTIN checksum and one-edit neighbours, all spec types (frozen, no extra fields), graph state with reducers.
- 03 · Dataset generator: 10 fictional businesses x 3 months (593 labelled cases), 8 injection variants, benign look-alikes, noisy scans, dev/test split by business, human review workflow.
- 04 · Intake agent: typed, normalised books and supplier rows, invoice-number keys, data-quality issues instead of crashes, line-ending-proof input hash.
- 05 · Matcher: five one-to-one passes (exact, normalised, no-year, GSTIN repair, fuzzy), duplicates, paise comparisons with the Rs 1 tolerance, order-independent, ties to a person; agrees with all 593 generated labels.
- 06 · Store and tracing: SQLite schema (WAL, foreign keys, numbered migrations), idempotent approvals and outbox (safe under 8 threads), paid-call ledger, append-only audit log, trace spans with cost and latency.
- 07 · LLM client layer: one client for Bedrock Nova, Claude and a fake; cost estimates, project and run budgets, full-jitter retries, pacer, paid-call ledger, record/replay. Live probes: USD 0.00084.
- 08 · Interfaces: Guard, RuleRetriever and Cache protocols; invoice injection rules, JSON validation, BM25 over citable CBIC snippets, SQLite cache; adapters for llm-guardrails (pinned v1.0.0), Regulatory RAG and semantic-cache with clear fallbacks.
- 09 · Extraction agent: renders the invoice, asks the vision model for JSON (no document text in prompts), schema guard with 2 retries, value normalisation, deterministic checks that lower confidence, one escalation, injection scan of text layer, fields and other text.
