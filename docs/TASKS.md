# Tasks

Build order for `docs/SPEC.md`. One task at a time, in this order, on branch `task/NN-slug`, self-reviewed against the acceptance criteria, then squash-merged. Each task lists the files it creates or edits; touching anything else needs a line in its changelog saying why. "Spec n" means section n of `docs/SPEC.md`.

Every task's acceptance criteria also include: `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest` pass; no network calls in unit tests; changelog `docs/changelogs/NN-slug.md` written; one line added to `docs/CHANGELOG.md`; commits authored by Goutham Tiwari only (`bash .githooks/audit-authors.sh` prints OK); no secrets, `*.sqlite` or `data/generated/` committed.

Live AWS or Anthropic calls happen only in tasks marked **LIVE**, and only after Goutham says yes to the printed call count and USD estimate.

## Before Task 01 (about 5 minutes, Goutham)

Nothing new is needed in the AWS console: Nova 2 Lite in `ap-south-1` already works from this PC, and the same Bedrock API key is reused.
1. ~~AWS budget alarm~~ done (USD 20/month, decision 8).
2. ~~Decisions~~ all answered (spec section 19). Goutham still makes `llm-guardrails` public on GitHub (repo Settings → Danger Zone → Change visibility) before Task 08.
3. Task 01 creates the public GitHub repo `GouthamTiwari/agent-orchestrator` with `gh`.

---

## 01 · Repo setup
**Files:** `.gitignore`, `.gitattributes`, `.githooks/*`, `.env.example`, `.env` (local only), `pyproject.toml`, `uv.lock`, `LICENSE`, `README.md`, `docs/SPEC.md`, `docs/POLICY_RULES.md`, `docs/TASKS.md`, `docs/CHANGELOG.md`, `docs/changelogs/01-repo-setup.md`, `.github/workflows/ci.yml`, `src/agent_orchestrator/__init__.py`, `tests/test_smoke.py`, `data/seed/*` (from the kit), `data/rules/sources/*.txt`
**Depends on:** –
**Why:** identity, hooks and ignores in place before anything else exists.
**Acceptance:**
- [ ] `git init -b main`; `git config user.name "Goutham Tiwari"`, `git config user.email "227244284+GouthamTiwari@users.noreply.github.com"`, `git config core.hooksPath .githooks` (each explained in one line).
- [ ] `.githooks/` from the kit with `OWNER_EMAIL` set to the noreply address in `_check.sh`; the kit's setup/check scripts are not copied (the steps are done here).
- [ ] `.gitignore` = kit list + `_kit/`, `.spend/`, `runs/`, `eval/human_feedback.jsonl`, `data/review/*.filled.csv`; local tool folders go in `.git/info/exclude`. `git check-ignore` proves `_kit/` and `.env` are ignored.
- [ ] `.gitattributes`: kit lines + LF for `*.jsonl *.json *.yaml *.txt` (CRLF broke hashes in guardrails).
- [ ] `.env.example`: `AWS_REGION=ap-south-1`, `AWS_BEARER_TOKEN_BEDROCK=`, `AWS_ACCESS_KEY_ID=`, `AWS_SECRET_ACCESS_KEY=`, `BEDROCK_VISION_MODEL_ID=global.amazon.nova-2-lite-v1:0`, `BEDROCK_VISION_ESCALATION_MODEL_ID=apac.amazon.nova-pro-v1:0`, `BEDROCK_TEXT_MODEL_ID=global.amazon.nova-2-lite-v1:0`, `ANTHROPIC_API_KEY=`, `ANTHROPIC_MODEL_ID=claude-haiku-5-5`, `LLM_MODE=replay`, `RUN_BUDGET_USD=1.00`, `RUN_BUDGET_TOKENS=500000`, `RUN_BUDGET_SECONDS=900`, `LIVE_BUDGET_USD=5`, `ANTHROPIC_BUDGET_USD=3`, `ORCH_MAX_CONCURRENCY=2`, `LLM_REQUESTS_PER_SECOND=2`, `APPROVER_NAME=`, `DATA_DIR=data`, `DB_PATH=data/orchestrator.sqlite`, optional package paths for the three sibling projects.
- [ ] Local `.env` created from the example; the Bedrock key copied from `..\llm-guardrails\.env` by a script that **never prints it**.
- [ ] `pyproject.toml`: name `agent-orchestrator`, import package `agent_orchestrator`, `requires-python = ">=3.12,<3.13"`, authors = Goutham Tiwari + noreply email, MIT, src layout, script `orchestrator = "agent_orchestrator.cli:app"`; deps per spec 2.1 with lower bounds at the versions checked; dev deps `pytest`, `hypothesis`, `ruff`, `reportlab`; ruff line length 100, rules `E,F,I,UP,B,SIM`, `*.md` and `_kit` excluded; pytest markers `live` and `slow` deselected by default.
- [ ] `LICENSE` MIT, `Copyright (c) 2026 Goutham Tiwari`. README skeleton with pitch, "status: in progress", results markers, "not tax advice" line, footer "Built by Goutham Tiwari".
- [ ] Kit seed copied to `data/seed/` byte for byte; `uv run --with reportlab python -I data/seed/make_seed_data.py` regenerates the same 18 labels and 17 PDFs (byte-identical CSV/JSON/labels).
- [ ] CI (ubuntu, Python 3.12, `astral-sh/setup-uv`): sync, ruff check, ruff format --check, pytest; second job runs the authorship audit with `fetch-depth: 0`.
- [ ] First commit; `git log --format=fuller` shows author and committer Goutham Tiwari, no trailer; audit prints OK; `gh repo create GouthamTiwari/agent-orchestrator --public --source . --remote origin`, push, CI green.

## 02 · Domain models, money and GSTIN helpers
**Files:** `src/agent_orchestrator/models/*.py`, `money.py`, `gstin.py`, `tests/test_models.py`, `tests/test_money.py`, `tests/test_gstin.py`
**Depends on:** 01
**Why:** every agent speaks these types; money as paise integers from day one.
**Acceptance:**
- [ ] All types of spec 5 with `extra="forbid"`; enums exactly as listed; `RunState` with reducers on every branch-written field.
- [ ] `money.parse("4257.0") == 425700`; floats rejected; > 2 decimals rejected; format back to `₹4,257.00`.
- [ ] `gstin.checksum_valid`, `check_char`, `state_code`, `pan_part`, `edit_distance_one` with the kit's reference implementation as test oracle (hypothesis: 1,000 random valid GSTINs pass; any single-char substitution fails or is a different valid GSTIN).
- [ ] Test: a test helper lists `RunState` fields without reducers; used later by Task 13.

## 03 · Dataset generator, splits and labels
**Files:** `src/agent_orchestrator/eval/generate.py`, `eval/scan_noise.py`, `data/splits.json`, `data/eval_config.yaml`, `tests/test_generate.py`, `cli.py` (dataset commands only)
**Depends on:** 02
**Why:** the evaluation needs ~600 labelled cases with extraction ground truth; dev/test split by business.
**Acceptance:**
- [ ] `orchestrator dataset generate` writes 30 business-months (10 businesses × 3 months) to `data/generated/` deterministically from one seed (two runs → identical hashes).
- [ ] Case mix per spec 15.1: all 11 seed types, 8 injection variants spread across months, benign hard negatives, both rates, intra/inter-state, 40% noisy scans, Hindi/Telugu supplier profiles.
- [ ] Labels per spec 15.1 including `auto_ok`, `injection`, `extraction_truth`; `auto_ok` true for clean `missing_in_2b` (decision 1).
- [ ] Every PDF says "Synthetic document generated for software testing"; every company name says "(fictional)"; GSTINs random but checksum-valid.
- [ ] `splits.json`: dev B01–B05, test B06–B10; a test proves no business or supplier GSTIN appears in both splits.
- [ ] `dataset review-sample` writes a 10% stratified sample CSV; `apply-review` applies a filled CSV and appends to `docs/DATASET_CHANGELOG.md`.

## 04 · Intake agent
**Files:** `src/agent_orchestrator/agents/intake.py`, `tests/test_intake.py`
**Depends on:** 02
**Why:** clean, typed records and data-quality issues before anything expensive runs.
**Acceptance:**
- [ ] Spec 4.1, 4.2, 4.4 and 6.1 exactly; `input_hash` stable across OS line endings.
- [ ] Seed month: 18 book lines, 17 portal rows, 1 checksum-invalid GSTIN issue (case13), duplicate line hashes detected.
- [ ] Bad CSV header → clear error; a bad row → `DQIssue` with line number, run continues.
- [ ] Normalisation table tests: `ABC/2026/0107`, `abc-2026-107`, ` abc 2026 0107 ` → same `inv_key`; FY tokens dropped only in `inv_key_noyear`.

## 05 · Matcher
**Files:** `src/agent_orchestrator/agents/matcher.py`, `tests/test_matcher.py`
**Depends on:** 04
**Why:** deterministic matching is most of the work and must be exactly right on money.
**Acceptance:**
- [ ] All five passes, duplicate detection, compare rules and priority of spec 6.3; comparisons in paise with `<=`.
- [ ] On the seed month (without PDFs), predicted mismatch type equals the label's underlying type for all 18 cases.
- [ ] Hypothesis: shuffling input order never changes the result; a pair is never matched twice; every book line and portal row ends in exactly one case.
- [ ] Ties in fuzzy matching → `needs_human`, never a guess.

## 06 · Store and tracing
**Files:** `src/agent_orchestrator/store/db.py`, `store/repo.py`, `store/idempotency.py`, `tracing.py`, `tests/test_store.py`, `tests/test_tracing.py`
**Depends on:** 02
**Why:** approvals, outbox, ledger and spans need one durable, idempotent place.
**Acceptance:**
- [ ] Tables of spec 9.1 created by an idempotent migration; WAL mode; foreign keys on.
- [ ] Idempotency keys per spec 9.2; inserting the same approval/outbox key twice returns `created=False` and the same row (also under 8 threads).
- [ ] Spans: start/finish context manager records agent, case_key, input hash, tokens, USD, latency, error; JSON export.
- [ ] `audit_log` is append-only (no update/delete function exists; test checks the repo API).

## 07 · LLM client layer (**LIVE: probes only, ≈ USD 0.01**)
**Files:** `src/agent_orchestrator/llm/client.py`, `llm/pricing.py`, `llm/budget.py`, `llm/retry.py`, `llm/ledger.py`, `llm/recording.py`, `llm/fake.py`, `tests/test_llm_*.py`, `tests/live/test_bedrock_live.py`
**Depends on:** 06
**Why:** one place for cost, budgets, retries, caching and replay, so agents stay simple.
**Acceptance:**
- [ ] Bedrock Converse client (text + PNG/JPEG image blocks), Anthropic client (`claude-haiku-5-5`, image blocks, no `temperature`), fake client (scripted responses, call counter). All return `LLMResult(text, tokens_in, tokens_out, usd, latency_ms, model_id, request_hash)`.
- [ ] Pricing table per spec 2.2 with `PRICES_CHECKED_ON`; unknown model raises.
- [ ] Budget: pre-call estimate check; 90% → `stop_new_llm_work`; project cap from `.spend/ledger.jsonl`; separate Anthropic cap.
- [ ] Retry: throttling/5xx/timeout backoff with full jitter (spec 8.3); `ValidationError` never retried here (test).
- [ ] Ledger: same request hash → stored response, no call, USD 0 (test with the fake's counter). Record/replay: replay mode with a missing recording raises.
- [ ] Pacer limits requests per second (uses `time.perf_counter`, not `monotonic`: Windows tick issue found in guardrails).
- [ ] LIVE probes (`pytest -m live`, after his yes): Nova 2 Lite reads the seed invoice image; Nova Pro (`apac.`) answers from `ap-south-1`; send 6 images in one Nova request to learn the real embedded-image limit; record results in the changelog and update spec 2.2 if needed.

## 08 · Interfaces, stubs and adapters
**Files:** `src/agent_orchestrator/interfaces/protocols.py`, `interfaces/stubs.py`, `interfaces/adapters.py`, `interfaces/injection_rules.yaml`, `tests/test_interfaces.py`
**Depends on:** 02
**Why:** the earlier projects plug in through three small shapes; the stubs make CI independent of them.
**Acceptance:**
- [ ] Protocols and stubs per spec 12. Stub retriever ranks the saved CBIC texts (sections split into snippets with stable ids like `cgst_act_s16#2aa`).
- [ ] Stub injection rules catch all 8 generator attack variants on dev data and flag none of the benign hard negatives (measured on dev only).
- [ ] Adapters import `llm_guardrails`, `regrag`, `semcache` lazily; missing package → clear message, stub used; a test runs each adapter only if the package is importable.
- [ ] `llm-guardrails` added as a git dependency pinned to `v1.0.0` (first check `gh repo view GouthamTiwari/llm-guardrails --json visibility` says PUBLIC); CI installs it and a test shows the stub and the real guard both catch the 8 attack variants.

## 09 · Extraction agent
**Files:** `src/agent_orchestrator/agents/extraction.py`, `llm/prompts/extract_system.txt`, `llm/prompts/extract_user.txt`, `src/agent_orchestrator/render.py`, `tests/test_extraction.py`, `data/recordings/extract_seed.jsonl`
**Depends on:** 07, 08
**Why:** reads the untrusted invoice into checked, typed fields.
**Acceptance:**
- [ ] Steps 1–7 of spec 6.2; render with pypdfium2 at 150 DPI.
- [ ] Fake-LLM tests: invalid JSON → 2 retries then `extraction_failed`; wrong GSTIN checksum lowers confidence; low confidence triggers exactly one escalation; injection in `other_text`, in a field, or only in the text layer sets `flagged`.
- [ ] The prompt contains no document text other than the image; `other_text` never leaves the extraction result except to the scanner, store and escaped UI.
- [ ] Seed recording made in Task 18 is replayable here; until then tests use the fake.

## 10 · Policy engine and policy file
**Files:** `src/agent_orchestrator/policy/engine.py`, `policy/schema.py`, `policy/rules/policy_v1.yaml`, `tests/test_policy.py`, `tests/test_policy_properties.py`, `tests/test_policy_doc_sync.py`
**Depends on:** 02
**Why:** the decision boundary: rules decide, the model never does.
**Acceptance:**
- [ ] YAML schema and evaluator per spec 7.1–7.2; loading fails for a legal rule without `url` + `checked_on`; no `eval`.
- [ ] Every rule in `docs/POLICY_RULES.md` section 2 exists in the YAML with the same id, kind and outcome (sync test parses the Markdown tables).
- [ ] Re-check the IGST s.19 full text and record it in `data/rules/sources/igst_act_s19.txt`.
- [ ] Property tests of spec 7.3 (2,000 examples; 20,000 under `-m slow`), including "free text never changes the outcome".
- [ ] Seed month decisions: case01–08 auto_allow; case09–10 auto_allow (decision 1); case11–17 require_approval; case18 block.

## R1 to R4 · Regulatory RAG upgrade (in `..\regulatory-rag`, its own branches and changelogs)

Built after Task 10 and before Task 11, following that repo's own loop (branch, changelog, self-review, squash-merge, push). Spec 12.1.

- **R1 · Bare law ingest (LIVE: Titan embeddings, about USD 0.02):** CGST Act, CGST Rules and IGST Act from the CBIC section pages into the manifest as `doc_type` act/rules, with "updated as on" dates; the downloader is polite (1 request/s, no fake browser user agent).
- **R2 · Section-aware chunker and exact reference lookup:** one chunk per sub-section with a citable `section_no`; questions naming a section or rule fetch that chunk first. Tests with "s.16(4)", "rule 36(4)", "section 17(5)(d)".
- **R3 · Newer documents (LIVE: embeddings, about USD 0.01):** CGST circulars after 250/07/2025 and the GSTN GSTR-2B / IMS advisories; manifest entries carry source URLs and sha256.
- **R4 · Library API, eval and tag (LIVE, about USD 0.15):** `regrag.api.search` and `regrag.api.answer`; about 20 new golden questions on the Act and Rules; hybrid+rerank re-run; README row added; tag `v1.1.0`.

## 11 · Rule research and resolution planner
**Files:** `src/agent_orchestrator/agents/research.py`, `agents/planner.py`, `llm/prompts/research_*.txt`, `llm/prompts/plan_*.txt`, `tests/test_research.py`, `tests/test_planner.py`
**Depends on:** 07, 08, 10
**Why:** cited reasoning, strict JSON, and proposals limited to allowed actions.
**Acceptance:**
- [ ] Spec 6.4 and 6.5; cache key `(mismatch_type, corpus_version)`; hit rate exposed.
- [ ] With `regrag` v1.1.0 installed, rule research uses `regrag.api.search`; without it, the saved CBIC text. Both paths tested (the RAG path is skipped when the package or its database is missing).
- [ ] Unknown `rule_ref` or `evidence_ref`, or an action outside the allowed list → schema retry → fallback `escalate_human` with `planner_failed`.
- [ ] A test asserts the planner prompt contains none of the document's `other_text` (uses the injection seed invoice's note text as the needle).

## 12 · Communicator and reporter
**Files:** `src/agent_orchestrator/agents/communicator.py`, `agents/reporter.py`, `llm/prompts/email_*.txt`, `llm/prompts/summary_*.txt`, `templates/email_fallback_*.txt`, `tests/test_communicator.py`, `tests/test_reporter.py`
**Depends on:** 06, 07
**Why:** drafts that only state facts we hold, and a report with numbers we can trace.
**Acceptance:**
- [ ] Spec 6.7 and 6.8; number grounding check falls back to the template; English/Hindi/Telugu templates.
- [ ] No SMTP or network code; grep test for `smtplib`, `sendmail`, `requests.post`.
- [ ] Report totals equal a hand-computed seed-month table in the test.

## 13 · Orchestrator graph, checkpoints and budgets
**Files:** `src/agent_orchestrator/orchestrator/state.py`, `orchestrator/graph.py`, `orchestrator/runner.py`, `tests/test_graph.py`, `tests/test_crash_resume.py`, `cli.py` (run/resume)
**Depends on:** 05, 09, 10, 11, 12
**Why:** the agents become one durable, resumable run.
**Acceptance:**
- [ ] Graph of spec 8.1 with `Send` fan-out, joins and conditional routes; nodes only call `agents/*` functions (a test greps `agents/` for `langgraph` imports: none).
- [ ] `SqliteSaver`, `thread_id = run_id`, `durability="sync"`.
- [ ] `orchestrator run --demo` completes the seed month with the fake LLM and pauses at the approval gate.
- [ ] Crash-resume test of spec 15.4 passes (subprocess, `ORCH_CRASH_AFTER_LLM_CALLS`).
- [ ] Budget test: a tiny budget stops new LLM work at 90%, remaining cases `needs_human` with `budget_stop`, report states which limit.
- [ ] Reducer test: no `Send` branch writes a field without a reducer.

## 14 · Approval gate, queue and resume
**Files:** `src/agent_orchestrator/orchestrator/graph.py` (gate node), `tools/toolfns.py`, `tests/test_approvals.py`, `tests/test_idempotent_rerun.py`, `cli.py` (approvals commands)
**Depends on:** 13
**Why:** the human-in-the-loop part, durable across processes.
**Acceptance:**
- [ ] Gate per spec 8.1 (no writes before `interrupt()`; database is the source of truth; partial decisions re-interrupt).
- [ ] Write tools of spec 10 refuse calls without the right `Decision` (tests for each refusal; `audit_log` row written).
- [ ] Approve / edit (allowed options only) / reject (reason required) via CLI; who/when/before→after logged; edits and rejections appended to `eval/human_feedback.jsonl`.
- [ ] Resume in a new process works; idempotent re-run test of spec 15.4 passes.
- [ ] The injection seed invoice: block, no approval row that could approve the LLM action, no outbox row, report lists it as blocked.

## 15 · MCP server
**Files:** `src/agent_orchestrator/tools/mcp_server.py`, `tests/test_mcp.py`, `docs/MCP.md`
**Depends on:** 14
**Why:** the same tools, usable from Claude Desktop or Claude Code.
**Acceptance:**
- [ ] Tools of spec 10 on `MCPServer` (mcp 2.3.x); argument schemas from type hints; tests call them through the SDK's in-process `Client` (verify the in-memory transport API in the installed version).
- [ ] Write tools via MCP obey the same refusals and idempotency (shared `toolfns`).
- [ ] `docs/MCP.md` shows the Claude Desktop config (`uv run orchestrator mcp`) and a sample session.

## 16 · Web UI, trace timeline and metrics
**Files:** `src/agent_orchestrator/web/app.py`, `web/templates/*.html`, `web/static/htmx.min.js`, `web/static/app.css`, `web/metrics.py`, `tests/test_web.py`, `cli.py` (serve)
**Depends on:** 14
**Why:** what a real accountant would use; also the demo GIF.
**Acceptance:**
- [ ] Pages and evidence view of spec 11; edit dropdown limited to allowed actions; resume button works end to end (test client).
- [ ] XSS test: an invoice whose `other_text` is `<script>alert(1)</script>` renders escaped.
- [ ] Form token required on POST; server binds 127.0.0.1.
- [ ] `/metrics` exposes the counters of spec 11; tripwire test: executing an action without an allowing decision increments `orchestrator_unsafe_auto_actions_total` and is refused.
- [ ] Ask box (spec 11) and `orchestrator ask`; a test with a fake `regrag.api.answer` shows citations render and that no document text is sent.
- [ ] Checked by hand in the browser pane: screenshot of the approval page in the changelog.

## 17 · Eval metrics and offline runner
**Files:** `src/agent_orchestrator/eval/metrics.py`, `eval/run.py`, `eval/report.py`, `tests/test_eval_metrics.py`, `cli.py` (eval)
**Depends on:** 03, 14
**Why:** exact formulas, tested on tiny hand-made examples before real numbers exist.
**Acceptance:**
- [ ] Every formula of spec 15.2 with a hand-computed test.
- [ ] `orchestrator eval --suite e2e --split dev` runs offline on the fake LLM (smoke) and writes reports to `reports/tmp/`.
- [ ] Reports regenerate byte for byte in replay mode (test).

## 18 · Dev-split live run and tuning (**LIVE ≈ USD 1.2**)
**Files:** `data/recordings/dev_*.jsonl`, `reports/dev/*`, `data/eval_config.yaml` (tuned values), `docs/changelogs/18-dev-run.md`
**Depends on:** 17
**Why:** real numbers on dev, tuning of tolerance, fuzzy threshold and confidence threshold; test split untouched.
**Acceptance:**
- [ ] Estimate printed and approved; run in steps under 10 minutes each (foreground).
- [ ] Sweeps of spec 14 run in replay (no new spend); chosen values written to config with the reason; defaults updated and tested equal to the config.
- [ ] Review sample produced for Goutham's spot-check.

## 19 · Single-LLM baseline (**LIVE ≈ USD 0.7 on dev**)
**Files:** `src/agent_orchestrator/eval/baseline.py`, `llm/prompts/baseline_*.txt`, `tests/test_baseline.py`, `data/recordings/baseline_dev.jsonl`, `reports/dev/baseline.*`
**Depends on:** 18
**Why:** the comparison that makes the README honest.
**Acceptance:**
- [ ] Spec 15.5: 2×2 packing on Nova (or one image per invoice if Task 07 found a higher limit), split rule, same metrics, extra "baseline + policy" row.
- [ ] Baseline's unsafe auto-actions counted with the same formula.

## 20 · Robustness suite and reports
**Files:** `src/agent_orchestrator/eval/robustness.py`, `reports/robustness.*`, `tests/test_robustness.py`
**Depends on:** 18
**Why:** crash-resume, pause/resume, idempotency and injection results as committed numbers.
**Acceptance:**
- [ ] All checks of spec 15.4 run from `orchestrator eval --suite robustness` and write `reports/robustness.json/md`.
- [ ] Injection documents auto-actioned = 0 on dev; benign false-positive rate reported.

## 21 · Test-split final runs (**LIVE ≈ USD 1.6 Nova; Claude ≈ USD 0.6 of his key**)
**Files:** `data/recordings/test_*.jsonl`, `reports/*` (final), `reports/claude_main_test.*`
**Depends on:** 19, 20
**Why:** the numbers that go on the resume, run once.
**Acceptance:**
- [ ] Config frozen before the run (hash recorded in the report); Nova run on test; then, after a separate yes, the Claude Haiku 5.5 main test (pipeline + baseline) with the Anthropic price re-checked that day.
- [ ] Unsafe auto-action rate on test = 0 (if not, stop and report; no silent fixes on test data).
- [ ] Reports regenerate offline from recordings.

## 22 · README, CI polish, demo GIF, release
**Files:** `README.md`, `docs/ARCHITECTURE.md`, `docs/demo/approval-flow.gif`, `.github/workflows/ci.yml`, `tests/test_readme_numbers.py`
**Depends on:** 21
**Why:** what recruiters read first.
**Acceptance:**
- [ ] README: pitch, Mermaid diagram, 3-command quickstart (`uv sync`, `uv run orchestrator run --demo`, `uv run orchestrator serve`), results block generated from `reports/`, rules-with-citations table (from `docs/POLICY_RULES.md`), design decisions, "What this does not do", spot-check status, not-tax-advice line.
- [ ] README numbers test passes; CI green; audit OK; secret scan of the full history clean.
- [ ] Definition-of-done checklist from the brief ticked item by item in the changelog; then tell Goutham it is ready to make public (he decides; tagging a release is asked first).
