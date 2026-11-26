# 11 · Rule research and resolution planner

## What changed
- `agents/research.py` (spec 6.4):
  - **One fixed query per mismatch type** (8 non-clean types). Each names the provisions that matter, such as "section 16(2)(aa)" and "rule 36(4)", so both retrievers put them first.
  - **Retrievers:** Regulatory RAG `regrag.api.search` (retrieval only, no LLM) when it is installed and its database answers; otherwise BM25 over the saved CBIC texts.
  - **Cache key** `research:v1:<mismatch_type>:<corpus_version>`. This is the spec's pair plus a version number, which is bumped when a query or the prompt changes so stale entries are not reused. `corpus_version` comes from the retriever that answered, so RAG and snapshot results never mix. `hit_rate(cache)` exposes the hit rate.
  - **The summary model** may cite only the snippet ids it was given, in the id list and as `[id]` in the text. An unknown id, no citation or bad JSON is re-asked with the problems (at most 2 retries). After that the result has no summary (`note="summary_failed"`) but keeps the snippets for the planner. Failures are not cached, so the next run tries again.
  - **Fails closed:** a retriever failure (`retrieval_failed`) or no snippets (`no_snippets`) means no model call. A budget stop and a provider error give no summary. Configuration errors (missing recording, unknown model) still stop the run, as in Task 09.
- `agents/planner.py` (spec 6.5):
  - **Clean matches** (`none`, `rounding_within_tolerance`, `invoice_no_format_difference`) get a rule proposal of `none` and no model call.
  - **The prompt has facts built by code only:**
    - money in paise;
    - match pass and score;
    - flags;
    - extraction check names;
    - invoice amounts as integer paise and the rate as a number;
    - the allowed actions from the policy file, with their meaning;
    - the evidence refs that exist for the case (`books:line:N`, `2b:index:N`, `pdf:field:<name>` for fields that were read);
    - the rule passages and the research summary.

    **No document string reaches the prompt:** not `other_text`, not the supplier name, not the invoice number.
  - **Checks on the answer:**
    - the action is allowed for the mismatch type;
    - there is at least one evidence ref, and every one exists for the case;
    - there is at least one rule ref, and every one is a passage that was shown.

    Any violation is re-asked (at most 2 retries). After that, the fallback is `escalate_human` with `proposer="rule"`, `fallback=True` and flag `planner_failed`, and the policy then requires approval (SAFE-PLAN-01).
  - **No rule text** gives the same fallback with `no_rule_text` and no model call. A budget stop gives `budget_stop`; a provider error gives `planner_failed` and `llm_error`.
- `llm/prompts/research_system.txt`, `research_user.txt`, `plan_system.txt`, `plan_user.txt`: the prompts. Values are filled with `string.Template`, so braces or dollar signs in law text cannot break them.

## Outside the listed files
- `pyproject.toml`, `uv.lock`: an optional extra `rag = ["regrag"]`, pinned to the public tag `v1.1.0` (`uv sync --extra rag`). CI runs `uv sync --frozen` without extras, so it uses the snapshot path.
- `interfaces/adapters.py`, `RegragRetriever`:
  - **Probes the database once on creation;** if it is not reachable, `load_retriever()` falls back to the snapshot with a note that names only the error type.
  - **Reads `corpus_version` once per run.**
  - **`REGULATORY_RAG_ENV_FILE`:** an installed package cannot find its project's `.env` by itself, so this variable points to regrag's `.env`. Only regrag's database and model settings are copied (never AWS keys, which this project's `.env` has), and nothing already set is overwritten.
- `.env.example`: `REGULATORY_RAG_ENV_FILE=` with how to set it up. The local `.env` got the path line; no values were printed.
- `tests/test_interfaces.py`:
  - **The Task 08 regrag test is now `live`.** With the package installed it calls AWS, and it skips if the database is down.
  - **Three new offline tests:** fallback when the database is unreachable, `corpus_version` read once, and the env-file copy rules.

## Found while testing
- **regrag's reranker is paced to this account's quota** (about 2 requests a minute in Tokyo), so each uncached RAG search waits about 30 s. The 8 research queries take about 4 minutes the first time. After that the research cache hits and no search runs. This is fine for the cached design, and slow for free-text questions (later `orchestrator ask`): to note in the README.

## Checks
- `uv run ruff check .`, `uv run ruff format --check .`: pass.
- `uv run pytest`: 412 passed, 1 skipped (semcache not installed), 7 deselected (63 new: research 33 + 2 live, planner 27, interfaces 3 + 1 now live).
  - **Snapshot path:** every query finds 4 passages and the provisions it names (s.16(2)(aa) and r.36(4) for missing_in_2b, s.16(2)(b), s.16(4) and r.60(7), s.77 and IGST s.19). Also covered: cache hit on the second call and hit rate 0.5, no reuse across corpus versions, retries and fallback, budget stop, provider error, configuration error, retriever failure.
  - **Planner:**
    - clean matches skip the model;
    - 8 kinds of violation are each retried, then fall back after 3 calls;
    - the fallback is decided `require_approval` with SAFE-PLAN-01;
    - allowed actions follow the policy file;
    - invoice amounts arrive as paise.
  - **Acceptance needle test:** the injection seed invoice IMA_2026_0117 is read through the real extraction agent, with its planted note in `other_text` and an attack in the supplier name. Across three planner calls (two retries), no prompt contains the note, "pre-verified", "approve full ITC", the supplier name or the invoice number.
- LIVE (`uv run pytest -m live tests/test_research.py tests/test_interfaces.py::test_regrag_adapter_when_installed`): 3 passed in 303 s against the real database (corpus `regrag-261b4851cca9`). Every query returned 4 GST passages. missing_in_2b got CGST Act 16(2)(aa) and CGST Rules 36(4), date_mismatch_other_period got 16(4), and tax_head_mismatch reached the IGST Act. Research with a fake summary model cited regrag snippet ids and hit the cache on the second call. About 11 searches (embedding + rerank), under USD 0.01; no LLM call.

## Out of scope / follow-ups
- Wiring research and planning into the graph (Task 13); the research cache file path comes with the runner.
- A seed recording of real research and planner answers comes with the live runs (Task 18).
