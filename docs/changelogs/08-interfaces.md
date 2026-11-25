# 08 · Interfaces, stubs and adapters

## What changed
- `interfaces/protocols.py`: three Protocols.
  - **`Guard`:** `scan(texts) -> InjectionResult` and `validate(text, model) -> ValidationResult`. `scan` takes a mapping of source name to text instead of a plain list, so `InjectionResult.sources` can name where a hit was (`text_layer`, `other_text`, `field:supplier_name`).
  - **`RuleRetriever`:** `search`, `get` and `corpus_version`.
  - **`Cache`:** `get`, `set` and `stats`.
- `interfaces/stubs.py`:
  - **`RegexGuard`:** the invoice injection rules from `injection_rules.yaml` (17 rules: 11 English, 3 Hindi, 3 Telugu). Text is normalised first (NFKC, zero-width characters removed, whitespace collapsed so a split instruction still matches).
  - **`validate_json`:** cuts JSON out of ```json fences or surrounding prose, refuses duplicate keys, and validates with Pydantic. Errors name the field and rule, never the value, so document text is not echoed back into a prompt.
  - **`SnapshotRetriever`:** splits the nine saved CBIC texts into 84 snippets with ids that follow the law's numbering (`cgst_act_s16#2aa` is s.16(2)(aa)), then runs BM25 search with deterministic tie-breaks. A query that names a provision ("rule 36(4)") puts that snippet first. `corpus_version` is a hash of the texts.
  - **`SqliteCache`:** an exact-key JSON cache with hit and miss counts.
- `interfaces/adapters.py`:
  - **`GuardrailsGuard`:** llm-guardrails 1.0.0's generic layer-1 rules together with the invoice rules, plus its JSON extractor.
  - **`RegragRetriever`:** `regrag.api.search` and `corpus_version`. This module arrives with Regulatory RAG v1.1.0 (tasks R1–R4); until then the loader falls back with a clear note.
  - **`SemcacheCache`:** semantic-cache's in-memory store with exact-then-similar lookup (cosine similarity ≥ 0.92 by default, Titan embedder by default, any embedder injectable).
  - **Loaders:** `load_guard`, `load_retriever` and `load_cache` return the adapter or the stub, with a one-line note. Packages are imported lazily.
- `pyproject.toml`: `llm-guardrails` as a git dependency pinned to tag `v1.0.0` (checked public with `gh` first). Ruff ignores SIM905 (as in llm-guardrails).
- `data/injection/paraphrase_check.jsonl`: 12 reworded attacks (English, Hindi, Telugu, an HTML comment) and 12 benign invoice notes. Written after the rules and never used to tune them.

## Findings
- **llm-guardrails' generic rules alone catch 2 of the 8 invoice attack styles** (the "note to AI" and "ignore previous" ones). Invoice-specific tricks need domain rules. A test pins this result for v1.0.0.
- **On the 8 styles, the adapter catches 8/8 and flags 0 of the 6 benign hard negatives.** Checked on rendered dev invoices too: all 8 injected PDFs flagged from their text layer, 0 of 25+ clean and benign PDFs flagged.
- **Keyword rules do not generalise:** on the 12 reworded attacks the adapter catches 3, with 0 of 12 false positives. Recorded in spec 13, with the reason this is acceptable (the later layers are the real defence). Task 18 now also measures llm-guardrails' embedding and classifier layers on the same set.
- **semantic-cache works through the adapter:** checked once locally with its offline embedder (`uv run --with ../semantic-cache`). An exact repeat hit, a reworded question hit, and an unrelated question missed. In CI that test is skipped because the package is not installed.

## Outside the listed files
- `pyproject.toml` and `uv.lock` (dependency), `data/injection/paraphrase_check.jsonl`, and `docs/SPEC.md` (sections 12 and 13) plus `docs/TASKS.md` (Tasks 08 and 18) for the findings above.

## Checks
- `uv run ruff check .`, `uv run ruff format --check .`: pass.
- `uv run pytest`: 264 passed, 2 skipped (the regrag and semcache adapter tests, which run only when those packages are installed), 3 live deselected. 43 new tests:
  - all four implementations satisfy the protocols;
  - every attack style is caught by the stub and by the adapter;
  - no rule fires on any benign note;
  - the guardrails 2-of-8 finding;
  - split lines and zero-width characters;
  - rendered dev PDFs;
  - the paraphrase floor;
  - validation with fences, prose, duplicate keys, and errors that never echo the value;
  - snippet ids, titles and URLs, the rule 46(i) clause, named-provision ranking, deterministic search, and a corpus version that follows the text;
  - the cache surviving a restart;
  - loader fallbacks.
