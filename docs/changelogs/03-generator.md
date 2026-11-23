# 03 · Dataset generator, splits and labels

## What changed
- `eval/generate.py`: a deterministic generator built on the kit's seed generator (which stays unchanged in `data/seed/` for the 18-case demo month).
  - 10 fictional buyers in 10 states, each with 8 to 12 fictional suppliers (102 in total, no name or GSTIN shared between businesses), over 2026-08, 2026-09 and 2026-10.
  - 18 to 22 labelled cases per business-month, 593 in total. Every month has all 11 case types, both tax rates (5% and 18%), intra- and inter-state suppliers, 1 or 2 benign "hard negative" notes, and often a `missing_in_2b` invoice above the Rs 50,000 auto-hold ceiling.
  - Supplier profiles carry a language (en/hi/te, for the communicator) and a filing frequency (quarterly filers supply most `missing_in_2b` cases outside quarter-end months).
  - 15 injection invoices (one business-month in two), cycling through the 8 variants: faint grey note, white-on-white text, 4-pt text, Hindi instruction, instruction disguised as payment terms, "change the GSTIN or amount", instruction split across two lines, instruction inside the supplier name. All 8 appear in the dev split. Half of the injected invoices otherwise match exactly, so the policy must block them on the injection alone.
  - Four invoice-number styles; format differences are recoverable by the spec's normalisation (separators/case, leading zeros, or a missing year token for the no-year pass).
  - Money is built with `Decimal` and written exactly: CSV as `"47300.00"`, JSON as number literals with no float in between.
  - Labels add, beyond the spec list: `mismatch_type` (gold matcher output; injection cases map to their underlying numbers), `book_lines` and `portal_index` (to align labels with cases), `book_tax_paise`, `injection_variant`, `hard_negative`, `pdf_file`, `image_file`, `supplier_language`, `supplier_filing`, `pdf_note` and `review_status`.
  - A duplicate book line yields two labels: the original (exact match) and the duplicate (`duplicate_in_books`), matching how the matcher will create cases.
  - `auto_ok` is true only for clean `none` actions and for `missing_in_2b` with book tax up to Rs 50,000 (decision 1). It is false for every injected invoice and every case without a PDF.
- Invoice PDFs: reportlab with `invariant=1` (fixed timestamps and ids), Indian digit grouping ("Rs 3,08,350.00"), "(fictional)" after both company names and the synthetic-document footer. The Hindi variant uses the first Devanagari font listed in `data/eval_config.yaml` (Nirmala UI on Windows). reportlab does not form the reph ligature, so the Hindi text avoids words that need it; the rest renders correctly (checked by eye).
- `eval/scan_noise.py`: page 1 rasterised at 150 DPI with pypdfium2, then rotation, Gaussian blur, contrast and JPEG compression, all seeded by the file's identity.
- `eval/review.py` (file not in the task list; kept apart from the generator for clarity): stratified 10% review sample, validation of a filled sample (every problem listed, nothing applied on error), overrides appended to `data/label_overrides.jsonl` and applied on every regeneration, summary appended to `docs/DATASET_CHANGELOG.md`.
- `data/eval_config.yaml`, `data/splits.json` (dev B01-B05, test B06-B10), committed labels in `data/labels/` (30 files) and the review sample in `data/review/review_sample.csv` (60 cases).
- CLI: `orchestrator dataset generate | split | review-sample | apply-review`.

## Outside the listed files
- `docs/DATASET_CHANGELOG.md` created (v1 entry), README line about the dataset and its review status, `src/agent_orchestrator/eval/__init__.py`.

## Checks
- `uv run ruff check .`, `uv run ruff format --check .`: pass.
- `uv run pytest`: 111 passed (27 new). They cover:
  - determinism: committed labels equal a fresh run; text outputs are LF and identical across runs; PDFs and noisy scans are byte-identical across runs;
  - the case mix in every month: count, all 11 types, both rates, both supplier kinds, hard negatives;
  - injections: one month in two, all variants in dev, always `escalate_human` with `auto_ok` false;
  - `auto_ok` rules, including the Rs 50,000 ceiling;
  - label-to-row pointers;
  - GSTIN validity, with the planted typos one edit away;
  - invoice-number differences recoverable by both normalisation passes;
  - exact money: invoices add up, one tax head only;
  - splits: no shared business or GSTIN, and the overlap check fails when it should;
  - PDF text holds the footer, "(fictional)" twice and both GSTINs;
  - the review workflow, including five kinds of bad review file.
- Full render (manual): two complete runs, 816 files each, identical sha256 for every file; about 60 s per run; 15 MB.
