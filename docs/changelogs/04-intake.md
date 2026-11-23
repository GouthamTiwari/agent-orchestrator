# 04 · Intake agent

## What changed
- `agents/intake.py`: reads one business-month into typed records with no model call.
  - **Books** (`purchase_register.csv`): the exact header is required (a wrong header stops the run and prints expected vs found). Each data row gets a 1-based `line_no`, a normalised GSTIN, integer-paise amounts, `inv_key`, `inv_key_noyear` and a 12-character content hash (`row_hash`, over the normalised key fields, not the supplier name or PDF path).
  - **Supplier statement** (`gstr2b_like.json`): numbers are read as `Decimal` straight from the JSON text (never through float). The buyer GSTIN must pass the checksum and the `period` must match the run, otherwise the run stops. `PortalRecord.index` stays the position in the file even when an earlier row is skipped.
  - **Row problems are data, not crashes.** A row whose key fields cannot be read is left out with a `DQIssue` naming the line and field (`bad_date`, `bad_amount`, `bad_rate`, `bad_row`, `missing_invoice_no`, `missing_fields`, `bad_period`) plus `row_skipped`.
  - **Soft issues keep the row**: `gstin_checksum`, `duplicate_row` (same content as an earlier line), `tax_heads_mixed`, `cgst_sgst_unequal`, `tax_not_rate_x_taxable` (more than Rs 2 off), `unusual_rate`, `pdf_missing`.
  - **`input_hash`**: sha256 over the books, the statement and the invoice files sorted by name. Text files have their BOM removed and CRLF turned into LF first, so the hash is the same on Windows and Linux.
  - **`inv_key`**: uppercase, split on non-alphanumerics, leading zeros removed from numeric tokens, joined with `|`. **`inv_key_noyear`** also drops year and financial-year tokens (`2026`, `26-27`, `2026-27`, `2026/2027`, `FY26`, `FY2026-27`). A pair counts as a financial year only if the second year follows the first, so `12-14` stays. The key is never empty.
- `models/records.py`: `rate` may now be up to 40 (GST has had a 40% slab since 22 September 2025; the old cap was 28).

## Outside the listed files
- `docs/TASKS.md`: the seed has 16 supplier rows, not 17 (18 cases minus the 2 missing from 2B); acceptance text corrected.
- `models/records.py` rate bound (above).

## Checks
- `uv run ruff check .`, `uv run ruff format --check .`: pass.
- `uv run pytest`: 139 passed (28 new).
  - Key normalisation table, including financial-year edge cases.
  - Seed month: 18 book lines, 16 supplier rows, the only checksum failure on line 13 (`TES/2026/0112`, case13), the duplicate pair `FIC/2026/0114` with equal hashes and different line numbers.
  - `input_hash` unchanged by CRLF and BOM, changed by a different invoice file.
  - File-level errors: header, period, buyer GSTIN, unexpected buyer, broken JSON, missing file, bad period text.
  - Row-level issues for books and supplier rows; exact decimals from JSON.
  - All 30 generated months read cleanly: the row counts match the labels, and every data-quality issue sits on the case type that should cause it. Across the dataset that is 30 GSTIN typos, 30 duplicates, 30 inflated-tax rows and 18 unequal CGST/SGST roundings. All 30 invoice-number format differences are recoverable (26 by `inv_key`, 4 by `inv_key_noyear`).
