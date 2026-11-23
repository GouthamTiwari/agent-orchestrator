# 05 · Matcher

## What changed
- `agents/matcher.py`: `match(books, portal, business_id, period, extractions=None) -> list[Case]`.
  - **Duplicates first:** book lines with the same (GSTIN, `inv_key`) after the first become `duplicate_in_books` cases (status `duplicate`); the first line matches normally.
  - **Five passes, one-to-one, each on what is still unmatched:** exact (raw number), normalised (`inv_key`), no-year (`inv_key_noyear`), GSTIN repair, fuzzy.
    - GSTIN repair needs a checksum-failing book GSTIN, a portal GSTIN that passes and is one substitution or adjacent swap away, and the same `inv_key`. If two corrections fit, it does not guess.
    - Fuzzy needs the same PAN, `fuzz.ratio >= 90` on the keys, and taxable values within 1%.
  - **Order independence:** inputs are sorted by line number and file index. The fuzzy pass accepts a pair only when it is the single best choice for both the book line and the portal row, repeated until nothing changes. Anything left with a tie becomes `needs_human` with flag `ambiguous_match`; a tie is never resolved by guessing.
  - **Comparison of a matched pair** in paise against the Rs 1.00 tolerance (`<=`). Priority: GSTIN typo > tax head > taxable value > tax amount > other period. Then: invoice-number format difference (normalised, no-year or fuzzy pass, status `matched_after_normalisation`) > rounding > none.
  - **Leftovers:** book lines become `missing_in_2b`, portal rows become `in_2b_not_in_books`.
  - **Flags:**
    - `gstin_invalid` (books GSTIN fails the checksum);
    - `fuzzy_match`;
    - `ambiguous_match`;
    - from the PDF, when an extraction is given: `injection`, `low_confidence` (below 0.85), and `pdf_disagrees` (the PDF disagrees with every side present on taxable value, total tax, GSTIN or invoice key).

    The PDF never changes the mismatch type or status: it is evidence, not a vote.
  - `case_key` = sha256(business, period, the corrected supplier GSTIN, the supplier's `inv_key`)[:16]. Duplicates add `dup<n>`; a rare collision (the same invoice reported twice) gets a deterministic suffix.
  - `facts_hash` covers the type, status, diffs, the book row hash, the portal row, the flags and the extraction (without `other_text`). It is used by idempotency keys later.
  - `itc_at_risk_paise` = book tax for every case that is not matched (0 when there is no book line).
- `diffs` are book minus portal, per head, plus `total_tax`.

## Checks
- `uv run ruff check .`, `uv run ruff format --check .`: pass.
- `uv run pytest`: 162 passed (23 new).
  - **Seed month:** 19 cases (the 18 labels plus the original line of the duplicate pair). Every label gets its underlying type: the injection case is `taxable_value_mismatch`; the injection itself is the policy's job.
  - **All 593 generated labels:** mismatch type, status and portal row agree exactly.
  - **Hypothesis:** 60 random reorderings of real months give identical cases. 30 random subsets keep every book line and portal row in exactly one case, never pair a row twice and keep case keys unique.
  - **Rules:** the Rs 1.00 boundary (100 paise is rounding, 101 is a mismatch, both signs), the priorities, the other-period case, GSTIN repair, and refusing to repair when two corrections fit.
  - **Fuzzy:** acceptance with flag, the 1% taxable limit, ties going to `needs_human`. Also duplicates, ITC at risk, and every PDF flag, including that a PDF agreeing with one side is not "disagreeing".
