# 02 · Domain models, money and GSTIN helpers

## What changed
- `money.py`: amounts become integer paise. `parse` accepts text (`"4257.0"` -> `425700`), `Decimal` or whole-rupee `int`; it rejects floats, booleans, more than two decimal places, negatives (unless `allow_negative=True`), non-finite values and amounts of Rs 10^12 or more. `format_inr` prints Indian grouping (`₹1,18,000.00`). `within_tolerance` compares with the inclusive Rs 1.00 tolerance (`TOLERANCE_PAISE = 100`).
- `gstin.py`: `normalise` (uppercase, no spaces or hyphens), `check_char`, `checksum_valid`, `state_code`, `pan_part`, and `edit_distance_one` (exactly one substitution or one adjacent swap; same length).
- `models/`: every type of spec section 5, split by topic (`records`, `extraction`, `cases`, `rules`, `actions`, `state`), all `extra="forbid"` and frozen. Money fields are strict integers, so a float cannot slip in. Shared vocabularies (`MismatchType`, `CaseStatus`, `ActionKind`) are `Literal`s with tuple copies for code that needs to list them.
- `ApprovalRecord` checks its own consistency: pending has no outcome; approved keeps the proposed action; edited changes it and needs a reason; rejected needs a reason and leaves the case with a person (`escalate_human`); every action must be in `allowed_options`.
- `Decision` needs at least one fired rule id and a timezone-aware time.
- `models/state.py`: `RunState` (the graph state) with `operator.add` reducers on the six fields parallel branches write, plus `BudgetSnapshot` and `RunError`. Helpers `fields_with_reducers` / `fields_without_reducers` list which fields merge parallel writes; the graph test in Task 13 uses them.

## Outside the listed files
- `docs/SPEC.md`: the graph state now lives in `models/state.py` (the spec named both `models/` and `orchestrator/state.py`; one home avoids an import cycle), and "Reject" is defined as final action `escalate_human` (the spec required a reason but did not say what happens to the case).

## Checks
- `uv run ruff check .`, `uv run ruff format --check .`: pass.
- `uv run pytest`: 84 passed.
  - Money: parsing and formatting tables, float and bad-text rejection, round trip over 0 to 10^14 paise (hypothesis), exact agreement with `Decimal` arithmetic.
  - GSTIN: 1,000 random valid GSTINs from the kit's reference generator pass and agree with the reference check character; 1,000 random single-character substitutions are all caught (the weight-2 fold is a permutation of 0..35, so one changed character always changes the total); agreement with the reference on arbitrary strings; edit-distance table.
  - Models: all 15 models forbid extra fields and are frozen; floats and negatives rejected in money fields; JSON round trip; exact vocabularies; approval state rules (4 valid, 9 invalid); reducer lists.
  - A real five-branch LangGraph fan-out shows the reducer keeps all five writes, and that the same graph without a reducer is refused by LangGraph.
- Seed sanity (manual): the only books GSTIN that fails the checksum is `TES/2026/0112`, the seed's `gstin_typo_in_books` case, and `edit_distance_one` finds its correct portal GSTIN. Every seed amount parses.
