# 10 · Policy engine and policy file

## What changed
- `policy/schema.py`: the policy file's schema and its condition language.
  - **Conditions:** `all`, `any`, `not` and leaves with the operators eq, ne, in, not_in, lt, le, gt, ge and contains, over 21 named facts.
  - **Checked at load time:** an unknown fact, operator or mismatch type, a wrong value type, a legal rule without a source `url` and `checked_on`, a non-legal rule without `design_ref`, duplicate ids, an `allowed_actions` table that does not cover every mismatch type, and a default outcome of `auto_allow` (fail closed). There is no `eval`.
- `policy/rules/policy_v1.yaml`: version 1.0.0 with 19 rules:
  - 9 legal, each citing the CBIC source URL and the date checked;
  - 8 safety;
  - 1 business (the Rs 50,000 ceiling from decision 1);
  - 1 data quality.
  - Plus the allowed actions per mismatch type and the defaults (fail closed, 0.85 confidence, the Rs 50,000 ceiling).
- `policy/engine.py`:
  - **`build_facts`:** money, flags and confidence from the matcher and extraction only. The proposal contributes its action, its proposer, whether the action is allowed, and whether it is the planner's fallback. No usable extraction counts as confidence 0.
  - **`decide`:** every rule is evaluated; the strictest outcome that fired wins (block > require_approval > auto_allow); if nothing fired, `require_approval` with `DEFAULT-NO-RULE`. Every fired rule id and the policy version are recorded.
- `data/rules/sources/igst_act_s19.txt`: the full text of IGST Act s.19, taken from CBIC's consolidated IGST Act page (27 March 2020; s.19 carries no amendment notes). This replaces the search excerpt. The rule retriever now has 87 snippets, and a wrong-tax-head question finds s.19(1) first.

## Found by the property tests
Hypothesis found an input the first version auto-allowed but the oracle would not: a "matched" case with no book line, no PDF and no extraction. The confidence rule required a PDF and the no-PDF rule required a book line, so this case slipped between them. Real data probably never has that shape, but the boundary must fail closed for every input. Fixed: `SAFE-CONF-01` now fires whenever there is no confident extraction (none at all counts as 0), and `SAFE-NOPDF-01` whenever there is no PDF.

## Added rule
- `SAFE-MATCH-01` (require approval): the pair was matched by similarity (the fuzzy pass), the match was ambiguous, or the matcher already marked the case `needs_human`. A person confirms it is the same invoice before anything is closed. The spec's oracle asks for a *clean* match; a fuzzy match is not one.

## Outside the listed files
- `docs/POLICY_RULES.md`:
  - the IGST s.19 source row (URL and the 2026-11-25 check);
  - the SAFE-CONF-01 and SAFE-NOPDF-01 wording;
  - the SAFE-MATCH-01 row.
- `data/rules/sources/igst_act_s19.txt`, `src/agent_orchestrator/policy/rules/__init__.py`.

## Checks
- `uv run ruff check .`, `uv run ruff format --check .`: pass.
- `uv run pytest`: 349 passed, 2 skipped, 4 deselected (41 new).
  - **Seed month:** case01–10 `auto_allow`, case11–17 `require_approval`, case18 `block` (fired: GST-ITC-VAL-01 and SAFE-INJ-01; block wins).
  - **All 593 generated labels:** with the expected action proposed and a clean read of each PDF, the outcome is `auto_allow` exactly when the label says `auto_ok`, and `block` exactly for the injected invoices.
  - **Loading:** 14 kinds of broken policy file refuse to load.
  - **Rules:** each safety rule; precedence; fail closed when no rule fires; the inclusive Rs 50,000 ceiling.
  - **Properties (2,000 examples each):**
    - `auto_allow` happens exactly when an independent oracle says it may (both directions);
    - changing a proposal's reason, evidence or rule references never changes the outcome;
    - unknown action names fail validation before reaching the engine;
    - the outcome is always the strictest fired rule.
  - **20,000-example run** (`POLICY_SLOW=1`): passed locally, 185 s.
  - **Doc sync:** every rule in `POLICY_RULES.md` section 2 has the same id, kind and outcome in the YAML (and the reverse), the allowed-actions table matches, and every cited source is listed in section 1.
- Not here: the "write tools refuse a call with no decision" property belongs to the tool layer (Task 15).
