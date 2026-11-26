# 12 · Communicator and reporter

## What changed
- `agents/communicator.py` (spec 6.7):
  - **`final_action` / `eligible`:**
    - A draft is made only for `chase_supplier_and_hold_itc` or `verify_with_supplier`, and only when the policy auto-allowed it or a person approved or edited the action to it.
    - Pending, rejected and blocked cases never get one. A blocked case stays blocked even if an approval record says otherwise.
  - **`email_facts`:** supplier name and GSTIN, invoice number and date, and amounts. These come from the books and GSTR-2B only; the GSTR-2B GSTIN corrects a typo in the books. The function takes no extraction at all, so text read from an invoice image (`other_text`) cannot reach a draft.
  - **Grounding check (`check_draft`):** every invoice id or GSTIN, date and number in the draft must appear in the facts (Devanagari and Telugu digits are converted and checked too). The draft must also:
    - name the invoice;
    - be in the requested script;
    - contain no link, e-mail address or markup.

    Return names such as GSTR-2B and GSTR-1 are not counted as numbers. A failed check uses the template, with no retry: it is cheap and always correct. Bad JSON is re-asked up to 2 times.
  - **Template fallback** with no model, also used when there is no client, a budget stop or a provider error. Configuration errors still stop the run.
  - **Idempotency key** `outbox_key(business, period, case, final action, supplier GSTIN)` (spec 9.2). The wording is not in the key, so a re-run finds the first draft (tested against the store).
  - **`load_supplier_languages`:** reads the generated `business.json`. English is the default, and unknown languages fall back to English.
  - **No sending code.** A test greps `src/` for `smtplib`, `sendmail`, `requests.post`, `SMTP(` and `send_message(`.
- `templates/email_fallback_{chase,verify}_{en,hi,te}.txt`: six fixed templates (`string.Template`). The verify template shows both sets of figures. Every one is tested to be fully filled and to pass the same grounding check.
- `agents/reporter.py` (spec 6.8):
  - **`build_report`:** deterministic JSON with:
    - cases by status, mismatch type, outcome and final action;
    - **ITC safe / held / at risk** in paise and ₹ (the GST on each book line, put in one bucket by the action that stands);
    - the auto-actioned list;
    - approvals: pending, approved, edited, rejected and not yet requested, plus the pending list;
    - the blocked list, with each fired rule's description as its reason;
    - data-quality issues, drafts by language, budget used and the stop reason, cost per agent;
    - `cases_detail`, so every total can be traced to its cases.
  - **`summarise`:** one model call on a totals-only JSON (no case lists, ids or invoice numbers). It is rejected if it has any number not in that JSON, or not 3 to 5 sentences; the summary is then left out. Budget stops and errors also leave it out and never stop the report.
  - **`render_markdown`, `write_report`:** write `report.json`, `report.md` and `trace.json`. **`record_in_store`** updates the `runs` row. `runs/` is gitignored, and a test checks that.
- `llm/prompts/email_system.txt`, `email_user.txt`, `summary_system.txt`, `summary_user.txt`.

## Outside the listed files
- `src/agent_orchestrator/templates/__init__.py`, so the templates load as package data.

## Checks
- `uv run ruff check .`, `uv run ruff format --check .`: pass.
- `uv run pytest`: 460 passed, 1 skipped (semcache not installed), 7 deselected (48 new: communicator 32, reporter 16).
  - **Seed month vs a hand-computed table:** the table was worked out from `purchase_register.csv` and the labels and is written in the test with its arithmetic.
    - **Cases:** 19 (8 matched, 1 after normalisation, 3 unmatched, 6 mismatch, 1 duplicate). Outcomes: 11 auto_allow, 7 require_approval, 1 block.
    - **ITC:** safe ₹61,984.00, held ₹48,294.50, at risk ₹96,051.00. A second test reads the CSV directly and checks that the three buckets add up to all the GST in the books.
    - **Lists:** 7 pending approvals in book order; IMA/2026/0117 blocked by GST-ITC-VAL-01 and SAFE-INJ-01; 3 data-quality issues; 2 chase drafts (SPE/2026/0108 and EXA/2026/0109).
  - **Decided approvals** (approve, edit, reject) move the final actions and the approval counts.
  - **Communicator:**
    - an 11-row eligibility table;
    - 8 kinds of ungrounded draft, each falling back to the template: an extra number, a wrong amount, a wrong date, an unknown invoice id, no invoice number, a link, an e-mail address, a section number;
    - a wrong language;
    - native digits;
    - bad JSON;
    - budget stop, provider error and configuration error.

## Known limits
- **The grounding check is about values, not wording.** A number that is in the facts can still be used in the wrong place (for example the rate, 18, as "18 days"). A person reviews every draft, and nothing is sent.
- **The Hindi and Telugu templates were written by the assistant.** They need a check by a native speaker before anyone relies on them, which the README will say.

## Out of scope / follow-ups
- Writing drafts through the idempotent `write_draft_email` tool (Task 14/15) and calling the reporter from the graph (Task 13).
- Live e-mail and summary calls come with the live runs (Task 18).
