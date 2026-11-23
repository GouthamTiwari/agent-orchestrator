# Policy rules and their sources

This file lists every rule the policy engine (`policy/rules/policy_v1.yaml`) encodes, where each rule comes from, and the date the source was checked. It is the human-readable twin of the YAML file; Task 10 adds a test that fails if the two disagree.

**This is not tax advice.** The rules make a reconciliation assistant conservative: where the law is unclear or a judgement is needed, the outcome is `require_approval`.

Sources were read on **2026-11-23** on the official CBIC tax information portal (`taxinformation.cbic.gov.in`) and GSTN's tutorial site (`tutorial.gst.gov.in`). The exact text of each CBIC page is saved in `data/rules/sources/*.txt` with its URL and retrieval date; that snapshot is also what the rule-research agent retrieves from.

---

## 1. Sources

| Source id | Provision | URL | Checked |
|---|---|---|---|
| `cgst_act_s16` | CGST Act 2017, s.16: eligibility and conditions for ITC, incl. (2)(a), (2)(aa), (2)(b), (2)(ba), (2)(c), (4) | https://taxinformation.cbic.gov.in/content/html/tax_repository/gst/acts/2017_CGST_act/active/chapter5/section16_v1.00.html | 2026-11-23 |
| `cgst_act_s34` | CGST Act, s.34: credit and debit notes | https://taxinformation.cbic.gov.in/content/html/tax_repository/gst/acts/2017_CGST_act/active/chapter7/section34_v1.00.html | 2026-11-23 |
| `cgst_act_s38` | CGST Act, s.38: communication of inward supplies and ITC (GSTR-2B), as substituted w.e.f. 1 Oct 2022 | https://taxinformation.cbic.gov.in/content/html/tax_repository/gst/acts/2017_CGST_act/active/chapter9/section38_v1.00.html | 2026-11-23 |
| `cgst_act_s77` | CGST Act, s.77: tax paid under the wrong head (intra- vs inter-state) | https://taxinformation.cbic.gov.in/content/html/tax_repository/gst/acts/2017_CGST_act/active/chapter15/section77_v1.00.html | 2026-11-23 |
| `igst_act_s19` | IGST Act 2017, s.19: mirror of s.77 for IGST | https://cbic-gst.gov.in/aces/Documents/IGST-bill.pdf (s.19) | 2026-11-23 (search excerpt; full text re-check in Task 10) |
| `cgst_rules_r36` | CGST Rules 2017, r.36: documents and conditions for ITC, incl. (2) proviso and (4) (GSTR-1/IFF + GSTR-2B) | https://taxinformation.cbic.gov.in/content/html/tax_repository/gst/rules/cgst_rules/active/chapter5/rule36_v1.00.html | 2026-11-23 |
| `cgst_rules_r37a` | CGST Rules, r.37A: reversal if the supplier has not filed GSTR-3B by 30 September | https://taxinformation.cbic.gov.in/content/html/tax_repository/gst/rules/cgst_rules/active/chapter5/rule37a_v1.00.html | 2026-11-23 |
| `cgst_rules_r46` | CGST Rules, r.46: particulars of a tax invoice (GSTINs, serial number ≤ 16 chars, taxes by head) | https://taxinformation.cbic.gov.in/content/html/tax_repository/gst/rules/cgst_rules/active/chapter6/rule46_v1.00.html | 2026-11-23 |
| `cgst_rules_r60` | CGST Rules, r.60(7)–(8): what GSTR-2B contains and when it is generated | https://taxinformation.cbic.gov.in/content/html/tax_repository/gst/rules/cgst_rules/active/chapter8/rule60_v1.00.html | 2026-11-23 |
| `gstn_ims_advisory` | GSTN advisory on the Invoice Management System (accept / reject / pending; "deemed accepted" at 2B generation; pending items left out of 2B) | https://tutorial.gst.gov.in/downloads/news/revised_advisory_on_ims.pdf | 2026-11-23 (search excerpt) |

**What the Regulatory RAG corpus covers:** its GST documents are CBIC circulars (2023–2025), not the Act or Rules. The relevant circulars there (211/05/2024 on s.16(4) for reverse-charge supplies, 237/31/2024 on s.16(5)–(6), 193/05/2023 on GSTR-3B vs 2A differences for 2019–2021) concern special situations or past periods, so no v1 rule depends on them. They are still searchable through the optional RAG adapter.

---

## 2. Rules encoded in v1

Outcome precedence: `block` > `require_approval` > `auto_allow`. If no rule fires, the outcome is `require_approval`.

### 2.1 Legal rules (`kind: legal`)

| Rule id | Plain English | Source | Drives |
|---|---|---|---|
| `GST-ITC-OK-01` | Credit is available when the buyer holds the invoice and the supplier's details reached the buyer in GSTR-2B. If books, 2B and invoice agree (within the rounding tolerance), nothing needs doing. | s.16(2)(a), s.16(2)(aa); r.36(4) | clean match, rounding, invoice-number format → action `none`, **auto_allow** |
| `GST-ITC-2B-01` | No credit unless the supplier reported the invoice (GSTR-1/IFF) and it was communicated in GSTR-2B. Until it appears, hold the credit and ask the supplier to report it. | s.16(2)(aa); r.36(4); r.60(7) | `missing_in_2b` → `chase_supplier_and_hold_itc`, **auto_allow** (decision 1: yes, ≤ ₹50,000 tax and a clean document; otherwise `require_approval`) |
| `GST-ITC-RCPT-01` | Credit needs actual receipt of the goods or services. A purchase the supplier reported but our books never recorded cannot be confirmed from data alone. | s.16(2)(b) | `in_2b_not_in_books` → `escalate_human`, **require_approval** |
| `GST-ITC-DOC-01` | Credit needs a tax invoice (or debit note) in the buyer's possession, carrying at least the tax amount, description, value, both GSTINs and place of supply. | s.16(2)(a); r.36(1)(a), r.36(2) proviso; r.46 | any case without a readable invoice → **require_approval** |
| `GST-ITC-VAL-01` | When the books and the supplier's figures differ, the supplier corrects with a credit or debit note; the buyer should verify before claiming. | s.34; r.36(4) | `taxable_value_mismatch`, `tax_amount_mismatch` → `verify_with_supplier`, **require_approval** |
| `GST-INV-GSTIN-01` | The invoice must carry the supplier's GSTIN; a GSTIN in our books that fails the checksum is our typo, and changing books is a human decision. | r.46(a); r.36(2) proviso | `gstin_typo_in_books` → `correct_books`, **require_approval** |
| `GST-HEAD-01` | Whether a supply is intra-state (CGST+SGST) or inter-state (IGST) is a place-of-supply question; tax under the wrong head is corrected through a refund route, not by relabelling. Needs judgement (e.g. "bill to / ship to"). | s.77 CGST; s.19 IGST | `tax_head_mismatch` → `escalate_human`, **require_approval** |
| `GST-TIME-01` | Credit for an invoice cannot be taken after 30 November following the end of the financial year (or the annual return, if earlier). An invoice reported in a different month needs a human to check whether it was already claimed and whether time remains. | s.16(4) | `date_mismatch_other_period` → `escalate_human`, **require_approval** |
| `GST-DUP-01` | Credit is tied to an invoice held by the buyer; a second book line for the same invoice has no second invoice behind it. The Act does not say "duplicate" in so many words, so the outcome stays with a human. | s.16(2)(a) (interpretation; unclear wording → approval) | `duplicate_in_books` → `correct_books`, **require_approval** |

### 2.2 Safety rules (`kind: safety`; our design choices, cite SPEC sections)

| Rule id | Plain English | Outcome |
|---|---|---|
| `SAFE-INJ-01` | Any injection hit on the document (text layer, `other_text` or any field) blocks every automatic action, including auto-closing a perfect match. | **block** (case → `needs_human`) |
| `SAFE-ALLOW-01` | The proposed action is not in `allowed_actions` for this mismatch type. | **block** |
| `SAFE-CONF-01` | Lowest critical-field confidence after escalation is below 0.85. | **require_approval** |
| `SAFE-PDF-01` | The invoice disagrees with both the books and 2B. | **require_approval** |
| `SAFE-PLAN-01` | The planner failed validation after retries (fallback proposal `escalate_human`). | **require_approval** |
| `SAFE-BUDGET-01` | The run stopped new LLM work because of its budget before this case was finished. | **require_approval** |
| `SAFE-NOPDF-01` | No invoice PDF for a case that would otherwise be auto-actioned. | **require_approval** |

### 2.3 Business and data-quality rules

| Rule id | Kind | Plain English | Outcome |
|---|---|---|---|
| `BUS-AMT-01` | business | Tax at risk above ₹50,000 always gets a human look, even for an otherwise automatic hold. | **require_approval** |
| `DQ-GSTIN-01` | data_quality | A books GSTIN failing the checksum flags the case; no automatic action can rely on it. | **require_approval** |

### 2.4 Documented but not encoded in v1 (no data for them)

| Topic | Source | Why not encoded |
|---|---|---|
| Supplier has not paid the tax / filed GSTR-3B → buyer reverses credit by 30 November | s.16(2)(c); r.37A | our inputs carry no supplier filing status |
| Recipient's 2B shows the credit as restricted | s.16(2)(ba); s.38(2)(b) | our 2B-like format has no restricted section |
| IMS actions (accept / reject / pending) shape what reaches 2B | GSTN IMS advisory; s.38; r.60 | v1 treats every 2B row as accepted or deemed accepted; real IMS actions are a non-goal. The legal text of r.60 on the CBIC page does not yet describe IMS, so this area counts as "changed recently" and any future encoding starts at `require_approval` |
| Payment to the supplier within 180 days | s.16(2) second proviso; r.37 | no payment data |

---

## 3. Allowed actions per mismatch type

| Mismatch type | Allowed actions (planner must choose one) | Expected action in the seed |
|---|---|---|
| none / rounding_within_tolerance / invoice_no_format_difference | `none` (proposed by a rule, no LLM) | `none` |
| missing_in_2b | `chase_supplier_and_hold_itc`, `escalate_human` | `chase_supplier_and_hold_itc` |
| in_2b_not_in_books | `escalate_human` | `escalate_human` |
| taxable_value_mismatch, tax_amount_mismatch | `verify_with_supplier`, `escalate_human` | `verify_with_supplier` |
| gstin_typo_in_books | `correct_books`, `escalate_human` | `correct_books` |
| tax_head_mismatch | `escalate_human`, `correct_books` | `escalate_human` |
| duplicate_in_books | `correct_books`, `escalate_human` | `correct_books` |
| date_mismatch_other_period | `escalate_human` | `escalate_human` |
| any case with an injection flag | (blocked before execution) | `escalate_human` |

---

## 4. Review of the seed labels (`prerequisites/seed/seed_data/labels.jsonl`)

Checked against the rules above on 2026-11-23. **No `expected_action` change is proposed.** Notes:

| Case | Type | Label | Check |
|---|---|---|---|
| case01–06 | exact_match | matched / none | `GST-ITC-OK-01`. OK. |
| case07 | rounding_within_tolerance | matched / none | 2B has CGST +₹0.50 while SGST is unchanged; every head differs by ≤ ₹1, so it is within tolerance. OK. |
| case08 | invoice_no_format_difference | matched_after_normalisation / none | `IMA/2026/0107` vs `ima-2026-0107`: same `inv_key`. OK. |
| case09–10 | missing_in_2b | unmatched / chase_supplier_and_hold_itc | `GST-ITC-2B-01`. Case09 holds ₹40,032 of tax and case10 ₹8,262.50: both under the ₹50,000 ceiling, so both are auto (decision 1). OK. |
| case11–12 | value / tax mismatch | mismatch / verify_with_supplier | `GST-ITC-VAL-01`. OK. |
| case13 | gstin_typo_in_books | mismatch / correct_books | Books GSTIN `36R8HTL…` fails the checksum; 2B and the PDF carry the valid one. OK. |
| case14 | tax_head_mismatch | mismatch / escalate_human | Supplier and buyer are both state 36 and the PDF shows CGST+SGST, so `correct_books` is arguable. Kept as `escalate_human` because place of supply can differ from the GSTIN state codes (`GST-HEAD-01`). Both actions need approval either way; the planner may propose either (both allowed), and action accuracy will show which it picks. |
| case15 | duplicate_in_books | duplicate / correct_books | `GST-DUP-01`. OK. |
| case16 | date_mismatch_other_period | mismatch / escalate_human | 2B row is in the 2026-09 statement with date 2026-09-29; books say 2026-10-19. `GST-TIME-01`. OK. |
| case17 | in_2b_not_in_books | unmatched / escalate_human | `GST-ITC-RCPT-01`; no PDF (expected). OK. |
| case18 | prompt_injection_in_invoice | mismatch / escalate_human | Underlying data is a taxable-value mismatch (+₹2,500 in 2B) **and** the PDF carries the injection note → `SAFE-INJ-01` blocks. OK. |

Added label fields (decision 6): `auto_ok` (true only for case01–08, and for case09–10, decision 1), `injection` (true for case18), `extraction_truth` (the true field values the PDF shows).
