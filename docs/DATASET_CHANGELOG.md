# Dataset changelog

Changes to the generated evaluation dataset and the results of human review.

## 2026-11-23: v1 generated

- 10 fictional businesses x 3 months (2026-08 to 2026-10): 593 labelled cases, 502 invoice PDFs (214 with a noisy scan), 15 injection invoices covering all 8 attack variants in the dev split.
- Split by business: dev B01-B05, test B06-B10; no supplier GSTIN in both.
- Labels are assistant-reviewed; the human spot-check of the 60-case sample (`data/review/review_sample.csv`) is pending.
