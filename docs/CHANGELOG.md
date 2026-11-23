# Changelog

One line per merged task. Details in `docs/changelogs/`.

- 01 · Repo setup: git identity and authorship hooks, ignores, project skeleton, CI, seed data, official CBIC rule texts.
- 02 · Domain models, money and GSTIN helpers: integer-paise money, GSTIN checksum and one-edit neighbours, all spec types (frozen, no extra fields), graph state with reducers.
- 03 · Dataset generator: 10 fictional businesses x 3 months (593 labelled cases), 8 injection variants, benign look-alikes, noisy scans, dev/test split by business, human review workflow.
- 04 · Intake agent: typed, normalised books and supplier rows, invoice-number keys, data-quality issues instead of crashes, line-ending-proof input hash.
- 05 · Matcher: five one-to-one passes (exact, normalised, no-year, GSTIN repair, fuzzy), duplicates, paise comparisons with the Rs 1 tolerance, order-independent, ties to a person; agrees with all 593 generated labels.
