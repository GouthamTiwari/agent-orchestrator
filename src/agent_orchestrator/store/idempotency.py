"""Deterministic idempotency keys (docs/SPEC.md section 9.2).

A key names *what* an action is about, so doing it twice (a re-run, a retry after a crash, a
double click) finds the first result instead of creating a second one.

- Approval: business, period, case, proposed action and the facts it was based on. If the facts
  change (new data), it is a new question for the person, so a new key.
- E-mail: business, period, case, final action and supplier. The LLM's wording is deliberately
  NOT part of the key: a re-run that words the e-mail differently still finds the first draft.
"""

from __future__ import annotations

import hashlib


def _key(*parts: str) -> str:
    for p in parts:
        if "|" in p:
            raise ValueError(f"key part may not contain '|': {p!r}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def approval_key(
    business_id: str, period: str, case_key: str, proposed_action: str, facts_hash: str
) -> str:
    return _key("approval", business_id, period, case_key, proposed_action, facts_hash)


def outbox_key(
    business_id: str, period: str, case_key: str, final_action: str, supplier_gstin: str
) -> str:
    return _key("email", business_id, period, case_key, final_action, supplier_gstin)
