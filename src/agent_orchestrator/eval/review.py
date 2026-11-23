"""Human spot-check of the generated labels (docs/SPEC.md section 15.1).

1. ``orchestrator dataset review-sample`` writes about 10% of the labels, stratified by case type,
   to ``data/review/review_sample.csv``.
2. A person fills the ``verdict`` column with OK or FIX (and the ``fix_*`` columns for a FIX),
   and saves it, for example as ``data/review/review_sample.filled.csv``.
3. ``orchestrator dataset apply-review FILE`` validates the file, appends the verdicts to
   ``data/label_overrides.jsonl`` (committed) and a summary to ``docs/DATASET_CHANGELOG.md``.
4. The generator applies the overrides every time it writes labels, so a fix survives
   regeneration and is visible in git.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent_orchestrator.models import ACTION_KINDS, CASE_STATUSES

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE = ROOT / "data" / "review" / "review_sample.csv"
DEFAULT_OVERRIDES = ROOT / "data" / "label_overrides.jsonl"
DEFAULT_CHANGELOG = ROOT / "docs" / "DATASET_CHANGELOG.md"

SAMPLE_COLUMNS = [
    "id",
    "business_id",
    "period",
    "case_type",
    "invoice_no",
    "supplier_gstin",
    "expected_status",
    "expected_action",
    "auto_ok",
    "injection",
    "pdf_file",
    "pdf_note",
    "verdict",
    "fix_expected_status",
    "fix_expected_action",
    "fix_auto_ok",
    "notes",
]
FIXABLE = {
    "fix_expected_status": ("expected_status", CASE_STATUSES),
    "fix_expected_action": ("expected_action", ACTION_KINDS),
    "fix_auto_ok": ("auto_ok", ("true", "false")),
}


class ReviewError(ValueError):
    """The filled review file has a problem; nothing was applied."""


def stratified_sample(
    labels: list[dict[str, Any]], share: float = 0.10, seed: int = 2611
) -> list[dict[str, Any]]:
    """About ``share`` of the labels, at least one of every case type, chosen deterministically."""
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lab in labels:
        by_type[lab["case_type"]].append(lab)
    rng = random.Random(f"{seed}:review-sample")
    chosen = []
    for case_type in sorted(by_type):
        group = sorted(by_type[case_type], key=lambda lab: lab["id"])
        k = max(1, math.ceil(len(group) * share))
        chosen.extend(rng.sample(group, k))
    return sorted(chosen, key=lambda lab: lab["id"])


def write_sample(labels: list[dict[str, Any]], out: Path = DEFAULT_SAMPLE, **kw: Any) -> int:
    rows = stratified_sample(labels, **kw)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SAMPLE_COLUMNS, lineterminator="\n")
        w.writeheader()
        for lab in rows:
            w.writerow({c: _cell(lab.get(c, "")) for c in SAMPLE_COLUMNS})
    return len(rows)


def _cell(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


def parse_review(path: Path, known_ids: set[str]) -> list[dict[str, Any]]:
    """Read and validate a filled review CSV. Raises ReviewError listing every problem."""
    entries, problems = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            verdict = (row.get("verdict") or "").strip().upper()
            if not verdict:
                continue  # not reviewed yet
            rid = row.get("id", "")
            if rid not in known_ids:
                problems.append(f"line {n}: unknown id {rid!r}")
                continue
            if verdict not in ("OK", "FIX"):
                problems.append(f"line {n}: verdict must be OK or FIX, not {verdict!r}")
                continue
            changes = {}
            for col, (field, allowed) in FIXABLE.items():
                value = (row.get(col) or "").strip().lower()
                if not value:
                    continue
                if value not in allowed:
                    problems.append(f"line {n}: {col}={value!r} is not one of {list(allowed)}")
                    continue
                changes[field] = value == "true" if field == "auto_ok" else value
            if verdict == "FIX" and not changes:
                problems.append(f"line {n}: FIX needs at least one fix_* value")
            if verdict == "OK" and changes:
                problems.append(f"line {n}: OK must not have fix_* values")
            entries.append(
                {
                    "id": rid,
                    "verdict": verdict,
                    "changes": changes,
                    "notes": (row.get("notes") or "").strip(),
                }
            )
    if problems:
        raise ReviewError("\n".join(problems))
    return entries


def apply_review(
    filled: Path,
    labels: list[dict[str, Any]],
    overrides_path: Path = DEFAULT_OVERRIDES,
    changelog_path: Path = DEFAULT_CHANGELOG,
    today: dt.date | None = None,
) -> dict[str, int]:
    by_id = {lab["id"]: lab for lab in labels}
    entries = parse_review(filled, set(by_id))
    today = today or dt.date.today()
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    with open(overrides_path, "a", encoding="utf-8", newline="\n") as f:
        for e in entries:
            f.write(json.dumps({**e, "reviewed_on": today.isoformat()}, ensure_ascii=False) + "\n")
    fixes = [e for e in entries if e["verdict"] == "FIX"]
    lines = [
        f"\n## {today.isoformat()}: human spot-check of {len(entries)} labels",
        "",
        f"- OK: {len(entries) - len(fixes)}",
        f"- Fixed: {len(fixes)}",
    ]
    for e in fixes:
        before = {k: by_id[e["id"]][k] for k in e["changes"]}
        lines.append(f"  - `{e['id']}`: {before} -> {e['changes']}. {e['notes']}".rstrip())
    if not changelog_path.exists():
        changelog_path.write_text(
            "# Dataset changelog\n\nHuman review results for the generated labels.\n",
            encoding="utf-8",
        )
    with open(changelog_path, "a", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return {"reviewed": len(entries), "fixed": len(fixes)}


def load_overrides(path: Path = DEFAULT_OVERRIDES) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def apply_overrides(
    labels: list[dict[str, Any]], overrides: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return labels with review results applied (later entries win). Adds ``review_status``."""
    latest = {o["id"]: o for o in overrides}
    out = []
    for lab in labels:
        o = latest.get(lab["id"])
        if o is None:
            out.append({**lab, "review_status": "assistant_reviewed"})
        elif o["verdict"] == "OK":
            out.append({**lab, "review_status": "human_ok"})
        else:
            out.append({**lab, **o["changes"], "review_status": "human_fixed"})
    return out
