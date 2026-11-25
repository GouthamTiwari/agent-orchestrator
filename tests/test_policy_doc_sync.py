"""docs/POLICY_RULES.md is the human-readable twin of policy_v1.yaml: they must agree.

Parses the Markdown tables of section 2 (rule id, kind, outcome) and section 3 (allowed
actions) and compares them with the loaded policy.
"""

import re
from pathlib import Path

from agent_orchestrator.models import MISMATCH_TYPES
from agent_orchestrator.policy.engine import load_policy

DOC = (Path(__file__).resolve().parents[1] / "docs" / "POLICY_RULES.md").read_text(encoding="utf-8")
END = "\n---\n"  # a horizontal rule line (table dividers also contain ---)
OUTCOME = re.compile(r"\*\*(auto_allow|require_approval|block)\*\*")


def _section(start: str, end: str) -> str:
    i = DOC.index(start)
    return DOC[i : DOC.index(end, i)]


def _rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if line.startswith("|") and not set(line) <= set("|- "):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows[1:] if rows else rows  # drop the header


def doc_rules() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for start, end, kind in (
        ("### 2.1", "### 2.2", "legal"),
        ("### 2.2", "### 2.3", "safety"),
        ("### 2.3", "### 2.4", None),
    ):
        for row in _rows(_section(start, end)):
            rule_id = row[0].strip("`")
            row_kind = kind or row[1]
            outcome = OUTCOME.search(" ".join(row)).group(1)
            out[rule_id] = (row_kind, outcome)
    return out


def test_every_documented_rule_is_in_the_policy_and_vice_versa():
    policy = {r.id: (r.kind, r.outcome) for r in load_policy().rules}
    documented = doc_rules()
    assert set(documented) == set(policy)
    for rule_id, (kind, outcome) in documented.items():
        assert policy[rule_id] == (kind, outcome), rule_id


def test_allowed_actions_table_matches():
    table = _section("## 3. Allowed actions", END)
    doc: dict[str, set[str]] = {}
    for row in _rows(table):
        types, actions = row[0], row[1]
        if types.startswith("any case"):
            continue
        acts = set(re.findall(r"`([a-z_]+)`", actions))
        for t in re.split(r"\s*[/,]\s*", types):
            doc[t.strip("` ")] = acts
    policy = load_policy().allowed_actions
    assert set(doc) == set(MISMATCH_TYPES)
    for t in MISMATCH_TYPES:
        assert doc[t] == set(policy[t]), t


def test_every_legal_source_is_listed_in_section_one():
    listed = set(re.findall(r"^\| `([a-z0-9_]+)` \|", _section("## 1. Sources", END), re.M))
    used = {s.id for r in load_policy().rules for s in r.sources}
    assert used <= listed
