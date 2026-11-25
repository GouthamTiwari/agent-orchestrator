"""In-repo implementations of the three interfaces: offline, deterministic, free.

- ``RegexGuard``: the domain injection rules in ``injection_rules.yaml`` plus JSON extraction
  and Pydantic validation.
- ``SnapshotRetriever``: BM25 keyword search over the official CBIC texts saved in
  ``data/rules/sources/``, split into citable snippets (``cgst_act_s16#2aa``).
- ``SqliteCache``: an exact-key cache in its own SQLite table.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import sqlite3
import threading
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from agent_orchestrator.interfaces.protocols import ValidationResult
from agent_orchestrator.models import InjectionResult, RuleSnippet

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCES = ROOT / "data" / "rules" / "sources"

# --- text normalisation for scanning ----------------------------------------------------------

_ZERO_WIDTH = re.compile("[​‌‍⁠﻿­]")
_SPACES = re.compile(r"\s+")


def normalise_for_scan(text: str) -> str:
    """NFKC, no zero-width characters, one space for any whitespace run, lowercase."""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    return _SPACES.sub(" ", text).strip().lower()


@dataclass(frozen=True)
class InjectionRule:
    id: str
    why: str
    pattern: re.Pattern[str]


def load_injection_rules(path: Path | None = None) -> tuple[InjectionRule, ...]:
    if path is None:
        text = (
            resources.files("agent_orchestrator.interfaces")
            .joinpath("injection_rules.yaml")
            .read_text(encoding="utf-8")
        )
    else:
        text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    rules = tuple(
        InjectionRule(r["id"], r["why"], re.compile(r["pattern"], re.IGNORECASE))
        for r in data["rules"]
    )
    ids = [r.id for r in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate injection rule id")
    return rules


# --- JSON extraction --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


class _DuplicateKey(ValueError):
    pass


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise _DuplicateKey(f"duplicate key {k!r}")
        out[k] = v
    return out


def extract_json_text(text: str) -> tuple[str | None, bool]:
    """Find the one JSON object in a model answer: inside ```json fences, or the outermost
    ``{...}``. Returns (json text or None, whether anything around it was stripped)."""
    stripped = text.strip()
    fenced = _FENCE.findall(stripped)
    if len(fenced) == 1:
        inner = fenced[0].strip()
        return inner, True
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end < start:
        return None, False
    return stripped[start : end + 1], (start > 0 or end < len(stripped) - 1)


def validate_json(text: str, model: type[BaseModel]) -> ValidationResult:
    """Validate a model answer. Error messages name the field and the rule, never the value,
    so untrusted document text is not echoed back into the next prompt."""
    json_text, stripped = extract_json_text(text)
    if json_text is None:
        return ValidationResult(None, ["no JSON object found in the answer"])
    try:
        data = json.loads(json_text, object_pairs_hook=_no_duplicates)
    except _DuplicateKey as exc:
        return ValidationResult(None, [str(exc)], stripped)
    except json.JSONDecodeError as exc:
        return ValidationResult(None, [f"not valid JSON: {exc.msg}"], stripped)
    try:
        value = model.model_validate(data)
    except ValidationError as exc:
        errors = [
            f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}"
            for e in exc.errors(include_input=False, include_url=False)
        ]
        return ValidationResult(None, errors, stripped)
    return ValidationResult(value, [], stripped)


# --- Guard stub -------------------------------------------------------------------------------


class RegexGuard:
    name = "regex-stub"

    def __init__(self, rules: tuple[InjectionRule, ...] | None = None) -> None:
        self.rules = rules if rules is not None else load_injection_rules()

    def hits(self, text: str) -> list[str]:
        norm = normalise_for_scan(text)
        return [r.id for r in self.rules if r.pattern.search(norm)]

    def scan(self, texts: Mapping[str, str]) -> InjectionResult:
        sources, rule_ids = [], set()
        for source, text in texts.items():
            found = self.hits(text or "")
            if found:
                sources.append(source)
                rule_ids.update(found)
        return InjectionResult(flagged=bool(sources), sources=sources, rule_ids=sorted(rule_ids))

    def validate(self, text: str, model: type[BaseModel]) -> ValidationResult:
        return validate_json(text, model)


# --- Retriever stub ---------------------------------------------------------------------------

_MARKER = re.compile(r"^(?:\d+\[)?\((\d+[A-Z]?|[a-z]{1,3})\)\s*(.*)$", re.DOTALL)
_ROMAN = re.compile(r"^(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)a?$")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the of and or to a in for by any such as be is on with which under this that shall "
    "has have been other from at an it its may not".split()
)


def _source_meta(source_id: str) -> tuple[str, str]:
    """('CGST Act', '16') from 'cgst_act_s16'; ('CGST Rules', '37A') from 'cgst_rules_r37a'."""
    if source_id.startswith("cgst_act_s"):
        return "CGST Act", source_id.removeprefix("cgst_act_s").upper()
    if source_id.startswith("cgst_rules_r"):
        return "CGST Rules", source_id.removeprefix("cgst_rules_r").upper()
    if source_id.startswith("igst_act_s"):
        return "IGST Act", source_id.removeprefix("igst_act_s").upper()
    return source_id, ""


def split_source(path: Path) -> list[RuleSnippet]:
    """Split one saved CBIC page into snippets: one per sub-section and per clause.

    Clause ids follow the law's own numbering, e.g. ``cgst_act_s16#2aa`` for section 16(2)(aa).
    Provisos, explanations and roman-numbered items stay with the clause they belong to.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    url = lines[0].removeprefix("Source:").strip()
    checked = dt.date.fromisoformat(lines[1].removeprefix("Retrieved:").strip()[:10])
    source_id = path.stem
    act, number = _source_meta(source_id)
    kind = "s." if act.endswith("Act") else "r."
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", "\n".join(lines[2:]))]
    sub, clause, last_letter = "", "", ""
    chunks: dict[str, list[str]] = {}
    order: list[str] = []
    for para in paragraphs:
        if not para:
            continue
        m = _MARKER.match(para)
        if m:
            token = m.group(1)
            if token[0].isdigit():
                sub, clause, last_letter = token, "", ""
            elif (
                _ROMAN.match(token)
                and not (last_letter and len(token) == 1 and ord(token) == ord(last_letter) + 1)
                and clause
            ):
                pass  # a roman item inside the current clause
            else:
                clause = token
                last_letter = token if len(token) == 1 else last_letter
        key = f"{sub}{clause}" or "0"
        if key not in chunks:
            chunks[key] = []
            order.append(key)
        chunks[key].append(para)
    snippets = []
    for key in order:
        body = " ".join(chunks[key])
        sub_part = re.match(r"\d+[A-Z]?", key)
        section = number
        if key != "0" and sub_part:
            section += f"({sub_part.group(0)})"
            rest = key[sub_part.end() :]
            if rest:
                section += f"({rest})"
        snippets.append(
            RuleSnippet(
                snippet_id=f"{source_id}#{key}",
                source_id=source_id,
                title=f"{act} {kind}{section}",
                url=url,
                section=section,
                text=body,
                checked_on=checked,
            )
        )
    return snippets


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class SnapshotRetriever:
    """BM25 over the saved official texts (k1 = 1.5, b = 0.75). Deterministic: ties are broken
    by snippet id. A query that names a provision ("section 16(2)(aa)", "rule 36(4)") puts that
    snippet first."""

    name = "cbic-snapshot"
    K1, B = 1.5, 0.75

    def __init__(self, sources_dir: Path = DEFAULT_SOURCES) -> None:
        files = sorted(Path(sources_dir).glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"no rule texts in {sources_dir}")
        h = hashlib.sha256()
        self.snippets: list[RuleSnippet] = []
        for f in files:
            h.update(f.name.encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n"))
            self.snippets += split_source(f)
        self._version = "cbic-" + h.hexdigest()[:12]
        self._by_id = {s.snippet_id: s for s in self.snippets}
        self._docs = [Counter(_tokens(f"{s.title} {s.text}")) for s in self.snippets]
        self._lens = [sum(d.values()) for d in self._docs]
        self._avg = sum(self._lens) / len(self._lens)
        df = Counter(t for d in self._docs for t in d)
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    @property
    def corpus_version(self) -> str:
        return self._version

    def get(self, snippet_id: str) -> RuleSnippet | None:
        return self._by_id.get(snippet_id)

    def _named(self, query: str) -> set[str]:
        """Snippet ids a query names explicitly, e.g. 'section 16(2)(aa)' or 'rule 36(4)'."""
        out = set()
        for m in re.finditer(
            r"\b(section|sec\.?|s\.|rule|r\.)\s*(\d+[a-z]?)((?:\(\w{1,4}\))*)", query, re.I
        ):
            prefix = "cgst_act_s" if m.group(1).lower().startswith("s") else "cgst_rules_r"
            parts = re.findall(r"\((\w{1,4})\)", m.group(3))
            key = "".join(parts) or "0"
            sid = f"{prefix}{m.group(2).lower()}#{key}"
            if sid in self._by_id:
                out.add(sid)
        return out

    def search(self, query: str, k: int = 4) -> list[RuleSnippet]:
        q = _tokens(query)
        named = self._named(query)
        scored = []
        for i, doc in enumerate(self._docs):
            score = 0.0
            for t in q:
                tf = doc.get(t, 0)
                if tf:
                    norm = tf + self.K1 * (1 - self.B + self.B * self._lens[i] / self._avg)
                    score += self._idf[t] * tf * (self.K1 + 1) / norm
            sid = self.snippets[i].snippet_id
            if sid in named:
                score += 1000.0
            if score > 0:
                scored.append((-score, sid, i))
        scored.sort()
        return [self.snippets[i] for _, _, i in scored[:k]]


# --- Cache stub -------------------------------------------------------------------------------


class SqliteCache:
    """Exact-key cache. Values must be JSON-serialisable."""

    name = "sqlite-exact"

    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " created_at TEXT NOT NULL)"
        )
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv_cache WHERE key = ?", (key,)).fetchone()
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
            return json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        now = dt.datetime.now(dt.UTC).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO kv_cache (key, value, created_at) VALUES (?,?,?)"
                " ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, text, now),
            )

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}
