"""Adapters to Goutham's sibling projects, loaded only when the package is installed.

- ``GuardrailsGuard`` wraps ``llm-guardrails`` 1.0.0 (a pinned git dependency). Offline it runs
  that project's generic injection rules TOGETHER with this repo's invoice rules: on our
  generated attacks the generic rules alone catch 2 of 8 styles (Task 08 finding), so the
  domain rules are kept. Its JSON extraction is used for answers.
- ``RegragRetriever`` wraps ``regrag.api.search`` (Regulatory RAG v1.1.0, after tasks R1-R4).
- ``SemcacheCache`` wraps ``semcache``'s in-memory store and an embedder for free-text lookups.

``load_guard()``, ``load_retriever()`` and ``load_cache()`` return the adapter when it can be
built and otherwise the in-repo stub, with a one-line note saying which and why.
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
import logging
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent_orchestrator.interfaces.protocols import Cache, Guard, RuleRetriever, ValidationResult
from agent_orchestrator.interfaces.stubs import (
    RegexGuard,
    SnapshotRetriever,
    SqliteCache,
    validate_json,
)
from agent_orchestrator.models import InjectionResult, RuleSnippet

log = logging.getLogger("agent_orchestrator.interfaces")


class AdapterUnavailable(RuntimeError):
    """The sibling package is missing or too old; the stub is used instead."""


def _import(module: str, hint: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise AdapterUnavailable(f"{module} is not installed ({hint})") from exc


# --- llm-guardrails ---------------------------------------------------------------------------


class GuardrailsGuard:
    name = "llm-guardrails+invoice-rules"

    def __init__(self, domain: RegexGuard | None = None) -> None:
        self._normalize = _import(
            "llm_guardrails.guards.normalize", "pip install llm-guardrails"
        ).normalize
        injection = _import("llm_guardrails.guards.injection", "pip install llm-guardrails")
        self._score_rules = injection.score_rules
        self._rules = injection.load_rules()
        self._extract = _import("llm_guardrails.guards.schema", "llm-guardrails").extract_json
        self.domain = domain or RegexGuard()

    def generic_hits(self, text: str) -> list[str]:
        """llm-guardrails layer-1 verdict: a strong rule hit blocks (score 1.0)."""
        rs = self._score_rules(self._normalize(text), self._rules)
        return [f"GR-{rid}" for rid in rs.rule_ids] if rs.score >= 1.0 else []

    def scan(self, texts: Mapping[str, str]) -> InjectionResult:
        sources, rule_ids = [], set()
        for source, text in texts.items():
            found = self.generic_hits(text or "") + self.domain.hits(text or "")
            if found:
                sources.append(source)
                rule_ids.update(found)
        return InjectionResult(flagged=bool(sources), sources=sources, rule_ids=sorted(rule_ids))

    def validate(self, text: str, model: type[BaseModel]) -> ValidationResult:
        extracted = self._extract(text)
        if extracted.json_text is None:
            return validate_json(text, model)  # same messages as the stub
        result = validate_json(extracted.json_text, model)
        return ValidationResult(result.value, result.errors, extracted.prose_stripped)


# --- Regulatory RAG ---------------------------------------------------------------------------


REGRAG_ENV_VAR = "REGULATORY_RAG_ENV_FILE"
# Only the settings regrag needs beyond AWS credentials (which this project's own .env has).
REGRAG_KEYS = (
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "POSTGRES_PORT",
    "DATABASE_URL",
    "BEDROCK_RERANK_REGION",
    "BEDROCK_EMBED_MODEL_ID",
    "BEDROCK_RERANK_MODEL_ID",
)


def load_regrag_env(path: str | None = None) -> int:
    """Copy regrag's database and model settings from its own ``.env`` into this process,
    without overwriting anything already set. Returns how many were set (values never logged).
    An installed package cannot find its project's ``.env`` by itself."""
    path = path or os.environ.get(REGRAG_ENV_VAR, "").strip()
    if not path:
        return 0
    if not Path(path).is_file():
        raise AdapterUnavailable(f"{REGRAG_ENV_VAR} points to a missing file")
    from dotenv import dotenv_values

    count = 0
    for key, value in dotenv_values(path).items():
        if key in REGRAG_KEYS and value and not os.environ.get(key):
            os.environ[key] = value
            count += 1
    return count


class RegragRetriever:
    name = "regulatory-rag"

    def __init__(self, env_file: str | None = None, probe: bool = True) -> None:
        load_regrag_env(env_file)
        api = _import("regrag.api", "Regulatory RAG v1.1.0 with regrag.api (tasks R1-R4)")
        for fn in ("search", "corpus_version"):
            if not hasattr(api, fn):
                raise AdapterUnavailable(f"regrag.api has no {fn}(); need regrag v1.1.0")
        self._api = api
        self._seen: dict[str, RuleSnippet] = {}
        self._version: str | None = None
        if probe:  # the package without its database is no retriever: use the snapshot
            try:
                self._version = f"regrag-{api.corpus_version()}"
            except Exception as exc:
                raise AdapterUnavailable(
                    f"regrag database not reachable ({type(exc).__name__}); "
                    "start it with `docker compose up -d` in the regulatory-rag folder"
                ) from exc

    @property
    def corpus_version(self) -> str:
        # Read once: the corpus does not change during a run, and every cache key uses it.
        if self._version is None:
            self._version = f"regrag-{self._api.corpus_version()}"
        return self._version

    def search(self, query: str, k: int = 4) -> list[RuleSnippet]:
        out = []
        for hit in self._api.search(query, k=k, corpus="gst"):
            snippet = RuleSnippet(
                snippet_id=str(hit.snippet_id),
                source_id=str(hit.doc_id),
                title=str(hit.title),
                url=str(hit.url),
                section=str(hit.section),
                text=str(hit.text),
                checked_on=hit.checked_on,
            )
            self._seen[snippet.snippet_id] = snippet
            out.append(snippet)
        return out

    def get(self, snippet_id: str) -> RuleSnippet | None:
        return self._seen.get(snippet_id)


# --- Semantic cache ---------------------------------------------------------------------------


class SemcacheCache:
    """Free-text cache: exact normalised match first, then nearest stored query by embedding
    cosine similarity at or above ``threshold``."""

    name = "semcache"
    SCOPE = "agent-orchestrator:rules"

    def __init__(self, embedder: Any = None, threshold: float = 0.92) -> None:
        self._store_mod = _import("semcache.store.memory", "pip install semcache")
        self._types = _import("semcache.types", "semcache")
        self._norm = _import("semcache.normalize", "semcache")
        if embedder is None:
            embedder = _import("semcache.embed", "semcache").TitanEmbedder()
        self.embedder = embedder
        self.store = self._store_mod.InMemoryStore()
        self.threshold = threshold
        self.hits = self.misses = 0

    def _now(self) -> dt.datetime:
        return dt.datetime.now(dt.UTC)

    def get(self, key: str) -> Any | None:
        norm = self._norm.normalize(key)
        now = self._now()
        entry = self.store.get_exact(self.SCOPE, self._norm.exact_key(self.SCOPE, norm), now)
        if entry is None:
            vec = self.embedder.embed(norm).vector
            found = self.store.search(self.SCOPE, vec, 1, now)
            if found and found[0][1] >= self.threshold:
                entry = found[0][0]
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self.store.record_hit(entry.id, now)
        return json.loads(entry.answer_text)

    def set(self, key: str, value: Any) -> None:
        norm = self._norm.normalize(key)
        now = self._now()
        ttl = self._types.TTLClass.STATIC
        entry = self._types.CacheEntry(
            id=uuid.uuid4().hex,
            scope_key=self.SCOPE,
            namespace="rules",
            model_id=getattr(self.embedder, "model_id", "embedder"),
            exact_key=self._norm.exact_key(self.SCOPE, norm),
            query_text=key,
            query_norm=norm,
            embedding=[float(x) for x in self.embedder.embed(norm).vector],
            answer_text=json.dumps(value, ensure_ascii=False, sort_keys=True),
            stop_reason="end_turn",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            ttl_class=ttl,
            created_at=now,
            expires_at=now + self._types.TTL_DURATIONS[ttl],
        )
        self.store.put(entry)

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


# --- loaders ----------------------------------------------------------------------------------


def _load(kind: str, prefer: bool, make_adapter, make_stub) -> tuple[Any, str]:
    if prefer:
        try:
            obj = make_adapter()
            return obj, f"{kind}: using {obj.name}"
        except AdapterUnavailable as exc:
            note = f"{kind}: {exc}; using the in-repo stub"
            log.info(note)
            return make_stub(), note
    return make_stub(), f"{kind}: using the in-repo stub"


def load_guard(prefer_adapter: bool = True) -> tuple[Guard, str]:
    return _load("guard", prefer_adapter, GuardrailsGuard, RegexGuard)


def load_retriever(prefer_adapter: bool = True) -> tuple[RuleRetriever, str]:
    return _load("retriever", prefer_adapter, RegragRetriever, SnapshotRetriever)


def load_cache(
    prefer_adapter: bool = False, path: str = ":memory:", embedder: Any = None
) -> tuple[Cache, str]:
    """The exact-key SQLite cache by default: rule research uses templated keys (spec 6.4).
    The semantic cache is for free-text queries (the MCP ``search_rules`` tool)."""
    return _load(
        "cache", prefer_adapter, lambda: SemcacheCache(embedder), lambda: SqliteCache(path)
    )
