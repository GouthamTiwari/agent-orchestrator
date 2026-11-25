"""The three small shapes the sibling projects plug into (docs/SPEC.md section 12).

Every agent depends on these Protocols only. The in-repo stubs (``stubs.py``) implement them
with no network and no extra services, so the whole system runs and tests offline; the
adapters (``adapters.py``) wrap Goutham's other projects when they are installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from agent_orchestrator.models import InjectionResult, RuleSnippet


@dataclass(frozen=True)
class ValidationResult:
    """A model answer checked against a Pydantic model. Errors name fields, never values."""

    value: BaseModel | None
    errors: list[str] = field(default_factory=list)
    prose_stripped: bool = False  # the JSON had to be cut out of fences or surrounding text

    @property
    def ok(self) -> bool:
        return self.value is not None


@runtime_checkable
class Guard(Protocol):
    name: str

    def scan(self, texts: Mapping[str, str]) -> InjectionResult:
        """Scan untrusted texts, keyed by where they came from ("text_layer", "other_text",
        "field:supplier_name", ...). ``sources`` in the result names the keys that hit."""
        ...

    def validate(self, text: str, model: type[BaseModel]) -> ValidationResult:
        """Cut the JSON out of a model answer and validate it against ``model``."""
        ...


@runtime_checkable
class RuleRetriever(Protocol):
    name: str

    @property
    def corpus_version(self) -> str:
        """Changes whenever the searchable text changes (part of every cache key)."""
        ...

    def search(self, query: str, k: int = 4) -> list[RuleSnippet]: ...

    def get(self, snippet_id: str) -> RuleSnippet | None: ...


@runtime_checkable
class Cache(Protocol):
    name: str

    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any) -> None: ...

    def stats(self) -> dict[str, int]:
        """At least ``hits`` and ``misses``, for the reported hit rate."""
        ...
