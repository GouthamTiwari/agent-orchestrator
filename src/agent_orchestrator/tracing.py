"""Tracing: one span per unit of agent work, with tokens, cost, latency and outcome.

Usage::

    tracer = Tracer(run_id, repo)
    with tracer.span("extraction", case_key=key, input_hash=h) as s:
        result = call_model(...)
        s.add_usage(input_tokens=1200, output_tokens=300, usd=0.0013)
        s.set_output("11 fields, min confidence 0.97")

A span records even when the work raises: the error is stored and the exception still
propagates. Spans nest (``parent_id``) through a context variable, so concurrent branches each
keep their own parent. Latency uses ``time.perf_counter`` (``time.monotonic`` ticks in about
15 ms steps on Windows).

Never put document text in ``set_output`` or in exception messages: traces are shown in the UI
and exported to JSON.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import json
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MAX_TEXT = 500
_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_span", default=None
)


@dataclass
class Span:
    run_id: str
    span_id: str
    agent: str
    name: str
    parent_id: str | None = None
    case_key: str | None = None
    input_hash: str | None = None
    output_summary: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    latency_ms: float = 0.0
    decision: str | None = None
    error: str | None = None
    started_at: str = ""
    finished_at: str = ""
    _t0: float = field(default=0.0, repr=False)

    def add_usage(self, input_tokens: int = 0, output_tokens: int = 0, usd: float = 0.0) -> None:
        if min(input_tokens, output_tokens, usd) < 0:
            raise ValueError("usage cannot be negative")
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.usd += usd

    def set_output(self, summary: str) -> None:
        self.output_summary = summary[:MAX_TEXT]

    def set_decision(self, decision: str) -> None:
        self.decision = decision[:MAX_TEXT]

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_t0")
        return d


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds")


class Tracer:
    def __init__(self, run_id: str, repo: Any | None = None, span_prefix: str = "") -> None:
        self.run_id = run_id
        self.repo = repo  # a store.repo.Repo, or None to keep spans in memory only
        # A resumed run is a new process whose counter starts again at 1: a per-process
        # prefix keeps its span ids distinct from the first process's.
        self.span_prefix = span_prefix
        self.spans: list[Span] = []
        self._lock = threading.Lock()
        self._counter = 0

    def _next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"{self.span_prefix}s{self._counter:05d}"

    @contextmanager
    def span(
        self,
        agent: str,
        name: str | None = None,
        *,
        case_key: str | None = None,
        input_hash: str | None = None,
    ) -> Iterator[Span]:
        s = Span(
            run_id=self.run_id,
            span_id=self._next_id(),
            agent=agent,
            name=name or agent,
            parent_id=_current_span.get(),
            case_key=case_key,
            input_hash=input_hash,
            started_at=_now(),
            _t0=time.perf_counter(),
        )
        token = _current_span.set(s.span_id)
        try:
            yield s
        except BaseException as exc:
            s.error = f"{type(exc).__name__}: {exc}"[:MAX_TEXT]
            raise
        finally:
            _current_span.reset(token)
            s.latency_ms = round((time.perf_counter() - s._t0) * 1000, 3)
            s.finished_at = _now()
            with self._lock:
                self.spans.append(s)
            if self.repo is not None:
                self.repo.insert_span(s.as_dict())

    # --- summaries ----------------------------------------------------------------------------

    def totals_by_agent(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "spans": 0,
                "errors": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "usd": 0.0,
                "latency_ms": 0.0,
            }
        )
        for s in self.spans:
            t = out[s.agent]
            t["spans"] += 1
            t["errors"] += 1 if s.error else 0
            t["input_tokens"] += s.input_tokens
            t["output_tokens"] += s.output_tokens
            t["usd"] += s.usd
            t["latency_ms"] += s.latency_ms
        return {k: dict(v) for k, v in sorted(out.items())}

    def to_dict(self) -> dict[str, Any]:
        spans = sorted(self.spans, key=lambda s: (s.started_at, s.span_id))
        by_agent = self.totals_by_agent()
        return {
            "run_id": self.run_id,
            "spans": [s.as_dict() for s in spans],
            "by_agent": by_agent,
            "total": {
                "spans": len(spans),
                "errors": sum(1 for s in spans if s.error),
                "input_tokens": sum(s.input_tokens for s in spans),
                "output_tokens": sum(s.output_tokens for s in spans),
                "usd": round(sum(s.usd for s in spans), 6),
            },
        }

    def export_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            f.write("\n")
        return path
