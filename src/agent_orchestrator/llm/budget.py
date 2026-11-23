"""Spend control (docs/SPEC.md sections 2.3 and 8.3).

Two independent limits, both checked BEFORE a call with a worst-case estimate:

- ``SpendLedger``: the project's lifetime cap per provider, kept in ``.spend/ledger.jsonl``
  (gitignored, survives across runs and days). Bedrock: USD 5. Anthropic: USD 3 of the USD 4 key.
- ``RunBudget``: one run's limits on USD, tokens and wall time. When spend reaches 90% of any
  limit, ``stop_new_llm_work`` is set and no new call starts; deterministic steps carry on.

Both are thread-safe: parallel graph branches share one instance, so totals cannot be lost.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import threading
import time
from pathlib import Path

from agent_orchestrator.models import BudgetSnapshot

DEFAULT_LEDGER = Path(".spend/ledger.jsonl")
STOP_AT = 0.90
_PURPOSE = re.compile(r"^[a-z][a-z0-9_.-]{0,40}$")


class BudgetExceeded(RuntimeError):
    """The call would go over a spending limit; it was not made."""


class BudgetStop(BudgetExceeded):
    """The run reached 90% of a limit; no new LLM work starts."""


class SpendLedger:
    """Lifetime spend per provider. Lines hold purpose, model, tokens and USD only, never text."""

    def __init__(self, path: Path | str = DEFAULT_LEDGER, caps: dict[str, float] | None = None):
        self.path = Path(path)
        self.caps = caps or {"bedrock": 5.0, "anthropic": 3.0}
        self._lock = threading.Lock()

    def entries(self) -> list[dict]:
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def total(self, provider: str | None = None) -> float:
        return round(
            sum(
                float(e["usd"])
                for e in self.entries()
                if provider is None or e["provider"] == provider
            ),
            9,
        )

    def check(self, provider: str, estimate_usd: float) -> None:
        if provider not in self.caps:
            raise BudgetExceeded(f"no spending cap configured for provider {provider!r}")
        spent = self.total(provider)
        cap = self.caps[provider]
        if spent + estimate_usd > cap:
            raise BudgetExceeded(
                f"{provider} project cap: spent ${spent:.4f} + estimated ${estimate_usd:.4f} "
                f"> ${cap:.2f}"
            )

    def record(
        self, *, provider: str, purpose: str, model_id: str, tokens_in: int, tokens_out: int,
        usd: float,
    ) -> None:  # fmt: skip
        if not _PURPOSE.match(purpose):
            raise ValueError("purpose must be a short lowercase identifier")
        line = json.dumps(
            {
                "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "provider": provider,
                "purpose": purpose,
                "model": model_id,
                "tokens_in": int(tokens_in),
                "tokens_out": int(tokens_out),
                "usd": round(float(usd), 9),
            }
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")


class RunBudget:
    def __init__(
        self,
        usd: float = 1.0,
        tokens: int = 500_000,
        seconds: float = 900.0,
        started_at: dt.datetime | None = None,
        clock=time.perf_counter,
    ) -> None:
        self.limits = {"usd": usd, "tokens": float(tokens), "seconds": seconds}
        self.usd = 0.0
        self.tokens = 0
        self.started_at = started_at or dt.datetime.now(dt.UTC)
        self._clock = clock
        self._t0 = clock()
        self._lock = threading.Lock()
        self.stop_reason: str | None = None

    def _elapsed(self) -> float:
        return self._clock() - self._t0

    def _update_stop(self) -> None:
        used = {"usd": self.usd, "tokens": float(self.tokens), "seconds": self._elapsed()}
        for name, value in used.items():
            if self.stop_reason is None and value >= STOP_AT * self.limits[name]:
                self.stop_reason = (
                    f"{name} at {value:.4g} of {self.limits[name]:.4g} "
                    f"({value / self.limits[name]:.0%})"
                )

    @property
    def stop_new_llm_work(self) -> bool:
        with self._lock:
            self._update_stop()
            return self.stop_reason is not None

    def check(self, estimate_usd: float, estimate_tokens: int) -> None:
        """Raise BudgetStop if the run is stopped or this call could cross a limit."""
        with self._lock:
            self._update_stop()
            if self.stop_reason:
                raise BudgetStop(f"run budget stop: {self.stop_reason}")
            if self.usd + estimate_usd > self.limits["usd"]:
                raise BudgetStop(
                    f"run budget: ${self.usd:.4f} + ${estimate_usd:.4f} would "
                    f"exceed ${self.limits['usd']:.2f}"
                )
            if self.tokens + estimate_tokens > self.limits["tokens"]:
                raise BudgetStop(
                    f"run budget: {self.tokens} + {estimate_tokens} tokens would "
                    f"exceed {self.limits['tokens']:.0f}"
                )

    def record(self, usd: float, tokens: int) -> None:
        with self._lock:
            self.usd += usd
            self.tokens += tokens
            self._update_stop()

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            self._update_stop()
            return BudgetSnapshot(
                usd_spent=round(self.usd, 9),
                tokens=self.tokens,
                started_at=self.started_at,
                stop_new_llm_work=self.stop_reason is not None,
                reason=self.stop_reason,
            )
