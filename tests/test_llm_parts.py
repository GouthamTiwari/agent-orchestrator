"""Pricing, budgets, retries and the pacer (no network)."""

import datetime as dt
import random

import pytest
from pydantic import BaseModel, ValidationError

from agent_orchestrator.llm import pricing
from agent_orchestrator.llm.budget import BudgetExceeded, BudgetStop, RunBudget, SpendLedger
from agent_orchestrator.llm.retry import Pacer, backoff_delay, call_with_retry, is_retryable

# --- pricing ----------------------------------------------------------------------------------


def test_prices_match_the_spec_table():
    assert dt.date(2026, 11, 23) == pricing.PRICES_CHECKED_ON
    assert pricing.price("global.amazon.nova-2-lite-v1:0") == (0.35, 2.95)
    assert pricing.price("apac.amazon.nova-pro-v1:0") == (0.94, 3.76)
    assert pricing.price("claude-haiku-5-5") == (0.10, 0.50)


def test_unknown_model_raises_instead_of_costing_zero():
    with pytest.raises(pricing.UnknownModel):
        pricing.cost_usd("some.new-model", 100, 100)
    with pytest.raises(pricing.UnknownModel):
        pricing.estimate_cost("some.new-model", 100, 0, 10)


def test_cost_and_estimates():
    assert pricing.cost_usd("global.amazon.nova-2-lite-v1:0", 1_000_000, 0) == pytest.approx(0.35)
    assert pricing.estimate_input_tokens("global.amazon.nova-2-lite-v1:0", 300, 2) == 100 + 460
    est = pricing.estimate_cost("global.amazon.nova-2-lite-v1:0", 3000, 1, 800)
    assert est == pytest.approx((1230 * 0.35 + 800 * 2.95) / 1e6)


# --- project spend ledger ---------------------------------------------------------------------


def test_project_cap_per_provider(tmp_path):
    ledger = SpendLedger(tmp_path / "ledger.jsonl", caps={"bedrock": 0.01, "anthropic": 0.02})
    ledger.check("bedrock", 0.009)
    ledger.record(
        provider="bedrock", purpose="probe", model_id="m", tokens_in=1, tokens_out=1, usd=0.008
    )
    with pytest.raises(BudgetExceeded, match="bedrock project cap"):
        ledger.check("bedrock", 0.003)
    ledger.check("anthropic", 0.015)  # separate cap
    with pytest.raises(BudgetExceeded):
        ledger.check("openai", 0.0)  # no cap configured: refuse
    assert ledger.total("bedrock") == 0.008 and ledger.total("anthropic") == 0


def test_spend_file_has_no_text_and_survives_restarts(tmp_path):
    path = tmp_path / "ledger.jsonl"
    SpendLedger(path).record(
        provider="bedrock", purpose="extract", model_id="m", tokens_in=10, tokens_out=2, usd=0.001
    )
    again = SpendLedger(path)
    assert again.total() == 0.001
    assert set(again.entries()[0]) == {
        "ts",
        "provider",
        "purpose",
        "model",
        "tokens_in",
        "tokens_out",
        "usd",
    }
    with pytest.raises(ValueError):
        again.record(
            provider="bedrock",
            purpose="Has Spaces!",
            model_id="m",
            tokens_in=1,
            tokens_out=1,
            usd=0,
        )


# --- run budget -------------------------------------------------------------------------------


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_run_budget_refuses_calls_that_would_cross_a_limit():
    b = RunBudget(usd=1.0, tokens=1000, seconds=100, clock=Clock())
    b.check(0.5, 500)
    with pytest.raises(BudgetStop, match=r"\$"):
        b.check(1.01, 10)
    with pytest.raises(BudgetStop, match="tokens"):
        b.check(0.01, 1001)


@pytest.mark.parametrize(
    ("usd", "tokens", "seconds", "reason"),
    [(0.9, 0, 0, "usd"), (0, 900, 0, "tokens"), (0, 0, 90, "seconds")],
)
def test_stop_new_llm_work_at_ninety_percent(usd, tokens, seconds, reason):
    clock = Clock()
    b = RunBudget(usd=1.0, tokens=1000, seconds=100, clock=clock)
    b.record(usd * 0.999, int(tokens * 0.999))
    clock.t = seconds * 0.999
    assert not b.stop_new_llm_work
    b.record(usd * 0.001 + (0.0001 if usd else 0), 1 if tokens else 0)
    clock.t = seconds
    assert b.stop_new_llm_work
    snap = b.snapshot()
    assert snap.stop_new_llm_work and snap.reason.startswith(reason)
    with pytest.raises(BudgetStop):
        b.check(0.0, 0)


# --- retries ----------------------------------------------------------------------------------


class ClientError(Exception):  # shaped like botocore's
    def __init__(self, code, status=400):
        super().__init__(code)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class RateLimitError(Exception):  # named like anthropic's
    status_code = 429


class ReadTimeoutError(Exception):
    pass


class M(BaseModel):
    x: int


def _validation_error():
    try:
        M(x="not a number")
    except ValidationError as exc:
        return exc


@pytest.mark.parametrize(
    "exc",
    [
        ClientError("ThrottlingException"),
        ClientError("ServiceUnavailableException", 503),
        ClientError("Whatever", 502),
        RateLimitError(),
        ReadTimeoutError(),
        TimeoutError(),
    ],
)
def test_transient_errors_are_retryable(exc):
    assert is_retryable(exc)


@pytest.mark.parametrize(
    "exc",
    [
        ClientError("ValidationException"),
        ClientError("AccessDeniedException", 403),
        ValueError(),
        KeyError("x"),
    ],
)
def test_other_errors_are_not_retryable(exc):
    assert not is_retryable(exc)


def test_validation_error_is_never_retried_here():
    calls = []

    def fn():
        calls.append(1)
        raise _validation_error()

    with pytest.raises(ValidationError):
        call_with_retry(fn, sleep=lambda s: None)
    assert len(calls) == 1


def test_retries_with_full_jitter_then_succeeds():
    delays, attempts = [], []
    outcomes = [ClientError("ThrottlingException"), ClientError("ThrottlingException"), "ok"]

    def fn():
        item = outcomes[len(attempts)]
        attempts.append(1)
        if isinstance(item, Exception):
            raise item
        return item

    assert call_with_retry(fn, sleep=delays.append, rng=lambda: 1.0) == "ok"
    assert delays == [1.0, 2.0]  # rng=1.0 shows the ceiling: base * 2**attempt


def test_gives_up_after_five_attempts():
    delays = []
    with pytest.raises(ClientError):
        call_with_retry(
            lambda: (_ for _ in ()).throw(ClientError("ThrottlingException")),
            sleep=delays.append,
            rng=lambda: 1.0,
        )
    assert delays == [1.0, 2.0, 4.0, 8.0]  # 5 attempts, 4 waits


def test_backoff_is_capped_and_jittered():
    rng = random.Random(1).random
    samples = [backoff_delay(10, rng=rng) for _ in range(200)]
    assert max(samples) <= 20.0 and min(samples) >= 0.0
    assert len({round(s, 6) for s in samples}) > 150  # spread out, not lock-step


# --- pacer ------------------------------------------------------------------------------------


def test_pacer_limits_requests_per_second():
    clock = Clock()
    waits = []

    def sleep(s):
        waits.append(s)
        clock.t += s

    pacer = Pacer(rate=2.0, burst=1, clock=clock, sleep=sleep)
    for _ in range(5):
        pacer.acquire()
    assert clock.t == pytest.approx(2.0)  # 5 requests at 2/s: the first is free
    assert all(w == pytest.approx(0.5) for w in waits)


def test_pacer_uses_perf_counter_by_default():
    import time

    assert Pacer(1.0)._clock is time.perf_counter
    with pytest.raises(ValueError):
        Pacer(0)
