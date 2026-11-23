"""Retries for transient provider errors, and a request pacer (docs/SPEC.md section 8.3).

Retried: throttling, "service unavailable", 5xx, 429, timeouts and dropped connections, with
exponential backoff and FULL jitter (sleep a random time between 0 and min(cap, base * 2^n)),
so parallel callers spread out instead of retrying in lock-step. At most 5 attempts.

Never retried here: anything else, and in particular a Pydantic ``ValidationError`` (a model
answer that did not fit the schema). That goes to the agent's schema-retry loop (cap 2), which
re-asks with the validation errors; retrying the identical request would just pay again.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable

from pydantic import ValidationError

RETRYABLE_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
    "ModelTimeoutException",
}
RETRYABLE_CLASS_NAMES = {
    # botocore
    "ReadTimeoutError",
    "ConnectTimeoutError",
    "EndpointConnectionError",
    "ConnectionClosedError",
    # anthropic
    "RateLimitError",
    "InternalServerError",
    "OverloadedError",
    "ServiceUnavailableError",
    "GatewayTimeoutError",
    "APITimeoutError",
    "APIConnectionError",
    "DeadlineExceededError",
    # standard library
    "TimeoutError",
    "ConnectionError",
}


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ValidationError):
        return False
    response = getattr(exc, "response", None)
    if isinstance(response, dict):  # botocore ClientError
        code = response.get("Error", {}).get("Code", "")
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        if code in RETRYABLE_CODES or status == 429 or status >= 500:
            return True
    status = getattr(exc, "status_code", None)  # anthropic APIStatusError
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True
    return any(cls.__name__ in RETRYABLE_CLASS_NAMES for cls in type(exc).__mro__)


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 20.0, rng=random.random) -> float:
    """Full jitter: uniform in [0, min(cap, base * 2**attempt)]. ``attempt`` starts at 0."""
    return rng() * min(cap, base * (2**attempt))


def call_with_retry[T](
    fn: Callable[[], T],
    *,
    max_attempts: int = 5,
    base: float = 1.0,
    cap: float = 20.0,
    sleep: Callable[[float], None] = time.sleep,
    rng=random.random,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            if not is_retryable(exc) or attempt == max_attempts - 1:
                raise
            delay = backoff_delay(attempt, base, cap, rng)
            if on_retry:
                on_retry(attempt + 1, exc, delay)
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


class Pacer:
    """Token bucket: at most ``rate`` requests per second on average, bursts of ``burst``.

    Uses ``time.perf_counter``: ``time.monotonic`` advances in ~15 ms steps on Windows, which
    made the guardrails project's pacer let bursts through.
    """

    def __init__(
        self,
        rate: float,
        burst: int = 1,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate, self.burst = rate, burst
        self._clock, self._sleep = clock, sleep
        self._tokens = float(burst)
        self._last = clock()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a request may start. Returns the time waited in seconds."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return waited
                need = (1 - self._tokens) / self.rate
            self._sleep(need)
            waited += need
