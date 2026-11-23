"""The one door every model call goes through (docs/SPEC.md sections 2.3 and 8.3).

``LLMClient.complete(...)`` does, in order:

1. hash the request (provider, model, prompt, images, settings);
2. paid-call ledger: if this exact request was paid before, return that answer (USD 0);
3. replay mode: return the recorded answer, or raise if there is none (never a silent call);
4. live mode: estimate the worst-case cost, check the run budget and the project cap, wait for
   the pacer, call the provider with retries, then record real tokens and USD everywhere
   (project spend file, run budget, ledger, recording, trace span).

Agents only see ``complete`` and ``LLMResult``; providers are small backends with ``invoke``.
"""

from __future__ import annotations

import base64
import os
import time
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from agent_orchestrator.llm import pricing
from agent_orchestrator.llm.budget import RunBudget, SpendLedger
from agent_orchestrator.llm.fake import FakeBackend, RawAnswer
from agent_orchestrator.llm.ledger import PaidCallCache, request_hash
from agent_orchestrator.llm.recording import Recorder
from agent_orchestrator.llm.retry import Pacer, call_with_retry
from agent_orchestrator.llm.types import ImagePart, LLMResult, Message

Mode = Literal["fake", "replay", "live"]
BEARER_ENV_VAR = "AWS_BEARER_TOKEN_BEDROCK"


class ConfigError(RuntimeError):
    """Missing or contradictory settings."""


class Backend(Protocol):
    provider: str
    model_id: str

    def invoke(
        self, system: str, messages: Sequence[Message], max_tokens: int,
        temperature: float | None,
    ) -> RawAnswer: ...  # fmt: skip


# --- Bedrock (Converse API) -------------------------------------------------------------------


class BedrockBackend:
    provider = "bedrock"

    def __init__(self, model_id: str, region: str = "ap-south-1", client: Any = None) -> None:
        self.model_id = model_id
        self.region = region
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3
            from botocore.config import Config

            # botocore's own retries are off: ours (full jitter, logged) are the only ones.
            config = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=60)
            if not os.environ.get(BEARER_ENV_VAR, "").strip():
                os.environ.pop(BEARER_ENV_VAR, None)  # an empty value would still be "a token"
            self._client = boto3.client("bedrock-runtime", region_name=self.region, config=config)
        return self._client

    @staticmethod
    def _content(parts: Sequence[str | ImagePart]) -> list[dict]:
        out: list[dict] = []
        for p in parts:
            if isinstance(p, ImagePart):
                fmt = "png" if p.media_type == "image/png" else "jpeg"
                out.append({"image": {"format": fmt, "source": {"bytes": p.data}}})
            else:
                out.append({"text": p})
        return out

    def invoke(self, system, messages, max_tokens, temperature) -> RawAnswer:
        inference: dict[str, Any] = {"maxTokens": max_tokens}
        if temperature is not None:
            inference["temperature"] = temperature
        request: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": [{"role": m.role, "content": self._content(m.parts)} for m in messages],
            "inferenceConfig": inference,
        }
        if system:
            request["system"] = [{"text": system}]
        response = self.client.converse(**request)
        content = response.get("output", {}).get("message", {}).get("content", [])
        usage = response.get("usage", {})
        return RawAnswer(
            text="".join(b.get("text", "") for b in content if isinstance(b, dict)),
            tokens_in=int(usage.get("inputTokens", 0)),
            tokens_out=int(usage.get("outputTokens", 0)),
            stop_reason=str(response.get("stopReason", "")),
        )


# --- Anthropic (Messages API), main test only -------------------------------------------------


class AnthropicBackend:
    provider = "anthropic"

    def __init__(self, model_id: str = "claude-haiku-5-5", client: Any = None) -> None:
        self.model_id = model_id
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
                raise ConfigError("ANTHROPIC_API_KEY is not set (main test only)")
            import anthropic

            # SDK retries off: ours are the only ones, so the attempt count is known.
            self._client = anthropic.Anthropic(max_retries=0, timeout=60.0)
        return self._client

    @staticmethod
    def _content(parts: Sequence[str | ImagePart]) -> list[dict]:
        out: list[dict] = []
        for p in parts:
            if isinstance(p, ImagePart):
                data = base64.standard_b64encode(p.data).decode("ascii")
                out.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": p.media_type, "data": data},
                    }
                )
            else:
                out.append({"type": "text", "text": p})
        return out

    def invoke(self, system, messages, max_tokens, temperature) -> RawAnswer:
        # temperature is deliberately not sent: the current API rejects it for this model.
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": m.role, "content": self._content(m.parts)} for m in messages],
        }
        if system:
            kwargs["system"] = system
        response = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        return RawAnswer(
            text=text,
            tokens_in=int(response.usage.input_tokens),
            tokens_out=int(response.usage.output_tokens),
            stop_reason=str(response.stop_reason or ""),
        )


# --- the client -------------------------------------------------------------------------------


class LLMClient:
    def __init__(
        self,
        backend: Backend,
        *,
        mode: Mode = "replay",
        spend: SpendLedger | None = None,
        run_budget: RunBudget | None = None,
        cache: PaidCallCache | None = None,
        recorder: Recorder | None = None,
        pacer: Pacer | None = None,
        tracer: Any = None,
        run_id: str | None = None,
        retry_sleep=time.sleep,
    ) -> None:
        if mode == "fake" and not isinstance(backend, FakeBackend):
            raise ConfigError("fake mode needs the fake backend")
        if mode == "live" and isinstance(backend, FakeBackend):
            raise ConfigError("live mode needs a real backend")
        if mode == "replay" and recorder is None:
            raise ConfigError("replay mode needs a recorder")
        self.backend = backend
        self.mode = mode
        self.spend = spend
        self.run_budget = run_budget
        self.cache = cache
        self.recorder = recorder
        self.pacer = pacer
        self.tracer = tracer
        self.run_id = run_id
        self._retry_sleep = retry_sleep

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    def complete(
        self,
        purpose: str,
        messages: Sequence[Message],
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float | None = None,
        case_key: str | None = None,
    ) -> LLMResult:
        """Answer one request. Raises BudgetExceeded/BudgetStop before spending, and
        RecordingMissing in replay mode; provider errors propagate after retries."""
        if self.tracer is None:
            return self._complete(purpose, messages, system, max_tokens, temperature)
        with self.tracer.span("llm", purpose, case_key=case_key) as span:
            result = self._complete(purpose, messages, system, max_tokens, temperature)
            span.input_hash = result.request_hash[:16]
            paid = result.source in ("live", "fake")
            span.add_usage(
                result.tokens_in if paid else 0, result.tokens_out if paid else 0, result.usd
            )
            span.set_output(f"{result.model_id} {result.source} {result.tokens_out} tokens out")
            return result

    def _complete(self, purpose, messages, system, max_tokens, temperature) -> LLMResult:
        messages = list(messages)
        provider = self.backend.provider
        model = self.backend.model_id
        req = request_hash(provider, model, system, messages, max_tokens, temperature)

        if self.cache is not None and (hit := self.cache.get(req)) is not None:
            return hit
        if self.mode == "replay":
            return self.recorder.lookup(req)  # raises RecordingMissing

        images = sum(1 for m in messages for p in m.parts if isinstance(p, ImagePart))
        chars = len(system) + sum(len(p) for m in messages for p in m.parts if isinstance(p, str))
        estimate = pricing.estimate_cost(model, chars, images, max_tokens)
        est_tokens = pricing.estimate_input_tokens(model, chars, images) + max_tokens
        if self.run_budget is not None:
            self.run_budget.check(estimate, est_tokens)
        if self.mode == "live":
            if self.spend is None:
                raise ConfigError("live mode needs the project spend ledger")
            self.spend.check(provider, estimate)
        if self.pacer is not None:
            self.pacer.acquire()

        retries = 0

        def note_retry(attempt: int, exc: BaseException, delay: float) -> None:
            nonlocal retries
            retries = attempt

        start = time.perf_counter()
        raw = call_with_retry(
            lambda: self.backend.invoke(system, messages, max_tokens, temperature),
            sleep=self._retry_sleep,
            on_retry=note_retry,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        usd = pricing.cost_usd(model, raw.tokens_in, raw.tokens_out)
        result = LLMResult(
            text=raw.text,
            tokens_in=raw.tokens_in,
            tokens_out=raw.tokens_out,
            usd=usd,
            latency_ms=round(latency_ms, 3),
            model_id=model,
            request_hash=req,
            stop_reason=raw.stop_reason,
            source="fake" if self.mode == "fake" else "live",
            retries=retries,
        )
        if self.mode == "live":
            self.spend.record(
                provider=provider,
                purpose=purpose,
                model_id=model,
                tokens_in=raw.tokens_in,
                tokens_out=raw.tokens_out,
                usd=usd,
            )
        if self.run_budget is not None:
            self.run_budget.record(usd, raw.tokens_in + raw.tokens_out)
        if self.cache is not None:
            self.cache.put(result, purpose, self.run_id)
        if self.recorder is not None and self.mode == "live":
            self.recorder.record(result, purpose)
        return result


def make_backend(provider: str, model_id: str, region: str = "ap-south-1") -> Backend:
    if provider == "bedrock":
        return BedrockBackend(model_id, region)
    if provider == "anthropic":
        return AnthropicBackend(model_id)
    raise ConfigError(f"unknown provider {provider!r}")
