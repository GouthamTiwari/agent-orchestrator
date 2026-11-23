"""Request hashing and the paid-call cache (docs/SPEC.md sections 2.3 and 9).

The request hash covers everything that changes the model's answer: provider, model, system
prompt, every message part (images by their sha256), max tokens and temperature. The same hash
means the same question, so the stored answer is reused and nothing is paid twice, even after
a crash and resume.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agent_orchestrator.llm.types import ImagePart, LLMResult, Message


def request_hash(
    provider: str,
    model_id: str,
    system: str,
    messages: list[Message],
    max_tokens: int,
    temperature: float | None,
) -> str:
    def part(p: str | ImagePart) -> Any:
        if isinstance(p, ImagePart):
            return {"image": hashlib.sha256(p.data).hexdigest(), "type": p.media_type}
        return {"text": p}

    payload = {
        "provider": provider,
        "model": model_id,
        "system": system,
        "messages": [{"role": m.role, "parts": [part(p) for p in m.parts]} for m in messages],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PaidCallCache:
    """Thin wrapper over the store's ``llm_calls`` table (Repo.ledger_get / ledger_put)."""

    def __init__(self, repo: Any) -> None:
        self.repo = repo

    def get(self, req_hash: str) -> LLMResult | None:
        row = self.repo.ledger_get(req_hash)
        if row is None:
            return None
        r = row["response"]
        return LLMResult(
            text=r["text"],
            tokens_in=row["input_tokens"],
            tokens_out=row["output_tokens"],
            usd=0.0,  # already paid once; reusing it is free
            latency_ms=0.0,
            model_id=row["model_id"],
            request_hash=req_hash,
            stop_reason=r.get("stop_reason", ""),
            source="ledger",
        )

    def put(self, result: LLMResult, purpose: str, run_id: str | None = None) -> bool:
        return self.repo.ledger_put(
            result.request_hash,
            purpose=purpose,
            model_id=result.model_id,
            input_tokens=result.tokens_in,
            output_tokens=result.tokens_out,
            usd=result.usd,
            latency_ms=result.latency_ms,
            response={"text": result.text, "stop_reason": result.stop_reason},
            run_id=run_id,
        )
