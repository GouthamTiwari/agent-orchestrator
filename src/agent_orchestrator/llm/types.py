"""Plain data types shared by the LLM layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MediaType = Literal["image/png", "image/jpeg"]


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    media_type: MediaType = "image/png"

    def __repr__(self) -> str:  # never dump image bytes into logs
        return f"ImagePart({self.media_type}, {len(self.data)} bytes)"


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    parts: tuple[str | ImagePart, ...]

    @classmethod
    def user(cls, *parts: str | ImagePart) -> Message:
        return cls("user", tuple(parts))

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls("assistant", (text,))


@dataclass(frozen=True)
class LLMResult:
    text: str
    tokens_in: int
    tokens_out: int
    usd: float
    latency_ms: float
    model_id: str
    request_hash: str
    stop_reason: str = ""
    # where the answer came from: "live", "ledger" (paid earlier), "replay" (recording), "fake"
    source: str = "live"
    retries: int = field(default=0, compare=False)
