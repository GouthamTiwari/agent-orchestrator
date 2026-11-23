"""A scripted stand-in for a model, for tests and offline demos. It costs nothing.

Script entries are used in order: a string is the answer, an exception is raised (to test
retries), and a callable receives ``(system, messages)`` and returns a string. When the script
runs out, the last entry repeats.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agent_orchestrator.llm.types import Message

ScriptItem = str | BaseException | Callable[[str, Sequence[Message]], str]


@dataclass
class RawAnswer:
    text: str
    tokens_in: int
    tokens_out: int
    stop_reason: str = "end_turn"


class FakeBackend:
    provider = "fake"

    def __init__(self, script: Sequence[ScriptItem] | ScriptItem = "{}", model_id: str = "fake"):
        self.script = list(script) if isinstance(script, list | tuple) else [script]
        self.model_id = model_id
        self.calls: list[tuple[str, Sequence[Message]]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def invoke(
        self, system: str, messages: Sequence[Message], max_tokens: int,
        temperature: float | None,
    ) -> RawAnswer:  # fmt: skip
        self.calls.append((system, messages))
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(item, BaseException):
            raise item
        text = item(system, messages) if callable(item) else item
        chars = len(system) + sum(len(p) for m in messages for p in m.parts if isinstance(p, str))
        return RawAnswer(text=text, tokens_in=max(1, chars // 3), tokens_out=max(1, len(text) // 3))
