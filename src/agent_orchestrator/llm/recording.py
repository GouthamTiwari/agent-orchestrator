"""Record and replay model calls (docs/SPEC.md section 15.3).

In ``record`` mode every live answer is appended to a JSON-lines file in ``data/recordings/``
(committed: text answers and token counts only, never keys). In ``replay`` mode answers are
looked up by request hash, so evaluations rerun offline and give the same numbers. A missing
recording in replay mode is an error, never a silent live call.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from agent_orchestrator.llm.types import LLMResult


class RecordingMissing(LookupError):
    """Replay mode found no recording for this request."""


class Recorder:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._cache: dict[str, dict] | None = None

    def _load(self) -> dict[str, dict]:
        if self._cache is None:
            self._cache = {}
            if self.path.is_file():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        self._cache[row["request_hash"]] = row
        return self._cache

    def lookup(self, req_hash: str) -> LLMResult:
        with self._lock:
            row = self._load().get(req_hash)
        if row is None:
            raise RecordingMissing(
                f"no recording for request {req_hash[:12]} in {self.path}; rerun with --live "
                "(asks before spending) to record it"
            )
        return LLMResult(
            text=row["text"],
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            usd=0.0,
            latency_ms=0.0,
            model_id=row["model_id"],
            request_hash=req_hash,
            stop_reason=row.get("stop_reason", ""),
            source="replay",
        )

    def record(self, result: LLMResult, purpose: str) -> None:
        row = {
            "request_hash": result.request_hash,
            "purpose": purpose,
            "model_id": result.model_id,
            "text": result.text,
            "stop_reason": result.stop_reason,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "usd": round(result.usd, 9),
        }
        with self._lock:
            cache = self._load()
            if result.request_hash in cache:
                return
            cache[result.request_hash] = row
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
