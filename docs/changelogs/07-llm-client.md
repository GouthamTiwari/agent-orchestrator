# 07 · LLM client layer (LIVE: probes, USD 0.00084)

## What changed
- `llm/client.py`: `LLMClient.complete(purpose, messages, system=, max_tokens=, temperature=, case_key=)` is the only way to call a model.
  - **Order of work:** request hash → paid-call ledger (reuse, USD 0) → replay (recording, or `RecordingMissing`) → worst-case estimate → run budget and project cap checks → pacer → call with retries → record real tokens and USD in the spend file, run budget, ledger, recording and trace span.
  - **Modes:** `fake`, `replay`, `live`. A mismatched mode and backend is a config error, and so is replay without a recorder.
  - **`BedrockBackend`:** Converse API with text, PNG and JPEG image blocks. botocore's own retries are off, so ours are the only ones. An empty bearer variable is removed (botocore would treat it as a token).
  - **`AnthropicBackend`:** `claude-haiku-5-5` with base64 image blocks and no `temperature` (the installed SDK 2.12 has no such parameter). SDK retries are off. The key is checked before the SDK is imported. Main test only.
- `llm/types.py`: `ImagePart` (its repr never prints the bytes), `Message`, `LLMResult(text, tokens_in, tokens_out, usd, latency_ms, model_id, request_hash, stop_reason, source, retries)`.
- `llm/pricing.py`: the spec 2.2 price table with `PRICES_CHECKED_ON = 2026-11-23`; tokens per image (Nova flat 230, Claude about 1,600 for a page); worst-case estimates. An unknown model raises.
- `llm/budget.py`:
  - `SpendLedger`: lifetime cap per provider in `.spend/ledger.jsonl` (Bedrock 5, Anthropic 3). Lines hold no text.
  - `RunBudget`: USD, tokens and seconds limits. At 90% it sets `stop_new_llm_work` with the reason, and `snapshot()` returns a `BudgetSnapshot`.
  - Both are thread-safe.
- `llm/retry.py`: retryable throttling, unavailable, 5xx, 429, timeouts and dropped connections (botocore and anthropic names). Full-jitter backoff (base 1 s, cap 20 s, 5 attempts). A Pydantic `ValidationError` is never retried here. A token-bucket `Pacer` on `perf_counter`.
- `llm/ledger.py`: `request_hash` over provider, model, system prompt, messages (images by sha256), max tokens and temperature. `PaidCallCache` wraps the store's `llm_calls` table.
- `llm/recording.py`: JSON-lines record and replay by request hash; one line per request.
- `llm/fake.py`: scripted answers, exceptions or callables, with a call counter.

## Outside the listed files
- `llm/types.py` (shared dataclasses, to avoid an import cycle).
- `docs/SPEC.md`: probe results in 2.2 and 15.5, and departure 6 reworded.
- `data/recordings/probes_task07.jsonl` (the three probe answers).

## LIVE probes (2026-11-23, `uv run pytest -m live tests/live -s`)
| Probe | Model | Tokens in / out | USD | Result |
|---|---|---|---|---|
| Read the seed invoice (150 DPI PNG) | `global.amazon.nova-2-lite-v1:0` | 325 / 55 | 0.000276 | Correct invoice no `SAM/2026/0100`, GSTIN `36TSDKP2716L1ZA`, taxable `Rs 47,300.00` |
| Answer from Mumbai | `apac.amazon.nova-pro-v1:0` | 7 / 10 | 0.000044 | Works from `ap-south-1` |
| 6 images in one request | `global.amazon.nova-2-lite-v1:0` | 1,451 / 4 | 0.000520 | Accepted, answered "6" |
| **Total** | | | **0.00084** | project spend file: 3 entries |

Findings recorded in the spec:
- The image limit is at least 6, not 5. The baseline (Task 19) probes a full month's image count first.
- Images are billed at the flat 230 tokens: 325 = 230 + 95 text tokens.
- Nova wraps JSON in ```json fences and writes amounts as `Rs 47,300.00`. The extraction schema guard (Task 09) must strip fences and normalise amounts.
- Nova Pro ignores "exactly": the answer to "Reply with exactly the word OK" was a longer sentence. Prompts must not rely on exact-format compliance; the schema guard handles it.

## Checks
- `uv run ruff check .`, `uv run ruff format --check .`: pass.
- `uv run pytest`: 221 passed, 3 live deselected (38 new, all offline).
  - Pricing table and unknown-model refusal.
  - Project caps per provider, and a spend file with no text.
  - Run budget: refusal on all three limits, and the 90% stop.
  - Retryable and non-retryable errors; validation errors never retried; full-jitter delays and their cap; the give-up count.
  - The pacer at 2 requests per second with a fake clock.
  - The request hash; fake, replay and ledger reuse (the fake's counter proves no second call).
  - Live mode against stubbed Bedrock and Anthropic clients: request shapes, cost, spend, budget, ledger, trace span; refusal before any call when the cap would be exceeded; throttling retried and paid once.
