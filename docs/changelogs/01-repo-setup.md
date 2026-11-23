# 01 · Repo setup

## What changed
- Git repository on `main`, identity `Goutham Tiwari <227244284+GouthamTiwari@users.noreply.github.com>`, hooks path `.githooks`.
- `.githooks/` (identity check before each commit, co-author lines blocked in commit messages, every pushed commit checked, `audit-authors.sh` for the whole history), with the noreply address as the only allowed author and committer.
- `.gitignore`: secrets (`.env`), Python caches, local state (`*.sqlite`, `runs/`, `.spend/`, `outbox/`, generated data, human feedback) and the original project kit (`_kit/`, private notes).
- `.gitattributes`: LF line endings for hooks, JSON, JSONL, YAML, CSV and the rule texts (their sha256 is the rule corpus version, so bytes must match on every OS); PDFs and PNGs are binary.
- `.env.example` with region `ap-south-1`, the checked Nova model IDs, budgets and limits. The local `.env` was built from it and the Bedrock key copied from the guardrails project by a script that never prints the key.
- `pyproject.toml` (Python 3.12, uv, hatchling, src layout, `orchestrator` script), dependency lower bounds at the versions checked on 2026-11-23. Installed: langgraph 1.2.24, mcp 2.3.0, pydantic 2.14.4, fastapi 0.165.0, anthropic 2.12.0.
- `LICENSE` (MIT), `README.md` skeleton with the results markers and the "not tax advice" line.
- `src/agent_orchestrator/` with a minimal Typer CLI (`orchestrator version`).
- `data/seed/`: the kit's seed generator, checksum reference and its 18-case dataset (17 invoice PDFs), copied byte for byte.
- `data/rules/sources/`: nine official CBIC texts (CGST Act s.16, 34, 38, 77; CGST Rules r.36, 37, 37A, 46, 60), each with its source URL and retrieval date.
- CI: lint, format check and tests on Python 3.12, plus an authorship audit job over the full history.

## Checks
- `uv run ruff check .`, `uv run ruff format --check .`: pass.
- `uv run pytest`: 5 passed (CLI version, seed dataset complete, injection invoice present, rule texts carry source and date, rule texts are LF only).
- `git check-ignore` confirms `.env` and `_kit/` are ignored.
- Seed regeneration: running `make_seed_data.py` again gives the same CSV byte for byte, and the same JSON and JSONL content (only line endings differ, because Python writes CRLF on Windows). The 17 PDFs differ only in their embedded creation timestamp and document id. Task 03's generator will write LF and fixed PDF metadata so its output is byte-identical on every run.

## Notes
- The first commit goes straight to `main` because there is nothing to branch from yet; from Task 02 every task uses a `task/NN-slug` branch.
