# 14 · Approval gate, queue and resume

## What changed
- `tools/toolfns.py`, the one set of tool functions for the graph, CLI, web app and MCP server (spec 10):
  - **`request_approval`** refuses unless the stored `Decision` for the case has outcome `require_approval`. The caller must name that exact decision (`decision_ref`: a hash of run, case, outcome, action, fired rules and policy version) and the deterministic key (spec 9.2). The proposed action must be the decision's. A second call returns the first row (`created: False`).
  - **`write_draft_email`** refuses unless the decision is `auto_allow` or a person approved or edited the case to a chase/verify action. A blocked case is refused even if an approval record exists.
    - The draft is built inside the tool from the run's own books and GSTR-2B, so a caller cannot pass in text.
    - The key is checked *before* drafting, so a wrong call never costs a model call.
    - Idempotent.
  - **Every refusal** writes an `audit_log` row (`refused:<tool>`, the rule that refused) and counts in `REFUSED_WRITES` (exported as a metric in Task 16).
  - **`latest_approval`:** the case's approval for its current facts, including one decided in an earlier run of the same month.
  - **Human actions `approve`, `edit`, `reject`** (CLI and web UI only):
    - an edit must be another allowed option and give a reason;
    - a rejection needs a reason and leaves the case with a person (`escalate_human`);
    - an approval can be decided only once.

    Who, when, before → after and the reason are audited (by the store). Edits and rejections are appended to `eval/human_feedback.jsonl` (gitignored; `ORCH_FEEDBACK_FILE` overrides the path). The line holds ids and action names only.
  - **`AGENT_TOOLS` vs `HUMAN_ACTIONS`:** what an agent may call, and what only a person may do. No agent tool approves anything.
- `orchestrator/graph.py`:
  - **Writes go through the tools:** `decide` requests approvals through `request_approval`, and `communicate` drafts through `write_draft_email`. It asks only for eligible cases, so the audit log isn't flooded with refusals.
  - **A blocked case's status is `needs_human`,** and an extraction that saw an injection adds the case flag `injection` (spec 13).
  - **Any case that needs a person goes through the gate,** even when its approval was already decided in an earlier run. The gate then passes straight on, and `apply_approvals` still applies the decision. Before this, a re-run skipped `apply_approvals` and left those decisions unapplied.
- `orchestrator/runner.py`: `start_run` / `resume_run` take an optional fake `script`, used by the injection test.
- `cli.py`:
  - `orchestrator approvals list [--run ID] [--all]`;
  - `approvals approve ID [--by NAME]`;
  - `approvals edit ID ACTION --reason ...`;
  - `approvals reject ID --reason ...`.

  The default name is `APPROVER_NAME`, else the OS user. A refused action prints why and exits 1.

## Outside the listed files
- `src/agent_orchestrator/tools/__init__.py`.
- `tests/conftest.py`: points `ORCH_FEEDBACK_FILE` at a temporary file for every test. One manual CLI check while building wrote a line to the real `eval/human_feedback.jsonl`; the file held only that line and was deleted.

## Checks
- `uv run ruff check .`, `uv run ruff format --check .`: pass.
- `uv run pytest`: 506 passed, 1 skipped (semcache not installed), 7 deselected (30 new: approvals 28, re-run 2).
  - **`request_approval`:**
    - idempotent;
    - refused for an auto-allowed decision, a blocked decision, a wrong `decision_ref`, another run, a wrong key, or a different proposed action. Each refusal has its audit row and adds to the counter.
  - **`write_draft_email`:**
    - an auto chase gets exactly one draft;
    - an approved verify gets one only after approval;
    - refused for a blocked case, a clean match, a rejected or edited-away approval, or a wrong `decision_ref`;
    - a wrong key is refused with 0 model calls.
  - **Human actions:**
    - approve, edit and reject record who, when, before → after and the reason;
    - `human_feedback.jsonl` gets the edit and the rejection only;
    - refused: an action outside the allowed list, an edit to the same action, a missing reason, an unknown id, deciding twice.
  - **The gate:** while it waits, a resume writes nothing to approvals, decisions, outbox, cases or the audit log.
  - **CLI:** list, approve, edit (a non-allowed action is refused) and reject (a reason is required). A missing database is reported.
  - **New processes:** run, decide with the CLI, resume, each step its own process. It pauses while one approval is pending and finishes once it is rejected.
  - **The injection invoice with a model that obeys it:** the reader hides the note and claims full confidence, and the planner proposes "none, approve full ITC". Result:
    - IMA/2026/0117 is still blocked by SAFE-INJ-01 (the free scan of the text layer caught it);
    - status `needs_human`, flag `injection`;
    - no approval row a person could approve into the model's action, and no outbox row;
    - `request_approval` refuses it, and the report lists it as blocked;
    - every "none" for a mismatch fell back to `escalate_human`.
  - **Idempotent re-run (spec 15.4):**
    - Running the finished month again under a new run id finishes without pausing, with identical case rows, 7 approvals for 7 keys, 4 drafts for 4 keys, and the same final actions.
    - Two runs started before anyone decides ask the same 7 questions; one decision answers both.

## Out of scope / follow-ups
- Read tools (`lookup_gstin`, `search_rules`, `list_cases`, `get_case`, `read_invoice`) and the MCP server: Task 15.
- The `orchestrator_refused_writes_total` metric and the web UI buttons: Task 16.
