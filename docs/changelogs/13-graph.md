# 13 · Orchestrator graph, checkpoints and budgets

## What changed
- `orchestrator/state.py`:
  - **`GraphState`:** `RunState` plus `buyer_gstin`, `plan_notes` (with a reducer) and `approval_ids`.
  - **Branch payloads:** `ExtractTask` and `PlanTask`.
  - **`BRANCH_WRITES`:** the fields each parallel branch may return.
  - **`CHECKPOINT_TYPES`:** the explicit list of model classes the checkpoint may hold. LangGraph's serializer only rebuilds registered classes, so a checkpoint cannot make the process build an arbitrary object.
- `orchestrator/graph.py`, the only module that imports LangGraph (tested). It has the spec 8.1 graph:
  - **Extraction:** `intake → fanout_extract ─Send per PDF→ extract_one → join_extract → match`.
  - **Planning:** `route_cases`: `Send` per non-clean case `→ research_and_plan`, plus `rule_propose` for clean matches (no model), both `→ decide`.
  - **Gate and finish:** `decide → approval_gate` only if an approval is pending, then `apply_approvals → communicate → report`.

  How the nodes behave:
  - **Thin nodes:** they call `agents/*`, the policy engine and the store, and nothing else.
  - **`decide`:**
    - runs the policy for every case;
    - records decisions and upserts cases;
    - requests approvals idempotently (`approval_key`, with `escalate_human` always among the options so a rejection is possible);
    - marks cases that a budget stop cut short as `needs_human`, flagged `budget_stop`.
  - **`approval_gate`:** writes nothing before `interrupt()`. It reads the approvals' status from SQLite and interrupts again while any is pending; the resume payload (`ApprovalResume`) only says "go".
  - **Research:** one lock per mismatch type, so two branches never research the same type at once and the second finds the cached summary.
  - **`cost_by_agent`:** sums the spans of every process of the run from the store.
  - **`write_run_report`:** writes the partial report while paused and the final one at the end.
- `orchestrator/runner.py`:
  - **`start_run` / `resume_run`:** `SqliteSaver` on the store's own file, `thread_id = run_id`, `durability="sync"`, `max_concurrency` 2. Run settings go to `runs/<run_id>/run.json`, so a new process can resume. The run budget continues from the run's rows in the paid-call ledger.
  - **`ThreadRepo`:** one SQLite connection per thread.
  - **`demo_script`:** a deterministic fake model for every prompt. It reads an invoice from its book line and puts the PDF's own text into `other_text`, so the planted note is seen and blocked.
  - **`ORCH_CRASH_AFTER_LLM_CALLS`:** `os._exit(1)` right after the N-th paid call, once that call is stored.
- `cli.py`:
  - `orchestrator run --demo | --data DIR --period P [--live --yes --provider --rag --run-id --db --runs-dir]`. A live run asks first.
  - `orchestrator resume RUN_ID`. It refuses cleanly when nothing is left to resume.

## Found and fixed while building
1. **PDFium is not thread-safe, even behind a lock.**
   - Two invoice branches rendering at once crashed the process (access violation). With a lock, about one run in two still rendered a page slightly differently (around 9,000 pixel bytes changed). With garbage collection turned off, it segfaulted.
   - A changed render also changes the paid-call ledger's request hash, which covers the image, so a crash-resume could have paid for a page twice.
   - **Fix:** every PDFium call now runs on one dedicated thread (`render.py`; `eval/scan_noise.py` uses it too). Bitmaps are also kept alive until their pixels are copied. After the fix: 11 demo runs in a row (5 with garbage collection off) were identical, and a test renders from 4 threads and compares the bytes.
2. **Span ids collided after a resume:** each process counted from `s00001`. Each process's tracer now gets its own id prefix (`tracing.py`).
3. **A crash loses in-flight calls.** With 2 parallel branches, the other branch's model call can be with the model when the process dies. That call is paid, its answer is lost, and it is made again after resume. Every *finished* call is answered from the ledger, so none is paid twice.

   The test therefore measures `paid calls = uninterrupted calls + calls in flight at the crash`, with at most 1 in flight (concurrency 2). Spec 15.4's "the paid-call counter equals the uninterrupted count" holds exactly when nothing was in flight.

## Outside the listed files
- `src/agent_orchestrator/render.py`, `eval/scan_noise.py` (the PDFium thread).
- `store/db.py`: `connect(check_same_thread=False)`, used only by `ThreadRepo` so the main thread can close every thread's connection.
- `tracing.py`: `span_prefix`.
- `orchestrator/__init__.py`.

## Checks
- `uv run ruff check .`, `uv run ruff format --check .`: pass.
- `uv run pytest`: 476 passed, 1 skipped (semcache not installed), 7 deselected (16 new: graph 14, crash-resume 2).
  - **Structure:** only `orchestrator/graph.py` imports LangGraph, and nothing in `agents/` does. Every field in `BRANCH_WRITES` has a reducer, and in a real streamed run `extract_one` and `research_and_plan` return only those fields.
  - **The demo:**
    - It pauses at the gate with 7 pending approvals: 11 auto-allowed, 1 blocked (IMA/2026/0117), no draft yet. The partial report says "paused".
    - The checkpoint is a `SqliteSaver` thread named after the run, and the run uses `durability="sync"` (spied).
  - **The gate:**
    - Resuming with nothing decided interrupts again with the same list. After one decision it waits for the other 6.
    - With all approved it finishes: 2 chase, 2 verify, 2 correct-books, 4 to a person, 9 closed, 4 drafts, report and trace written, run `done`. A second resume is refused.
    - An edit and a rejection change the outcome and the number of drafts.
  - **Budget:**
    - With 6,000 tokens, new model work stops at 90%. The cases cut short are `needs_human`, flagged `budget_stop` and fired by SAFE-BUDGET-01. No call crossed the limit.
    - The report says "stopped: tokens at …", and the deterministic steps and the gate still ran.
  - **CLI:** run then resume; a live run asks first and starts nothing on "n".
  - **Crash-resume** (subprocesses, about 35 s): crashing after the 5th call (while invoices are being read) or the 24th (while cases are being planned), then resuming in a new process, gives the same cases, decisions, approvals, status and ledger rows as an uninterrupted run. Paid calls equal the uninterrupted 35 plus the calls in flight. Passed 3 times in a row.

## Out of scope / follow-ups
- The write tools (`request_approval`, `write_draft_email`) and the CLI approval commands: Task 14. `decide` and `communicate` write through the store for now.
- `run --data` without `--live` replays a recording, which comes with the live runs (Task 18).
- The run budget's time limit restarts in a resumed process (spend and tokens carry over).
