# agent-orchestrator

**GST input-tax-credit (ITC) reconciliation agents.** Each month an Indian business has to check its purchase books against what its suppliers reported to the GST portal, and against the invoices themselves. This project does that with a small team of agents:

> **The LLM reads and proposes. A deterministic, cited policy decides. A person approves.**

- A vision model reads each invoice image; deterministic code matches books, supplier data and invoices.
- Every proposed action goes through a versioned policy file whose rules cite the CGST Act and Rules.
- Clean matches close on their own; anything that changes money or the books waits for a human on an approval page.
- An invoice that tries to instruct the model gets no automatic action at all.
- A crashed run resumes where it stopped, and never pays twice for the same model call.

**Status: in progress.** The design is in [docs/SPEC.md](docs/SPEC.md), the cited rules in [docs/POLICY_RULES.md](docs/POLICY_RULES.md) and the build order in [docs/TASKS.md](docs/TASKS.md).

## Results

<!-- RESULTS:START -->
Results appear here after the evaluation tasks.
<!-- RESULTS:END -->

## Quick start

```
uv sync
uv run orchestrator version
uv run pytest
```

All business data in this repo is synthetic: company names, GSTINs and invoices are fictional. The evaluation set (`orchestrator dataset generate`) has 593 labelled cases from 10 fictional businesses over 3 months; its labels are assistant-reviewed, and the human spot-check is pending.

**This is not tax advice.** It is an engineering project that shows how to keep LLM agents inside rules a person can audit.

---

Built by Goutham Tiwari · MIT licence
