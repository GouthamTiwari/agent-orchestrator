"""Rule research: the law that applies to one mismatch type, in plain English (spec 6.4).

Steps:
1. A fixed search query per mismatch type, naming the provisions that matter ("section
   16(2)(aa)", "rule 36(4)") so both retrievers put them first.
2. ``RuleRetriever.search``: Regulatory RAG (``regrag.api.search``, no LLM call) when it is
   installed and its database is up, else BM25 over the saved CBIC texts.
3. Cache lookup on ``(mismatch_type, corpus_version)``. The query is fixed, so after about one
   call per mismatch type every lookup is a hit. Because ``corpus_version`` comes from the
   retriever that answered, RAG and snapshot results never mix.
4. On a miss, a small LLM writes a short summary that may cite only the snippet ids it was
   given. A bad answer is re-asked with the problems (at most 2 retries); after that the
   result has no summary, and the planner still gets the snippets.

Nothing from an invoice is in these prompts: the input is a mismatch type, and the text is
official law.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from string import Template

from pydantic import BaseModel, ConfigDict, Field

from agent_orchestrator.interfaces.protocols import Cache, Guard, RuleRetriever
from agent_orchestrator.llm.budget import BudgetExceeded
from agent_orchestrator.llm.client import ConfigError, LLMClient
from agent_orchestrator.llm.pricing import UnknownModel
from agent_orchestrator.llm.recording import RecordingMissing
from agent_orchestrator.llm.types import Message
from agent_orchestrator.models import MismatchType, RuleResearch, RuleSnippet

# Bump when a query, the prompt or the answer shape changes, so old cache entries are not reused.
RESEARCH_VERSION = "1"
K = 4
MAX_SCHEMA_RETRIES = 2
MAX_OUTPUT_TOKENS = 500
PASSAGE_CHARS = 1500  # per snippet in a prompt; law passages are long, the clause comes first

# Clean matches never reach research or the planner (spec 6.5).
CLEAN_TYPES: frozenset[str] = frozenset(
    {"none", "rounding_within_tolerance", "invoice_no_format_difference"}
)

MEANING: dict[str, str] = {
    "none": "Books, GSTR-2B and the invoice agree.",
    "rounding_within_tolerance": "The records agree within Rs 1 of rounding.",
    "invoice_no_format_difference": "The invoice numbers differ only in format.",
    "missing_in_2b": "The purchase is in the books but the supplier's invoice is not in GSTR-2B.",
    "in_2b_not_in_books": "The supplier reported the invoice in GSTR-2B but the books have no "
    "matching purchase.",
    "taxable_value_mismatch": "Books and GSTR-2B show different taxable values for the invoice.",
    "tax_amount_mismatch": "Books and GSTR-2B show different tax amounts for the invoice.",
    "gstin_typo_in_books": "The supplier GSTIN in the books looks like a typing error of the "
    "GSTIN in GSTR-2B.",
    "tax_head_mismatch": "Books and GSTR-2B put the tax under different heads (IGST versus "
    "CGST and SGST).",
    "duplicate_in_books": "The same invoice is recorded more than once in the books.",
    "date_mismatch_other_period": "The invoice appears in GSTR-2B for a different tax period.",
}

# One fixed query per mismatch type. Naming the provisions makes both retrievers pin them.
QUERIES: dict[str, str] = {
    "missing_in_2b": (
        "section 16(2)(aa) input tax credit only when the details of the invoice are furnished "
        "by the supplier and communicated in GSTR-2B; rule 36(4) credit restricted to invoices "
        "in GSTR-2B"
    ),
    "in_2b_not_in_books": (
        "section 16(2)(b) input tax credit requires actual receipt of the goods or services; "
        "invoice reported by the supplier but not recorded in the recipient's books"
    ),
    "taxable_value_mismatch": (
        "section 34 credit note or debit note where the taxable value or tax charged in the "
        "tax invoice is found to exceed or be less than the correct value; section 16(2)(aa)"
    ),
    "tax_amount_mismatch": (
        "section 34 credit note or debit note where the tax charged in the tax invoice exceeds "
        "or is less than the tax payable; rule 36(4) input tax credit limited to GSTR-2B"
    ),
    "gstin_typo_in_books": (
        "rule 46 tax invoice particulars name address and GSTIN of the recipient; section "
        "16(2)(a) possession of a tax invoice; rule 36(4) details in GSTR-2B"
    ),
    "tax_head_mismatch": (
        "section 77 of the CGST Act tax paid as central and state tax on a transaction "
        "considered inter-State, refund of tax wrongfully collected; IGST section 19(1)"
    ),
    "duplicate_in_books": (
        "section 16(2) conditions for input tax credit on a tax invoice; credit taken more "
        "than once on the same invoice; rule 36(4) details in GSTR-2B"
    ),
    "date_mismatch_other_period": (
        "section 16(4) time limit for taking input tax credit on an invoice; rule 60(7) "
        "GSTR-2B made available for each tax period"
    ),
}


def _prompt(name: str) -> str:
    return (
        resources.files("agent_orchestrator.llm.prompts").joinpath(name).read_text("utf-8").strip()
    )


SYSTEM_PROMPT = _prompt("research_system.txt")
USER_PROMPT = Template(_prompt("research_user.txt"))


class ResearchAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1, max_length=1500)
    snippet_ids: list[str] = Field(min_length=1)


@dataclass(frozen=True)
class ResearchResult:
    """What the planner needs: the summary and the snippets themselves (their text)."""

    research: RuleResearch
    snippets: tuple[RuleSnippet, ...]
    retriever: str
    corpus_version: str
    # "" when the summary is good; else why there is none: "no_snippets", "retrieval_failed",
    # "summary_failed", "budget_stop" or "llm_error"
    note: str = ""


def cache_key(mismatch_type: str, corpus_version: str) -> str:
    return f"research:v{RESEARCH_VERSION}:{mismatch_type}:{corpus_version}"


def hit_rate(cache: Cache) -> float | None:
    """Share of cache lookups that hit, or None before the first lookup."""
    s = cache.stats()
    total = s.get("hits", 0) + s.get("misses", 0)
    return s.get("hits", 0) / total if total else None


def format_passages(snippets: Sequence[RuleSnippet], max_chars: int = PASSAGE_CHARS) -> str:
    """The passages as the models see them: id, title, date, then the (possibly cut) text."""
    blocks = []
    for s in snippets:
        text = s.text if len(s.text) <= max_chars else s.text[:max_chars].rstrip() + " [...]"
        blocks.append(f"[{s.snippet_id}] {s.title} (checked {s.checked_on.isoformat()})\n{text}")
    return "\n\n".join(blocks)


_CITED = re.compile(r"\[([^\[\]\s]+)\]")


def check_answer(answer: ResearchAnswer, given: Sequence[str]) -> list[str]:
    """Every id the answer uses, in the list or in the text, must be one it was shown."""
    known = set(given)
    errors = [
        f"snippet_ids[{i}]: not one of the passage ids given"
        for i, sid in enumerate(answer.snippet_ids)
        if sid not in known
    ]
    cited = _CITED.findall(answer.summary)
    if not cited:
        errors.append("summary: cites no passage id in square brackets")
    errors += [f"summary: cites an id that was not given ({n})" for n, c in
               enumerate(cited, start=1) if c not in known]  # fmt: skip
    return errors


def _summarise(
    client: LLMClient, guard: Guard, mismatch_type: str, snippets: Sequence[RuleSnippet]
) -> tuple[ResearchAnswer | None, list[str]]:
    given = [s.snippet_id for s in snippets]
    user = USER_PROMPT.substitute(
        mismatch_type=mismatch_type,
        meaning=MEANING[mismatch_type],
        ids=", ".join(given),
        passages=format_passages(snippets),
    )
    messages = [Message.user(user)]
    errors: list[str] = []
    for attempt in range(1 + MAX_SCHEMA_RETRIES):
        result = client.complete(
            "research" if attempt == 0 else "research-fix",
            messages,
            system=SYSTEM_PROMPT,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.0,
        )
        checked = guard.validate(result.text, ResearchAnswer)
        if checked.ok:
            errors = check_answer(checked.value, given)
            if not errors:
                return checked.value, []
        else:
            errors = checked.errors
        feedback = "; ".join(errors[:8])[:600]
        messages = [
            *messages,
            Message.assistant(result.text[:4000]),
            Message.user(
                f"Your reply was not accepted. Problems: {feedback}. The only passage ids you "
                f"may use are: {', '.join(given)}. Reply again with the corrected JSON object only."
            ),
        ]
    return None, errors


def _ordered_ids(answer: ResearchAnswer) -> list[str]:
    return list(dict.fromkeys([*answer.snippet_ids, *_CITED.findall(answer.summary)]))


def research_rules(
    mismatch_type: MismatchType,
    *,
    retriever: RuleRetriever,
    cache: Cache,
    client: LLMClient,
    guard: Guard,
    k: int = K,
) -> ResearchResult:
    """Research one mismatch type. Never raises for a bad answer, a budget stop, a provider
    failure or a retriever failure: those give a result with no summary and a ``note``.
    Configuration errors (missing recording, unknown model) still stop the run."""
    if mismatch_type in CLEAN_TYPES:
        raise ValueError(f"{mismatch_type} is a clean match: it needs no rule research")
    if mismatch_type not in QUERIES:
        raise ValueError(f"no research query for mismatch type {mismatch_type!r}")
    name = retriever.name
    try:
        version = retriever.corpus_version
        key = cache_key(mismatch_type, version)
        cached = cache.get(key)
        if cached is not None:
            snippets = tuple(RuleSnippet.model_validate(s) for s in cached["snippets"])
            research = RuleResearch.model_validate({**cached["research"], "cached": True})
            return ResearchResult(research, snippets, name, version)
        snippets = tuple(retriever.search(QUERIES[mismatch_type], k=k))
    except (RecordingMissing, UnknownModel, ConfigError):
        raise
    except Exception:  # database down, AWS error: fail closed, the planner escalates
        empty = RuleResearch(mismatch_type=mismatch_type, summary="")
        return ResearchResult(empty, (), name, "unavailable", note="retrieval_failed")

    all_ids = [s.snippet_id for s in snippets]
    if not snippets:
        empty = RuleResearch(mismatch_type=mismatch_type, summary="")
        return ResearchResult(empty, (), name, version, note="no_snippets")

    def without_summary(note: str) -> ResearchResult:
        research = RuleResearch(mismatch_type=mismatch_type, summary="", snippet_ids=all_ids)
        return ResearchResult(research, snippets, name, version, note=note)

    try:
        answer, _ = _summarise(client, guard, mismatch_type, snippets)
    except BudgetExceeded:
        return without_summary("budget_stop")
    except (RecordingMissing, UnknownModel, ConfigError):
        raise
    except Exception:  # provider failure after retries
        return without_summary("llm_error")
    if answer is None:
        return without_summary("summary_failed")  # not cached: the next run tries again

    research = RuleResearch(
        mismatch_type=mismatch_type, summary=answer.summary, snippet_ids=_ordered_ids(answer)
    )
    cache.set(
        key,
        {
            "research": research.model_dump(mode="json"),
            "snippets": [s.model_dump(mode="json") for s in snippets],
        },
    )
    return ResearchResult(research, snippets, name, version)
