import datetime as dt
import json
import os
from pathlib import Path

import pytest

from agent_orchestrator.agents import research
from agent_orchestrator.agents.research import (
    CLEAN_TYPES,
    MEANING,
    QUERIES,
    SYSTEM_PROMPT,
    cache_key,
    format_passages,
    hit_rate,
    research_rules,
)
from agent_orchestrator.interfaces.stubs import RegexGuard, SnapshotRetriever, SqliteCache
from agent_orchestrator.llm.budget import RunBudget
from agent_orchestrator.llm.client import LLMClient
from agent_orchestrator.llm.fake import FakeBackend
from agent_orchestrator.llm.recording import Recorder, RecordingMissing
from agent_orchestrator.models import MISMATCH_TYPES, RuleSnippet

ROOT = Path(__file__).resolve().parents[1]
RESEARCHED = sorted(set(MISMATCH_TYPES) - CLEAN_TYPES)


@pytest.fixture(scope="module")
def snapshot():
    return SnapshotRetriever()


def client(*script):
    backend = FakeBackend(list(script))
    return LLMClient(backend, mode="fake"), backend


def good_answer(ids):
    """A fake model that cites the first two ids it was given."""

    def answer(system, messages):
        return json.dumps(
            {
                "summary": f"Credit needs the invoice in GSTR-2B [{ids[0]}]; see also [{ids[1]}].",
                "snippet_ids": ids[:2],
            }
        )

    return answer


def run(retriever, script, cache=None, mt="missing_in_2b"):
    c, backend = client(*script)
    cache = cache if cache is not None else SqliteCache()
    out = research_rules(mt, retriever=retriever, cache=cache, client=c, guard=RegexGuard())
    return out, backend, cache


# --- queries ------------------------------------------------------------------------------------


def test_every_mismatch_type_has_a_meaning_and_every_non_clean_one_a_query():
    assert set(MEANING) == set(MISMATCH_TYPES)
    assert set(QUERIES) == set(RESEARCHED)


@pytest.mark.parametrize("mt", sorted(CLEAN_TYPES))
def test_clean_matches_need_no_research(snapshot, mt):
    with pytest.raises(ValueError, match="clean match"):
        run(snapshot, ["{}"], mt=mt)


@pytest.mark.parametrize(
    ("mt", "expected"),
    [
        ("missing_in_2b", {"cgst_act_s16#2aa", "cgst_rules_r36#4"}),
        ("in_2b_not_in_books", {"cgst_act_s16#2b"}),
        ("date_mismatch_other_period", {"cgst_act_s16#4", "cgst_rules_r60#7"}),
        ("gstin_typo_in_books", {"cgst_act_s16#2a"}),
        ("tax_amount_mismatch", {"cgst_rules_r36#4"}),
    ],
)
def test_snapshot_queries_find_the_provisions_they_name(snapshot, mt, expected):
    ids = {s.snippet_id for s in snapshot.search(QUERIES[mt], k=4)}
    assert expected <= ids


@pytest.mark.parametrize("mt", RESEARCHED)
def test_every_query_finds_rule_text_in_the_snapshot(snapshot, mt):
    assert len(snapshot.search(QUERIES[mt], k=4)) == 4


def test_tax_head_query_finds_section_77_and_igst_19(snapshot):
    sources = {s.source_id for s in snapshot.search(QUERIES["tax_head_mismatch"], k=4)}
    assert {"cgst_act_s77", "igst_act_s19"} <= sources


# --- summary, citations and cache -----------------------------------------------------------------


def ids_for(retriever, mt="missing_in_2b"):
    return [s.snippet_id for s in retriever.search(QUERIES[mt], k=4)]


def test_a_cited_summary_is_returned_and_cached(snapshot):
    ids = ids_for(snapshot)
    out, backend, cache = run(snapshot, [good_answer(ids)])
    assert out.note == "" and out.research.summary.startswith("Credit needs")
    assert out.research.snippet_ids == ids[:2] and not out.research.cached
    assert [s.snippet_id for s in out.snippets] == ids
    assert out.retriever == "cbic-snapshot" and out.corpus_version == snapshot.corpus_version
    assert backend.call_count == 1

    again, backend2, _ = run(snapshot, ["never called"], cache=cache)
    assert again.research.cached and backend2.call_count == 0
    assert again.research.summary == out.research.summary and again.snippets == out.snippets
    assert hit_rate(cache) == 0.5  # one miss, then one hit


def test_hit_rate_is_none_before_any_lookup():
    assert hit_rate(SqliteCache()) is None


def test_the_cache_key_is_the_mismatch_type_and_the_corpus_version():
    assert cache_key("missing_in_2b", "cbic-abc") == "research:v1:missing_in_2b:cbic-abc"
    assert cache_key("missing_in_2b", "regrag-abc") != cache_key("missing_in_2b", "cbic-abc")


class Renamed:
    """The same snippets under another corpus version (as if the other retriever answered)."""

    name = "other"

    def __init__(self, inner, version):
        self.inner, self._version = inner, version

    @property
    def corpus_version(self):
        return self._version

    def search(self, query, k=4):
        return self.inner.search(query, k)

    def get(self, snippet_id):
        return self.inner.get(snippet_id)


def test_a_different_corpus_version_never_reuses_the_cache(snapshot):
    ids = ids_for(snapshot)
    _, _, cache = run(snapshot, [good_answer(ids)])
    out, backend, _ = run(Renamed(snapshot, "regrag-x"), [good_answer(ids)], cache=cache)
    assert not out.research.cached and backend.call_count == 1


def test_the_prompt_carries_the_passages_and_their_ids(snapshot):
    ids = ids_for(snapshot)
    _, backend, _ = run(snapshot, [good_answer(ids)])
    system, messages = backend.calls[0]
    assert system == SYSTEM_PROMPT
    user = messages[0].parts[0]
    assert "missing_in_2b" in user and MEANING["missing_in_2b"] in user
    for s in snapshot.search(QUERIES["missing_in_2b"], k=4):
        assert f"[{s.snippet_id}]" in user


def test_an_unknown_id_is_retried_with_the_problem(snapshot):
    ids = ids_for(snapshot)
    bad = json.dumps({"summary": "See [cgst_act_s99#1].", "snippet_ids": ["cgst_act_s99#1"]})
    out, backend, _ = run(snapshot, [bad, good_answer(ids)])
    assert out.note == "" and backend.call_count == 2
    feedback = backend.calls[1][1][-1].parts[0]
    assert "not one of the passage ids given" in feedback and ids[0] in feedback


def test_a_summary_without_citations_is_rejected(snapshot):
    ids = ids_for(snapshot)
    bare = json.dumps({"summary": "Credit needs GSTR-2B.", "snippet_ids": ids[:1]})
    out, backend, _ = run(snapshot, [bare, good_answer(ids)])
    assert out.note == "" and backend.call_count == 2
    assert "cites no passage id" in backend.calls[1][1][-1].parts[0]


def test_after_two_retries_there_is_no_summary_but_the_snippets_stay(snapshot):
    ids = ids_for(snapshot)
    bad = json.dumps({"summary": "See [made#up].", "snippet_ids": ["made#up"]})
    out, backend, cache = run(snapshot, [bad])
    assert backend.call_count == 3
    assert out.note == "summary_failed" and out.research.summary == ""
    assert out.research.snippet_ids == ids and len(out.snippets) == 4
    # a failure is not cached: the next run asks again
    again, backend2, _ = run(snapshot, [good_answer(ids)], cache=cache)
    assert not again.research.cached and backend2.call_count == 1


def test_a_model_answer_that_is_not_json_is_retried(snapshot):
    ids = ids_for(snapshot)
    out, backend, _ = run(snapshot, ["I think credit is allowed.", good_answer(ids)])
    assert out.note == "" and backend.call_count == 2


def test_a_budget_stop_gives_no_summary_and_is_not_cached(snapshot):
    stopped = RunBudget(usd=1.0)
    stopped.record(0.95, 0)  # past 90%: no new LLM work
    c = LLMClient(FakeBackend(["{}"]), mode="fake", run_budget=stopped)
    cache = SqliteCache()
    out = research_rules(
        "missing_in_2b", retriever=snapshot, cache=cache, client=c, guard=RegexGuard()
    )
    assert out.note == "budget_stop" and out.research.summary == "" and len(out.snippets) == 4
    assert cache.get(cache_key("missing_in_2b", snapshot.corpus_version)) is None


def test_a_provider_failure_gives_no_summary(snapshot):
    out, _, _ = run(snapshot, [RuntimeError("bedrock down")])
    assert out.note == "llm_error" and out.snippets


def test_configuration_errors_stop_the_run(snapshot, tmp_path):
    c = LLMClient(FakeBackend(["{}"]), mode="replay", recorder=Recorder(tmp_path / "r.jsonl"))
    with pytest.raises(RecordingMissing):
        research_rules(
            "missing_in_2b", retriever=snapshot, cache=SqliteCache(), client=c, guard=RegexGuard()
        )


class Broken:
    name = "broken"

    @property
    def corpus_version(self):
        raise ConnectionError("database down")

    def search(self, query, k=4):
        raise ConnectionError("database down")

    def get(self, snippet_id):
        return None


class Empty(Broken):
    name = "empty"

    @property
    def corpus_version(self):
        return "empty-1"

    def search(self, query, k=4):
        return []


def test_a_retriever_failure_fails_closed_without_a_model_call():
    out, backend, _ = run(Broken(), ["never"])
    assert out.note == "retrieval_failed" and out.snippets == () and backend.call_count == 0


def test_no_snippets_means_no_model_call():
    out, backend, _ = run(Empty(), ["never"])
    assert out.note == "no_snippets" and backend.call_count == 0


def test_long_passages_are_cut_in_the_prompt():
    s = RuleSnippet(
        snippet_id="x#1", source_id="x", title="X", url="https://example.invalid/x",
        section="1", text="word " * 1000, checked_on=dt.date(2026, 11, 23),
    )  # fmt: skip
    text = format_passages([s])
    assert text.startswith("[x#1] X (checked 2026-11-23)") and text.endswith("[...]")
    assert len(text) < research.PASSAGE_CHARS + 100


# --- the Regulatory RAG path (LIVE: one embedding and one rerank per query, < USD 0.01) -----------


def regrag_retriever():
    """The real adapter, or a skip when the package, its settings or its database is missing."""
    pytest.importorskip("regrag.api")
    from dotenv import load_dotenv

    from agent_orchestrator.interfaces import adapters

    load_dotenv(ROOT / ".env")  # AWS credentials and REGULATORY_RAG_ENV_FILE; never printed
    if not os.environ.get(adapters.REGRAG_ENV_VAR):
        pytest.skip(f"{adapters.REGRAG_ENV_VAR} is not set")
    try:
        return adapters.RegragRetriever()
    except adapters.AdapterUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.live
def test_regrag_path_finds_the_named_provisions_for_every_mismatch_type():
    r = regrag_retriever()
    assert r.corpus_version.startswith("regrag-")
    found = {mt: r.search(QUERIES[mt], k=4) for mt in RESEARCHED}
    for mt, hits in found.items():
        assert len(hits) == 4, mt
        assert all(h.source_id.startswith("gst-") for h in hits), mt
    sections = {mt: {(h.source_id, h.section) for h in hits} for mt, hits in found.items()}
    assert ("gst-cgst-act-2017", "16(2)(aa)") in sections["missing_in_2b"]
    assert ("gst-cgst-rules-2017", "36(4)") in sections["missing_in_2b"]
    assert ("gst-cgst-act-2017", "16(4)") in sections["date_mismatch_other_period"]
    assert any(s == "gst-igst-act-2017" for s, _ in sections["tax_head_mismatch"])


@pytest.mark.live
def test_regrag_path_research_with_a_fake_summary_model():
    r = regrag_retriever()
    ids = [h.snippet_id for h in r.search(QUERIES["missing_in_2b"], k=4)]
    out, backend, cache = run(r, [good_answer(ids)])
    assert out.retriever == "regulatory-rag" and out.corpus_version == r.corpus_version
    assert out.note == "" and out.research.snippet_ids == ids[:2]
    assert all(r.get(i) is not None for i in ids)
    again, _, _ = run(r, ["never"], cache=cache)
    assert again.research.cached
