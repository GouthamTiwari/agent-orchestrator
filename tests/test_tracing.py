import json
import threading
import time

import pytest

from agent_orchestrator.store.repo import Repo
from agent_orchestrator.tracing import Tracer


@pytest.fixture
def repo(tmp_path):
    r = Repo.open(tmp_path / "t.sqlite")
    r.create_run("r1", "B01", "2026-10", "h" * 64, "policy_v1")
    yield r
    r.close()


def test_span_records_usage_latency_and_output(repo):
    tracer = Tracer("r1", repo)
    with tracer.span("extraction", case_key="ck", input_hash="ih") as s:
        time.sleep(0.02)
        s.add_usage(input_tokens=1200, output_tokens=300, usd=0.0013)
        s.add_usage(input_tokens=10)
        s.set_output("11 fields")
        s.set_decision("ok")
    stored = repo.list_spans("r1")
    assert len(stored) == 1
    row = stored[0]
    assert (row["agent"], row["case_key"], row["input_hash"]) == ("extraction", "ck", "ih")
    assert (row["input_tokens"], row["output_tokens"], row["usd"]) == (1210, 300, 0.0013)
    assert row["latency_ms"] >= 15 and row["error"] is None
    assert row["output_summary"] == "11 fields" and row["decision"] == "ok"


def test_span_records_errors_and_reraises(repo):
    tracer = Tracer("r1", repo)
    with pytest.raises(RuntimeError), tracer.span("planner", case_key="ck"):
        raise RuntimeError("model said no")
    row = repo.list_spans("r1")[0]
    assert row["error"] == "RuntimeError: model said no"


def test_nested_spans_have_parents():
    tracer = Tracer("r1")
    with tracer.span("graph", "run") as outer:
        with tracer.span("matcher") as inner:
            pass
        with tracer.span("reporter") as second:
            pass
    assert inner.parent_id == outer.span_id == second.parent_id
    assert outer.parent_id is None


def test_threads_keep_their_own_parent():
    tracer = Tracer("r1")
    seen = {}

    def branch(i):
        with tracer.span("extraction", case_key=f"c{i}") as s, tracer.span("guard") as g:
            seen[i] = (s.span_id, g.parent_id)

    threads = [threading.Thread(target=branch, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(span_id == parent for span_id, parent in seen.values())
    assert len({s.span_id for s in tracer.spans}) == 12


def test_negative_usage_is_refused():
    tracer = Tracer("r1")
    with pytest.raises(ValueError), tracer.span("x") as s:
        s.add_usage(usd=-1)


def test_totals_and_json_export(tmp_path):
    tracer = Tracer("r1")
    for usd in (0.001, 0.002):
        with tracer.span("extraction") as s:
            s.add_usage(input_tokens=100, output_tokens=10, usd=usd)
    with pytest.raises(KeyError), tracer.span("planner"):
        raise KeyError("x")
    by_agent = tracer.totals_by_agent()
    assert by_agent["extraction"]["spans"] == 2
    assert by_agent["extraction"]["usd"] == pytest.approx(0.003)
    assert by_agent["planner"]["errors"] == 1
    path = tracer.export_json(tmp_path / "runs" / "r1" / "trace.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == "r1"
    assert data["total"] == {
        "spans": 3,
        "errors": 1,
        "input_tokens": 200,
        "output_tokens": 20,
        "usd": 0.003,
    }
    assert [s["agent"] for s in data["spans"]] == ["extraction", "extraction", "planner"]


def test_long_text_is_truncated():
    tracer = Tracer("r1")
    with tracer.span("x") as s:
        s.set_output("y" * 2000)
    assert len(s.output_summary) == 500
