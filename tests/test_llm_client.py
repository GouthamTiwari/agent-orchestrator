"""The LLM client with fake and stubbed backends (no network)."""

import base64
import json
from types import SimpleNamespace

import pytest

from agent_orchestrator.llm.budget import BudgetExceeded, BudgetStop, RunBudget, SpendLedger
from agent_orchestrator.llm.client import (
    AnthropicBackend,
    BedrockBackend,
    ConfigError,
    LLMClient,
)
from agent_orchestrator.llm.fake import FakeBackend
from agent_orchestrator.llm.ledger import PaidCallCache, request_hash
from agent_orchestrator.llm.recording import Recorder, RecordingMissing
from agent_orchestrator.llm.types import ImagePart, Message
from agent_orchestrator.store.repo import Repo
from agent_orchestrator.tracing import Tracer

NOVA = "global.amazon.nova-2-lite-v1:0"
PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"


@pytest.fixture
def repo(tmp_path):
    r = Repo.open(tmp_path / "db.sqlite")
    r.create_run("r1", "B01", "2026-10", "h" * 64, "v1")
    yield r
    r.close()


# --- request hash -----------------------------------------------------------------------------


def test_request_hash_covers_what_changes_the_answer():
    base = dict(
        provider="bedrock",
        model_id=NOVA,
        system="s",
        messages=[Message.user("hi", ImagePart(PNG))],
        max_tokens=10,
        temperature=None,
    )
    h = request_hash(**base)
    assert h == request_hash(**base)
    assert h != request_hash(**{**base, "system": "t"})
    assert h != request_hash(**{**base, "max_tokens": 11})
    assert h != request_hash(**{**base, "model_id": "apac.amazon.nova-pro-v1:0"})
    assert h != request_hash(**{**base, "messages": [Message.user("hi", ImagePart(PNG + b"x"))]})
    assert "fake-image" not in repr(ImagePart(PNG))  # bytes never printed


# --- modes ------------------------------------------------------------------------------------


def test_fake_mode_returns_scripted_answers():
    fake = FakeBackend(['{"a": 1}', "second"])
    client = LLMClient(fake, mode="fake")
    r1 = client.complete("extract", [Message.user("q1")])
    r2 = client.complete("extract", [Message.user("q2")])
    r3 = client.complete("extract", [Message.user("q3")])
    assert (r1.text, r2.text, r3.text) == ('{"a": 1}', "second", "second")
    assert r1.usd == 0 and r1.source == "fake" and fake.call_count == 3


def test_mode_and_backend_must_agree():
    with pytest.raises(ConfigError):
        LLMClient(FakeBackend(), mode="live")
    with pytest.raises(ConfigError):
        LLMClient(BedrockBackend(NOVA, client=object()), mode="fake")
    with pytest.raises(ConfigError):
        LLMClient(FakeBackend(), mode="replay")  # no recorder


def test_ledger_reuses_a_paid_answer_without_calling(repo):
    fake = FakeBackend(["answer"])
    client = LLMClient(fake, mode="fake", cache=PaidCallCache(repo))
    first = client.complete("plan", [Message.user("same question")], system="sys")
    again = client.complete("plan", [Message.user("same question")], system="sys")
    assert fake.call_count == 1
    assert again.text == "answer" and again.source == "ledger" and again.usd == 0
    assert again.request_hash == first.request_hash
    client.complete("plan", [Message.user("other question")], system="sys")
    assert fake.call_count == 2


def test_replay_returns_recordings_and_never_calls(tmp_path):
    rec = Recorder(tmp_path / "rec.jsonl")
    req = request_hash("fake", "fake", "", [Message.user("q")], 1024, None)
    rec._load()[req] = {
        "request_hash": req,
        "model_id": "fake",
        "text": "recorded",
        "tokens_in": 3,
        "tokens_out": 2,
    }
    fake = FakeBackend(["live answer"])
    client = LLMClient(fake, mode="replay", recorder=rec)
    result = client.complete("extract", [Message.user("q")])
    assert result.text == "recorded" and result.source == "replay" and fake.call_count == 0
    with pytest.raises(RecordingMissing, match="no recording"):
        client.complete("extract", [Message.user("never recorded")])
    assert fake.call_count == 0


def test_recorder_appends_once_per_request(tmp_path):
    path = tmp_path / "rec.jsonl"
    stub = StubBedrock()
    client = LLMClient(
        BedrockBackend(NOVA, client=stub),
        mode="live",
        spend=SpendLedger(tmp_path / "s.jsonl"),
        recorder=Recorder(path),
    )
    client.complete("probe", [Message.user("q")])
    client.recorder.record(client.complete("probe", [Message.user("q")]), "probe")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["text"] == "OK from stub"
    replay = LLMClient(FakeBackend(), mode="replay", recorder=Recorder(path))
    replay.backend = BedrockBackend(NOVA, client=None)  # same provider/model as recorded
    assert replay.complete("probe", [Message.user("q")]).text == "OK from stub"


# --- live mode with a stubbed Bedrock client ---------------------------------------------------


class StubBedrock:
    def __init__(self, text="OK from stub", fail=()):
        self.requests = []
        self.text = text
        self.fail = list(fail)

    def converse(self, **request):
        self.requests.append(request)
        if self.fail:
            raise self.fail.pop(0)
        return {
            "output": {"message": {"content": [{"text": self.text}]}},
            "usage": {"inputTokens": 1000, "outputTokens": 100},
            "stopReason": "end_turn",
        }


def test_live_bedrock_request_shape_and_cost(tmp_path, repo):
    stub = StubBedrock()
    spend = SpendLedger(tmp_path / "s.jsonl")
    budget = RunBudget(usd=1.0)
    tracer = Tracer("r1", repo)
    client = LLMClient(
        BedrockBackend(NOVA, client=stub),
        mode="live",
        spend=spend,
        run_budget=budget,
        cache=PaidCallCache(repo),
        tracer=tracer,
        run_id="r1",
    )
    result = client.complete(
        "extract",
        [Message.user(ImagePart(PNG), "read it")],
        system="untrusted",
        max_tokens=500,
        temperature=0.0,
        case_key="ck",
    )
    req = stub.requests[0]
    assert req["modelId"] == NOVA and req["system"] == [{"text": "untrusted"}]
    assert req["messages"][0]["content"][0] == {
        "image": {"format": "png", "source": {"bytes": PNG}}
    }
    assert req["inferenceConfig"] == {"maxTokens": 500, "temperature": 0.0}
    expected = (1000 * 0.35 + 100 * 2.95) / 1e6
    assert result.usd == pytest.approx(expected) and result.source == "live"
    assert spend.total("bedrock") == pytest.approx(expected)
    assert budget.usd == pytest.approx(expected) and budget.tokens == 1100
    assert repo.ledger_get(result.request_hash)["usd"] == pytest.approx(expected)
    span = repo.list_spans("r1")[0]
    assert (span["agent"], span["name"], span["case_key"]) == ("llm", "extract", "ck")
    assert span["usd"] == pytest.approx(expected)


def test_live_refuses_before_calling_when_the_cap_would_be_exceeded(tmp_path):
    stub = StubBedrock()
    spend = SpendLedger(tmp_path / "s.jsonl", caps={"bedrock": 0.0001})
    client = LLMClient(BedrockBackend(NOVA, client=stub), mode="live", spend=spend)
    with pytest.raises(BudgetExceeded):
        client.complete("extract", [Message.user("x" * 3000)], max_tokens=1000)
    assert stub.requests == []


def test_run_budget_stop_blocks_new_calls(tmp_path):
    stub = StubBedrock()
    budget = RunBudget(usd=1.0)
    budget.record(0.95, 0)
    client = LLMClient(
        BedrockBackend(NOVA, client=stub),
        mode="live",
        spend=SpendLedger(tmp_path / "s.jsonl"),
        run_budget=budget,
    )
    with pytest.raises(BudgetStop):
        client.complete("plan", [Message.user("q")])
    assert stub.requests == []


def test_throttling_is_retried_and_counted(tmp_path):
    class Throttle(Exception):
        response = {
            "Error": {"Code": "ThrottlingException"},
            "ResponseMetadata": {"HTTPStatusCode": 400},
        }

    stub = StubBedrock(fail=[Throttle(), Throttle()])
    client = LLMClient(
        BedrockBackend(NOVA, client=stub),
        mode="live",
        spend=SpendLedger(tmp_path / "s.jsonl"),
        retry_sleep=lambda s: None,
    )
    result = client.complete("extract", [Message.user("q")])
    assert result.retries == 2 and len(stub.requests) == 3
    assert len(SpendLedger(tmp_path / "s.jsonl").entries()) == 1  # paid once


def test_temperature_is_left_out_when_not_given():
    stub = StubBedrock()
    BedrockBackend(NOVA, client=stub).invoke("", [Message.user("q")], 10, None)
    assert stub.requests[0]["inferenceConfig"] == {"maxTokens": 10}
    assert "system" not in stub.requests[0]


# --- Anthropic backend with a stubbed SDK client -----------------------------------------------


class StubAnthropic:
    def __init__(self):
        self.kwargs = None
        self.messages = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="{}")],
            usage=SimpleNamespace(input_tokens=2000, output_tokens=50),
            stop_reason="end_turn",
        )


def test_anthropic_request_has_image_blocks_and_no_temperature(tmp_path):
    stub = StubAnthropic()
    client = LLMClient(
        AnthropicBackend(client=stub), mode="live", spend=SpendLedger(tmp_path / "s.jsonl")
    )
    result = client.complete(
        "extract", [Message.user(ImagePart(PNG), "read")], system="sys", temperature=0.0
    )
    kw = stub.kwargs
    assert kw["model"] == "claude-haiku-5-5" and kw["system"] == "sys"
    assert "temperature" not in kw
    image = kw["messages"][0]["content"][0]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/png"
    assert base64.standard_b64decode(image["source"]["data"]) == PNG
    assert result.usd == pytest.approx((2000 * 0.10 + 50 * 0.50) / 1e6)
    assert SpendLedger(tmp_path / "s.jsonl").total("anthropic") == pytest.approx(result.usd)


def test_anthropic_without_a_key_is_a_config_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        AnthropicBackend().client  # noqa: B018
