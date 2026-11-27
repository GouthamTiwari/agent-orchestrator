import pytest


@pytest.fixture(autouse=True)
def _feedback_in_tmp(tmp_path, monkeypatch):
    """No test ever appends to the real eval/human_feedback.jsonl."""
    monkeypatch.setenv("ORCH_FEEDBACK_FILE", str(tmp_path / "human_feedback.jsonl"))
