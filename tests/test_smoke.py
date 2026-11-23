import json
from pathlib import Path

from typer.testing import CliRunner

import agent_orchestrator
from agent_orchestrator.cli import app

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed" / "seed_data"


def test_version_command_prints_package_version():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == agent_orchestrator.__version__


def test_seed_dataset_is_complete():
    labels = [json.loads(line) for line in (SEED / "labels.jsonl").read_text().splitlines()]
    assert len(labels) == 18
    assert len(list((SEED / "invoices").glob("*.pdf"))) == 17
    assert (SEED / "purchase_register.csv").read_text().count("\n") == 19  # header + 18 rows


def test_seed_injection_invoice_is_present():
    lines = (SEED / "labels.jsonl").read_text().splitlines()
    labels = {row["invoice_no"]: row for row in map(json.loads, lines)}
    assert "IMA/2026/0117" in labels
    assert (SEED / "invoices" / "IMA_2026_0117.pdf").exists()


def test_rule_sources_have_source_and_date_header():
    files = sorted((ROOT / "data" / "rules" / "sources").glob("*.txt"))
    assert len(files) >= 9
    for path in files:
        head = path.read_text(encoding="utf-8").splitlines()[:3]
        assert any(line.startswith("Source: https://") for line in head), path.name
        assert any(line.startswith("Retrieved: ") for line in head), path.name


def test_rule_sources_are_lf_only():
    # sha256 of these files is the corpus version, so Windows CRLF must never sneak in.
    for path in (ROOT / "data" / "rules" / "sources").glob("*.txt"):
        assert b"\r\n" not in path.read_bytes(), path.name
