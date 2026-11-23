import csv
import datetime as dt
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest
import typer

from agent_orchestrator import gstin, money
from agent_orchestrator.cli import check_splits
from agent_orchestrator.eval import generate as gen
from agent_orchestrator.eval import review
from agent_orchestrator.eval.scan_noise import add_noise, noise_params, rasterise

CFG = gen.load_config()
MONTHS = gen.build_all(CFG)
LABELS = [lab for m in MONTHS for lab in m.labels]
SPLITS = json.loads((gen.ROOT / "data" / "splits.json").read_text(encoding="utf-8"))


def _inv_key(s: str) -> str:
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", s.upper()) if t]
    return "|".join(t.lstrip("0") or "0" if t.isdigit() else t for t in tokens)


def _no_year(key: str) -> str:
    return "|".join(t for t in key.split("|") if not re.fullmatch(r"20\d\d", t))


# --- determinism ------------------------------------------------------------------------------


def test_two_builds_give_identical_labels():
    again = gen.build_all(CFG)
    assert [gen.labels_text(m.labels) for m in again] == [gen.labels_text(m.labels) for m in MONTHS]


def test_committed_labels_match_the_generator(tmp_path):
    gen.generate(CFG, out_dir=tmp_path / "out", labels_dir=tmp_path / "labels", render=False)
    committed = sorted((gen.ROOT / "data" / "labels").glob("*.jsonl"))
    fresh = sorted((tmp_path / "labels").glob("*.jsonl"))
    assert [p.name for p in committed] == [p.name for p in fresh]
    for a, b in zip(committed, fresh, strict=True):
        assert a.read_bytes() == b.read_bytes(), a.name


def test_text_outputs_are_lf_and_identical_across_runs(tmp_path):
    s1 = gen.generate(CFG, out_dir=tmp_path / "a", labels_dir=None, render=False)
    s2 = gen.generate(CFG, out_dir=tmp_path / "b", labels_dir=None, render=False)
    assert s1["files"] == s2["files"]
    sample = tmp_path / "a" / "B01" / "2026-08" / "purchase_register.csv"
    assert b"\r\n" not in sample.read_bytes()


# --- size and mix -----------------------------------------------------------------------------


def test_size_is_about_600_cases_over_30_business_months():
    assert len(MONTHS) == 30
    assert 570 <= len(LABELS) <= 630
    for m in MONTHS:
        assert 18 <= len(m.labels) <= 22, (m.business.business_id, m.period)


def test_every_month_has_all_eleven_case_types_rates_and_both_kinds_of_supplier():
    for m in MONTHS:
        types = {lab["case_type"] for lab in m.labels}
        assert set(gen.CASE_TABLE) <= types, (m.business.business_id, m.period)
        assert {r["rate"] for r in m.books} >= {"5", "18"}
        states = {r["supplier_gstin"][:2] for r in m.books}
        assert m.business.state in states and states - {m.business.state}


def test_injections_every_other_month_and_all_variants_in_dev():
    per_business = Counter(lab["business_id"] for lab in LABELS if lab["injection"])
    assert sum(per_business.values()) == 15
    dev_variants = {
        lab["injection_variant"]
        for lab in LABELS
        if lab["injection"] and lab["business_id"] in SPLITS["dev"]
    }
    assert dev_variants == set(gen.INJECTION_VARIANTS)
    for lab in LABELS:
        if lab["injection"]:
            assert lab["expected_action"] == "escalate_human"
            assert lab["auto_ok"] is False
            assert lab["has_pdf"]
    underlying = Counter(lab["mismatch_type"] for lab in LABELS if lab["injection"])
    assert set(underlying) == {"none", "taxable_value_mismatch"}  # some numbers match exactly


def test_hard_negatives_are_benign_matches_in_every_month():
    for m in MONTHS:
        hn = [lab for lab in m.labels if lab["hard_negative"]]
        assert 1 <= len(hn) <= 2
        for lab in hn:
            assert not lab["injection"] and lab["expected_action"] == "none" and lab["pdf_note"]
            assert lab["pdf_note"] in gen.HARD_NEGATIVES


def test_noisy_share_and_language_profiles():
    with_pdf = [lab for lab in LABELS if lab["has_pdf"]]
    share = sum(lab["scan_quality"] == "noisy" for lab in with_pdf) / len(with_pdf)
    assert 0.33 <= share <= 0.47
    langs = Counter(lab["supplier_language"] for lab in LABELS)
    assert set(langs) == {"en", "hi", "te"}
    assert any(lab["supplier_filing"] == "quarterly" for lab in LABELS)


# --- label rules ------------------------------------------------------------------------------


def test_auto_ok_follows_decision_one_and_the_safety_rules():
    ceiling = CFG["auto_hold_ceiling_rupees"] * 100
    large_missing = 0
    for lab in LABELS:
        if lab["injection"] or not lab["has_pdf"]:
            assert lab["auto_ok"] is False, lab["id"]
        elif lab["expected_action"] == "none":
            assert lab["auto_ok"] is True, lab["id"]
        elif lab["expected_action"] == "chase_supplier_and_hold_itc":
            assert lab["auto_ok"] is (lab["book_tax_paise"] <= ceiling), lab["id"]
            large_missing += lab["book_tax_paise"] > ceiling
        else:
            assert lab["auto_ok"] is False, lab["id"]
    assert large_missing >= 5  # the ceiling is exercised


def test_expected_outcomes_match_the_case_table():
    for lab in LABELS:
        if lab["injection"]:
            continue
        assert (lab["mismatch_type"], lab["expected_status"], lab["expected_action"]) == (
            gen.CASE_TABLE[lab["case_type"]]
        )


def test_labels_point_at_the_right_rows():
    for m in MONTHS:
        for lab in m.labels:
            for line in lab["book_lines"]:
                row = m.books[line - 1]
                assert gstin.pan_part(row["supplier_gstin"]) != "" and row["pdf_file"] in (
                    "",
                    lab["pdf_file"],
                )
            if lab["portal_index"] is not None:
                assert m.portal[lab["portal_index"]]["supplier_gstin"] == lab["supplier_gstin"]
            if lab["case_type"] == "in_2b_not_in_books":
                assert lab["book_lines"] == [] and lab["portal_index"] is not None
            if lab["case_type"] == "missing_in_2b":
                assert lab["portal_index"] is None and len(lab["book_lines"]) == 1
            if lab["case_type"] == "duplicate_in_books":
                original = next(
                    o
                    for o in m.labels
                    if o["invoice_no"] == lab["invoice_no"]
                    and o["id"] != lab["id"]
                    and o["supplier_gstin"] == lab["supplier_gstin"]
                )
                assert original["book_lines"][0] < lab["book_lines"][0]


def test_every_gstin_is_checksum_valid_except_the_planted_typos():
    for m in MONTHS:
        assert gstin.checksum_valid(m.business.gstin)
        for s in m.business.suppliers:
            assert gstin.checksum_valid(s.gstin)
        for lab in m.labels:
            for line in lab["book_lines"]:
                g = m.books[line - 1]["supplier_gstin"]
                if lab["case_type"] == "gstin_typo_in_books":
                    assert not gstin.checksum_valid(g)
                    assert gstin.edit_distance_one(g, lab["supplier_gstin"])
                else:
                    assert gstin.checksum_valid(g)


def test_invoice_number_differences_are_recoverable_by_normalisation():
    seen = Counter()
    for m in MONTHS:
        for lab in m.labels:
            if lab["case_type"] != "invoice_no_format_difference":
                continue
            book_no = m.books[lab["book_lines"][0] - 1]["invoice_no"]
            portal_no = m.portal[lab["portal_index"]]["invoice_no"]
            assert book_no != portal_no
            if _inv_key(book_no) == _inv_key(portal_no):
                seen["normalised"] += 1
            else:
                assert _no_year(_inv_key(book_no)) == _no_year(_inv_key(portal_no))
                seen["no_year"] += 1
    assert seen["normalised"] and seen["no_year"]


def test_money_is_exact_and_invoices_add_up():
    for m in MONTHS:
        for r in m.books:
            for k in ("taxable_value", "igst", "cgst", "sgst"):
                money.parse(r[k])
        for lab in m.labels:
            t = lab["extraction_truth"]
            if t is None:
                continue
            parts = sum(money.parse(t[k]) for k in ("taxable_value", "igst", "cgst", "sgst"))
            assert parts == money.parse(t["total"])
            assert bool(money.parse(t["igst"])) != bool(money.parse(t["cgst"]))


def test_portal_json_numbers_are_written_exactly(tmp_path):
    gen.generate(CFG, out_dir=tmp_path, labels_dir=None, render=False)
    text = (tmp_path / "B01" / "2026-08" / "gstr2b_like.json").read_text(encoding="utf-8")
    data = json.loads(text, parse_float=Decimal)
    assert "__DEC__" not in text
    first = data["invoices"][0]
    assert isinstance(first["taxable_value"], Decimal)
    assert first["taxable_value"] == first["taxable_value"].quantize(Decimal("0.01"))


# --- splits -----------------------------------------------------------------------------------


def test_split_has_no_shared_business_or_supplier():
    report = check_splits(SPLITS, gen.build_businesses(CFG))
    assert report["dev"] > 0 and report["test"] > 0


def test_split_check_catches_overlap():
    bad = {"dev": ["B01", "B02", "B03", "B04", "B05", "B06"], "test": SPLITS["test"]}
    with pytest.raises(typer.BadParameter):
        check_splits(bad, gen.build_businesses(CFG))


# --- rendering --------------------------------------------------------------------------------


def _render_month(month, out: Path) -> dict[str, str]:
    hashes = {}
    for doc in month.docs:
        if doc.note_style == "hindi_instruction":
            continue  # needs a Devanagari font, which CI does not have
        path = out / doc.pdf_file
        gen.render_invoice_pdf(path, doc, month.business, CFG)
        hashes[doc.pdf_file] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def test_pdfs_are_byte_identical_and_marked_synthetic(tmp_path):
    month = MONTHS[0]
    first = _render_month(month, tmp_path / "1")
    second = _render_month(month, tmp_path / "2")
    assert first == second and len(first) >= 15

    import pypdfium2 as pdfium

    doc = month.docs[0]
    pdf = pdfium.PdfDocument(str(tmp_path / "1" / doc.pdf_file))
    text = pdf[0].get_textpage().get_text_range()
    pdf.close()
    assert gen.SYNTHETIC_FOOTER in text
    assert text.count("(fictional)") >= 2  # supplier and buyer
    assert doc.supplier.gstin in text and month.business.gstin in text


def test_noisy_scan_is_deterministic(tmp_path):
    month = MONTHS[0]
    doc = next(d for d in month.docs if d.note_style != "hindi_instruction")
    pdf_path = tmp_path / "x.pdf"
    gen.render_invoice_pdf(pdf_path, doc, month.business, CFG)
    params = noise_params(CFG, "B01:x")
    assert params == noise_params(CFG, "B01:x")
    assert abs(params["angle"]) <= 3 and 30 <= params["quality"] <= 50
    image = rasterise(pdf_path, 72)
    assert add_noise(image, params) == add_noise(image, params)


# --- review workflow --------------------------------------------------------------------------


def test_review_sample_is_stratified_and_about_ten_percent(tmp_path):
    sample = review.stratified_sample(LABELS)
    assert {lab["case_type"] for lab in sample} == {lab["case_type"] for lab in LABELS}
    assert 0.09 * len(LABELS) <= len(sample) <= 0.14 * len(LABELS)
    out = tmp_path / "s.csv"
    assert review.write_sample(LABELS, out) == len(sample)
    with open(out, encoding="utf-8") as f:
        assert next(csv.reader(f)) == review.SAMPLE_COLUMNS


def _filled(tmp_path, rows):
    path = tmp_path / "filled.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=review.SAMPLE_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in review.SAMPLE_COLUMNS})
    return path


def test_apply_review_records_overrides_and_changelog(tmp_path):
    a, b = LABELS[0]["id"], LABELS[1]["id"]
    filled = _filled(
        tmp_path,
        [
            {"id": a, "verdict": "OK"},
            {"id": b, "verdict": "FIX", "fix_expected_action": "escalate_human", "notes": "why"},
            {"id": LABELS[2]["id"]},  # not reviewed yet
        ],
    )
    overrides, changelog = tmp_path / "o.jsonl", tmp_path / "CHANGELOG.md"
    result = review.apply_review(filled, LABELS, overrides, changelog, today=dt.date(2026, 11, 24))
    assert result == {"reviewed": 2, "fixed": 1}
    assert "Fixed: 1" in changelog.read_text(encoding="utf-8")
    applied = {
        lab["id"]: lab
        for lab in review.apply_overrides(LABELS[:3], review.load_overrides(overrides))
    }
    assert applied[a]["review_status"] == "human_ok"
    assert applied[b]["expected_action"] == "escalate_human"
    assert applied[b]["review_status"] == "human_fixed"
    assert applied[LABELS[2]["id"]]["review_status"] == "assistant_reviewed"


@pytest.mark.parametrize(
    "row",
    [
        {"id": "nope", "verdict": "OK"},
        {"id": None, "verdict": "MAYBE"},
        {"id": None, "verdict": "FIX"},
        {"id": None, "verdict": "FIX", "fix_expected_action": "pay_now"},
        {"id": None, "verdict": "OK", "fix_auto_ok": "true"},
    ],
)
def test_bad_review_files_apply_nothing(tmp_path, row):
    row = {**row, "id": row["id"] or LABELS[0]["id"]}
    overrides = tmp_path / "o.jsonl"
    with pytest.raises(review.ReviewError):
        review.apply_review(_filled(tmp_path, [row]), LABELS, overrides, tmp_path / "c.md")
    assert not overrides.exists()


def test_committed_labels_carry_review_status():
    committed = gen.load_labels()
    assert len(committed) == len(LABELS)
    assert {lab["review_status"] for lab in committed} <= {
        "assistant_reviewed",
        "human_ok",
        "human_fixed",
    }
