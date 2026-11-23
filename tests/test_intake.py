import json
import shutil
from pathlib import Path

import pytest

from agent_orchestrator.agents import intake
from agent_orchestrator.agents.intake import IntakeError, inv_key, inv_key_noyear, run_intake_dir
from agent_orchestrator.eval import generate as gen

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed" / "seed_data"
HEADER = ",".join(intake.BOOK_HEADER)


def _seed_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "month"
    shutil.copytree(SEED, dst)
    return dst


def _codes(result, ref=None):
    return [i.code for i in result.dq_issues if ref is None or i.row_ref == ref]


# --- invoice-number keys ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw", ["ABC/2026/0107", "abc-2026-107", " abc 2026 0107 ", "ABC.2026.107"]
)
def test_same_inv_key_for_formatting_differences(raw):
    assert inv_key(raw) == "ABC|2026|107"


def test_token_boundaries_are_kept():
    assert inv_key("12/3") != inv_key("1/23")
    assert inv_key("0000") == "0"
    assert inv_key("INV000105") == "INV000105"  # mixed tokens keep their zeros


@pytest.mark.parametrize(
    ("raw", "noyear"),
    [
        ("ABC/2026/0107", "ABC|107"),
        ("ABC/0107", "ABC|107"),
        ("ABC/26-27/107", "ABC|107"),
        ("ABC/2026-27/107", "ABC|107"),
        ("ABC/2026/2027/107", "ABC|107"),
        ("FY26/ABC/107", "ABC|107"),
        ("FY2026-27/ABC/107", "ABC|107"),
        ("ABC-12-14-107", "ABC|12|14|107"),  # 12-14 is not a financial year
        ("ABC/0107/2026", "ABC|107"),
        ("2026", "2026"),  # never empty
    ],
)
def test_year_tokens_dropped_only_in_noyear_key(raw, noyear):
    assert inv_key_noyear(raw) == noyear


def test_fy_tokens_stay_in_the_main_key():
    assert inv_key("ABC/26-27/107") == "ABC|26|27|107"
    assert inv_key("FY26/ABC/107") == "FY26|ABC|107"


# --- the seed month ---------------------------------------------------------------------------


def test_seed_month_counts_and_issues():
    r = run_intake_dir(SEED, "2026-10")
    assert len(r.books) == 18
    assert len(r.portal) == 16
    assert r.buyer_gstin == "36PKWZU0959A1ZQ"
    gstin_issues = [i for i in r.dq_issues if i.code == "gstin_checksum"]
    assert [i.row_ref for i in gstin_issues] == ["books:line:13"]  # case13, TES/2026/0112
    typo = next(b for b in r.books if b.line_no == 13)
    assert typo.invoice_no == "TES/2026/0112" and typo.gstin_valid is False
    dup = [b for b in r.books if b.invoice_no == "FIC/2026/0114"]
    assert len(dup) == 2 and dup[0].row_hash == dup[1].row_hash
    assert dup[0].line_no != dup[1].line_no
    assert "duplicate_row" in _codes(r, f"books:line:{dup[1].line_no}")


def test_seed_money_is_paise():
    r = run_intake_dir(SEED, "2026-10")
    first = r.books[0]
    assert (first.taxable_paise, first.cgst_paise, first.sgst_paise) == (4730000, 425700, 425700)
    assert r.portal[0].taxable_paise == 4730000


def test_input_hash_ignores_line_endings_and_bom(tmp_path):
    a = _seed_copy(tmp_path / "a")
    b = _seed_copy(tmp_path / "b")
    csv_b = b / "purchase_register.csv"
    text = csv_b.read_bytes().replace(b"\r\n", b"\n")
    csv_b.write_bytes(b"\xef\xbb\xbf" + text.replace(b"\n", b"\r\n"))
    json_b = b / "gstr2b_like.json"
    json_b.write_bytes(json_b.read_bytes().replace(b"\n", b"\r\n"))
    ha = run_intake_dir(a, "2026-10").input_hash
    assert ha == run_intake_dir(b, "2026-10").input_hash
    (b / "invoices" / "SAM_2026_0100.pdf").write_bytes(b"%PDF-changed")
    assert ha != run_intake_dir(b, "2026-10").input_hash


# --- file-level errors stop the run -----------------------------------------------------------


def test_bad_header_is_a_clear_error(tmp_path):
    month = _seed_copy(tmp_path)
    path = month / "purchase_register.csv"
    path.write_text(path.read_text().replace("taxable_value", "taxable"), encoding="utf-8")
    with pytest.raises(IntakeError, match="unexpected header"):
        run_intake_dir(month, "2026-10")


def test_wrong_period_or_bad_buyer_gstin_stops_the_run(tmp_path):
    month = _seed_copy(tmp_path)
    with pytest.raises(IntakeError, match="period"):
        run_intake_dir(month, "2026-09")
    path = month / "gstr2b_like.json"
    doc = json.loads(path.read_text())
    doc["buyer_gstin"] = "36PKWZU0959A1ZX"
    path.write_text(json.dumps(doc))
    with pytest.raises(IntakeError, match="checksum"):
        run_intake_dir(month, "2026-10")


def test_expected_buyer_is_checked(tmp_path):
    with pytest.raises(IntakeError, match="expected"):
        run_intake_dir(SEED, "2026-10", expected_buyer_gstin="36TSDKP2716L1ZA")


def test_missing_or_broken_files(tmp_path):
    month = _seed_copy(tmp_path)
    (month / "gstr2b_like.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(IntakeError, match="JSON"):
        run_intake_dir(month, "2026-10")
    (month / "purchase_register.csv").unlink()
    with pytest.raises(IntakeError, match="not found"):
        run_intake_dir(month, "2026-10")
    with pytest.raises(IntakeError, match="YYYY-MM"):
        run_intake_dir(month, "Oct-2026")


# --- row-level problems become issues ---------------------------------------------------------


def _books(tmp_path, rows, portal_rows=None):
    month = _seed_copy(tmp_path)
    (month / "purchase_register.csv").write_text(
        HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    if portal_rows is not None:
        path = month / "gstr2b_like.json"
        doc = json.loads(path.read_text())
        doc["invoices"] = portal_rows
        path.write_text(json.dumps(doc))
    return run_intake_dir(month, "2026-10")


GOOD = "SAM/2026/0100,2026-10-10,Sample Packaging Co,36TSDKP2716L1ZA,47300.0,18,0.0,4257.0,4257.0,"


def test_bad_rows_become_issues_and_the_run_continues(tmp_path):
    r = _books(
        tmp_path,
        [
            GOOD,
            GOOD.replace("2026-10-10", "2026-02-30"),  # impossible date
            GOOD.replace("47300.0", "47300.0x"),  # bad amount
            GOOD.replace(",18,", ",18.5,"),  # not a whole rate
            "too,few,cells",
            ",2026-10-10,X,36TSDKP2716L1ZA,1.00,18,0,0.09,0.09,",  # no invoice number
            GOOD.replace("47300.0", "47300.123"),  # three decimals
        ],
    )
    assert [b.line_no for b in r.books] == [1]
    codes = {i.row_ref: _codes(r, i.row_ref) for i in r.dq_issues}
    assert "bad_date" in codes["books:line:2"] and "row_skipped" in codes["books:line:2"]
    assert "bad_amount" in codes["books:line:3"]
    assert "bad_rate" in codes["books:line:4"]
    assert "bad_row" in codes["books:line:5"]
    assert "missing_invoice_no" in codes["books:line:6"]
    assert "bad_amount" in codes["books:line:7"]


def test_soft_issues_keep_the_row(tmp_path):
    r = _books(
        tmp_path,
        [
            GOOD.replace(",0.0,4257.0,4257.0,", ",100.0,4257.0,4257.0,"),  # IGST and CGST both
            GOOD.replace(",18,", ",17,"),  # not a GST slab
            GOOD.replace("4257.0,4257.0,", "4257.0,4257.0,invoices/none.pdf"),
            GOOD.replace("36TSDKP2716L1ZA", " 36tsdkp 2716-l1za "),  # needs normalising only
        ],
    )
    assert len(r.books) == 4
    assert "tax_heads_mixed" in _codes(r, "books:line:1")
    assert "unusual_rate" in _codes(r, "books:line:2")
    assert "pdf_missing" in _codes(r, "books:line:3")
    # Same content as line 3 (the PDF path is not part of the content hash): only flagged.
    assert _codes(r, "books:line:4") == ["duplicate_row"]
    assert r.books[3].supplier_gstin == "36TSDKP2716L1ZA" and r.books[3].gstin_valid


def test_bad_portal_rows_become_issues(tmp_path):
    good = json.loads((SEED / "gstr2b_like.json").read_text())["invoices"][0]
    float_text = dict(good, taxable_value="47300.001")
    r = _books(
        tmp_path,
        [GOOD],
        portal_rows=[
            good,
            {k: v for k, v in good.items() if k != "rate"},
            dict(good, reported_in_period="Oct"),
            float_text,
            "not an object",
            dict(good, supplier_gstin="36TSDKP2716L1ZB"),
        ],
    )
    assert [p.index for p in r.portal] == [0, 5]  # indexes stay the file positions
    assert "missing_fields" in _codes(r, "2b:index:1")
    assert "bad_period" in _codes(r, "2b:index:2")
    assert "bad_amount" in _codes(r, "2b:index:3")
    assert "bad_row" in _codes(r, "2b:index:4")
    assert "gstin_checksum" in _codes(r, "2b:index:5")


def test_portal_numbers_never_pass_through_float(tmp_path):
    good = json.loads((SEED / "gstr2b_like.json").read_text())["invoices"][0]
    month = _seed_copy(tmp_path)
    path = month / "gstr2b_like.json"
    doc = json.loads(path.read_text())
    doc["invoices"] = [good]
    # 0.1 + 0.2 style values: exact in Decimal, inexact in float
    text = json.dumps(doc).replace('"taxable_value": 47300.0', '"taxable_value": 47300.30')
    path.write_text(text)
    r = run_intake_dir(month, "2026-10")
    assert r.portal[0].taxable_paise == 4730030


# --- the generated dataset --------------------------------------------------------------------


def test_generated_months_read_cleanly(tmp_path):
    cfg = gen.load_config()
    gen.generate(cfg, out_dir=tmp_path, labels_dir=None, render=False)
    expected_issue_types = {
        "gstin_checksum": "gstin_typo_in_books",
        "duplicate_row": "duplicate_in_books",
        "tax_not_rate_x_taxable": "tax_amount_mismatch",
        "cgst_sgst_unequal": "rounding_within_tolerance",
    }
    for month_dir in sorted(tmp_path.glob("B*/2026-*")):
        labels = [
            json.loads(x)
            for x in (month_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        r = intake.run_intake(
            month_dir / "purchase_register.csv", month_dir / "gstr2b_like.json", month_dir.name
        )
        assert len(r.books) == sum(len(lab["book_lines"]) for lab in labels)
        assert len(r.portal) == sum(lab["portal_index"] is not None for lab in labels)
        for issue in r.dq_issues:
            kind, _, n = issue.row_ref.split(":")
            owners = [
                lab["case_type"]
                for lab in labels
                if (kind == "books" and int(n) in lab["book_lines"])
                or (kind == "2b" and lab["portal_index"] == int(n))
            ]
            assert expected_issue_types[issue.code] in owners, (month_dir, issue)
