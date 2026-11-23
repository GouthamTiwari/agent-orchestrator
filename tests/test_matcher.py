import datetime as dt
import json
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_orchestrator import gstin
from agent_orchestrator.agents.intake import inv_key, inv_key_noyear, run_intake_dir
from agent_orchestrator.agents.matcher import match
from agent_orchestrator.eval import generate as gen
from agent_orchestrator.models import (
    BookRecord,
    ExtractedInvoice,
    FieldValue,
    InjectionResult,
    PortalRecord,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed" / "seed_data"
SEED_TYPE = {"exact_match": "none", "prompt_injection_in_invoice": "taxable_value_mismatch"}
G1 = "36TSDKP2716L1ZA"  # valid seed GSTIN
PERIOD = "2026-10"


def book(line, no="ABC/2026/0101", g=G1, taxable=1000000, igst=0, cgst=90000, sgst=90000, pdf=None):
    g = gstin.normalise(g)
    return BookRecord(
        line_no=line,
        row_hash=f"{line:012d}",
        invoice_no=no,
        inv_key=inv_key(no),
        inv_key_noyear=inv_key_noyear(no),
        date=dt.date(2026, 10, 5),
        supplier_name="S",
        supplier_gstin=g,
        gstin_valid=gstin.checksum_valid(g),
        taxable_paise=taxable,
        rate=18,
        igst_paise=igst,
        cgst_paise=cgst,
        sgst_paise=sgst,
        pdf_file=pdf,
    )


def portal(
    index, no="ABC/2026/0101", g=G1, taxable=1000000, igst=0, cgst=90000, sgst=90000, period=PERIOD
):
    return PortalRecord(
        index=index,
        invoice_no=no,
        inv_key=inv_key(no),
        inv_key_noyear=inv_key_noyear(no),
        date=dt.date(2026, 10, 5),
        supplier_name="S",
        supplier_gstin=g,
        taxable_paise=taxable,
        rate=18,
        igst_paise=igst,
        cgst_paise=cgst,
        sgst_paise=sgst,
        reported_in_period=period,
    )


def one(cases):
    assert len(cases) == 1
    return cases[0]


# --- against the labels -----------------------------------------------------------------------


def test_seed_month_matches_every_label():
    r = run_intake_dir(SEED, PERIOD)
    cases = match(r.books, r.portal, "SEED", PERIOD)
    labels = [json.loads(x) for x in (SEED / "labels.jsonl").read_text().splitlines()]
    assert len(cases) == 19  # 18 labels + the original line of the duplicate pair
    for lab in labels:
        want = SEED_TYPE.get(lab["case_type"], lab["case_type"])
        candidates = [
            c
            for c in cases
            if any(r.books[ln - 1].invoice_no == lab["invoice_no"] for ln in c.book_lines)
            or (
                c.portal_index is not None
                and next(p for p in r.portal if p.index == c.portal_index).invoice_no
                == lab["invoice_no"]
            )
        ]
        if lab["case_type"] == "duplicate_in_books":
            candidates = [c for c in candidates if c.status == "duplicate"]
        c = one(candidates)
        assert c.mismatch_type == want, lab["id"]
        assert c.status == lab["expected_status"] or lab["case_type"].startswith("prompt"), lab


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen")
    gen.generate(gen.load_config(), out_dir=out, labels_dir=None, render=False)
    months = []
    for md in sorted(out.glob("B*/2026-*")):
        labels = [json.loads(x) for x in (md / "labels.jsonl").read_text("utf-8").splitlines()]
        r = run_intake_dir(md, md.name)
        months.append((md.parent.name, md.name, r, labels))
    return months


def test_every_generated_label_is_matched_exactly(generated):
    total = 0
    for business, period, r, labels in generated:
        cases = match(r.books, r.portal, business, period)
        assert len(cases) == len(labels)
        by_line = {c.book_lines[0]: c for c in cases if c.book_lines}
        by_portal = {c.portal_index: c for c in cases if not c.book_lines}
        for lab in labels:
            c = (
                by_line[lab["book_lines"][0]]
                if lab["book_lines"]
                else by_portal[lab["portal_index"]]
            )
            assert (c.mismatch_type, c.status, c.portal_index) == (
                lab["mismatch_type"],
                lab["expected_status"],
                lab["portal_index"],
            ), lab["id"]
            total += 1
    assert total == 593


def test_case_keys_are_unique_and_stable(generated):
    business, period, r, _ = generated[0]
    first = [c.case_key for c in match(r.books, r.portal, business, period)]
    again = [c.case_key for c in match(r.books, r.portal, business, period)]
    assert first == again and len(set(first)) == len(first)
    other = [c.case_key for c in match(r.books, r.portal, "OTHER", period)]
    assert not set(first) & set(other)


# --- properties -------------------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_input_order_never_changes_the_result(generated, data):
    business, period, r, _ = data.draw(st.sampled_from(generated))
    books = data.draw(st.permutations(r.books))
    rows = data.draw(st.permutations(r.portal))
    assert match(books, rows, business, period) == match(r.books, r.portal, business, period)


@settings(max_examples=30, deadline=None)
@given(data=st.data())
def test_every_row_ends_in_exactly_one_case(generated, data):
    business, period, r, _ = data.draw(st.sampled_from(generated))
    keep_b = data.draw(st.lists(st.booleans(), min_size=len(r.books), max_size=len(r.books)))
    keep_p = data.draw(st.lists(st.booleans(), min_size=len(r.portal), max_size=len(r.portal)))
    books = [b for b, k in zip(r.books, keep_b, strict=True) if k]
    rows = [p for p, k in zip(r.portal, keep_p, strict=True) if k]
    cases = match(books, rows, business, period)
    lines = Counter(ln for c in cases for ln in c.book_lines)
    portal_idx = Counter(c.portal_index for c in cases if c.portal_index is not None)
    assert lines == Counter(b.line_no for b in books)  # each line once, none invented
    assert portal_idx == Counter(p.index for p in rows)  # each row once: never matched twice
    assert len({c.case_key for c in cases}) == len(cases)


# --- passes and rules -------------------------------------------------------------------------


def test_exact_and_normalised_and_no_year_passes():
    c = one(match([book(1)], [portal(0)], "B", PERIOD))
    assert (c.match_pass, c.mismatch_type, c.status) == ("exact", "none", "matched")
    c = one(match([book(1, no="abc-2026-101")], [portal(0)], "B", PERIOD))
    assert (c.match_pass, c.mismatch_type) == ("normalised", "invoice_no_format_difference")
    assert c.status == "matched_after_normalisation"
    c = one(match([book(1, no="ABC/0101")], [portal(0)], "B", PERIOD))
    assert c.match_pass == "no_year"


def test_same_number_from_another_supplier_does_not_match():
    other = "36YWMAS3306K1ZC"
    cases = match([book(1, g=other)], [portal(0)], "B", PERIOD)
    assert sorted(c.mismatch_type for c in cases) == ["in_2b_not_in_books", "missing_in_2b"]


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (0, "none"),
        (1, "rounding_within_tolerance"),
        (100, "rounding_within_tolerance"),
        (101, "tax_amount_mismatch"),
        (-101, "tax_amount_mismatch"),
    ],
)
def test_tolerance_is_one_rupee_inclusive(delta, expected):
    c = one(match([book(1)], [portal(0, cgst=90000 + delta)], "B", PERIOD))
    assert c.mismatch_type == expected
    if delta:
        assert c.diffs["cgst"] == -delta


def test_taxable_mismatch_and_its_priority_over_tax_amount():
    c = one(match([book(1)], [portal(0, taxable=1100000, cgst=99000, sgst=99000)], "B", PERIOD))
    assert c.mismatch_type == "taxable_value_mismatch"
    assert c.diffs["taxable"] == -100000


def test_tax_head_mismatch():
    c = one(match([book(1, igst=180000, cgst=0, sgst=0)], [portal(0)], "B", PERIOD))
    assert c.mismatch_type == "tax_head_mismatch" and c.status == "mismatch"


def test_other_period():
    c = one(match([book(1)], [portal(0, period="2026-09")], "B", PERIOD))
    assert c.mismatch_type == "date_mismatch_other_period"


def test_gstin_repair_needs_exactly_one_candidate():
    typo = "36TSDKP2716L1ZB"  # one character off G1, fails the checksum
    c = one(match([book(1, g=typo)], [portal(0)], "B", PERIOD))
    assert (c.match_pass, c.mismatch_type, c.status) == (
        "gstin_repair",
        "gstin_typo_in_books",
        "mismatch",
    )
    assert "gstin_invalid" in c.flags


def test_gstin_repair_refuses_when_two_corrections_fit():
    typo = "36TSDKP2716L1ZB"
    alternatives = []
    for pos in range(15):
        for ch in gstin.ALPHABET:
            cand = typo[:pos] + ch + typo[pos + 1 :]
            if cand != typo and gstin.checksum_valid(cand):
                alternatives.append(cand)
    assert G1 in alternatives and len(alternatives) >= 2
    other = next(a for a in alternatives if a != G1)
    cases = match([book(1, g=typo)], [portal(0), portal(1, g=other)], "B", PERIOD)
    missing = one([c for c in cases if c.book_lines])
    assert missing.mismatch_type == "missing_in_2b" and "gstin_invalid" in missing.flags


def test_fuzzy_match_unique_best_is_accepted_and_flagged():
    c = one(match([book(1, no="ABC/2026/1071")], [portal(0, no="ABC/2026/1017")], "B", PERIOD))
    assert c.match_pass == "fuzzy" and "fuzzy_match" in c.flags
    assert c.mismatch_type == "invoice_no_format_difference"


def test_fuzzy_requires_taxable_within_one_percent():
    cases = match(
        [book(1, no="ABC/2026/1071")],
        [portal(0, no="ABC/2026/1017", taxable=1011000)],
        "B",
        PERIOD,
    )
    assert len(cases) == 2


def test_fuzzy_tie_goes_to_a_person():
    cases = match(
        [book(1, no="ABC/2026/1071")],
        [portal(0, no="ABC/2026/1072"), portal(1, no="ABC/2026/1073")],
        "B",
        PERIOD,
    )
    assert len(cases) == 3
    assert all(c.status == "needs_human" and "ambiguous_match" in c.flags for c in cases)
    assert all(c.match_pass == "none" for c in cases)


def test_duplicates_beyond_the_first_line():
    cases = match([book(1), book(2), book(3)], [portal(0)], "B", PERIOD)
    first = one([c for c in cases if c.book_lines == [1]])
    assert first.mismatch_type == "none"
    dups = [c for c in cases if c.status == "duplicate"]
    assert [c.book_lines for c in dups] == [[2], [3]]
    assert len({c.case_key for c in cases}) == 3
    assert all(c.itc_at_risk_paise == 180000 for c in dups)


def test_itc_at_risk():
    matched = one(match([book(1)], [portal(0)], "B", PERIOD))
    assert matched.itc_at_risk_paise == 0
    missing = one(match([book(1)], [], "B", PERIOD))
    assert missing.itc_at_risk_paise == 180000
    extra = one(match([], [portal(0)], "B", PERIOD))
    assert extra.itc_at_risk_paise == 0


# --- PDF evidence -----------------------------------------------------------------------------


def _ex(
    taxable="10000.00",
    cgst="900.00",
    sgst="900.00",
    g=G1,
    no="ABC/2026/0101",
    conf=0.95,
    injected=False,
):
    def fv(v):
        return FieldValue(value=v, confidence=conf)

    return ExtractedInvoice(
        pdf_file="invoices/a.pdf",
        model_id="fake",
        escalated=False,
        supplier_name=fv("S"),
        supplier_gstin=fv(g),
        buyer_gstin=fv("X"),
        invoice_no=fv(no),
        invoice_date=fv("2026-10-05"),
        taxable_value=fv(taxable),
        rate=fv("18"),
        igst=fv("0.00"),
        cgst=fv(cgst),
        sgst=fv(sgst),
        total=fv("11800.00"),
        min_critical_confidence=conf,
        injection=InjectionResult(flagged=injected, sources=["other_text"] if injected else []),
    )


def test_pdf_flags_are_evidence_only():
    b = book(1, pdf="invoices/a.pdf")
    clean = one(match([b], [portal(0)], "B", PERIOD, {"invoices/a.pdf": _ex()}))
    assert clean.flags == [] and clean.mismatch_type == "none"
    injected = one(match([b], [portal(0)], "B", PERIOD, {"invoices/a.pdf": _ex(injected=True)}))
    assert "injection" in injected.flags and injected.mismatch_type == "none"
    assert injected.facts_hash != clean.facts_hash
    low = one(match([b], [portal(0)], "B", PERIOD, {"invoices/a.pdf": _ex(conf=0.5)}))
    assert "low_confidence" in low.flags
    off = one(match([b], [portal(0)], "B", PERIOD, {"invoices/a.pdf": _ex(taxable="12000.00")}))
    assert "pdf_disagrees" in off.flags and off.status == "matched"  # flagged, not re-decided
    one_side = one(
        match(
            [b],
            [portal(0, taxable=1200000, cgst=108000, sgst=108000)],
            "B",
            PERIOD,
            {"invoices/a.pdf": _ex(taxable="12000.00", cgst="1080.00", sgst="1080.00")},
        )
    )
    assert "pdf_disagrees" not in one_side.flags  # agrees with the supplier's side
