"""Matcher: pair book lines with supplier-reported rows and classify each pair (spec 6.3).

Deterministic and free. Five passes, each looking only at what is still unmatched, every
pairing one-to-one:

1. exact        same GSTIN, same invoice number as typed
2. normalised   same GSTIN, same ``inv_key``
3. no_year      same GSTIN, same ``inv_key_noyear``
4. gstin_repair book GSTIN fails the checksum; a portal GSTIN one edit away that passes it,
                with the same ``inv_key``
5. fuzzy        same PAN, ``fuzz.ratio`` of the keys >= 90, taxable values within 1%

The result never depends on the order of the input lists: inputs are sorted by line number and
file index first, and the fuzzy pass only accepts a pair that is the single best choice for both
sides. A tie is never guessed; it goes to a person (``needs_human``).

Money is compared in integer paise with ``<=`` against the Rs 1.00 tolerance.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from agent_orchestrator import gstin, money
from agent_orchestrator.agents.intake import inv_key
from agent_orchestrator.models import BookRecord, Case, ExtractedInvoice, PortalRecord

FUZZY_MIN_SCORE = 90.0
FUZZY_TAXABLE_PCT = 1
LOW_CONFIDENCE = 0.85


@dataclass
class _Pair:
    book: BookRecord | None
    portal: PortalRecord | None
    match_pass: str
    score: float
    flags: set[str] = field(default_factory=set)


# --- helpers ----------------------------------------------------------------------------------


def _tax(r: BookRecord | PortalRecord) -> int:
    return r.igst_paise + r.cgst_paise + r.sgst_paise


def _is_igst(r: BookRecord | PortalRecord) -> bool:
    return r.igst_paise > 0


def _within_pct(a: int, b: int, pct: int) -> bool:
    return abs(a - b) * 100 <= max(abs(a), abs(b)) * pct


def case_key(business_id: str, period: str, supplier_gstin: str, key: str, extra: str = "") -> str:
    text = "|".join((business_id, period, supplier_gstin, key, extra))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# --- passes -----------------------------------------------------------------------------------


def _pair_by(
    books: list[BookRecord],
    portal: list[PortalRecord],
    book_key,
    portal_key,
    match_pass: str,
    pairs: list[_Pair],
) -> tuple[list[BookRecord], list[PortalRecord]]:
    """Pair equal keys first-come by line number / file index. Returns what is left."""
    by_key: dict[object, list[PortalRecord]] = defaultdict(list)
    for p in portal:
        k = portal_key(p)
        if k is not None:
            by_key[k].append(p)
    left_books = []
    used: set[int] = set()
    for b in books:
        k = book_key(b)
        candidates = [p for p in by_key.get(k, []) if p.index not in used] if k is not None else []
        if candidates:
            p = candidates[0]
            used.add(p.index)
            pairs.append(_Pair(b, p, match_pass, 100.0))
        else:
            left_books.append(b)
    return left_books, [p for p in portal if p.index not in used]


def _fuzzy(
    books: list[BookRecord], portal: list[PortalRecord], pairs: list[_Pair]
) -> tuple[list[BookRecord], list[PortalRecord], set[int], set[int]]:
    """Accept only pairs that are the unique best for both sides; report ties."""
    scores: dict[tuple[int, int], float] = {}
    for b in books:
        for p in portal:
            if gstin.pan_part(b.supplier_gstin) != gstin.pan_part(p.supplier_gstin):
                continue
            if not _within_pct(b.taxable_paise, p.taxable_paise, FUZZY_TAXABLE_PCT):
                continue
            s = fuzz.ratio(b.inv_key, p.inv_key)
            if s >= FUZZY_MIN_SCORE:
                scores[(b.line_no, p.index)] = s
    book_by_line = {b.line_no: b for b in books}
    portal_by_index = {p.index: p for p in portal}
    involved_books = {bl for bl, _ in scores}
    involved_portal = {pi for _, pi in scores}
    changed = True
    while changed:
        changed = False
        best_b: dict[int, list[tuple[float, int]]] = defaultdict(list)
        best_p: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for (bl, pi), s in scores.items():
            best_b[bl].append((s, pi))
            best_p[pi].append((s, bl))
        for (bl, pi), s in sorted(scores.items()):
            top_b = max(x for x, _ in best_b[bl])
            top_p = max(x for x, _ in best_p[pi])
            unique_b = sum(1 for x, _ in best_b[bl] if x == top_b) == 1
            unique_p = sum(1 for x, _ in best_p[pi] if x == top_p) == 1
            if s == top_b == top_p and unique_b and unique_p:
                pairs.append(
                    _Pair(book_by_line[bl], portal_by_index[pi], "fuzzy", s, {"fuzzy_match"})
                )
                scores = {k: v for k, v in scores.items() if k[0] != bl and k[1] != pi}
                changed = True
                break
    matched_b = {pr.book.line_no for pr in pairs if pr.match_pass == "fuzzy" and pr.book}
    matched_p = {pr.portal.index for pr in pairs if pr.match_pass == "fuzzy" and pr.portal}
    tied_books = {bl for bl, _ in scores} & (involved_books - matched_b)
    tied_portal = {pi for _, pi in scores} & (involved_portal - matched_p)
    left_b = [b for b in books if b.line_no not in matched_b]
    left_p = [p for p in portal if p.index not in matched_p]
    return left_b, left_p, tied_books, tied_portal


# --- classification ---------------------------------------------------------------------------


def _classify(pair: _Pair, period: str) -> tuple[str, str, dict[str, int]]:
    """(mismatch_type, status, diffs) for a matched pair. Priority per spec 6.3."""
    b, p = pair.book, pair.portal
    assert b is not None and p is not None
    diffs = {
        "taxable": b.taxable_paise - p.taxable_paise,
        "igst": b.igst_paise - p.igst_paise,
        "cgst": b.cgst_paise - p.cgst_paise,
        "sgst": b.sgst_paise - p.sgst_paise,
        "total_tax": _tax(b) - _tax(p),
    }
    t = money.TOLERANCE_PAISE
    if pair.match_pass == "gstin_repair":
        return "gstin_typo_in_books", "mismatch", diffs
    if _is_igst(b) != _is_igst(p) and abs(diffs["total_tax"]) <= t:
        return "tax_head_mismatch", "mismatch", diffs
    if abs(diffs["taxable"]) > t:
        return "taxable_value_mismatch", "mismatch", diffs
    if any(abs(diffs[h]) > t for h in ("igst", "cgst", "sgst")):
        return "tax_amount_mismatch", "mismatch", diffs
    if p.reported_in_period != period:
        return "date_mismatch_other_period", "mismatch", diffs
    if pair.match_pass in ("normalised", "no_year", "fuzzy"):
        return "invoice_no_format_difference", "matched_after_normalisation", diffs
    if any(diffs.values()):
        return "rounding_within_tolerance", "matched", diffs
    return "none", "matched", diffs


def _pdf_flags(
    book: BookRecord | None, portal: PortalRecord | None, ex: ExtractedInvoice | None
) -> set[str]:
    """The PDF is evidence, never a vote that closes a case (spec 6.3)."""
    if ex is None:
        return set()
    flags = set()
    if ex.injection.flagged:
        flags.add("injection")
    if ex.min_critical_confidence < LOW_CONFIDENCE:
        flags.add("low_confidence")

    def read(fv) -> int | None:
        try:
            return money.parse((fv.value or "").replace(",", "")) if fv.value else None
        except money.MoneyError:
            return None

    pdf_taxable, pdf_tax = read(ex.taxable_value), [read(ex.igst), read(ex.cgst), read(ex.sgst)]
    pdf_gstin = gstin.normalise(ex.supplier_gstin.value or "")
    pdf_key = None
    if ex.invoice_no.value:
        pdf_key = inv_key(ex.invoice_no.value)

    def disagrees(side: BookRecord | PortalRecord) -> bool:
        t = money.TOLERANCE_PAISE
        if pdf_taxable is not None and abs(pdf_taxable - side.taxable_paise) > t:
            return True
        if None not in pdf_tax and abs(sum(pdf_tax) - _tax(side)) > t:
            return True
        if pdf_gstin and pdf_gstin != side.supplier_gstin:
            return True
        return bool(pdf_key and pdf_key != side.inv_key)

    sides = [s for s in (book, portal) if s is not None]
    if sides and all(disagrees(s) for s in sides):
        flags.add("pdf_disagrees")
    return flags


def _facts_hash(c: dict) -> str:
    text = json.dumps(c, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# --- entry point ------------------------------------------------------------------------------


def match(
    books: Iterable[BookRecord],
    portal: Iterable[PortalRecord],
    business_id: str,
    period: str,
    extractions: Mapping[str, ExtractedInvoice] | None = None,
) -> list[Case]:
    """Match one business-month. ``extractions`` maps a book ``pdf_file`` to what the vision
    agent read from it (optional; without it no PDF flags are set)."""
    books = sorted(books, key=lambda b: b.line_no)
    portal = sorted(portal, key=lambda p: p.index)
    extractions = extractions or {}
    pairs: list[_Pair] = []

    # Duplicates: book lines with the same (GSTIN, inv_key) beyond the first.
    first_seen: set[tuple[str, str]] = set()
    singles: list[BookRecord] = []
    duplicates: list[BookRecord] = []
    for b in books:
        k = (b.supplier_gstin, b.inv_key)
        (duplicates if k in first_seen else singles).append(b)
        first_seen.add(k)

    rest_b, rest_p = _pair_by(
        singles,
        portal,
        lambda b: (b.supplier_gstin, b.invoice_no.strip()),
        lambda p: (p.supplier_gstin, p.invoice_no.strip()),
        "exact",
        pairs,
    )
    rest_b, rest_p = _pair_by(
        rest_b,
        rest_p,
        lambda b: (b.supplier_gstin, b.inv_key),
        lambda p: (p.supplier_gstin, p.inv_key),
        "normalised",
        pairs,
    )
    rest_b, rest_p = _pair_by(
        rest_b,
        rest_p,
        lambda b: (b.supplier_gstin, b.inv_key_noyear),
        lambda p: (p.supplier_gstin, p.inv_key_noyear),
        "no_year",
        pairs,
    )
    rest_b, rest_p = _gstin_repair(rest_b, rest_p, pairs)
    rest_b, rest_p, tied_b, tied_p = _fuzzy(rest_b, rest_p, pairs)

    for b in rest_b:
        flags = {"ambiguous_match"} if b.line_no in tied_b else set()
        pairs.append(_Pair(b, None, "none", 0.0, flags))
    for p in rest_p:
        flags = {"ambiguous_match"} if p.index in tied_p else set()
        pairs.append(_Pair(None, p, "none", 0.0, flags))

    cases = [_to_case(pr, business_id, period, extractions) for pr in pairs]
    dup_count: dict[tuple[str, str], int] = defaultdict(int)
    for b in duplicates:
        k = (b.supplier_gstin, b.inv_key)
        dup_count[k] += 1
        cases.append(_duplicate_case(b, business_id, period, dup_count[k], extractions))

    cases.sort(
        key=lambda c: (
            c.book_lines[0] if c.book_lines else 10**9,
            c.portal_index if c.portal_index is not None else -1,
        )
    )
    _ensure_unique_keys(cases)
    return cases


def _gstin_repair(
    books: list[BookRecord], portal: list[PortalRecord], pairs: list[_Pair]
) -> tuple[list[BookRecord], list[PortalRecord]]:
    used: set[int] = set()
    left = []
    for b in books:
        if b.gstin_valid:
            left.append(b)
            continue
        candidates = [
            p
            for p in portal
            if p.index not in used
            and p.inv_key == b.inv_key
            and gstin.checksum_valid(p.supplier_gstin)
            and gstin.edit_distance_one(b.supplier_gstin, p.supplier_gstin)
        ]
        if len(candidates) == 1:
            used.add(candidates[0].index)
            pairs.append(_Pair(b, candidates[0], "gstin_repair", 100.0))
        else:
            left.append(b)  # none, or more than one plausible correction: do not guess
    return left, [p for p in portal if p.index not in used]


def _to_case(
    pr: _Pair, business_id: str, period: str, extractions: Mapping[str, ExtractedInvoice]
) -> Case:
    b, p = pr.book, pr.portal
    flags = set(pr.flags)
    if b is not None and not b.gstin_valid:
        flags.add("gstin_invalid")
    ex = extractions.get(b.pdf_file) if b is not None and b.pdf_file else None
    flags |= _pdf_flags(b, p, ex)
    if b is not None and p is not None:
        mismatch, status, diffs = _classify(pr, period)
    elif b is not None:
        mismatch, status, diffs = "missing_in_2b", "unmatched", {}
    else:
        mismatch, status, diffs = "in_2b_not_in_books", "unmatched", {}
    if "ambiguous_match" in flags:
        status = "needs_human"
    gst = p.supplier_gstin if p is not None else b.supplier_gstin  # the corrected GSTIN if any
    key_src = p.inv_key if p is not None else b.inv_key
    at_risk = (
        _tax(b) if b is not None and status not in ("matched", "matched_after_normalisation") else 0
    )
    facts = {
        "mismatch_type": mismatch,
        "status": status,
        "diffs": diffs,
        "book": b.row_hash if b else None,
        "portal": p.model_dump(mode="json") if p else None,
        "flags": sorted(flags),
        "pdf": ex.model_dump(mode="json", exclude={"other_text"}) if ex else None,
    }
    return Case(
        case_key=case_key(business_id, period, gst, key_src),
        business_id=business_id,
        period=period,
        mismatch_type=mismatch,
        status=status,
        book_lines=[b.line_no] if b else [],
        portal_index=p.index if p else None,
        pdf_file=b.pdf_file if b else None,
        match_pass=pr.match_pass,
        match_score=round(pr.score, 2),
        diffs=diffs,
        itc_at_risk_paise=at_risk,
        facts_hash=_facts_hash(facts),
        flags=sorted(flags),
    )


def _duplicate_case(
    b: BookRecord,
    business_id: str,
    period: str,
    n: int,
    extractions: Mapping[str, ExtractedInvoice],
) -> Case:
    flags = {"gstin_invalid"} if not b.gstin_valid else set()
    ex = extractions.get(b.pdf_file) if b.pdf_file else None
    flags |= {f for f in _pdf_flags(b, None, ex) if f != "pdf_disagrees"}
    facts = {
        "mismatch_type": "duplicate_in_books",
        "book": b.row_hash,
        "n": n,
        "flags": sorted(flags),
    }
    return Case(
        case_key=case_key(business_id, period, b.supplier_gstin, b.inv_key, f"dup{n}"),
        business_id=business_id,
        period=period,
        mismatch_type="duplicate_in_books",
        status="duplicate",
        book_lines=[b.line_no],
        portal_index=None,
        pdf_file=b.pdf_file,
        match_pass="duplicate",
        match_score=100.0,
        diffs={},
        itc_at_risk_paise=_tax(b),
        facts_hash=_facts_hash(facts),
        flags=sorted(flags),
    )


def _ensure_unique_keys(cases: list[Case]) -> None:
    """Two cases can share (GSTIN, inv_key) only in odd data (e.g. the supplier reported the
    same invoice twice). Give the later one a distinct, still deterministic key."""
    seen: set[str] = set()
    for i, c in enumerate(cases):
        if c.case_key in seen:
            extra = f"line{c.book_lines[0]}" if c.book_lines else f"p{c.portal_index}"
            key = case_key(c.business_id, c.period, c.case_key, extra)
            cases[i] = c.model_copy(update={"case_key": key})
        seen.add(cases[i].case_key)
