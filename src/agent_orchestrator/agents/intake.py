"""Intake agent: read the books and the supplier statement into typed, normalised records.

Deterministic and free (no model call). A file that cannot be read correctly stops the run
with a clear `IntakeError`; a single bad row becomes a `DQIssue` and the run continues
(docs/SPEC.md sections 4 and 6.1).
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from agent_orchestrator import gstin, money
from agent_orchestrator.models import BookRecord, DQIssue, PortalRecord

BOOK_HEADER = [
    "invoice_no",
    "date",
    "supplier_name",
    "supplier_gstin",
    "taxable_value",
    "rate",
    "igst",
    "cgst",
    "sgst",
    "pdf_file",
]
PORTAL_FIELDS = (
    "invoice_no",
    "date",
    "supplier_name",
    "supplier_gstin",
    "taxable_value",
    "rate",
    "igst",
    "cgst",
    "sgst",
    "reported_in_period",
)
GST_RATES = frozenset({0, 3, 5, 12, 18, 28, 40})
INVOICE_FILE_SUFFIXES = (".pdf", ".jpg", ".jpeg", ".png")
_PERIOD = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


class IntakeError(ValueError):
    """An input file is unreadable or is not the file this run expects."""


@dataclass(frozen=True)
class IntakeResult:
    books: list[BookRecord]
    portal: list[PortalRecord]
    dq_issues: list[DQIssue]
    input_hash: str
    buyer_gstin: str
    period: str


# --- invoice-number keys (spec 4.4) -----------------------------------------------------------

_TOKEN_SPLIT = re.compile(r"[^A-Z0-9]+")
_EDGE = r"(?<![A-Z0-9])", r"(?![A-Z0-9])"
_FY_PAIR = re.compile(_EDGE[0] + r"(?:FY)?(\d{2}|\d{4})[-/](\d{2}|\d{4})" + _EDGE[1])
_FY_SINGLE = re.compile(_EDGE[0] + r"FY(\d{2}|\d{4})" + _EDGE[1])
_YEAR = re.compile(_EDGE[0] + r"(?:19|20)\d{2}" + _EDGE[1])


def inv_key(raw: str) -> str:
    """Comparable invoice-number key: uppercase, split on any non-alphanumeric character,
    leading zeros removed from purely numeric tokens, tokens joined with ``|``.

    ``ABC/2026/0107``, ``abc-2026-107`` and `` abc 2026 0107 `` all give ``ABC|2026|107``.
    Keeping the token boundaries means ``12/3`` and ``1/23`` stay different.
    """
    tokens = [t for t in _TOKEN_SPLIT.split(raw.upper()) if t]
    return "|".join((t.lstrip("0") or "0") if t.isdigit() else t for t in tokens)


def _is_fy_pair(a: str, b: str) -> bool:
    """True for consecutive years written as a financial year: 26-27, 2026-27, 2026-2027."""
    first, second = int(a), int(b)
    if len(a) == 2 and len(b) == 2:
        return second == (first + 1) % 100
    if len(a) == 4 and len(b) == 2:
        return second == (first + 1) % 100
    if len(a) == 4 and len(b) == 4:
        return second == first + 1
    return False


def inv_key_noyear(raw: str) -> str:
    """``inv_key`` without tokens that look like a year or a financial year.

    Removes ``2026``, ``26-27``, ``2026-27``, ``2026/2027``, ``FY26`` and ``FY2026-27``. Used
    only by a later matching pass, always together with the same supplier GSTIN. If nothing
    would be left, the full key is returned so a key is never empty.
    """
    text = raw.upper()
    text = _FY_PAIR.sub(lambda m: " " if _is_fy_pair(m[1], m[2]) else m[0], text)
    text = _FY_SINGLE.sub(" ", text)
    text = _YEAR.sub(" ", text)
    return inv_key(text) or inv_key(raw)


# --- hashing ----------------------------------------------------------------------------------


def _normalise_text_bytes(data: bytes) -> bytes:
    """Same bytes for the same text on every OS: no BOM, LF line endings."""
    return data.removeprefix(b"\xef\xbb\xbf").replace(b"\r\n", b"\n")


def compute_input_hash(books_path: Path, portal_path: Path, invoices_dir: Path | None) -> str:
    """sha256 over the books, the supplier statement and the invoice files (sorted by name).

    Text files are normalised first, so the hash does not change when git converts line
    endings on Windows. Invoice images are hashed as raw bytes.
    """
    h = hashlib.sha256()
    for label, path in (("books", books_path), ("portal", portal_path)):
        h.update(f"{label}\0".encode())
        h.update(_normalise_text_bytes(Path(path).read_bytes()))
        h.update(b"\0")
    if invoices_dir is not None and Path(invoices_dir).is_dir():
        files = sorted(
            p for p in Path(invoices_dir).iterdir() if p.suffix.lower() in INVOICE_FILE_SUFFIXES
        )
        for p in files:
            h.update(f"file\0{p.name}\0".encode())
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def _row_hash(
    inv: str, date: dt.date, g: str, taxable: int, rate: int, igst: int, cgst: int, sgst: int
) -> str:
    """Content hash of a normalised book row: identical entries give identical hashes."""
    text = "|".join(map(str, (inv, date.isoformat(), g, taxable, rate, igst, cgst, sgst)))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# --- field parsing ----------------------------------------------------------------------------


class _RowProblems:
    """Collects data-quality issues for one row."""

    def __init__(self, source: str, ref: str) -> None:
        self.source, self.ref, self.items = source, ref, []

    def add(self, code: str, detail: str) -> None:
        self.items.append(DQIssue(code=code, source=self.source, row_ref=self.ref, detail=detail))


def _parse_date(value: Any, field: str, p: _RowProblems) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(value).strip())
    except (ValueError, TypeError):
        p.add("bad_date", f"{field}={value!r} is not a valid YYYY-MM-DD date")
        return None


def _parse_money(value: Any, field: str, p: _RowProblems) -> int | None:
    try:
        return money.parse(value if isinstance(value, Decimal | int | str) else repr(value))
    except money.MoneyError as exc:
        p.add("bad_amount", f"{field}: {exc}")
        return None


def _parse_rate(value: Any, p: _RowProblems) -> int | None:
    try:
        d = Decimal(str(value).strip())
        if d != d.to_integral_value():
            raise ValueError
        rate = int(d)
    except (ValueError, ArithmeticError):
        p.add("bad_rate", f"rate={value!r} is not a whole number")
        return None
    if rate not in GST_RATES:
        p.add("unusual_rate", f"rate {rate}% is not a GST slab {sorted(GST_RATES)}")
        if not 0 <= rate <= 40:
            return None
    return rate


def _check_amounts(
    taxable: int, rate: int, igst: int, cgst: int, sgst: int, p: _RowProblems
) -> None:
    if igst and (cgst or sgst):
        p.add("tax_heads_mixed", "both IGST and CGST/SGST are filled in")
    if cgst != sgst:
        p.add("cgst_sgst_unequal", f"CGST {cgst} paise != SGST {sgst} paise")
    expected = taxable * rate  # paise * percent = 100x the tax in paise
    if abs((igst + cgst + sgst) * 100 - expected) > 200 * 100:
        p.add("tax_not_rate_x_taxable", "tax differs from rate x taxable value by more than Rs 2")


# --- books ------------------------------------------------------------------------------------


def read_books(path: Path, invoices_base: Path | None = None) -> tuple[list[BookRecord], list]:
    """Parse the purchase register. Returns (records, data-quality issues)."""
    text = _normalise_text_bytes(Path(path).read_bytes()).decode("utf-8", errors="strict")
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header is None:
        raise IntakeError(f"{path}: the purchase register is empty")
    header = [h.strip() for h in header]
    if header != BOOK_HEADER:
        raise IntakeError(
            f"{path}: unexpected header.\n  expected: {','.join(BOOK_HEADER)}\n"
            f"  found:    {','.join(header)}"
        )
    records: list[BookRecord] = []
    issues: list[DQIssue] = []
    first_line_of: dict[str, int] = {}
    line_no = 0
    for cells in reader:
        if not any(c.strip() for c in cells):
            continue  # blank line
        line_no += 1
        p = _RowProblems("books", f"books:line:{line_no}")
        if len(cells) != len(BOOK_HEADER):
            p.add("bad_row", f"expected {len(BOOK_HEADER)} columns, found {len(cells)}")
            issues += p.items
            continue
        row = dict(zip(BOOK_HEADER, (c.strip() for c in cells), strict=True))
        record = _book_record(row, line_no, p, invoices_base)
        if record is not None:
            if record.row_hash in first_line_of:
                p.add(
                    "duplicate_row",
                    f"same content as line {first_line_of[record.row_hash]}",
                )
            else:
                first_line_of[record.row_hash] = line_no
            records.append(record)
        else:
            p.add("row_skipped", "a key field could not be read; row left out")
        issues += p.items
    return records, issues


def _book_record(
    row: dict[str, str], line_no: int, p: _RowProblems, invoices_base: Path | None
) -> BookRecord | None:
    invoice_no = row["invoice_no"]
    if not invoice_no or not inv_key(invoice_no):
        p.add("missing_invoice_no", "invoice number is empty")
    g = gstin.normalise(row["supplier_gstin"])
    if not g:
        p.add("missing_gstin", "supplier GSTIN is empty")
    date = _parse_date(row["date"], "date", p)
    rate = _parse_rate(row["rate"], p)
    amounts = {k: _parse_money(row[k], k, p) for k in ("taxable_value", "igst", "cgst", "sgst")}
    if not invoice_no or not g or date is None or rate is None or None in amounts.values():
        return None
    valid = gstin.checksum_valid(g)
    if not valid:
        p.add("gstin_checksum", f"supplier GSTIN {g} fails the checksum")
    taxable, igst, cgst, sgst = (amounts[k] for k in ("taxable_value", "igst", "cgst", "sgst"))
    _check_amounts(taxable, rate, igst, cgst, sgst, p)
    pdf = row["pdf_file"] or None
    if pdf and invoices_base is not None and not (invoices_base / pdf).is_file():
        p.add("pdf_missing", f"{pdf} not found")
    key = inv_key(invoice_no)
    return BookRecord(
        line_no=line_no,
        row_hash=_row_hash(key, date, g, taxable, rate, igst, cgst, sgst),
        invoice_no=invoice_no,
        inv_key=key,
        inv_key_noyear=inv_key_noyear(invoice_no),
        date=date,
        supplier_name=row["supplier_name"],
        supplier_gstin=g,
        gstin_valid=valid,
        taxable_paise=taxable,
        rate=rate,
        igst_paise=igst,
        cgst_paise=cgst,
        sgst_paise=sgst,
        pdf_file=pdf,
    )


# --- supplier statement -----------------------------------------------------------------------


def read_portal(path: Path, period: str) -> tuple[list[PortalRecord], list[DQIssue], str]:
    """Parse the 2B-like statement. Returns (records, issues, buyer GSTIN).

    Numbers are read as ``Decimal`` straight from the JSON text, never as floats.
    """
    text = _normalise_text_bytes(Path(path).read_bytes()).decode("utf-8")
    try:
        doc = json.loads(text, parse_float=Decimal)
    except json.JSONDecodeError as exc:
        raise IntakeError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("invoices"), list):
        raise IntakeError(f"{path}: expected an object with an 'invoices' list")
    buyer = gstin.normalise(str(doc.get("buyer_gstin", "")))
    if not gstin.checksum_valid(buyer):
        raise IntakeError(f"{path}: buyer_gstin {buyer!r} fails the GSTIN checksum")
    if doc.get("period") != period:
        raise IntakeError(
            f"{path}: statement is for period {doc.get('period')!r}, but this run is for {period!r}"
        )
    records: list[PortalRecord] = []
    issues: list[DQIssue] = []
    for index, item in enumerate(doc["invoices"]):
        p = _RowProblems("portal", f"2b:index:{index}")
        record = _portal_record(item, index, p) if isinstance(item, dict) else None
        if not isinstance(item, dict):
            p.add("bad_row", "invoice entry is not an object")
        if record is None:
            p.add("row_skipped", "a key field could not be read; row left out")
        else:
            records.append(record)
        issues += p.items
    return records, issues, buyer


def _portal_record(item: dict[str, Any], index: int, p: _RowProblems) -> PortalRecord | None:
    missing = [k for k in PORTAL_FIELDS if k not in item]
    if missing:
        p.add("missing_fields", f"missing {', '.join(missing)}")
        return None
    invoice_no = str(item["invoice_no"]).strip()
    g = gstin.normalise(str(item["supplier_gstin"]))
    date = _parse_date(item["date"], "date", p)
    rate = _parse_rate(item["rate"], p)
    amounts = {k: _parse_money(item[k], k, p) for k in ("taxable_value", "igst", "cgst", "sgst")}
    reported = str(item["reported_in_period"])
    if not _PERIOD.match(reported):
        p.add("bad_period", f"reported_in_period={reported!r} is not YYYY-MM")
    if not invoice_no:
        p.add("missing_invoice_no", "invoice number is empty")
    if (
        not invoice_no
        or not g
        or date is None
        or rate is None
        or None in amounts.values()
        or not _PERIOD.match(reported)
    ):
        return None
    if not gstin.checksum_valid(g):
        p.add("gstin_checksum", f"supplier GSTIN {g} fails the checksum")
    taxable, igst, cgst, sgst = (amounts[k] for k in ("taxable_value", "igst", "cgst", "sgst"))
    _check_amounts(taxable, rate, igst, cgst, sgst, p)
    return PortalRecord(
        index=index,
        invoice_no=invoice_no,
        inv_key=inv_key(invoice_no),
        inv_key_noyear=inv_key_noyear(invoice_no),
        date=date,
        supplier_name=str(item["supplier_name"]).strip(),
        supplier_gstin=g,
        taxable_paise=taxable,
        rate=rate,
        igst_paise=igst,
        cgst_paise=cgst,
        sgst_paise=sgst,
        reported_in_period=reported,
    )


# --- entry point ------------------------------------------------------------------------------


def run_intake(
    books_path: Path,
    portal_path: Path,
    period: str,
    invoices_dir: Path | None = None,
    expected_buyer_gstin: str | None = None,
) -> IntakeResult:
    """Read one business-month. ``invoices_dir`` is the folder holding the invoice files; the
    books' ``pdf_file`` paths are relative to its parent (as in ``invoices/X.pdf``)."""
    if not _PERIOD.match(period):
        raise IntakeError(f"period {period!r} is not YYYY-MM")
    for path in (books_path, portal_path):
        if not Path(path).is_file():
            raise IntakeError(f"{path}: file not found")
    base = Path(invoices_dir).parent if invoices_dir is not None else None
    books, book_issues = read_books(Path(books_path), base)
    portal, portal_issues, buyer = read_portal(Path(portal_path), period)
    if expected_buyer_gstin and buyer != gstin.normalise(expected_buyer_gstin):
        raise IntakeError(
            f"{portal_path}: statement is for buyer {buyer}, expected {expected_buyer_gstin}"
        )
    return IntakeResult(
        books=books,
        portal=portal,
        dq_issues=book_issues + portal_issues,
        input_hash=compute_input_hash(Path(books_path), Path(portal_path), invoices_dir),
        buyer_gstin=buyer,
        period=period,
    )


def run_intake_dir(month_dir: Path, period: str, **kw: Any) -> IntakeResult:
    """Convenience: a folder with purchase_register.csv, gstr2b_like.json and invoices/."""
    month_dir = Path(month_dir)
    return run_intake(
        month_dir / "purchase_register.csv",
        month_dir / "gstr2b_like.json",
        period,
        invoices_dir=month_dir / "invoices",
        **kw,
    )
