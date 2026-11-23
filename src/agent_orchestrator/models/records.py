"""Input records after normalisation (spec sections 4 and 5)."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import Field

from agent_orchestrator.models.base import Frozen, NonNegativePaise, Period


class BookRecord(Frozen):
    """One row of the buyer's purchase register."""

    line_no: int = Field(ge=1)  # 1-based line in the CSV, for evidence links
    row_hash: str  # sha256 of the normalised row, first 12 hex characters
    invoice_no: str
    inv_key: str
    inv_key_noyear: str
    date: dt.date
    supplier_name: str
    supplier_gstin: str  # normalised (uppercase, no spaces) but possibly a typo
    gstin_valid: bool
    taxable_paise: NonNegativePaise
    rate: int = Field(ge=0, le=40)  # GST slabs include 40% since 22 Sep 2025
    igst_paise: NonNegativePaise
    cgst_paise: NonNegativePaise
    sgst_paise: NonNegativePaise
    pdf_file: str | None = None


class PortalRecord(Frozen):
    """One invoice from the supplier-reported statement (the 2B-like file)."""

    index: int = Field(ge=0)  # position in the file's invoice list
    invoice_no: str
    inv_key: str
    inv_key_noyear: str
    date: dt.date
    supplier_name: str
    supplier_gstin: str
    taxable_paise: NonNegativePaise
    rate: int = Field(ge=0, le=40)  # GST slabs include 40% since 22 Sep 2025
    igst_paise: NonNegativePaise
    cgst_paise: NonNegativePaise
    sgst_paise: NonNegativePaise
    reported_in_period: Period


class DQIssue(Frozen):
    """A data-quality problem. Recorded and reported; never stops the run on its own."""

    code: str  # e.g. "bad_date", "bad_amount", "gstin_checksum"
    source: Literal["books", "portal", "pdf"]
    row_ref: str  # e.g. "books:line:12", "2b:index:5"
    detail: str
