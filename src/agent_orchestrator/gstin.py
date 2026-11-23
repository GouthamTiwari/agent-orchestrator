"""GSTIN helpers: normalisation, checksum, state code, PAN part, one-edit neighbours.

A GSTIN has 15 characters: 2-digit state code, the holder's 10-character PAN, an entity number,
the letter ``Z`` by default, and a check character. The check character is a base-36 checksum
over the first 14 characters, so a one-character typo is caught by arithmetic, with no AI.

The checksum follows the reference implementation in ``data/seed/reference_checksums.py``,
which the tests use as the oracle.
"""

from __future__ import annotations

import re

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_INDEX = {ch: i for i, ch in enumerate(ALPHABET)}
_STRIP = re.compile(r"[\s\-]+")


def normalise(raw: str) -> str:
    """Uppercase and remove spaces and hyphens: ``" 36ab-cde..."`` -> ``"36ABCDE..."``."""
    return _STRIP.sub("", raw).upper()


def check_char(first14: str) -> str:
    """The check character for the first 14 characters of a GSTIN.

    Each character's base-36 value is multiplied by 1 or 2 (alternating, starting with 1);
    the product is folded back into base 36 (quotient + remainder) and summed; the check value
    is ``(36 - total mod 36) mod 36``.
    """
    if len(first14) != 14 or any(c not in _INDEX for c in first14):
        raise ValueError(f"need 14 base-36 characters, got {first14!r}")
    total = 0
    for i, ch in enumerate(first14):
        product = _INDEX[ch] * (1 if i % 2 == 0 else 2)
        total += product // 36 + product % 36
    return ALPHABET[(36 - total % 36) % 36]


def checksum_valid(gstin: str) -> bool:
    """True if ``gstin`` is 15 base-36 characters (uppercase) and its check character matches."""
    return (
        len(gstin) == 15 and all(c in _INDEX for c in gstin) and check_char(gstin[:14]) == gstin[14]
    )


def state_code(gstin: str) -> str:
    """The two-digit state code, e.g. ``"36"`` (Telangana)."""
    return gstin[:2]


def pan_part(gstin: str) -> str:
    """Characters 3 to 12: the holder's PAN. Two GSTINs of one business share it."""
    return gstin[2:12]


def edit_distance_one(a: str, b: str) -> bool:
    """True if ``b`` differs from ``a`` by exactly one substitution or one adjacent swap.

    These are the two common typing errors for a fixed-length code. Insertions and deletions
    change the length, which already fails the 15-character rule, so they are not considered.
    """
    if len(a) != len(b) or a == b:
        return False
    diff = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
    if len(diff) == 1:
        return True
    if len(diff) == 2:
        i, j = diff
        return j == i + 1 and a[i] == b[j] and a[j] == b[i]
    return False
