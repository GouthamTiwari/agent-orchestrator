"""Money as integer paise (1 rupee = 100 paise).

A float never touches money: `0.1 + 0.2 != 0.3` in binary floating point, and a reconciliation
that compares amounts to the paisa cannot afford that. Amounts are parsed from their text (or
from a `Decimal`) straight into an `int` number of paise, and compared as integers.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

# Differences up to and including Rs 1.00 are treated as rounding (spec section 6.3).
TOLERANCE_PAISE = 100

# Sanity ceiling for one amount (Rs 1 lakh crore). Larger values are data errors, and the cap
# keeps Decimal arithmetic inside its default 28-digit precision.
MAX_RUPEES = Decimal(10) ** 12


class MoneyError(ValueError):
    """The value is not a valid rupee amount."""


def parse(value: str | Decimal | int, *, allow_negative: bool = False) -> int:
    """Convert a rupee amount to integer paise.

    Accepts a string such as ``"4257.0"`` or ``"1233.75"``, a ``Decimal`` (for example from
    ``json.loads(text, parse_float=Decimal)``) or an ``int`` meaning whole rupees.
    Rejects floats, booleans, more than two decimal places, and negatives unless allowed.

    >>> parse("4257.0")
    425700
    """
    if isinstance(value, (bool, float)):
        raise MoneyError(f"money must be text or Decimal, not {type(value).__name__}: {value!r}")
    if isinstance(value, int):
        amount = Decimal(value)
    elif isinstance(value, Decimal):
        amount = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise MoneyError("empty amount")
        try:
            amount = Decimal(text)
        except InvalidOperation as exc:
            raise MoneyError(f"not a number: {value!r}") from exc
    else:
        raise MoneyError(f"unsupported money type {type(value).__name__}: {value!r}")

    if not amount.is_finite():
        raise MoneyError(f"not a finite amount: {value!r}")
    if abs(amount) >= MAX_RUPEES:
        raise MoneyError(f"amount too large: {value!r}")
    if amount != amount.quantize(Decimal("0.01")):
        raise MoneyError(f"more than 2 decimal places: {value!r}")
    if amount < 0 and not allow_negative:
        raise MoneyError(f"negative amount: {value!r}")
    return int(amount * 100)


def format_inr(paise: int) -> str:
    """Format paise as rupees with Indian digit grouping (lakh, crore).

    ``425700`` -> ``"₹4,257.00"``; ``11800000`` -> ``"₹1,18,000.00"``; negatives get a leading
    minus: ``-150`` -> ``"-₹1.50"``.
    """
    if isinstance(paise, bool) or not isinstance(paise, int):
        raise MoneyError(f"paise must be an int, not {type(paise).__name__}")
    sign = "-" if paise < 0 else ""
    rupees, rem = divmod(abs(paise), 100)
    digits = str(rupees)
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups) + "," + tail
    return f"{sign}₹{digits}.{rem:02d}"


def within_tolerance(a_paise: int, b_paise: int, tolerance: int = TOLERANCE_PAISE) -> bool:
    """True when the two amounts differ by at most ``tolerance`` paise (inclusive)."""
    return abs(a_paise - b_paise) <= tolerance
