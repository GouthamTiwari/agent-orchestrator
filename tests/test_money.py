from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agent_orchestrator import money
from agent_orchestrator.money import MoneyError


@pytest.mark.parametrize(
    ("text", "paise"),
    [
        ("4257.0", 425700),
        ("4257", 425700),
        ("1233.75", 123375),
        ("0.05", 5),
        ("0", 0),
        (" 47300.00 ", 4730000),
        ("8262.50", 826250),
        ("1.000", 100),  # trailing zeros are not extra precision
    ],
)
def test_parse_text(text, paise):
    assert money.parse(text) == paise


def test_parse_decimal_and_whole_rupee_int():
    assert money.parse(Decimal("4257.0")) == 425700
    assert money.parse(18) == 1800


@pytest.mark.parametrize("bad", [4257.0, 0.1, True, None, [1]])
def test_parse_rejects_non_text_types(bad):
    with pytest.raises(MoneyError):
        money.parse(bad)


@pytest.mark.parametrize(
    "bad", ["1.005", "12.345", "abc", "", "  ", "NaN", "Infinity", "1e400x", "1e30", "1e400"]
)
def test_parse_rejects_bad_text(bad):
    with pytest.raises(MoneyError):
        money.parse(bad)


def test_negative_only_when_allowed():
    with pytest.raises(MoneyError):
        money.parse("-10.00")
    assert money.parse("-10.00", allow_negative=True) == -1000


@pytest.mark.parametrize(
    ("paise", "text"),
    [
        (425700, "₹4,257.00"),
        (0, "₹0.00"),
        (5, "₹0.05"),
        (99999, "₹999.99"),
        (100000, "₹1,000.00"),
        (11800000, "₹1,18,000.00"),
        (1234567890, "₹1,23,45,678.90"),
        (-150, "-₹1.50"),
    ],
)
def test_format_inr_uses_indian_grouping(paise, text):
    assert money.format_inr(paise) == text


def test_format_rejects_float():
    with pytest.raises(MoneyError):
        money.format_inr(4257.0)


@given(st.integers(min_value=0, max_value=10**14 - 1))
def test_format_then_parse_round_trips(paise):
    text = money.format_inr(paise).removeprefix("₹").replace(",", "")
    assert money.parse(text) == paise


@given(st.decimals(min_value=0, max_value=10**9, places=2, allow_nan=False, allow_infinity=False))
def test_parse_matches_exact_decimal_arithmetic(amount):
    assert money.parse(str(amount)) == int(amount * 100)


def test_tolerance_is_inclusive_one_rupee():
    assert money.TOLERANCE_PAISE == 100
    assert money.within_tolerance(10000, 10100)
    assert not money.within_tolerance(10000, 10101)
