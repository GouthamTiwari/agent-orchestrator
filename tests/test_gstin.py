import importlib.util
import random
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_orchestrator import gstin

# The kit's reference implementation is the test oracle (kept byte for byte in data/seed).
_REF_PATH = Path(__file__).resolve().parents[1] / "data" / "seed" / "reference_checksums.py"
_spec = importlib.util.spec_from_file_location("reference_checksums", _REF_PATH)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

SEED_BUYER = "36PKWZU0959A1ZQ"
SEED_SUPPLIER = "36TSDKP2716L1ZA"

valid_gstins = st.builds(
    lambda seed, state: ref.make_gstin(random.Random(seed), state),
    st.integers(min_value=0, max_value=2**32),
    st.sampled_from(["01", "07", "24", "27", "29", "33", "36", "37"]),
)


def test_seed_gstins_are_valid():
    assert gstin.checksum_valid(SEED_BUYER)
    assert gstin.checksum_valid(SEED_SUPPLIER)


@settings(max_examples=1000)
@given(valid_gstins)
def test_random_valid_gstins_pass_and_agree_with_reference(g):
    assert gstin.checksum_valid(g)
    assert gstin.check_char(g[:14]) == ref.gstin_check_char(g[:14])


@settings(max_examples=1000)
@given(valid_gstins, st.integers(min_value=0, max_value=14), st.sampled_from(gstin.ALPHABET))
def test_any_single_substitution_is_caught(g, pos, ch):
    # Weight 2 folded into base 36 is a permutation of 0..35, so changing one character always
    # changes the total mod 36: the result can never be another valid GSTIN.
    if g[pos] == ch:
        return
    typo = g[:pos] + ch + g[pos + 1 :]
    assert not gstin.checksum_valid(typo)
    assert ref.gstin_valid(typo) == gstin.checksum_valid(typo)


@given(st.text(alphabet=gstin.ALPHABET + "abc-/ ", min_size=0, max_size=20))
def test_agrees_with_reference_on_arbitrary_strings(s):
    assert gstin.checksum_valid(s) == ref.gstin_valid(s)


def test_lowercase_is_invalid_until_normalised():
    assert not gstin.checksum_valid(SEED_SUPPLIER.lower())
    assert gstin.checksum_valid(gstin.normalise(" 36tsdkp-2716 l1za "))


def test_check_char_rejects_bad_input():
    with pytest.raises(ValueError):
        gstin.check_char("36ABC")
    with pytest.raises(ValueError):
        gstin.check_char("36tsdkp2716l1z")


def test_state_code_and_pan_part():
    assert gstin.state_code(SEED_SUPPLIER) == "36"
    assert gstin.pan_part(SEED_SUPPLIER) == "TSDKP2716L"


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("36TSDKP2716L1ZA", "36TSDKP2716L1ZB", True),  # one substitution
        ("36TSDKP2716L1ZA", "36TSDKP2761L1ZA", True),  # adjacent swap 16 -> 61
        ("36TSDKP2716L1ZA", "36TSDKP2716L1ZA", False),  # identical
        ("36TSDKP2716L1ZA", "36TSDKP2617L1ZA", False),  # non-adjacent swap
        ("36TSDKP2716L1ZA", "36TSDKP2716L2ZB", False),  # two substitutions
        ("36TSDKP2716L1ZA", "36TSDKP2716L1Z", False),  # different length
    ],
)
def test_edit_distance_one(a, b, expected):
    assert gstin.edit_distance_one(a, b) is expected
    assert gstin.edit_distance_one(b, a) is expected
