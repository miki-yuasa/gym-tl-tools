import numpy as np
import pytest

from gym_tl_tools.parser import Parser


@pytest.fixture
def parser():
    return Parser()


@pytest.fixture
def rob_dict():
    return {
        "psi_ba_ra": -5.0710678118654755,
        "psi_ba_rf": -6.5710678118654755,
        "psi_ba_rt": -3.5,
        "psi_ra_bf": -8.055385138137417,
        "psi_ba_ob": -1.5,
        "psi_ba_wa": -2.5,
        "psi_ra_bt": -3,
    }


@pytest.mark.parametrize(
    "spec,expected_tokens",
    [
        (
            "psi_ba_ra&!psi_ba_ob&!psi_ba_rt&!psi_ba_wa&!psi_ra_bt",
            [
                "psi_ba_ra",
                "&",
                "!",
                "psi_ba_ob",
                "&",
                "!",
                "psi_ba_rt",
                "&",
                "!",
                "psi_ba_wa",
                "&",
                "!",
                "psi_ra_bt",
            ],
        ),
        (
            "a|b",
            ["a", "|", "b"],
        ),
    ],
)
def test_tokenize(parser, spec, expected_tokens):
    tokens = parser.tokenize(spec)
    assert tokens == expected_tokens


def test_parse(parser, rob_dict):
    spec = "psi_ba_ra&!psi_ba_ob"
    tokens = parser.tokenize(spec)
    parsed = parser.parse(tokens, list(rob_dict.keys()))
    # Should be in Reverse Polish Notation
    assert parsed == ["psi_ba_ra", "psi_ba_ob", "!", "&"]


def test_evaluate(parser, rob_dict):
    parsed = ["psi_ba_ra", "psi_ba_ob", "!", "&"]
    result = parser.evaluate(parsed, rob_dict)
    # & is min, ! is negation
    expected = np.minimum(rob_dict["psi_ba_ra"], -rob_dict["psi_ba_ob"])
    assert np.isclose(result, expected)


def test_tl2rob(parser, rob_dict):
    spec = "psi_ba_ra&!psi_ba_ob"
    result = parser.tl2rob(spec, rob_dict)
    expected = np.minimum(rob_dict["psi_ba_ra"], -rob_dict["psi_ba_ob"])
    assert np.isclose(result, expected)


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("1&2", 1),
        ("1|2", 2),
        ("!3", -3),
        ("4<2", -2),
        ("4>2", 2),
    ],
)
def test_basic_ops(parser, spec, expected):
    rob_dict = {}
    result = parser.tl2rob(spec, rob_dict)
    assert np.isclose(result, expected)
