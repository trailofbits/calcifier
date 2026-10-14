"""Boundary behavior for shared numeric configuration checks."""

from __future__ import annotations

import math

import pytest

from calcifier.validation import is_positive_number


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        (1, True),
        (0.1, True),
        (math.nan, False),
        (0, False),
        (-1, False),
        (True, False),
        (False, False),
        ("1", False),
        (None, False),
    ],
)
def test_positive_limit_accepts_only_positive_numeric_values(value: object, accepted: bool) -> None:
    assert is_positive_number(value) is accepted


@pytest.mark.parametrize(
    ("value", "integral", "finite", "maximum", "accepted"),
    [
        (1, True, False, None, True),
        (1.0, True, False, None, False),
        (0.5, False, False, 1, True),
        (1, False, False, 1, True),
        (2, False, False, 1, False),
        (math.inf, False, False, None, True),
        (math.inf, False, True, None, False),
        (math.nan, False, True, None, False),
        pytest.param(10**400, True, False, None, True, id="large-integral-limit"),
        pytest.param(10**400, False, True, None, False, id="unrepresentable-finite-limit"),
    ],
)
def test_positive_limit_respects_optional_policies(
    value: object, integral: bool, finite: bool, maximum: float | None, accepted: bool
) -> None:
    assert is_positive_number(value, integral=integral, finite=finite, maximum=maximum) is accepted
