"""Structured-output known answers."""

from __future__ import annotations

import json
import re
from typing import Annotated, Any

import pytest

from calcifier import (
    word_count,
)
from calcifier.outputs import OutputParseError, parse_response, strict_schema_for

# -- structured output word counts ----------------------------------------------


@pytest.mark.parametrize(
    ("minimum", "maximum", "text"),
    [
        (0, None, ""),
        (0, 0, ""),
        (2, 2, "two words"),
        (1, 3, "one"),
        (1, 3, "one two three"),
    ],
)
def test_word_count_accepts_values_at_each_inclusive_limit(
    minimum: int, maximum: int | None, text: str
) -> None:
    annotation = Annotated[str, word_count(minimum, maximum)]

    assert parse_response(json.dumps(text), annotation) == text


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "must contain at least 1 words"),
        ("one two three four", "must contain at most 3 words"),
    ],
)
def test_word_count_rejects_values_one_word_outside_the_limits(text: str, message: str) -> None:
    with pytest.raises(OutputParseError, match=re.escape(message)):
        parse_response(json.dumps(text), Annotated[str, word_count(1, 3)])


def test_word_count_bounds_are_visible_in_the_provider_schema() -> None:
    annotation = Annotated[str, word_count(2, 4)]

    assert strict_schema_for(annotation)["description"] == "At least 2 words; at most 4 words."


_MINIMUM_ERROR = "word_count() minimum must be a non-negative integer"
_MAXIMUM_ERROR = "word_count() maximum must be at least the minimum"


@pytest.mark.parametrize(
    ("minimum", "maximum", "message"),
    [
        (-1, None, _MINIMUM_ERROR),
        (True, None, _MINIMUM_ERROR),
        (1.0, None, _MINIMUM_ERROR),
        (2, 1, _MAXIMUM_ERROR),
        (0, -1, _MAXIMUM_ERROR),
        (1, True, _MAXIMUM_ERROR),
        (1, 1.5, _MAXIMUM_ERROR),
    ],
)
def test_word_count_rejects_each_invalid_limit(minimum: Any, maximum: Any, message: str) -> None:
    with pytest.raises(ValueError) as raised:
        word_count(minimum, maximum)

    assert str(raised.value) == message
