from __future__ import annotations

import ast

import pytest

from calcifier.effect_methods import forbidden_method


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ('text.replace("old", "new")', None),
        ('text.replace("old", "new", 1)', None),
        ('text.replace(old="old", new="new")', None),
        ('transform(text).replace("old", "new")', None),
        ('destination.replace("other")', "replace"),
        ('destination.replace(target="other")', "replace"),
        ('destination.rename("other")', "rename"),
        ('destination.write_text("content")', "write_text"),
        ("text.strip()", None),
    ],
)
def test_direct_effect_syntax_distinguishes_moves_and_string_operations(expression, expected):
    call = ast.parse(expression, mode="eval").body
    assert isinstance(call, ast.Call)
    assert forbidden_method(call) == expected
