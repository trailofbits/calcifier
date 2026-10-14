"""AST assertions and source positions shared by compiler tests."""

from __future__ import annotations

import ast


def assert_same_program(actual: str, expected: str) -> None:
    assert ast.dump(ast.parse(actual), include_attributes=False) == ast.dump(
        ast.parse(expected), include_attributes=False
    )


def source_line(source: str, text: str) -> int:
    return next(number for number, line in enumerate(source.splitlines(), start=1) if text in line)
