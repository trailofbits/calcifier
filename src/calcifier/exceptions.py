"""Shared public exception types."""

from __future__ import annotations

from enum import StrEnum


class ErrorFamily(StrEnum):
    SETUP = "setup"
    RUNTIME = "runtime"
    COMPILATION = "compilation"
    VERIFICATION = "verification"


class CalcifierError(Exception):
    """Base for errors Calcifier can report without a Python traceback."""

    family = ErrorFamily.RUNTIME


class RecipeError(CalcifierError):
    """Raised when a recipe file cannot be loaded or doesn't expose a recipe."""

    family = ErrorFamily.SETUP


class RecipeFailure(CalcifierError):  # noqa: N818 - public surface; not just an "Error"
    """Raised by ``fail()`` to abort a recipe with an explicit reason."""
