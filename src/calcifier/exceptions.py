"""Shared public exception types."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal


class ErrorFamily(StrEnum):
    SETUP = "setup"
    RUNTIME = "runtime"
    COMPILATION = "compilation"
    VERIFICATION = "verification"


class CalcifierError(Exception):
    """Base for errors Calcifier can report without a Python traceback."""

    family = ErrorFamily.RUNTIME


class RecipeError(CalcifierError):
    """Raised for invalid recipe setup, operations, or command results.

    ``command_failure`` identifies a command launch failure or local timeout;
    other errors leave it ``None``. Shared budget exhaustion is a separate error.
    """

    family = ErrorFamily.SETUP
    command_failure: Literal["launch", "timeout"] | None = None


class RecipeFailure(CalcifierError):  # noqa: N818 - public surface; not just an "Error"
    """Raised by ``fail()`` to abort a recipe with an explicit reason."""


class UnmaterializedArtifactError(RecipeError):
    """A declared subprocess output is unavailable after compiler dry-run suppression."""
