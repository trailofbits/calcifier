"""Shared checks for positive numeric limits."""

from __future__ import annotations

import math


def is_positive_number(
    value: object, *, integral: bool = False, finite: bool = False, maximum: float | None = None
) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    if integral and not isinstance(value, int):
        return False
    if not value > 0:
        return False
    if finite:
        try:
            if not math.isfinite(value):
                return False
        except OverflowError:
            return False
    return maximum is None or value <= maximum


def bounded_timeout(configured: float | None, remaining: float) -> float:
    """Bound a call by positive time remaining, retaining a minimum subprocess timeout."""
    remaining = max(remaining, 0.001)
    return remaining if configured is None else min(configured, remaining)
