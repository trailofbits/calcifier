"""Shared CLI budget override behavior."""

from __future__ import annotations

import math

import pytest

from calcifier.budget import Budget
from calcifier.cli_options import shorten_budget


def test_wall_override_without_declared_budget_has_no_call_limit() -> None:
    assert shorten_budget(None, None) is None
    assert shorten_budget(None, 5) == Budget(max_agent_calls=None, max_wall_seconds=5)


@pytest.mark.parametrize("override", [math.inf, math.nan, -math.inf])
def test_budget_override_rejects_nonfinite_values(override: float) -> None:
    with pytest.raises(ValueError, match="--max-wall-seconds must be positive"):
        shorten_budget(Budget(2, 60), override)
