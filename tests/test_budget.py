"""Regression tests for shared recipe-run budgets."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from calcifier.budget import Budget, BudgetExceeded, BudgetState
from calcifier.clock import SuspendAwareTimer
from calcifier.exceptions import RecipeFailure
from calcifier.trace import active_run


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "1"])
def test_budget_rejects_nonpositive_noninteger_and_boolean_call_limits(value: object) -> None:
    with pytest.raises(ValueError, match="max_agent_calls must be a positive integer"):
        Budget(max_agent_calls=value, max_wall_seconds=1)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("value", [True, False, 0, -0.1, "1", None])
def test_budget_rejects_nonpositive_nonnumeric_and_boolean_wall_limits(value: object) -> None:
    with pytest.raises(ValueError, match="max_wall_seconds must be positive"):
        Budget(max_agent_calls=1, max_wall_seconds=value)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("wall", [1, 0.1, 1e100])
def test_budget_accepts_exact_positive_boundaries(wall: float) -> None:
    budget = Budget(max_agent_calls=1, max_wall_seconds=wall)

    assert budget.max_agent_calls == 1
    assert budget.max_wall_seconds == wall


def test_wall_only_budget_counts_attempts_without_limiting_them(monkeypatch) -> None:
    state = BudgetState(Budget(max_agent_calls=None, max_wall_seconds=30))
    monkeypatch.setattr(state, "remaining_wall_seconds", lambda: 1)

    for _ in range(100):
        state.charge_attempt()

    assert state.attempted_calls == 100
    monkeypatch.setattr(state, "remaining_wall_seconds", lambda: 0)
    with pytest.raises(BudgetExceeded, match="wall-clock budget exhausted"):
        state.charge_attempt()
    assert state.attempted_calls == 100


def test_budget_is_frozen() -> None:
    budget = Budget(max_agent_calls=1, max_wall_seconds=1)

    with pytest.raises(FrozenInstanceError):
        budget.max_agent_calls = 2  # ty: ignore[invalid-assignment]


def test_live_budget_query_requires_an_active_budgeted_run() -> None:
    with pytest.raises(RecipeFailure, match="active budgeted recipe run"):
        Budget.remaining_wall_seconds()
    with active_run(), pytest.raises(RecipeFailure, match="active budgeted recipe run"):
        Budget.remaining_wall_seconds()


@pytest.mark.parametrize(
    ("monotonic_elapsed", "wall_elapsed", "remaining"),
    [(2, 2, 8), (0, 7, 3), (3, -100, 7), (0, 11, -1)],
)
def test_live_budget_query_uses_run_clock_without_charging_attempts(
    monkeypatch: pytest.MonkeyPatch,
    monotonic_elapsed: float,
    wall_elapsed: float,
    remaining: float,
) -> None:
    monkeypatch.setattr("calcifier.clock._CONTINUOUS_CLOCK", None)
    monkeypatch.setattr("calcifier.clock.time.monotonic", lambda: 10 + monotonic_elapsed)
    monkeypatch.setattr("calcifier.clock.time.time", lambda: 100 + wall_elapsed)
    state = BudgetState(
        Budget(1, 10),
        _timer=SuspendAwareTimer(10, 100, None),
    )
    with active_run(budget_state=state):
        assert Budget.remaining_wall_seconds() == remaining
        assert Budget.remaining_wall_seconds() == remaining
    assert state.attempted_calls == 0


def test_live_budget_query_restores_parent_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("calcifier.clock._CONTINUOUS_CLOCK", None)
    monkeypatch.setattr("calcifier.clock.time.monotonic", lambda: 10)
    monkeypatch.setattr("calcifier.clock.time.time", lambda: 100)
    parent = BudgetState(Budget(1, 30))
    child = BudgetState(Budget(1, 7))
    with active_run(budget_state=parent):
        assert Budget.remaining_wall_seconds() == 30
        with active_run(budget_state=child):
            assert Budget.remaining_wall_seconds() == 7
        assert Budget.remaining_wall_seconds() == 30
    assert parent.attempted_calls == child.attempted_calls == 0


def test_budget_state_repr_excludes_timer_and_lock() -> None:
    state = BudgetState(Budget(max_agent_calls=1, max_wall_seconds=1))

    rendered = repr(state)
    assert "_timer" not in rendered
    assert "_lock" not in rendered


def test_wall_budget_allows_a_fractional_positive_remainder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("calcifier.clock._CONTINUOUS_CLOCK", None)
    monkeypatch.setattr("calcifier.clock.time.monotonic", lambda: 10.5)
    monkeypatch.setattr("calcifier.clock.time.time", lambda: 100.5)
    state = BudgetState(
        Budget(max_agent_calls=1, max_wall_seconds=1),
        _timer=SuspendAwareTimer(
            monotonic_started=10.0,
            wall_started=100.0,
            continuous_started=None,
        ),
    )

    assert state.remaining_wall_seconds() == 0.5
    state.check_wall()


def test_whole_run_budget_counts_suspended_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """A paused monotonic clock must not extend the declared wall budget."""
    monkeypatch.setattr("calcifier.clock._CONTINUOUS_CLOCK", None)
    monkeypatch.setattr("calcifier.clock.time.monotonic", lambda: 50.0)
    monkeypatch.setattr("calcifier.clock.time.time", lambda: 1_500.0)
    state = BudgetState(
        Budget(max_agent_calls=2, max_wall_seconds=300),
        _timer=SuspendAwareTimer(
            monotonic_started=50.0,
            wall_started=1_000.0,
            continuous_started=None,
        ),
    )

    assert state.remaining_wall_seconds() == -200
    with pytest.raises(BudgetExceeded, match=r"wall-clock budget exhausted \(300s\)"):
        state.charge_attempt()
    assert state.attempted_calls == 0


def test_whole_run_budget_uses_continuous_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A continuous platform clock takes precedence over an unchanged fallback clock."""
    monkeypatch.setattr("calcifier.clock._CONTINUOUS_CLOCK", lambda: 410.0)
    monkeypatch.setattr("calcifier.clock.time.monotonic", lambda: 10.0)
    monkeypatch.setattr("calcifier.clock.time.time", lambda: 1_000.0)
    state = BudgetState(
        Budget(max_agent_calls=1, max_wall_seconds=60),
        _timer=SuspendAwareTimer(
            monotonic_started=10.0,
            wall_started=1_000.0,
            continuous_started=350.0,
        ),
    )

    assert state.remaining_wall_seconds() == 0
    with pytest.raises(BudgetExceeded, match=r"wall-clock budget exhausted \(60s\)"):
        state.check_wall()


def test_whole_run_budget_still_charges_calls_before_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("calcifier.clock._CONTINUOUS_CLOCK", None)
    monkeypatch.setattr("calcifier.clock.time.monotonic", lambda: 15.0)
    monkeypatch.setattr("calcifier.clock.time.time", lambda: 105.0)
    state = BudgetState(
        Budget(max_agent_calls=1, max_wall_seconds=30),
        _timer=SuspendAwareTimer(
            monotonic_started=10.0,
            wall_started=100.0,
            continuous_started=None,
        ),
    )

    state.charge_attempt()

    assert state.attempted_calls == 1
    assert state.remaining_wall_seconds() == 25
    with pytest.raises(BudgetExceeded, match="agent-call budget exhausted"):
        state.charge_attempt()


@pytest.mark.parametrize(
    "value", [math.nan, math.inf, -math.inf, pytest.param(10**400, id="unrepresentable-wall")]
)
def test_budget_rejects_unbounded_or_unrepresentable_wall_limits(value: float) -> None:
    with pytest.raises(ValueError, match="max_wall_seconds"):
        Budget(max_agent_calls=1, max_wall_seconds=value)
