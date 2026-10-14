"""Shared wall-clock and optional agent-call limits for recipe runs."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .clock import SuspendAwareTimer
from .exceptions import RecipeFailure
from .validation import is_positive_number


class BudgetExceeded(RecipeFailure):
    """Raised when a recipe exhausts its wall-clock or agent-call budget."""


@dataclass(frozen=True)
class Budget:
    max_agent_calls: int | None
    max_wall_seconds: float

    def __post_init__(self) -> None:
        if self.max_agent_calls is not None and not is_positive_number(
            self.max_agent_calls, integral=True
        ):
            raise ValueError("max_agent_calls must be a positive integer or None")
        if not is_positive_number(self.max_wall_seconds, finite=True):
            raise ValueError("max_wall_seconds must be positive and finite")


@dataclass
class BudgetState:
    budget: Budget
    attempted_calls: int = 0
    _timer: SuspendAwareTimer = field(default_factory=SuspendAwareTimer.start, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def charge_attempt(self) -> None:
        with self._lock:
            self.check_wall()
            limit = self.budget.max_agent_calls
            if limit is not None and self.attempted_calls >= limit:
                raise BudgetExceeded(
                    f"agent-call budget exhausted ({self.budget.max_agent_calls} attempts)"
                )
            self.attempted_calls += 1

    def remaining_wall_seconds(self) -> float:
        return self.budget.max_wall_seconds - self._timer.elapsed()

    def check_wall(self) -> None:
        if self.remaining_wall_seconds() <= 0:
            raise BudgetExceeded(f"wall-clock budget exhausted ({self.budget.max_wall_seconds:g}s)")
