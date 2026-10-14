"""An exhausted declared budget is never a transient transport retry."""

from __future__ import annotations

import pytest

from calcifier import Budget, TransientAgentError
from calcifier.agent_events import _run_cli
from calcifier.budget import BudgetExceeded, BudgetState
from calcifier.subprocesses import ProcessTimeoutError
from calcifier.trace import active_run


@pytest.mark.parametrize("expired", [False, True])
def test_cli_timeout_respects_shared_wall_budget(monkeypatch, expired):
    state = BudgetState(Budget(3, 30))

    def timeout(*args, **kwargs):
        monkeypatch.setattr(state, "remaining_wall_seconds", lambda: 0 if expired else 20)
        raise ProcessTimeoutError("native process timeout")

    monkeypatch.setattr("calcifier.agent_events.run_process", timeout)
    expected = BudgetExceeded if expired else TransientAgentError
    with active_run(budget_state=state), pytest.raises(expected):
        _run_cli(["fixture"], "", timeout=30, name="fixture")
