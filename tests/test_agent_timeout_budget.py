"""An exhausted declared budget is never a transient transport retry."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from calcifier import Budget, TransientAgentError, agent
from calcifier.agent_events import _run_cli
from calcifier.agent_providers import CommandCli
from calcifier.budget import BudgetExceeded, BudgetState
from calcifier.registry import Registry
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


@pytest.mark.parametrize("call_timeout, expected", [(None, 3.25), (2.5, 2.5)])
def test_adapter_observes_timeout_after_call_override_and_shared_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    call_timeout: float | None,
    expected: float,
) -> None:
    state = BudgetState(Budget(1, 30))
    monkeypatch.setattr(state, "remaining_wall_seconds", lambda: 3.25)
    registry = Registry()
    registry.register(
        CommandCli(
            "worker",
            (
                sys.executable,
                "-c",
                "import json, os, sys; assert sys.stdin.read(); "
                "contract = json.loads(os.environ['CALCIFIER_AGENT_CONTRACT']); "
                "print(json.dumps({'version': 1, 'output': "
                "json.dumps({'value': contract['timeout_seconds']}), 'usage': {}}))",
            ),
            default_timeout=9,
        )
    )
    with active_run(registry=registry, budget_state=state):
        result = agent(
            "worker",
            "Report the observed timeout.",
            output=float,
            timeout=call_timeout,
            working_directory=tmp_path,
            readable_paths=(),
        )
    assert result == expected
