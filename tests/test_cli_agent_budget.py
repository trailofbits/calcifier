"""A runtime call override also bounds automatic structured-output retries."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from calcifier import Budget, agent, cli, recipe
from calcifier.registry import Registry
from tests.support.agents import StubProvider


@recipe(agents=("reviewer",), budget=Budget(None, 30))
def bounded_review(target: str) -> bool:
    return agent("reviewer", "Return a neutral verdict", working_directory=target, output=bool)


@pytest.mark.parametrize("entry_point", ["exec_budget", "recipe_cli"])
def test_cli_call_limit_blocks_retry_before_provider_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry_point: str
) -> None:
    provider = StubProvider(default="invalid JSON")
    provider.name = "reviewer"
    invocation = Mock(wraps=provider.invoke)
    monkeypatch.setattr(provider, "invoke", invocation)
    registry = Registry(configuration_hash="neutral-budget-test")
    registry.register(provider)
    if entry_point == "exec_budget":
        budget = cli._budget_for_args(
            bounded_review, argparse.Namespace(max_agent_calls=1, max_wall_seconds=None)
        )
        run = bounded_review.run({"target": str(tmp_path)}, registry=registry, budget=budget)
        assert not run.ok
        assert "agent-call budget exhausted" in run.render()
    else:
        config = tmp_path / "agents.json"
        config.write_bytes(
            json.dumps(
                {
                    "version": 1,
                    "agents": {
                        "reviewer": {
                            "target": {"command": [sys.executable]},
                            "runtime": {"kind": "host"},
                            "timeout_seconds": 5,
                        }
                    },
                }
            ).encode("utf-8")
            + b"\n"
        )
        module = importlib.import_module("calcifier.recipe")
        monkeypatch.setattr(module, "configured_registry", lambda *args: registry)
        assert (
            bounded_review.cli(
                ["--target", str(tmp_path), "--agent-config", str(config), "--max-agent-calls", "1"]
            )
            == 1
        )
    assert invocation.call_count == 1
