"""Public-neutral, provider-free regressions for runtime review findings."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from calcifier import (
    Budget,
    agent,
    emit,
    parallel,
    recipe,
    step,
    write_text,
)
from calcifier.agent_events import AgentActivity, AgentResult, _run_cli, capture_agent_activity
from calcifier.budget import BudgetState
from calcifier.filesystem_contract import AgentError
from calcifier.outputs import _extract_json
from calcifier.recipe import (
    RecipeRun,
    _checkpoint_due,
    _checkpoint_writer,
)
from calcifier.registry import Registry
from calcifier.tools import Capability, ToolRegistry, tool
from calcifier.trace import RunContext, StepRecord, active_run, fork_run_context
from calcifier.trace_export import _trace_validator
from tests.support.agents import StubProvider


class FakeTool:
    def available(self, name: str) -> bool:
        return True

    def invoke(self, name: str, **kwargs: Any) -> str:
        return name


class MeteredProvider(StubProvider):
    name = "worker"

    def __init__(self) -> None:
        super().__init__(default="1")
        self.calls = 0

    def invoke(self, prompt: str, **kwargs: Any) -> AgentResult:
        self.calls += 1
        return AgentResult(
            "1", "1", "", 0, "fake", usage={"input_tokens": 100, "output_tokens": 20}
        )


def registry_for(provider: StubProvider) -> Registry:
    registry = Registry()
    registry.register(provider)
    return registry


@pytest.mark.parametrize("allowed", [False, True])
def test_parallel_preserves_capability_allowlist(allowed: bool) -> None:
    @step
    def worker(value: int) -> str:
        return tool("Extra")

    @recipe(capabilities=("Extra",) if allowed else ("Read",))
    def main() -> list[str]:
        return parallel(worker, [1, 2])

    registry = ToolRegistry()
    registry.register(Capability("Extra", "fake"), FakeTool())
    result = main.run(tool_registry=registry)
    assert result.ok is allowed
    if allowed:
        assert result.output == ["Extra", "Extra"]
    else:
        assert "not declared" in str(result.error)


def test_failed_parallel_workers_keep_every_record_and_usage(tmp_path: Path) -> None:
    @step
    def before() -> bool:
        emit("parent", name="parent")
        return True

    @step
    def worker(value: int) -> int:
        response = agent("worker", "one", working_directory=".", readable_paths=())
        emit(value, name="worker")
        if value == 2:
            raise ValueError("failed worker")
        return response

    @recipe(agents=("worker",), budget=Budget(2, 10))
    def main() -> list[int]:
        before()
        return parallel(worker, [1, 2], max_workers=1)

    provider = MeteredProvider()
    path = tmp_path / "trace.json"
    activity = AgentActivity()
    with capture_agent_activity(activity):
        result = main.run(registry=registry_for(provider), trace_path=path)
    assert activity.attempts == 2
    assert activity.usage["input_tokens"] == 200
    assert result.ok is False
    assert len(result.records) == 3
    assert [item.value for item in result.artifacts] == ["parent", 1, 2]
    assert result.budget is not None and result.budget["attempted_agent_calls"] == 2
    assert (
        sum(
            call.usage["input_tokens"] or 0
            for record in result.records
            for call in record.agent_calls
        )
        == 200
    )
    trace = json.loads(path.read_text())
    assert len(trace["steps"]) == 3
    assert trace["state"] == "failed"
    assert provider.calls == 2


@pytest.mark.parametrize("failure", ["provider", "invalid_answer"])
def test_failed_agent_preserves_partial_artifact_without_finishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    provider = StubProvider(default='{"value": 42}')
    provider.name = "worker"
    if failure == "provider":

        def reject(prompt: str, **kwargs: Any) -> AgentResult:
            raise AgentError("neutral provider unavailable")

        monkeypatch.setattr(provider, "invoke", reject)

    @step
    def prepare() -> str:
        return str(write_text("partial.txt", "prepared café\n"))

    @step
    def inspect() -> str:
        return agent(
            "worker",
            "Return the required result.",
            working_directory=tmp_path,
            readable_paths=(),
            output=str,
        )

    @step
    def finish(answer: str) -> str:
        return str(write_text("result.txt", answer))

    output = tmp_path / "out"

    @recipe(agents=("worker",), capabilities=("Write",), output_root=output, budget=Budget(2, 30))
    def run() -> str:
        prepare()
        return finish(inspect())

    trace_path = tmp_path / "trace.json"
    result = run.run(registry=registry_for(provider), trace_path=trace_path)

    assert not result.ok
    assert result.output is None
    assert result.error
    if failure == "provider":
        assert "neutral provider unavailable" in result.error
    assert (output / "partial.txt").read_bytes() == b"prepared caf\xc3\xa9\n"
    assert not (output / "result.txt").exists()
    assert [record.name for record in result.records] == ["prepare", "inspect"]
    assert result.records[-1].error
    assert result.budget is not None
    assert result.budget["attempted_agent_calls"] == (1 if failure == "provider" else 2)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["state"] == "failed"
    assert [record["name"] for record in trace["steps"]] == ["prepare", "inspect"]


@given(
    st.recursive(
        st.none() | st.booleans() | st.integers() | st.text(),
        lambda s: st.lists(s, max_size=5) | st.dictionaries(st.text(), s, max_size=5),
        max_leaves=25,
    )
)
def test_embedded_json_preserves_the_entire_nested_answer(value: Any) -> None:
    expected = {"items": [value]}
    assert _extract_json("Answer: " + json.dumps(expected)) == expected


def test_embedded_json_chooses_last_outer_block_not_inner_object() -> None:
    assert _extract_json('draft {"old": []} final [{"new": "{literal}"}]') == [{"new": "{literal}"}]
    assert _extract_json('broken {prefix answer {"new": [1]} trailing') == {"new": [1]}


def test_every_retry_timeout_uses_the_remaining_wall_budget() -> None:
    class Timer:
        seconds = 0.0

        def elapsed(self) -> float:
            return self.seconds

    timer = Timer()
    timeouts = []

    class Retry(MeteredProvider):
        def invoke(self, prompt: str, *, timeout=None, **kwargs: Any) -> AgentResult:
            timeouts.append(timeout)
            if len(timeouts) == 1:
                timer.seconds = 6.0
                return AgentResult("bad", "bad", "", 0, "fake")
            return super().invoke(prompt)

    state = BudgetState(Budget(2, 10), _timer=timer)  # ty: ignore[invalid-argument-type]
    with active_run(registry=registry_for(Retry()), budget_state=state):
        assert (
            agent("worker", "one", output=int, timeout=10, working_directory=".", readable_paths=())
            == 1
        )
    assert timeouts == [10, 4]


@pytest.mark.parametrize(
    "event",
    [
        {"type": "turn.failed", "usage": {"input_tokens": 100, "output_tokens": 20}},
        {"type": "error"},
    ],
)
def test_nonzero_native_exit_retains_partial_usage(event: dict[str, Any]) -> None:
    activity = AgentActivity()
    with capture_agent_activity(activity), pytest.raises(AgentError, match="exited 1"):
        _run_cli(
            [
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1]); sys.exit(1)",
                json.dumps(event),
            ],
            "",
            timeout=5,
            name="fake",
            native_format="codex",
        )
    assert activity.usage["input_tokens"] == event.get("usage", {}).get("input_tokens")
    assert activity.usage["output_tokens"] == event.get("usage", {}).get("output_tokens")
    assert activity.usage_reported_attempts == {}


def test_fork_snapshot_keeps_policy_and_publishes_to_spawning_step() -> None:
    parent = RunContext(allowed_capabilities=frozenset({"Read"}))
    spawning = StepRecord("spawn", {})
    parent.record_stack.append(spawning)
    child = fork_run_context(parent, lane="child")
    parent.record_stack.clear()
    child.records.append(StepRecord("child", {}))
    assert child.checkpoint is not None
    child.checkpoint("worker_complete", child)
    child.checkpoint("worker_complete", child)
    assert child.allowed_capabilities == frozenset({"Read"})
    assert child.frames is not parent.frames
    assert [r.name for r in spawning.children] == ["child"]
    assert parent.records == []
    assert fork_run_context(None).parallel_worker is True


def test_checkpoints_coalesce_routine_events_but_flush_important_boundaries(
    tmp_path: Path, monkeypatch
) -> None:
    ticks = [1.0]
    writes = []
    monkeypatch.setattr(sys.modules["calcifier.recipe"].time, "monotonic", lambda: ticks[0])
    monkeypatch.setattr(
        sys.modules["calcifier.recipe"],
        "write_trace",
        lambda record, path, *, checkpoint: writes.append((len(record.records), checkpoint)),
    )
    writer = _checkpoint_writer(RecipeRun("main", {}), tmp_path / "trace.json", 0)
    assert writer is not None
    ctx = RunContext()
    writer("start", ctx)
    for _ in range(10):
        ctx.records.append(StepRecord("small", {}))
        writer("step", ctx)
    assert writes == [(0, True)]
    ticks[0] += 0.25
    writer("step", ctx)
    for event in ("agent", "attempt", "usage", "model"):
        writer(event, ctx)
    writer("side-effect", ctx)
    writer("artifact", ctx)
    writer("error", ctx)
    # step and attempt flush; agent, usage, and model coalesce; the rest flush.
    assert [count for count, _ in writes] == [0, 10, 10, 10, 10, 10]
    assert all(checkpoint for _, checkpoint in writes)
    assert _trace_validator() is _trace_validator()


@pytest.mark.parametrize("event", ["step", "prompt", "worker_complete", "agent", "usage", "model"])
def test_routine_checkpoint_events_wait_for_the_interval(event: str) -> None:
    assert _checkpoint_due(event, 0.249) is False
    assert _checkpoint_due(event, 0.25) is True


@pytest.mark.parametrize(
    "event",
    [
        "attempt",
        "side-effect",
        "side-effect-complete",
        "side-effect-failed",
        "artifact",
        "error",
        "interrupted",
        "run",
        "parallel_complete",
    ],
)
def test_host_effect_and_boundary_checkpoints_flush_immediately(event: str) -> None:
    assert _checkpoint_due(event, 0.0) is True
