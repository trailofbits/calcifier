"""Shared compiler limits are explicit and separate from per-call and runtime limits."""

from __future__ import annotations

import json

import pytest

from calcifier import cli
from calcifier.agent_events import AgentActivity
from calcifier.artifact import prepare_artifact
from calcifier.baseline import is_baseline
from calcifier.compiler_activity import compiler_activity_payload
from calcifier.diagnostics import CalcifyError
from calcifier.recipe_draft import draft_response
from calcifier.registry import Registry
from tests.support.recipes import _GOOD_RECIPE
from tests.test_compiler_deadline import timed_compilation


@pytest.mark.parametrize("limit", [None, 1200])
def test_slow_compilation_can_finish_with_default_or_explicit_shared_budget(
    tmp_path, monkeypatch, limit
):
    skill, registry, compiler, elapsed = timed_compilation(
        tmp_path, monkeypatch, [draft_response(_GOOD_RECIPE)]
    )
    invoke = compiler.invoke
    durations = iter([400, 450, 100])
    timeouts = []

    def slow(prompt, *, timeout=None, **kwargs):
        timeouts.append(timeout)
        elapsed[0] += next(durations)
        return invoke(prompt, timeout=timeout, **kwargs)

    monkeypatch.setattr(compiler, "invoke", slow)
    activity = AgentActivity()
    options = {} if limit is None else {"compilation_timeout": limit}
    artifact = prepare_artifact(skill, registry, activity=activity, **options)
    assert not is_baseline(artifact.source)
    assert compiler_activity_payload(activity)["outcome"] == "compiled"
    expected = 3600 if limit is None else limit
    assert timeouts == [expected]
    budget = next(e for e in activity.events if e["phase"] == "compiler.budget")
    assert json.loads(str(budget["detail"])) == {
        "max_model_calls": 5,
        "max_wall_seconds": expected,
    }


@pytest.mark.parametrize("command", ["calcify", "update", "build-graph"])
def test_shared_limit_is_a_separate_cli_option(command):
    args = cli.build_parser().parse_args([command, "input", "--compilation-timeout", "1200"])
    assert args.compilation_timeout == 1200


@pytest.mark.parametrize("command", ["calcify", "update", "build-graph"])
@pytest.mark.parametrize("limit", ["0", "-1", "1.5"])
def test_cli_rejects_invalid_shared_limits_before_work(command, limit):
    with pytest.raises(SystemExit) as caught:
        cli.build_parser().parse_args([command, "input", "--compilation-timeout", limit])
    assert caught.value.code == 2


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_api_rejects_invalid_shared_limits_before_provider(tmp_path, monkeypatch, limit):
    skill, registry, compiler, _ = timed_compilation(tmp_path, monkeypatch, [])
    with pytest.raises(CalcifyError, match="compilation timeout must be a positive integer"):
        prepare_artifact(skill, registry, compilation_timeout=limit)
    assert compiler.calls == []


@pytest.mark.parametrize("command", ["calcify", "update"])
def test_cli_passes_shared_limit_through_artifact_preparation(tmp_path, monkeypatch, command):
    skill, registry, compiler, _ = timed_compilation(
        tmp_path, monkeypatch, [draft_response(_GOOD_RECIPE)]
    )
    target = tmp_path / ".calcifier" / "recipe.py"
    if command == "update":
        target.parent.mkdir()
        target.write_bytes(_GOOD_RECIPE.encode())
    monkeypatch.setattr(cli, "_registry_for_args", lambda *_: registry)
    invoke = compiler.invoke
    timeouts = []

    def observed(prompt, *, timeout=None, **kwargs):
        timeouts.append(timeout)
        return invoke(prompt, timeout=timeout, **kwargs)

    monkeypatch.setattr(compiler, "invoke", observed)
    assert cli.main([command, str(skill), "--compilation-timeout", "1200"]) == 0
    assert timeouts == [1200]


def test_interrupted_repair_checkpoints_wall_time_and_known_partial_usage(tmp_path, monkeypatch):
    from types import SimpleNamespace

    skill, registry, compiler, elapsed = timed_compilation(
        tmp_path,
        monkeypatch,
        [json.dumps({"version": f"invalid-{i}", "source": _GOOD_RECIPE}) for i in range(3)],
    )
    compiler.record_usage = True
    original = compiler.invoke
    durations = iter([300, 400, 170, 10])
    trace = tmp_path / "trace.json"
    checkpoints = []

    def slow(prompt, **kwargs):
        if trace.exists():
            checkpoints.append(json.loads(trace.read_text(encoding="utf-8")))
        elapsed[0] += next(durations)
        if len(compiler.calls) == 3:
            raise KeyboardInterrupt
        return original(prompt, **kwargs)

    def start_timer():
        start = elapsed[0]
        return SimpleNamespace(elapsed=lambda: elapsed[0] - start)

    monkeypatch.setattr(compiler, "invoke", slow)
    monkeypatch.setattr("calcifier.agent_events.SuspendAwareTimer.start", start_timer)
    monkeypatch.setattr(cli, "_registry_for_args", lambda *_: registry)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["calcify", str(skill), "--compiler-trace", str(trace)])
    payload = json.loads(trace.read_text(encoding="utf-8"))
    assert payload["outcome"] == "interrupted"
    assert payload["wall_seconds"] == 880
    assert payload["model_calls"] == 4
    assert payload["usage"]["input_tokens"] == 6
    assert payload["usage"]["output_tokens"] == 3
    assert payload["usage"]["cost_usd"] is None
    assert payload["usage_reported_attempts"] == {"input_tokens": 3, "output_tokens": 3}
    assert [entry["wall_seconds"] for entry in checkpoints] == [0, 300, 700, 870]
    assert not (tmp_path / ".calcifier" / "recipe.py").exists()


def test_recipe_write_failure_is_not_reported_as_successful_optimization(tmp_path, monkeypatch):
    from calcifier.artifact import atomic_write

    skill, registry, compiler, _ = timed_compilation(
        tmp_path, monkeypatch, [draft_response(_GOOD_RECIPE)]
    )
    trace = tmp_path / "trace.json"

    def cannot_publish(path, data):
        if path.name == "recipe.py":
            raise OSError("recipe destination unavailable")
        atomic_write(path, data)

    monkeypatch.setattr("calcifier.artifact.atomic_write", cannot_publish)
    monkeypatch.setattr(cli, "_registry_for_args", lambda *_: registry)
    assert cli.main(["calcify", str(skill), "--compiler-trace", str(trace)]) == 1
    payload = json.loads(trace.read_text(encoding="utf-8"))
    assert payload["outcome"] == "failed"
    assert "recipe destination unavailable" in payload["events"][-1]["detail"]
    assert len(compiler.calls) == 1
    assert not (tmp_path / ".calcifier" / "recipe.py").exists()


def test_shared_file_compilation_limit_also_applies_without_a_runtime_role(tmp_path, monkeypatch):
    skill, _, compiler, elapsed = timed_compilation(
        tmp_path, monkeypatch, [draft_response(_GOOD_RECIPE)]
    )
    registry = Registry()
    registry.register(compiler)
    original = compiler.invoke

    def slow(prompt, **kwargs):
        elapsed[0] = 901
        return original(prompt, **kwargs)

    monkeypatch.setattr(compiler, "invoke", slow)
    with pytest.raises(CalcifyError, match="shared 900-second wall budget"):
        prepare_artifact(skill, registry, compilation_timeout=900)
    assert len(compiler.calls) == 1
