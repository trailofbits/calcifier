"""A shared compiler deadline bounds repairs without waiving final validation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest

from calcifier import cli
from calcifier.agent_events import AgentActivity
from calcifier.artifact import prepare_artifact
from calcifier.baseline import is_baseline
from calcifier.clock import SuspendAwareTimer
from calcifier.compiler_activity import (
    CompilationLimitError,
    CompilerDeadline,
    compiler_request_count,
    compiler_timeout,
)
from calcifier.recipe_draft import draft_response
from calcifier.registry import Registry
from tests.support.agents import StubProvider
from tests.support.compiler import ScriptedCompiler
from tests.support.recipes import _GOOD_RECIPE
from tests.support.skill_source import SKILL


def timed_compilation(tmp_path, monkeypatch, responses):
    elapsed = [0.0]
    timer = cast(SuspendAwareTimer, SimpleNamespace(elapsed=lambda: elapsed[0]))
    monkeypatch.setattr(
        "calcifier.artifact.CompilerDeadline", lambda seconds: CompilerDeadline(seconds, timer)
    )
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(SKILL.encode())
    registry = Registry()
    compiler = ScriptedCompiler(responses)
    compiler.default_timeout = None
    registry.register(compiler)
    registry.register(StubProvider())
    return skill, registry, compiler, elapsed


@pytest.mark.parametrize("accepted", [False, True])
def test_source_repairs_share_five_calls_and_remaining_time(tmp_path, monkeypatch, accepted):
    bad = [json.dumps({"version": f"invalid-{i}", "source": _GOOD_RECIPE}) for i in range(5)]
    skill, registry, compiler, elapsed = timed_compilation(
        tmp_path, monkeypatch, bad[:4] + [draft_response(_GOOD_RECIPE)] if accepted else bad
    )
    original = compiler.invoke
    timeouts = []

    def invoke(prompt, *, timeout=None, **kwargs):
        timeouts.append(timeout)
        elapsed[0] += 100
        return original(prompt, timeout=timeout, **kwargs)

    monkeypatch.setattr(compiler, "invoke", invoke)
    activity = AgentActivity()
    artifact = prepare_artifact(skill, registry, activity=activity, compilation_timeout=900)
    assert timeouts == [900, 800, 700, 600, 500]
    assert len(compiler.calls) == compiler_request_count(activity) == 5
    assert is_baseline(artifact.source) is not accepted


def test_early_repair_accepts_without_reserved_review_capacity(tmp_path, monkeypatch):
    skill, registry, compiler, _ = timed_compilation(
        tmp_path, monkeypatch, ["invalid", draft_response(_GOOD_RECIPE)]
    )
    artifact = prepare_artifact(skill, registry, max_compiler_calls=2)
    assert len(compiler.calls) == 2
    assert not is_baseline(artifact.source)


@pytest.mark.parametrize("accepted", [False, True])
def test_cli_minimum_budgets_preserve_one_call_and_one_second(tmp_path, monkeypatch, accepted):
    skill, registry, compiler, _ = timed_compilation(
        tmp_path, monkeypatch, [draft_response(_GOOD_RECIPE) if accepted else "invalid"]
    )
    original = compiler.invoke
    timeouts = []

    def invoke(prompt, *, timeout=None, **kwargs):
        timeouts.append(timeout)
        return original(prompt, timeout=timeout, **kwargs)

    monkeypatch.setattr(compiler, "invoke", invoke)
    monkeypatch.setattr(cli, "_registry_for_args", lambda args, providers: registry)
    trace = tmp_path / "compiler-trace.json"
    result = cli.main(
        [
            "calcify",
            str(skill),
            "--compilation-timeout",
            "1",
            "--max-compiler-calls",
            "1",
            "--require-compiled",
            "--compiler-trace",
            str(trace),
        ]
    )

    assert (result == 0) is accepted
    assert timeouts == [1]
    assert len(compiler.calls) == 1
    payload = json.loads(trace.read_bytes())
    assert payload["max_model_calls"] == payload["model_calls"] == 1
    assert payload["outcome"] == ("compiled" if accepted else "baseline_retained")
    configured = [event for event in payload["events"] if event["phase"] == "compiler.budget"]
    assert len(configured) == 1
    assert json.loads(configured[0]["detail"]) == {"max_model_calls": 1, "max_wall_seconds": 1}
    source = (skill.parent / ".calcifier" / "recipe.py").read_text(encoding="utf-8")
    assert is_baseline(source) is not accepted


@pytest.mark.parametrize(
    "expires_after,expected_calls", [("setup", 0), ("source", 1), ("validation", 1)]
)
def test_expiry_keeps_baseline_without_late_publication(
    tmp_path, monkeypatch, expires_after, expected_calls
):
    from calcifier import artifact as artifact_module

    skill, registry, compiler, elapsed = timed_compilation(
        tmp_path, monkeypatch, [draft_response(_GOOD_RECIPE)]
    )
    from calcifier import repair as repair_module

    original_invoke, original_check = compiler.invoke, artifact_module._checked_artifact
    original_dry_run = repair_module._require_dry_run_success
    checked = []

    def invoke(prompt, **kwargs):
        result = original_invoke(prompt, **kwargs)
        if expires_after == "source":
            elapsed[0] = 900
        return result

    def check(*args, **kwargs):
        result = original_check(*args, **kwargs)
        checked.append(result)
        if len(checked) == 1 and expires_after == "setup":
            elapsed[0] = 900
        return result

    def dry_run(*args, **kwargs):
        result = original_dry_run(*args, **kwargs)
        if expires_after == "validation":
            assert result.status == "passed"
            elapsed[0] = 900
        return result

    monkeypatch.setattr(compiler, "invoke", invoke)
    monkeypatch.setattr(artifact_module, "_checked_artifact", check)
    monkeypatch.setattr(repair_module, "_require_dry_run_success", dry_run)
    activity = AgentActivity()
    artifact = prepare_artifact(skill, registry, activity=activity, compilation_timeout=900)
    assert is_baseline(artifact.source)
    assert artifact.source == checked[0]
    assert artifact.outcome == activity.outcome == "baseline_retained"
    assert activity.dry_run is None
    assert repair_module.render_dry_run(activity) == "dry run: not run"
    assert len(compiler.calls) == compiler_request_count(activity) == expected_calls
    assert "shared 900-second wall budget" in str(activity.events[-1]["detail"])
    assert not any(event["status"] == "replaced" for event in activity.events)


@pytest.mark.parametrize(
    ("configured", "requested", "expected"),
    [(None, None, 10), (30, None, 10), (4, None, 4), (4, 6, 6), (4, 30, 10)],
)
def test_deadline_preserves_shorter_per_call_timeouts(configured, requested, expected):
    compiler = ScriptedCompiler([])
    compiler.default_timeout = configured
    elapsed = [890.0]
    timer = cast(SuspendAwareTimer, SimpleNamespace(elapsed=lambda: elapsed[0]))
    deadline = CompilerDeadline(900, timer)
    assert compiler_timeout(compiler, requested, deadline) == expected
    assert compiler_timeout(compiler, requested, None) == requested
    elapsed[0] = 900
    with pytest.raises(CompilationLimitError, match="shared 900-second wall budget"):
        compiler_timeout(compiler, requested, deadline)


def test_deadline_preserves_fractional_time_until_the_exact_limit() -> None:
    elapsed = [899.75]
    timer = cast(SuspendAwareTimer, SimpleNamespace(elapsed=lambda: elapsed[0]))
    deadline = CompilerDeadline(900, timer)
    compiler = ScriptedCompiler([])

    assert deadline.remaining() == 0.25
    assert compiler_timeout(compiler, 30, deadline) == 0.25
    for expired in (900, 900.25):
        elapsed[0] = expired
        with pytest.raises(CompilationLimitError, match="shared 900-second wall budget"):
            compiler_timeout(compiler, 30, deadline)


def test_source_mode_also_uses_shared_deadline(tmp_path, monkeypatch):
    skill, registry, compiler, elapsed = timed_compilation(tmp_path, monkeypatch, ["invalid"])
    original = compiler.invoke

    def invoke(prompt, *, timeout=None, **kwargs):
        assert timeout == 900
        elapsed[0] = 900
        return original(prompt, timeout=timeout, **kwargs)

    monkeypatch.setattr(compiler, "invoke", invoke)
    activity = AgentActivity()
    artifact = prepare_artifact(skill, registry, activity=activity, compilation_timeout=900)
    assert is_baseline(artifact.source)
    assert len(compiler.calls) == compiler_request_count(activity) == 1
    assert "shared 900-second wall budget" in str(activity.events[-1]["detail"])


def test_observers_cannot_reset_the_compiler_call_budget(tmp_path, monkeypatch):
    skill, registry, compiler, _ = timed_compilation(tmp_path, monkeypatch, ["invalid", "invalid"])
    activity = AgentActivity()

    def forget_events(event):
        activity.events.clear()
        activity.attempts = 0
        activity.max_model_calls = 1000

    activity.event_callback = forget_events
    artifact = prepare_artifact(skill, registry, activity=activity, max_compiler_calls=2)
    assert artifact.outcome == "baseline_retained"
    assert len(compiler.calls) == activity.model_calls == 2
    assert "global compiler-call budget" in (activity.retained_reason or "")
    assert activity.events == []


def test_agent_failure_emits_one_terminal_compiler_event(tmp_path, monkeypatch):
    from calcifier.filesystem_contract import AgentError

    skill, registry, compiler, _ = timed_compilation(tmp_path, monkeypatch, [])

    def fail(*args, **kwargs):
        raise AgentError("neutral transport failure")

    monkeypatch.setattr(compiler, "invoke", fail)
    activity = AgentActivity()
    artifact = prepare_artifact(skill, registry, activity=activity)
    assert artifact.outcome == "baseline_retained"
    failures = [
        event
        for event in activity.events
        if event["phase"] == "compiler" and event["status"] == "failed"
    ]
    assert len(failures) == 1
    assert activity.model_calls == 1
