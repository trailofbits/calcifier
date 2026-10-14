"""Usage observations must not masquerade as complete provider accounting."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest
from jsonschema import ValidationError

from calcifier.agent_events import (
    AgentActivity,
    AgentResult,
    _record_agent_usage,
    capture_agent_activity,
)
from calcifier.filesystem_contract import AgentError
from calcifier.native_events import parse_adapter_envelope, parse_claude_events, parse_codex_events
from calcifier.recipe import RecipeRun
from calcifier.trace import AgentCallRecord, active_run, agent_call_frame
from calcifier.trace_export import serialize_run, validate_trace
from calcifier.trace_profile import render_trace_profile
from calcifier.usage import normalized_usage
from tests.support.agents import StubProvider


@pytest.mark.parametrize("value", [None, True, False, -1, -0.1, float("nan"), float("inf"), "7"])
def test_invalid_or_absent_usage_does_not_become_zero(value: object) -> None:
    usage = normalized_usage({"input_tokens": value}, cost=value)

    assert usage["input_tokens"] is None
    assert usage["cost_usd"] is None
    assert usage["output_tokens"] is None


@pytest.mark.parametrize("value", [0, 0.0, 7, 1.25])
def test_valid_usage_including_explicit_zero_is_retained(value: int | float) -> None:
    usage = normalized_usage({"input_tokens": value}, cost=value)

    assert usage["input_tokens"] == value
    assert usage["cost_usd"] == value


@pytest.mark.parametrize("value", [True, -1, float("inf")])
def test_invalid_claude_thinking_details_do_not_become_usage(value: object) -> None:
    _, usage = parse_claude_events(
        json.dumps(
            {"type": "result", "usage": {"output_tokens_details": {"thinking_tokens": value}}}
        )
    )

    assert usage["nested_model_tokens"] is None


@pytest.mark.parametrize("provider", ["codex", "claude", "adapter"])
def test_native_and_adapter_results_distinguish_zero_from_missing(provider: str) -> None:
    raw = {"input_tokens": 0}
    if provider == "codex":
        _, usage = parse_codex_events(json.dumps({"type": "turn.completed", "usage": raw}))
    elif provider == "claude":
        _, usage = parse_claude_events(json.dumps({"type": "result", "usage": raw}))
    else:
        _, usage = parse_adapter_envelope(json.dumps({"version": 1, "output": "ok", "usage": raw}))

    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] is None
    assert usage["cache_write_tokens"] is None


def test_partial_per_model_fields_keep_terminal_usage() -> None:
    _, usage = parse_claude_events(
        json.dumps(
            {
                "type": "result",
                "usage": {"input_tokens": 20, "output_tokens": 4},
                "modelUsage": {
                    "main": {"inputTokens": 10, "outputTokens": 3},
                    "aux": {"outputTokens": 2},
                },
            }
        )
    )

    assert usage["input_tokens"] == 20
    assert usage["output_tokens"] == 5
    assert usage["cache_read_tokens"] is None


@pytest.mark.parametrize("terminal", [{"input_tokens": 20}, {}])
@pytest.mark.parametrize("invalid", [None, False, 0, "missing"])
def test_invalid_per_model_entry_keeps_terminal_usage_or_unknown(
    terminal: dict[str, int], invalid: object
) -> None:
    _, usage = parse_claude_events(
        json.dumps(
            {
                "type": "result",
                "usage": terminal,
                "modelUsage": {"main": {"inputTokens": 10}, "aux": invalid},
            }
        )
    )

    assert usage["input_tokens"] == terminal.get("input_tokens")


@pytest.mark.parametrize("reported", [None, 0, 7])
def test_structured_retry_counts_final_reports_even_for_rejected_responses(
    monkeypatch: pytest.MonkeyPatch, reported: int | None
) -> None:
    provider = StubProvider()
    activity = AgentActivity()
    results = iter(
        [
            replace(provider._make("not json"), usage={"input_tokens": 0}),
            replace(provider._make("true"), usage={"input_tokens": reported}),
        ]
    )

    def invoke(*args: object, **kwargs: object) -> AgentResult:
        del args, kwargs
        return next(results)

    monkeypatch.setattr(provider, "invoke", invoke)
    with active_run() as context, capture_agent_activity(activity):
        with agent_call_frame(
            role="worker", target_type="builtin", target_name="codex", runtime={"mode": "host"}
        ):
            assert provider.structured("boolean", annotation=bool) is True
        call = context.agent_calls[0]

    assert activity.attempts == call.attempts == 2
    assert activity.usage["input_tokens"] == call.usage["input_tokens"] == (reported or 0)
    assert (
        activity.usage_reported_attempts
        == call.usage_reported_attempts
        == {"input_tokens": 1 if reported is None else 2}
    )
    exported = serialize_run(RecipeRun(name="retry", inputs={}, agent_calls=[call]))
    assert exported["agent_calls"][0]["usage_reported_attempts"] == call.usage_reported_attempts


@pytest.mark.parametrize("partial", [None, 3])
def test_failed_attempt_keeps_observations_without_claiming_a_final_report(
    monkeypatch: pytest.MonkeyPatch, partial: int | None
) -> None:
    provider = StubProvider()
    activity = AgentActivity()

    def invoke(prompt: str, **kwargs: Any) -> AgentResult:
        del kwargs
        if prompt == "failed":
            if partial is not None:
                _record_agent_usage({"input_tokens": partial})
            raise AgentError("provider stopped")
        return replace(provider._make("ok"), usage={"input_tokens": 7})

    monkeypatch.setattr(provider, "invoke", invoke)
    with active_run() as context, capture_agent_activity(activity):
        with agent_call_frame(
            role="worker", target_type="builtin", target_name="codex", runtime={"mode": "host"}
        ):
            provider.complete("reported")
            with pytest.raises(AgentError, match="provider stopped"):
                provider.complete("failed")
        call = context.agent_calls[0]

    assert activity.attempts == call.attempts == 2
    assert activity.usage["input_tokens"] == call.usage["input_tokens"] == 7 + (partial or 0)
    assert activity.usage["output_tokens"] is call.usage["output_tokens"] is None
    assert activity.usage_reported_attempts == call.usage_reported_attempts == {"input_tokens": 1}
    profile = render_trace_profile(
        serialize_run(RecipeRun(name="partial", inputs={}, agent_calls=[call]))
    )
    assert f"input_tokens={7 + (partial or 0)} observed (1/2 final reports)" in profile
    assert "output_tokens=unknown (0/2 final reports)" in profile


def _call(value: int | None, reports: int, attempts: int = 1) -> AgentCallRecord:
    call = AgentCallRecord(
        role="worker",
        target_type="builtin",
        target_name="codex",
        runtime={"mode": "host"},
        attempts=attempts,
    )
    call.usage["input_tokens"] = value
    call.usage_reported_attempts["input_tokens"] = reports
    return call


@pytest.mark.parametrize(
    ("calls", "expected"),
    [
        ([], "input_tokens=0"),
        ([_call(None, 0, 0)], "input_tokens=0"),
        ([_call(0, 1)], "input_tokens=0"),
        ([_call(None, 0)], "input_tokens=unknown (0/1 final reports)"),
        ([_call(0, 1), _call(None, 0)], "input_tokens=0 observed (1/2 final reports)"),
        ([_call(7, 1), _call(None, 0)], "input_tokens=7 observed (1/2 final reports)"),
    ],
)
def test_profiles_separate_exact_zero_unknown_and_incomplete_observations(
    calls: list[AgentCallRecord], expected: str
) -> None:
    rendered = render_trace_profile(
        serialize_run(RecipeRun(name="usage", inputs={}, agent_calls=calls))
    )

    assert expected in next(line for line in rendered.splitlines() if line.startswith("usage:"))


def test_trace_requires_usage_reporting_coverage() -> None:
    payload = serialize_run(RecipeRun(name="usage", inputs={}, agent_calls=[_call(0, 1)]))
    del payload["agent_calls"][0]["usage_reported_attempts"]
    with pytest.raises(ValidationError, match="usage_reported_attempts"):
        render_trace_profile(payload)


def test_profile_rejects_report_counts_exceeding_attempts() -> None:
    payload = serialize_run(RecipeRun(name="invalid", inputs={}, agent_calls=[_call(7, 2)]))

    with pytest.raises(ValueError, match="report count exceeds"):
        render_trace_profile(payload)


@pytest.mark.parametrize("reports", [{"input_tokens": -1}, {"unknown": 1}, {"input_tokens": True}])
def test_trace_rejects_invalid_usage_report_counts(reports: dict[str, int]) -> None:
    payload = serialize_run(RecipeRun(name="invalid", inputs={}, agent_calls=[_call(7, 1)]))
    payload["agent_calls"][0]["usage_reported_attempts"] = reports

    with pytest.raises(ValidationError):
        validate_trace(payload)
