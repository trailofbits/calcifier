"""Compiler activity callbacks and telemetry."""

from __future__ import annotations

import pytest

from calcifier.agent_events import AgentActivity
from calcifier.compiler_activity import compiler_activity_payload, render_compiler_activity


def test_compiler_progress_callback_failure_is_diagnostic_only() -> None:
    activity = AgentActivity()

    def fail_callback(event: dict[str, object]) -> None:
        del event
        raise RuntimeError("observer failed")

    activity.event_callback = fail_callback

    activity.record_event("phase", "status", detail="detail")

    assert activity.events == [
        {
            "sequence": 1,
            "phase": "phase",
            "status": "status",
            "model_calls": 0,
            "detail": "detail",
        }
    ]


# -- compiler telemetry -------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "effort", "rendered_model", "rendered_effort"),
    [
        (None, None, "provider-default", "provider-default"),
        ("model-large", "high", "model-large", "high"),
    ],
)
def test_compiler_activity_renders_exact_telemetry(
    model: str | None, effort: str | None, rendered_model: str, rendered_effort: str
) -> None:
    activity = AgentActivity(
        attempts=3,
        duration_seconds=1.2345678,
        usage={"output_tokens": 7, "input_tokens": 5, "cost_usd": None},
        usage_reported_attempts={"input_tokens": 3, "output_tokens": 2},
        max_model_calls=8,
        model=model,
        reasoning_effort=effort,
        events=[{"phase": "compiler", "status": "started"}],
    )

    assert render_compiler_activity(activity) == (
        "compiler activity: model_calls=3 max_model_calls=8 "
        f'model="{rendered_model}" reasoning_effort="{rendered_effort}" '
        'wall_seconds=1.23 usage={"cost_usd":null,"input_tokens":5,"output_tokens":7}'
        ' usage_reported_attempts={"input_tokens":3,"output_tokens":2}'
    )
    assert compiler_activity_payload(activity) == {
        "outcome": "in_progress",
        "suitability_note": None,
        "model_calls": 3,
        "max_model_calls": 8,
        "model": rendered_model,
        "reasoning_effort": rendered_effort,
        "wall_seconds": 1.234568,
        "usage": {"output_tokens": 7, "input_tokens": 5, "cost_usd": None},
        "usage_reported_attempts": {"input_tokens": 3, "output_tokens": 2},
        "events": [{"phase": "compiler", "status": "started"}],
    }


@pytest.mark.parametrize(
    "outcome",
    ["in_progress", "source_accepted", "baseline_retained", "compiled", "failed", "interrupted"],
)
def test_compiler_outcome_is_independent_of_observer_events(outcome):
    activity = AgentActivity(outcome=outcome)
    assert compiler_activity_payload(activity)["outcome"] == outcome
    activity.events.append({"phase": "compiler", "status": "failed"})
    assert compiler_activity_payload(activity)["outcome"] == outcome
    activity.events.clear()
    assert compiler_activity_payload(activity)["outcome"] == outcome
