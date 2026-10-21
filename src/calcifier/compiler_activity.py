"""Compiler call counts, usage, and progress events."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .agent_events import AgentActivity, AgentProvider
from .clock import SuspendAwareTimer
from .diagnostics import CalcifyError
from .validation import bounded_timeout

logger = logging.getLogger(__name__)

DEFAULT_COMPILATION_TIMEOUT = 3600


class CompilationLimitError(CalcifyError):
    """The shared compilation deadline expired; another model call cannot repair it."""


@dataclass(frozen=True)
class CompilerDeadline:
    """One elapsed-time allowance shared by setup, model calls and validation."""

    seconds: int
    timer: SuspendAwareTimer = field(default_factory=SuspendAwareTimer.start)

    def __post_init__(self) -> None:
        if type(self.seconds) is not int or self.seconds <= 0:
            raise CalcifyError("compilation timeout must be a positive integer")

    def remaining(self) -> float:
        remaining = self.seconds - self.timer.elapsed()
        if remaining <= 0:
            raise CompilationLimitError(
                f"compilation exhausted the shared {self.seconds}-second wall budget"
            )
        return remaining


def check_compiler_deadline(deadline: CompilerDeadline | None) -> None:
    if deadline is not None:
        deadline.remaining()


def record_compiler_start(activity: AgentActivity, deadline: CompilerDeadline | None) -> None:
    activity.record_event("compiler", "started")
    if deadline is not None:
        activity.record_event(
            "compiler.budget",
            "configured",
            detail=json.dumps(
                {"max_model_calls": activity.max_model_calls, "max_wall_seconds": deadline.seconds}
            ),
        )


def compiler_timeout(
    provider: AgentProvider, timeout: float | None, deadline: CompilerDeadline | None
) -> float | None:
    """Preserve configured per-call limits while bounding them by time left."""
    if deadline is None:
        return timeout
    remaining = deadline.remaining()
    configured = provider.effective_timeout(timeout)
    return bounded_timeout(configured, remaining)


def render_compiler_activity(activity: AgentActivity) -> str:
    """Render stable, normalized compiler telemetry for success and failure paths."""
    usage = json.dumps(activity.usage, sort_keys=True, separators=(",", ":"))
    reported = json.dumps(activity.usage_reported_attempts, sort_keys=True, separators=(",", ":"))
    model = json.dumps(activity.model or "provider-default")
    reasoning_effort = json.dumps(activity.reasoning_effort or "provider-default")
    result = (
        f"compiler activity: model_calls={compiler_request_count(activity)} "
        f"max_model_calls={activity.max_model_calls} model={model} "
        f"reasoning_effort={reasoning_effort} "
        f"wall_seconds={activity.elapsed_seconds():.2f} usage={usage} "
        f"usage_reported_attempts={reported}"
    )
    if activity.outcome == "baseline_retained":
        result += "\ncompilation: baseline retained; compiler improvement failed: " + (
            activity.retained_reason or ""
        )
    if activity.suitability_note:
        result += (
            "\nsuitability: owner review required (compiler assessment): "
            + activity.suitability_note
        )
    return result


def compiler_activity_payload(activity: AgentActivity) -> dict[str, object]:
    """Return structured compiler telemetry, including per-stage diagnostics."""
    return {
        "outcome": activity.outcome,
        "suitability_note": activity.suitability_note,
        "model_calls": compiler_request_count(activity),
        "max_model_calls": activity.max_model_calls,
        "model": activity.model or "provider-default",
        "reasoning_effort": activity.reasoning_effort or "provider-default",
        "wall_seconds": round(activity.elapsed_seconds(), 6),
        "usage": dict(activity.usage),
        "usage_reported_attempts": dict(activity.usage_reported_attempts),
        "events": list(activity.events),
    }


def compiler_request_count(activity: AgentActivity) -> int:
    """Count host-issued compiler calls independently of provider telemetry."""
    return max(activity.model_calls, activity.attempts)
