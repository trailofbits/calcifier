"""One recipe invocation and its human-readable result."""

from __future__ import annotations

import pprint
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from . import trace as _trace


@dataclass(frozen=True)
class ErrorLocation:
    line: int
    source: str | None = None


@dataclass(frozen=True)
class RunException:
    type_name: str
    location: ErrorLocation | None = None
    filename: str | None = None
    causes: tuple[RunException, ...] = ()

    @classmethod
    def capture(cls, exc: BaseException, *, location: ErrorLocation | None = None) -> RunException:
        """Retain explicit causes without keeping exceptions or traceback frames alive."""
        causes: list[RunException] = []
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            causes.append(
                cls(
                    type(current).__name__,
                    filename=current.filename
                    if isinstance(current, FileNotFoundError) and isinstance(current.filename, str)
                    else None,
                )
            )
            current = current.__cause__
        return replace(causes[0], location=location, causes=tuple(causes[1:]))


@dataclass
class RecipeRun:
    """Record of a single recipe invocation."""

    name: str
    inputs: dict[str, Any]
    state: Literal["running", "completed", "failed", "interrupted"] = "running"
    output: Any | None = None
    error: str | None = None
    exception: RunException | None = None
    duration_seconds: float = 0.0
    budget: dict[str, float | int | None] | None = None
    records: list[_trace.StepRecord] = field(default_factory=list)
    artifacts: list[_trace.ArtifactRecord] = field(default_factory=list)
    agent_transport: list[dict[str, Any]] = field(default_factory=list)
    agent_calls: list[_trace.AgentCallRecord] = field(default_factory=list)
    agent_preflight: _trace.AgentPreflightReport | None = None
    tool_preflight: _trace.ToolPreflightReport | None = None
    agent_runtime: dict[str, Any] = field(default_factory=lambda: {"mode": "host"})
    recipe_hash: str = ""
    output_directory: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def error_type(self) -> str | None:
        return None if self.exception is None else self.exception.type_name

    def render(self) -> str:
        verdict = "OK" if self.ok else "FAIL"
        lines = [f"=== Calcifier run :: {self.name} :: {verdict} ==="]
        lines.append(f"duration: {self.duration_seconds:.2f}s")
        if self.output_directory is not None:
            lines.append(f"output directory: {self.output_directory}")
        lines.extend(_render_records(self.records, indent=0))
        lines.extend(_render_side_effects(self.records))
        for artifact in self.artifacts:
            lines.extend(_render_artifact(artifact))
        if self.error:
            lines.append(f"failure: {self.error}")
        elif self.output is not None:
            lines.append(f"output: {self.output!r}")
        return "\n".join(lines)


def _render_records(records: list[_trace.StepRecord], *, indent: int) -> list[str]:
    pad = "  " * indent
    out: list[str] = []
    for r in records:
        tag = "fail" if r.error else "pass"
        out.append(f"{pad}[{tag}] {r.name}  duration={r.duration_seconds:.2f}s")
        if r.error:
            out.append(f"{pad}  error: {r.error}")
        out.extend(_render_records(r.children, indent=indent + 1))
    return out


def _all_side_effects(records: list[_trace.StepRecord]) -> list[Any]:
    effects: list[Any] = []
    for record in records:
        effects.extend(record.side_effects)
        effects.extend(_all_side_effects(record.children))
    return effects


def _render_side_effects(records: list[_trace.StepRecord]) -> list[str]:
    effects = _all_side_effects(records)
    if not effects:
        return []
    out = ["side effects:"]
    for effect in effects:
        err = f" error={effect.error}" if effect.error is not None else " error=None"
        out.append(
            "- "
            f"{effect.kind} {effect.target} "
            f"idempotency_key={effect.idempotency_key} "
            f"completed={effect.completed}{err}"
        )
    return out


def _render_artifact(artifact: _trace.ArtifactRecord) -> list[str]:
    label = "artifact" if artifact.name == "artifact" else f"artifact {artifact.name}"
    rendered = _render_artifact_value(artifact.value)
    if "\n" not in rendered:
        return [f"{label}: {rendered}"]
    return [f"{label}:"] + rendered.splitlines()


def _render_artifact_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return pprint.pformat(value, sort_dicts=False)
