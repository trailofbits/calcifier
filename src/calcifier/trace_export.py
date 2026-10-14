"""Persistent JSON export for recipe traces."""

from __future__ import annotations

import base64
import dataclasses
import json
from datetime import date, datetime
from decimal import Decimal
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

from .atomic_io import atomic_write as _atomic_write_bytes
from .run_result import RecipeRun
from .trace import (
    AgentCallRecord,
    AgentPreflightReport,
    ArtifactRecord,
    StepRecord,
    ToolCallRecord,
    ToolPreflightReport,
    redact,
)

TRACE_SCHEMA_VERSION = "1"


def validate_trace(payload: object) -> None:
    """Validate a serialized trace against the shipped Draft 2020-12 schema."""
    version = (
        cast("dict[str, object]", payload).get("version") if isinstance(payload, dict) else None
    )
    if version != TRACE_SCHEMA_VERSION:
        raise ValueError(f"unsupported trace version: {version!r}")
    _trace_validator().validate(payload)


@cache
def _trace_validator() -> Any:
    from jsonschema import Draft202012Validator

    return Draft202012Validator(json.loads(_trace_schema_text()))


def _trace_schema_text() -> str:
    filename = "trace-schema.json"
    packaged = files("calcifier").joinpath(filename)
    try:
        return packaged.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (Path(__file__).parents[2] / "docs" / filename).read_text(encoding="utf-8")


def _jsonish(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(k): _jsonish(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonish(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonish(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime | date | Decimal):
        return str(value)
    return {"__repr__": repr(value)}


def serialize_step_record(record: StepRecord) -> dict[str, Any]:
    """Serialize one step record."""
    return {
        "name": record.name,
        "inputs": _jsonish(record.inputs),
        "output": _jsonish(record.output),
        "error": str(record.error) if record.error is not None else None,
        "duration_seconds": record.duration_seconds,
        "tool_calls": [serialize_tool_call(call) for call in record.tool_calls],
        "agent_calls": [serialize_agent_call(call) for call in record.agent_calls],
        "side_effects": [serialize_side_effect(effect) for effect in record.side_effects],
        "subrecipe": _jsonish(record.subrecipe) if record.subrecipe is not None else None,
        "children": [serialize_step_record(child) for child in record.children],
    }


def serialize_tool_call(record: ToolCallRecord) -> dict[str, Any]:
    return {
        "tool": record.tool,
        "kwargs": _jsonish(record.kwargs),
        "result": _jsonish(record.result),
        "error": record.error,
        "duration_seconds": record.duration_seconds,
    }


def serialize_agent_call(record: AgentCallRecord) -> dict[str, Any]:
    return {
        "role": record.role,
        "target_type": record.target_type,
        "target_name": record.target_name,
        "runtime": _jsonish(record.runtime),
        "filesystem": _jsonish(record.filesystem),
        "configuration_hash": record.configuration_hash,
        "configured_model": record.configured_model,
        "provider_reported_model": record.provider_reported_model,
        "reasoning_effort": record.reasoning_effort,
        "model_identity_source": record.model_identity_source,
        "attempts": record.attempts,
        "error": record.error,
        "duration_seconds": record.duration_seconds,
        "usage": _jsonish(record.usage),
        "usage_reported_attempts": dict(record.usage_reported_attempts),
        "prompt_bytes": list(record.prompt_bytes),
        "schema_bytes": list(record.schema_bytes),
    }


def serialize_side_effect(effect: Any) -> dict[str, Any]:
    return {
        "kind": effect.kind,
        "target": effect.target,
        "idempotency_key": effect.idempotency_key,
        "payload": _jsonish(effect.payload),
        "reversible": effect.reversible,
        "completed": effect.completed,
        "error": effect.error,
    }


def serialize_run(run: RecipeRun) -> dict[str, Any]:
    redacted_output = None if run.output is None else redact(run.output)
    recipe = {"name": run.name}
    if run.recipe_hash:
        recipe["hash"] = run.recipe_hash
    return {
        "version": TRACE_SCHEMA_VERSION,
        "state": run.state,
        "recipe": recipe,
        "inputs": _jsonish(redact(run.inputs)),
        "output": _jsonish(redacted_output),
        "output_directory": run.output_directory,
        "error": str(run.error) if run.error is not None else None,
        "ok": run.ok,
        "duration_seconds": run.duration_seconds,
        "budget": _jsonish(run.budget),
        "agent_runtime": _jsonish(run.agent_runtime),
        "agent_preflight": serialize_agent_preflight_report(run.agent_preflight),
        "tool_preflight": serialize_tool_preflight_report(run.tool_preflight),
        "agent_calls": [serialize_agent_call(record) for record in run.agent_calls],
        "steps": [serialize_step_record(record) for record in run.records],
        "artifacts": [serialize_artifact_record(artifact) for artifact in run.artifacts],
        "agent_transport": _jsonish(run.agent_transport),
    }


def serialize_agent_preflight_report(
    report: AgentPreflightReport | None,
) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "agents": [_jsonish(agent) for agent in report.agents],
        "ok": report.ok,
    }


def serialize_tool_preflight_report(
    report: ToolPreflightReport | None,
) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "missing": [_jsonish(miss) for miss in report.missing],
        "ok": report.ok,
    }


def serialize_artifact_record(record: ArtifactRecord) -> dict[str, Any]:
    return {"name": record.name, "value": _jsonish(record.value)}


def write_trace(run: RecipeRun, path: Path, *, checkpoint: bool = False) -> None:
    """Atomically write ``run`` as trace JSON.

    Args:
        run: The run record to serialize.
        path: Destination file; replaced atomically.
        checkpoint: Write an incremental crash-recovery snapshot. Schema
            validation is deferred to the terminal write, and the JSON is
            compact, so frequent checkpoints stay linear in trace size instead
            of quadratic.
    """
    serialized = serialize_run(run)
    if checkpoint:
        payload = json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()
    else:
        validate_trace(serialized)
        payload = json.dumps(serialized, indent=2, sort_keys=True).encode()
    _atomic_write_bytes(path, payload)
