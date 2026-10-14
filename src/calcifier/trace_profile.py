"""Read-only runtime summaries for exported recipe traces."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .trace_export import validate_trace


def _calls_in_steps(
    steps: list[dict[str, Any]], prefix: str = ""
) -> Iterator[tuple[str, dict[str, Any]]]:
    for step in steps:
        location = f"{prefix}/{step['name']}" if prefix else step["name"]
        for call in step["agent_calls"]:
            yield location, call
        yield from _calls_in_steps(step["children"], location)


def _tool_calls_in_steps(steps: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for step in steps:
        yield from step["tool_calls"]
        yield from _tool_calls_in_steps(step["children"])


def _usage_text(calls: list[dict[str, Any]], key: str) -> str:
    """Show observed usage and whether every attempt supplied a final report."""
    values = [call["usage"].get(key) for call in calls]
    observed = [value for value in values if value is not None]
    total = f"{sum(observed):,}" if observed else "unknown"
    counts = [call["usage_reported_attempts"].get(key, 0) for call in calls]
    if any(count > call["attempts"] for call, count in zip(calls, counts, strict=True)):
        raise ValueError("usage report count exceeds provider attempts")
    if all(
        count == call["attempts"] and (value is not None or call["attempts"] == 0)
        for call, count, value in zip(calls, counts, values, strict=True)
    ):
        return f"{sum(observed):,}"
    coverage = f"{sum(counts)}/{sum(call['attempts'] for call in calls)} final reports"
    return f"{total} observed ({coverage})" if observed else f"unknown ({coverage})"


def render_trace_profile(payload: dict[str, Any]) -> str:
    """Summarize a validated recipe trace."""
    validate_trace(payload)
    calls = [("(recipe)", call) for call in payload["agent_calls"]]
    calls.extend(_calls_in_steps(payload["steps"]))
    tools = list(_tool_calls_in_steps(payload["steps"]))
    attempts = sum(call["attempts"] for _, call in calls)
    extra_attempts = sum(max(0, call["attempts"] - 1) for _, call in calls)
    prompt_bytes = sum(sum(call["prompt_bytes"]) for _, call in calls)
    schema_bytes = sum(sum(call["schema_bytes"]) for _, call in calls)
    usage_keys = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
    agent_calls = [call for _, call in calls]
    lines = [
        f"recipe: {payload['recipe']['name']}",
        f"state: {payload['state']} (ok={str(payload['ok']).lower()})",
        f"wall: {payload['duration_seconds']:.1f}s",
        (
            f"agent calls: {len(calls)}, attempts: {attempts}, "
            f"extra attempts: {extra_attempts}, "
            f"summed agent wall: {sum(call['duration_seconds'] for _, call in calls):.1f}s"
        ),
        (
            f"tool calls: {len(tools)}, "
            f"summed tool wall: {sum(call['duration_seconds'] for call in tools):.1f}s"
        ),
        f"prompt bytes: {prompt_bytes:,}; schema bytes: {schema_bytes:,}",
        "usage: " + ", ".join(f"{key}={_usage_text(agent_calls, key)}" for key in usage_keys),
    ]
    if calls:
        lines.append("slowest agent calls:")
        for location, call in sorted(
            calls, key=lambda item: item[1]["duration_seconds"], reverse=True
        ):
            lines.append(
                f"  {call['duration_seconds']:.1f}s {call['role']} @ {location} "
                f"(attempts={call['attempts']}, prompt_bytes={sum(call['prompt_bytes']):,}, "
                f"input_tokens={_usage_text([call], 'input_tokens')}, "
                f"output_tokens={_usage_text([call], 'output_tokens')})"
            )
    return "\n".join(lines)
