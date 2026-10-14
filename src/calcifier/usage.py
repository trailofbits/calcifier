"""Add normalized provider usage without discarding unknown fields."""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from typing import Any, TypeGuard


def is_usage_value(value: object) -> TypeGuard[int | float]:
    """Accept a reported count or cost, including zero but excluding invalid numbers."""
    return (type(value) is int and value >= 0) or (
        type(value) is float and math.isfinite(value) and value >= 0
    )


def empty_usage() -> dict[str, float | int | None]:
    """Create an independent usage record with unreported fields set to None."""
    return {
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
        "nested_model_tokens": None,
        "cost_usd": None,
    }


def add_usage(
    aggregate: MutableMapping[str, float | int | None],
    usage: Mapping[str, float | int | None],
    *,
    reported_attempts: MutableMapping[str, int] | None = None,
) -> None:
    for key, value in usage.items():
        if is_usage_value(value):
            aggregate[key] = (aggregate.get(key) or 0) + value
            if reported_attempts is not None:
                reported_attempts[key] = reported_attempts.get(key, 0) + 1
        else:
            aggregate.setdefault(key, None)


def normalized_usage(
    raw: Mapping[str, Any],
    *,
    cost: object = None,
) -> dict[str, float | int | None]:
    aliases = {
        "input_tokens": ("input_tokens", "inputTokenCount"),
        "output_tokens": ("output_tokens", "outputTokenCount"),
        "cache_read_tokens": ("cache_read_input_tokens", "cached_input_tokens"),
        "cache_write_tokens": ("cache_creation_input_tokens", "cache_write_input_tokens"),
        "nested_model_tokens": ("reasoning_output_tokens", "nested_model_tokens"),
    }
    usage: dict[str, float | int | None] = {}
    for normalized, names in aliases.items():
        value = next((raw[name] for name in names if is_usage_value(raw.get(name))), None)
        usage[normalized] = value
    usage["cost_usd"] = cost if is_usage_value(cost) else None
    return usage
