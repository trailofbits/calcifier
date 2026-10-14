"""Side-effect manifests and idempotency helpers."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from . import trace as _trace
from .exceptions import RecipeError, RecipeFailure
from .hashing import sha256_hex
from .trace import DryRunMode as DryRunMode

logger = logging.getLogger(__name__)

SideEffectKind = Literal["file_write", "external_object", "publish", "delete", "other"]
_SIDE_EFFECT_KINDS = frozenset({"file_write", "external_object", "publish", "delete", "other"})


@dataclass
class SideEffect:
    kind: SideEffectKind
    target: str
    idempotency_key: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    reversible: bool = False
    completed: bool = False
    error: str | None = None
    _recorded_at_ns: int = field(default_factory=time.monotonic_ns, repr=False, compare=False)


class SideEffectBlocked(RecipeFailure):
    """Raised in HARD dry-run mode when a side effect is declared."""


def record_side_effect(
    kind: SideEffectKind,
    target: str,
    *,
    idempotency_key: str | None = None,
    payload: Mapping[str, Any] | None = None,
    reversible: bool = False,
    pending: bool = False,
) -> SideEffect:
    """Declare and record a side effect for the active step."""
    if kind not in _SIDE_EFFECT_KINDS:
        raise RecipeError(f"unsupported side effect kind: {kind!r}")
    ctx = _trace.current()
    if ctx is None or not ctx.record_stack:
        raise RecipeError("record_side_effect() requires an active step")
    if ctx.parallel_worker:
        raise RecipeError("parallel workers cannot mutate shared state")
    data = dict(payload or {})
    key = idempotency_key or _derive_idempotency_key(kind, target, data)
    if ctx.dry_run is DryRunMode.HARD:
        raise SideEffectBlocked(f"side effect blocked by hard dry-run: {kind} {target}")
    existing = _find_by_idempotency_key(ctx.records, key)
    if existing is not None:
        logger.warning("duplicate side effect idempotency key %s; reusing record", key)
        return existing
    effect = SideEffect(
        kind=kind,
        target=target,
        idempotency_key=key,
        payload=data,
        reversible=reversible,
        completed=ctx.dry_run is DryRunMode.OFF and not pending,
    )
    ctx.record_stack[-1].side_effects.append(effect)
    _trace.checkpoint("side-effect")
    return effect


def _derive_idempotency_key(kind: SideEffectKind, target: str, payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest_input = f"{kind}|{target}|{canonical}".encode()
    return sha256_hex(digest_input)


def _find_by_idempotency_key(records: list[_trace.StepRecord], key: str) -> SideEffect | None:
    for record in records:
        for effect in record.side_effects:
            if effect.idempotency_key == key:
                return effect
        found = _find_by_idempotency_key(record.children, key)
        if found is not None:
            return found
    return None
