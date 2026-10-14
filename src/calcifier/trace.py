"""Per-run context and records, stored in a ContextVar.

Step frames mirror the call tree and supply agent output annotations.
Secret-like field names are redacted before recording; ``redacted_kwargs``
extends the default patterns for a step.
"""

from __future__ import annotations

import contextvars
import dataclasses as _dc
import re
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .usage import add_usage, empty_usage

if TYPE_CHECKING:
    from .budget import BudgetState
    from .side_effects import SideEffect
    from .tools import ToolRegistry


class DryRunMode(Enum):
    OFF = "off"
    SOFT = "soft"
    HARD = "hard"


@dataclass(frozen=True)
class RunHooks:
    """Scoped runtime substitutions used by isolated candidate execution."""

    parent_recipe_path: Callable[[Path], Path] | None = None
    verify_dependency: Callable[[Path, Path, str], None] | None = None
    scratch_output_root: Path | None = None
    command_gap: Callable[[list[str], str | Path, str], bool] | None = None


_DEFAULT_REDACTED = frozenset(
    {
        "password",
        "token",
        "secret",
        "api_key",
        "apikey",
        "cred",
        "credential",
        "credentials",
        "auth",
        "key",
        "authorization",
        "cookie",
        "private_key",
        "session_id",
    }
)

_REDACTED_PLACEHOLDER = "<redacted>"


@dataclass
class StepFrame:
    name: str
    annotation: Any
    parent: StepFrame | None
    inputs: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    extra_redacted: tuple[str, ...] = ()


@dataclass
class ToolCallRecord:
    tool: str
    kwargs: dict[str, Any]
    result: Any | None = None
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class AgentCallRecord:
    role: str
    target_type: str
    target_name: str
    runtime: dict[str, str]
    filesystem: dict[str, Any] | None = None
    configuration_hash: str = ""
    configured_model: str | None = None
    provider_reported_model: str | None = None
    reasoning_effort: str | None = None
    model_identity_source: str = "unavailable"
    attempts: int = 0
    error: str | None = None
    duration_seconds: float = 0.0
    usage: dict[str, float | int | None] = field(default_factory=empty_usage)
    prompt_bytes: list[int] = field(default_factory=list)
    schema_bytes: list[int] = field(default_factory=list)
    usage_reported_attempts: dict[str, int] = field(default_factory=dict)


@dataclass
class StepRecord:
    name: str
    inputs: dict[str, Any]
    output: Any | None = None
    error: str | None = None
    subrecipe: dict[str, str] | None = None
    children: list[StepRecord] = field(default_factory=list)
    duration_seconds: float = 0.0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    agent_calls: list[AgentCallRecord] = field(default_factory=list)
    side_effects: list[SideEffect] = field(default_factory=list)


@dataclass
class ArtifactRecord:
    name: str
    value: Any


@dataclass(frozen=True)
class CapabilityMiss:
    name: str
    remediation: str | None = None


@dataclass(frozen=True)
class ToolPreflightReport:
    missing: tuple[CapabilityMiss, ...]

    @property
    def ok(self) -> bool:
        return not self.missing


@dataclass(frozen=True)
class AgentPreflightEntry:
    role: str
    target_type: str
    target_name: str
    runtime: dict[str, str]
    available: bool
    remediation: str | None = None


@dataclass(frozen=True)
class AgentPreflightReport:
    agents: tuple[AgentPreflightEntry, ...]

    @property
    def ok(self) -> bool:
        return all(agent.available for agent in self.agents)


@dataclass
class OutputWrites:
    """Run-owned destinations, shared by children and parallel lanes."""

    protect_existing: bool = False
    created: set[Path] = field(default_factory=set)
    lock: Any = field(default_factory=threading.RLock, repr=False)


@dataclass
class RunContext:
    frames: list[StepFrame] = field(default_factory=list)
    records: list[StepRecord] = field(default_factory=list)
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    agent_calls: list[AgentCallRecord] = field(default_factory=list)
    agent_call_stack: list[AgentCallRecord] = field(default_factory=list)
    record_stack: list[StepRecord] = field(default_factory=list)
    _recipe_body: StepRecord | None = None
    # The active agent registry. Set by ``Recipe.run`` before the run
    # begins so ``Agent.__call__`` can look up providers by name without
    # having to thread the registry through every step body.
    agent_transport: list[dict[str, Any]] = field(default_factory=list)
    registry: Any | None = None
    tool_registry: ToolRegistry | None = None
    dry_run: DryRunMode = DryRunMode.OFF
    hooks: RunHooks = field(default_factory=RunHooks)
    recipe_hash: str = ""
    parent_step_annotation: Any | None = None
    allowed_agents: frozenset[str] | None = None
    allowed_capabilities: frozenset[str] | None = None
    output_root: Path | None = None
    output_writes: OutputWrites = field(default_factory=OutputWrites)
    resource_root: Path | None = None
    budget_state: BudgetState | None = None
    checkpoint: Callable[[str, RunContext], None] | None = None
    parallel_worker: bool = False
    parallel_lane: str | None = None
    merge_lock: Any = field(default_factory=threading.RLock, repr=False)


_CURRENT: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "calcifier_run_context", default=None
)


def current() -> RunContext | None:
    return _CURRENT.get()


def current_frame() -> StepFrame | None:
    ctx = _CURRENT.get()
    return ctx.frames[-1] if ctx and ctx.frames else None


@contextmanager
def activate_context(ctx: RunContext):
    token = _CURRENT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT.reset(token)


@contextmanager
def async_step_context():
    """Isolate task-local ancestry while sharing this run's records and accounting."""
    parent = current()
    if parent is None:
        yield
        return
    child = _dc.replace(
        parent,
        frames=list(parent.frames),
        record_stack=list(parent.record_stack),
        agent_call_stack=list(parent.agent_call_stack),
    )
    with activate_context(child):
        yield


def fork_run_context(parent: RunContext | None, *, lane: str | None = None) -> RunContext:
    """Snapshot policy before dispatch; share budgets, never mutable frame stacks.

    Publish new records incrementally, including on worker failure. Capturing the
    destination now keeps late background records attached to their spawning step.
    """
    if parent is None:
        return RunContext(parallel_worker=True, parallel_lane=lane)
    destinations = _worker_destinations(parent)
    published = dict.fromkeys(destinations, 0)

    def publish(event: str, child: RunContext) -> None:
        with parent.merge_lock:
            for name, destination in destinations.items():
                records = getattr(child, name)
                destination.extend(records[published[name] :])
                published[name] = len(records)
            _checkpoint(parent, event)

    return _dc.replace(
        parent,
        frames=[],
        records=[],
        artifacts=[],
        agent_calls=[],
        agent_call_stack=[],
        record_stack=[],
        _recipe_body=None,
        parent_step_annotation=(
            parent.frames[-1].annotation if parent.frames else parent.parent_step_annotation
        ),
        checkpoint=publish,
        parallel_worker=True,
        parallel_lane=lane,
    )


def order_worker_records(parent: RunContext | None, workers: list[RunContext]) -> None:
    """Keep final parallel traces in input order without dropping partial checkpoints."""
    if parent is None:
        return
    with parent.merge_lock:
        for name, destination in _worker_destinations(parent).items():
            ordered = [record for worker in workers for record in getattr(worker, name)]
            identities = {id(record) for record in ordered}
            indices = [i for i, record in enumerate(destination) if id(record) in identities]
            if indices:
                retained = [record for record in destination if id(record) not in identities]
                retained[indices[0] : indices[0]] = ordered
                destination[:] = retained


def _worker_destinations(parent: RunContext) -> dict[str, list[Any]]:
    """Where a parallel worker's records land in its parent's trace."""
    return {
        "records": parent.record_stack[-1].children if parent.record_stack else parent.records,
        "agent_calls": (
            parent.record_stack[-1].agent_calls if parent.record_stack else parent.agent_calls
        ),
        "artifacts": parent.artifacts,
        "agent_transport": parent.agent_transport,
    }


@contextmanager
def active_run(
    registry: Any | None = None,
    *,
    tool_registry: ToolRegistry | None = None,
    dry_run: DryRunMode = DryRunMode.OFF,
    hooks: RunHooks | None = None,
    recipe_hash: str = "",
    parent_step_annotation: Any | None = None,
    allowed_agents: frozenset[str] | None = None,
    allowed_capabilities: frozenset[str] | None = None,
    output_root: Path | None = None,
    output_writes: OutputWrites | None = None,
    resource_root: Path | None = None,
    budget_state: BudgetState | None = None,
    checkpoint: Callable[[str, RunContext], None] | None = None,
    parallel_worker: bool = False,
    parallel_lane: str | None = None,
):
    ctx = RunContext(
        registry=registry,
        tool_registry=tool_registry,
        dry_run=dry_run,
        hooks=hooks or RunHooks(),
        recipe_hash=recipe_hash,
        parent_step_annotation=parent_step_annotation,
        allowed_agents=allowed_agents,
        allowed_capabilities=allowed_capabilities,
        output_root=output_root,
        output_writes=output_writes if output_writes is not None else OutputWrites(),
        resource_root=resource_root,
        budget_state=budget_state,
        checkpoint=checkpoint,
        parallel_worker=parallel_worker,
        parallel_lane=parallel_lane,
    )
    token = _CURRENT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT.reset(token)


@contextmanager
def agent_call_frame(
    *,
    role: str,
    target_type: str,
    target_name: str,
    runtime: dict[str, str],
    filesystem: dict[str, Any] | None = None,
    configuration_hash: str = "",
    configured_model: str | None = None,
    reasoning_effort: str | None = None,
    model_identity_source: str = "unavailable",
):
    """Record metadata for one logical typed agent call without its contents."""
    ctx = _CURRENT.get()
    if ctx is None:
        yield None
        return
    record = AgentCallRecord(
        role=role,
        target_type=target_type,
        target_name=target_name,
        runtime=dict(runtime),
        filesystem=None if filesystem is None else dict(filesystem),
        configuration_hash=configuration_hash,
        configured_model=configured_model,
        reasoning_effort=reasoning_effort,
        model_identity_source=model_identity_source,
    )
    if ctx.record_stack:
        ctx.record_stack[-1].agent_calls.append(record)
    else:
        ctx.agent_calls.append(record)
    ctx.agent_call_stack.append(record)
    started = time.time()
    try:
        yield record
    except BaseException as exc:
        record.error = type(exc).__name__
        raise
    finally:
        record.duration_seconds = time.time() - started
        ctx.agent_call_stack.pop()
        _checkpoint(ctx, "agent")


def mark_agent_attempt() -> None:
    ctx = _CURRENT.get()
    if ctx is not None and ctx.budget_state is not None:
        ctx.budget_state.charge_attempt()
    if ctx is not None and ctx.agent_call_stack:
        ctx.agent_call_stack[-1].attempts += 1
    if ctx is not None:
        _checkpoint(ctx, "attempt")


def record_agent_usage(usage: Mapping[str, float | int | None], *, final: bool = False) -> None:
    ctx = _CURRENT.get()
    if ctx is None or not ctx.agent_call_stack:
        return
    call = ctx.agent_call_stack[-1]
    add_usage(
        call.usage,
        usage,
        reported_attempts=call.usage_reported_attempts if final else None,
    )
    _checkpoint(ctx, "usage")


def record_agent_model(model: str | None) -> None:
    """Record the provider's resolved model separately from configuration."""
    ctx = _CURRENT.get()
    if ctx is None or not ctx.agent_call_stack or not model:
        return
    record = ctx.agent_call_stack[-1]
    record.provider_reported_model = model
    record.model_identity_source = "provider_reported"
    _checkpoint(ctx, "model")


def record_agent_prompt(prompt_bytes: int, schema_bytes: int) -> None:
    """Record sizes only; prompts and schemas remain absent from persisted traces."""
    ctx = _CURRENT.get()
    if ctx is None or not ctx.agent_call_stack:
        return
    record = ctx.agent_call_stack[-1]
    record.prompt_bytes.append(prompt_bytes)
    record.schema_bytes.append(schema_bytes)
    _checkpoint(ctx, "prompt")


@contextmanager
def step_frame(
    name: str,
    annotation: Any,
    inputs: dict[str, Any],
    *,
    extra_redacted: tuple[str, ...] = (),
):
    """Push a frame for the duration of a ``@step`` body."""
    ctx = _CURRENT.get()
    if ctx is None:
        # Step called outside of a recipe run — yield without recording. The
        # @step wrapper still runs the body; trace is just unavailable.
        yield None
        return

    parent = ctx.frames[-1] if ctx.frames else None
    redacted_inputs = redact(inputs, extra_redacted)
    frame = StepFrame(
        name=name,
        annotation=annotation,
        parent=parent,
        inputs=redacted_inputs,
        extra_redacted=extra_redacted,
    )
    record = StepRecord(name=name, inputs=redacted_inputs)
    ctx.frames.append(frame)
    if ctx.record_stack:
        ctx.record_stack[-1].children.append(record)
    else:
        ctx.records.append(record)
    ctx.record_stack.append(record)
    started = time.time()
    try:
        yield record
    finally:
        record.duration_seconds = time.time() - started
        ctx.frames.pop()
        ctx.record_stack.pop()
        _checkpoint(ctx, "error" if record.error is not None else "step")


def checkpoint(event: str) -> None:
    ctx = _CURRENT.get()
    if ctx is not None:
        _checkpoint(ctx, event)


def _checkpoint(ctx: RunContext, event: str) -> None:
    if ctx.checkpoint is not None:
        ctx.checkpoint(event, ctx)


def redact(value: Any, extra_redacted: tuple[str, ...] = ()) -> Any:
    """Return a copy of ``value`` with redacted-key entries replaced."""
    extras = {k.lower() for k in extra_redacted}
    return _redact_value(value, _DEFAULT_REDACTED | extras)


def is_redacted_name(name: str, redacted: frozenset[str] | set[str]) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return normalized in redacted or any(
        item != "key" and normalized.endswith(f"_{item}") for item in redacted
    )


def _redact_value(value: Any, redacted: frozenset[str] | set[str]) -> Any:
    if _dc.is_dataclass(value) and not isinstance(value, type):
        # Convert dataclass to a plain dict so trace serializers can ship it
        # safely. Field-name redaction applies, then recurse for nested
        # dataclasses / dicts / lists.
        return {
            f.name: (
                _REDACTED_PLACEHOLDER
                if is_redacted_name(f.name, redacted)
                else _redact_value(getattr(value, f.name), redacted)
            )
            for f in _dc.fields(value)
        }
    if isinstance(value, dict):
        return {
            k: (
                _REDACTED_PLACEHOLDER
                if isinstance(k, str) and is_redacted_name(k, redacted)
                else _redact_value(v, redacted)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(v, redacted) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(v, redacted) for v in value)
    return value
