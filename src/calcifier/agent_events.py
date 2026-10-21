"""Agent invocation transport, accounting, and observer telemetry."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from . import trace as trace_module
from .agent_config import AgentRuntime
from .clock import SuspendAwareTimer
from .filesystem_contract import AgentError, FilesystemContract, TransientAgentError
from .native_events import NativeEventStream, provider_reported_model
from .outputs import (
    StructuredAttemptError,
    needs_json_transport,
    strict_schema_for,
    validated_structured,
    wire_schema,
)
from .subprocesses import (
    ProcessError,
    ProcessOutputLimitError,
    ProcessResult,
    ProcessTimeoutError,
    run_process,
)
from .usage import add_usage, empty_usage
from .validation import bounded_timeout

if TYPE_CHECKING:
    from .dry_run import DryRunResult
    from .native_events import NativeEventStream


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentResult:
    text: str
    raw_stdout: str
    raw_stderr: str
    exit_code: int
    provider: str
    usage: dict[str, float | int | None] = field(default_factory=dict)
    reported_model: str | None = None


@dataclass
class AgentActivity:
    """Aggregate attempted model calls, normalized usage, and elapsed wall time."""

    attempts: int = 0
    duration_seconds: float = 0.0
    usage: dict[str, float | int | None] = field(default_factory=empty_usage)
    max_model_calls: int | None = None
    model_calls: int = 0
    outcome: str = "in_progress"
    retained_reason: str | None = None
    dry_run: DryRunResult | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    events: list[dict[str, object]] = field(default_factory=list)
    event_callback: Callable[[dict[str, object]], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    usage_reported_attempts: dict[str, int] = field(default_factory=dict)
    _active_timer: SuspendAwareTimer | None = field(default=None, repr=False, compare=False)

    def elapsed_seconds(self) -> float:
        """Include the running capture in incremental checkpoints."""
        return self.duration_seconds + (
            self._active_timer.elapsed() if self._active_timer is not None else 0.0
        )

    def record_event(
        self,
        phase: str,
        status: str,
        *,
        detail: str | None = None,
    ) -> None:
        event: dict[str, object] = {
            "sequence": len(self.events) + 1,
            "phase": phase,
            "status": status,
            "model_calls": max(self.model_calls, self.attempts),
        }
        if detail is not None:
            event["detail"] = detail
        self.events.append(event)
        if self.event_callback is not None:
            try:
                self.event_callback(dict(event))
            except Exception:  # noqa: BLE001 - diagnostics must never break compilation
                logger.exception("compiler progress callback failed")


_ACTIVITY_COLLECTORS: contextvars.ContextVar[tuple[AgentActivity, ...]] = contextvars.ContextVar(
    "calcifier_agent_activity_collectors", default=()
)
_ACTIVITY_LOCK = threading.RLock()


@contextmanager
def capture_agent_activity(activity: AgentActivity):
    """Capture agent activity in ``activity``, including failed invocations."""
    collectors = _ACTIVITY_COLLECTORS.get()
    token = _ACTIVITY_COLLECTORS.set((*collectors, activity))
    timer = SuspendAwareTimer.start()
    previous_timer, activity._active_timer = activity._active_timer, timer
    try:
        yield activity
    finally:
        activity.duration_seconds += timer.elapsed()
        activity._active_timer = previous_timer
        _ACTIVITY_COLLECTORS.reset(token)


def captured_agent_activities() -> tuple[AgentActivity, ...]:
    """Return active telemetry collectors for deterministic compiler stages."""
    return _ACTIVITY_COLLECTORS.get()


_STRUCTURED_DIRECTIVE = (
    "Return ONLY a JSON value that conforms to the supplied JSON schema. "
    "Do not include prose, markdown fences, or explanation — emit JSON only."
)
_RETRY_DIRECTIVE = (
    "Your previous response failed to parse: {error}. "
    "Return ONLY a JSON value strictly matching the schema."
)


class AgentProvider(ABC):
    name: str = "agent"
    native_structured_output: bool = False
    target_type: str = "builtin"
    target_name: str = "agent"
    runtime: AgentRuntime = AgentRuntime(kind="host")
    default_timeout: float | None = None
    remediation: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    # None means the adapter has not declared its tool support. It is not a grant.
    capabilities: tuple[str, ...] | None = None

    @abstractmethod
    def invoke(
        self,
        prompt: str,
        *,
        timeout: float | None = None,
        filesystem: FilesystemContract | None = None,
        schema: dict[str, Any] | None = None,
    ) -> AgentResult: ...

    def structured(
        self,
        prompt: str,
        *,
        annotation: Any,
        timeout: float | None = None,
        filesystem: FilesystemContract | None = None,
    ) -> Any:
        """Invoke with a JSON-schema directive derived from ``annotation``,
        parse the response into that type. One automatic retry appends the
        parser's error to the prompt. Providers receive an object-root schema:
        any other root is wrapped as ``{"value": ...}`` and unwrapped here.
        Open mappings and unconstrained values use a JSON string in that
        envelope; strict host parsing validates the decoded value."""
        schema = self._structured_schema(annotation)
        directive = _STRUCTURED_DIRECTIVE
        if not self.native_structured_output:
            directive += "\n\nSchema:\n" + json.dumps(schema, separators=(",", ":")) + "\n"
        full_prompt = f"{prompt}\n\n{directive}"
        _record_agent_prompt_size(full_prompt, schema)
        result = self.complete(
            full_prompt,
            timeout=timeout,
            filesystem=filesystem,
            schema=schema,
        )
        try:
            return self._parse_structured(result.text, annotation)
        except StructuredAttemptError as first_error:
            retry_prompt = f"{full_prompt}\n\n{_RETRY_DIRECTIVE.format(error=first_error)}"
            _record_agent_prompt_size(retry_prompt, schema)
            retry_result = self.complete(
                retry_prompt,
                timeout=timeout,
                filesystem=filesystem,
                schema=schema,
            )
            try:
                return self._parse_structured(retry_result.text, annotation)
            except StructuredAttemptError as second_error:
                # Stages only: validator messages can quote model output, which must not
                # reach run errors or traces.
                raise AgentError(
                    f"{self.name} returned no valid structured response after one retry "
                    f"(first attempt {first_error.stage}; second attempt {second_error.stage})"
                ) from second_error

    def _structured_schema(self, annotation: Any) -> dict[str, Any]:
        return wire_schema(strict_schema_for(annotation))

    def _parse_structured(self, text: str, annotation: Any) -> Any:
        return validated_structured(
            text, annotation, json_encoded=needs_json_transport(strict_schema_for(annotation))
        )

    def complete(
        self,
        prompt: str,
        *,
        timeout: float | None = None,
        filesystem: FilesystemContract | None = None,
        schema: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Invoke once while recording the attempt and any normalized usage."""
        _mark_agent_attempt()
        ctx = trace_module.current()
        if ctx is not None and ctx.budget_state is not None:
            remaining = ctx.budget_state.remaining_wall_seconds()
            configured = self.effective_timeout(timeout)
            timeout = bounded_timeout(configured, remaining)
        result = self.invoke(
            prompt,
            timeout=timeout,
            filesystem=filesystem,
            schema=schema,
        )
        _record_agent_usage(result.usage, final=True)
        trace_module.record_agent_model(result.reported_model)
        return result

    def is_available(self) -> bool:  # pragma: no cover - trivial
        return True

    def trace_metadata(self) -> dict[str, Any]:
        return {
            "target_type": self.target_type,
            "target_name": self.target_name,
            "runtime": self.runtime.as_dict(),
            "configured_model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "model_identity_source": "configured" if self.model is not None else "unavailable",
        }

    def effective_timeout(self, timeout: float | None) -> float | None:
        return self.default_timeout if timeout is None else timeout


def _mark_agent_attempt() -> None:
    with _ACTIVITY_LOCK:
        activities = _ACTIVITY_COLLECTORS.get()
        trace_module.mark_agent_attempt()
        for activity in activities:
            activity.attempts += 1


def _record_agent_usage(usage: Mapping[str, float | int | None], *, final: bool = False) -> None:
    with _ACTIVITY_LOCK:
        for activity in _ACTIVITY_COLLECTORS.get():
            add_usage(
                activity.usage,
                usage,
                reported_attempts=activity.usage_reported_attempts if final else None,
            )
    trace_module.record_agent_usage(usage, final=final)


def _record_agent_prompt_size(prompt: str, schema: Mapping[str, Any]) -> None:
    trace_module.record_agent_prompt(
        len(prompt.encode("utf-8")),
        len(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")),
    )


# -- shared cli runner --------------------------------------------------------


def _check_timeout_budget() -> None:
    ctx = trace_module.current()
    if ctx is not None and ctx.budget_state is not None:
        ctx.budget_state.check_wall()


def _run_cli(
    cmd: list[str],
    stdin: str,
    *,
    timeout: float | None,
    name: str,
    filesystem: FilesystemContract | None = None,
    env: dict[str, str] | None = None,
    event_parser: Callable[[str], tuple[str, dict[str, float | int | None]]] | None = None,
    native_format: Literal["claude", "codex"] | None = None,
) -> AgentResult:
    if timeout is None:
        raise AgentError(f"{name} CLI requires an explicit timeout")
    stream = None if native_format is None else NativeEventStream(native_format)
    timeout_seconds = timeout
    try:
        completed = run_process(
            cmd,
            stdin=stdin.encode(),
            timeout_seconds=timeout_seconds,
            max_output_bytes=8 * 1024 * 1024,
            cwd=None if filesystem is None else filesystem.working_directory,
            env=env,
            stdout_consumer=None if stream is None else stream.consume,
        )
        if stream is not None:
            stream.finish()
    except KeyboardInterrupt:
        _failed_stream(stream, b"", event_parser, name)
        raise
    except ProcessTimeoutError as exc:
        _failed_stream(stream, exc.stdout, event_parser, name)
        _report_agent_diagnostic(name, exc.stderr_tail)
        _check_timeout_budget()
        raise TransientAgentError(f"{name} CLI timed out after {timeout_seconds:g}s") from exc
    except ProcessError as exc:
        _failed_stream(stream, exc.stdout, event_parser, name)
        _report_agent_diagnostic(name, exc.stderr_tail)
        _raise_transport_error(stream, exc, name)
    _report_agent_diagnostic(name, completed.stderr_tail)
    if completed.returncode != 0:
        _failed_stream(stream, completed.stdout, event_parser, name)
        raise AgentError(f"{name} CLI exited {completed.returncode}")
    if stream is not None:
        return _native_cli_result(stream, completed.returncode, name)
    return _text_cli_result(completed, event_parser, name)


def _raise_transport_error(
    stream: NativeEventStream | None, error: ProcessError, name: str
) -> None:
    if stream is not None:
        reason = (
            "native output rejected"
            if isinstance(error, ProcessOutputLimitError)
            else "transport failed"
        )
        raise AgentError(
            f"{name} CLI {reason}: {error}; " + json.dumps(stream.diagnostics(), sort_keys=True)
        ) from error
    if isinstance(error, ProcessOutputLimitError):
        raise AgentError(f"{name} CLI output exceeded 8388608 bytes") from error
    raise AgentError(f"{name} CLI failed to start: {error}") from error


def _native_cli_result(stream: NativeEventStream, exit_code: int, name: str) -> AgentResult:
    try:
        stream.require_completion()
    except AgentError:
        _failed_stream(stream, b"", None, name)
        raise
    _record_native_transport(stream, name, "completed")
    # Preserve earlier observations without claiming they were finally reported,
    # and without adding a partial count to the provider's final total.
    _record_agent_usage(
        {key: value for key, value in stream.partial_usage.items() if stream.usage.get(key) is None}
    )
    return AgentResult(
        text=stream.text,
        raw_stdout="",  # Native auxiliary events are deliberately not retained.
        raw_stderr="",
        exit_code=exit_code,
        provider=name,
        usage=stream.usage,
        reported_model=stream.diagnostics()["reported_model"],
    )


def _text_cli_result(
    completed: ProcessResult,
    event_parser: Callable[[str], tuple[str, dict[str, float | int | None]]] | None,
    name: str,
) -> AgentResult:
    try:
        stdout = completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AgentError(f"{name} CLI stdout was not valid UTF-8") from exc
    text, usage = (stdout.strip(), {}) if event_parser is None else event_parser(stdout)
    reported_model = provider_reported_model(stdout)
    return AgentResult(
        text=text,
        raw_stdout=stdout,
        raw_stderr="",
        exit_code=completed.returncode,
        provider=name,
        usage=usage,
        reported_model=reported_model,
    )


def _failed_stream(
    stream: NativeEventStream | None,
    stdout: bytes,
    parser: Callable[[str], tuple[str, dict[str, float | int | None]]] | None,
    name: str,
) -> None:
    if stream is None:
        _record_partial_usage(stdout, parser)
    else:
        _record_agent_usage(stream.partial_usage)
        _record_native_transport(stream, name, "failed")


def _record_native_transport(stream: NativeEventStream, name: str, status: str) -> None:
    diagnostics = {"provider": name, **stream.diagnostics()}
    activities = captured_agent_activities()
    if activities:
        for activity in activities:
            activity.record_event("native_transport", status, detail=json.dumps(diagnostics))
    trace_module.record_agent_model(diagnostics["reported_model"])
    ctx = trace_module.current()
    if ctx is not None:
        ctx.agent_transport.append({"status": status, **diagnostics})
        trace_module.checkpoint("agent-transport")


def _record_partial_usage(
    stdout: bytes,
    event_parser: Callable[[str], tuple[str, dict[str, float | int | None]]] | None,
) -> None:
    if not stdout or event_parser is None:
        return
    try:
        _, usage = event_parser(stdout.decode("utf-8"))
    except (UnicodeDecodeError, AgentError):
        return
    _record_agent_usage(usage)


def _report_agent_diagnostic(name: str, data: bytes) -> None:
    detail = data.decode("utf-8", errors="replace").strip()
    if detail:
        print(f"{name} CLI stderr: {detail}", file=sys.stderr)
