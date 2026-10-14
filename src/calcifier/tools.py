"""Declarative tool capability registry and trace plumbing."""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from . import trace
from .exceptions import CalcifierError, ErrorFamily
from .json_config import load_json_config
from .path_syntax import PathSyntax
from .subprocesses import (
    MAX_TIMEOUT_SECONDS,
    ProcessError,
    ProcessOutputLimitError,
    ProcessTimeoutError,
    run_process,
)
from .validation import is_positive_number


@dataclass(frozen=True)
class Capability:
    name: str
    provider: str
    redact_args: tuple[str, ...] = ()
    redact_result: bool = False
    remediation: str | None = None


class ToolUnavailableError(CalcifierError):
    """Raised when a declared tool capability is unavailable."""


class ToolConfigError(CalcifierError, ValueError):
    """Raised when a CLI tool-provider configuration is invalid."""

    family = ErrorFamily.SETUP


class ToolExecutionError(CalcifierError, RuntimeError):
    """Raised when a configured tool subprocess fails its wire contract."""


@dataclass(frozen=True)
class SubprocessToolSpec:
    command: tuple[str, ...]
    timeout_seconds: float = 60.0
    redact_args: tuple[str, ...] = ()
    redact_result: bool = False
    remediation: str | None = None


class SubprocessToolProvider:
    """JSON-over-stdin capability provider backed by one argv command."""

    _MAX_STDOUT_BYTES = 1024 * 1024
    _MAX_DIAGNOSTIC_BYTES = 4096

    def __init__(self, spec: SubprocessToolSpec) -> None:
        self.spec = spec

    def available(self, name: str) -> bool:
        del name
        return shutil.which(self.spec.command[0]) is not None

    def invoke(self, name: str, /, **kwargs: Any) -> Any:
        payload = json.dumps(
            {"capability": name, "arguments": kwargs},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return self._invoke_payload(name, payload)

    def _invoke_payload(self, name: str, payload: bytes) -> Any:
        if not self.available(name):
            raise ToolUnavailableError(
                f"tool capability {name!r} executable not found: {self.spec.command[0]}"
            )
        try:
            completed = run_process(
                self.spec.command,
                stdin=payload,
                timeout_seconds=self.spec.timeout_seconds,
                max_output_bytes=self._MAX_STDOUT_BYTES,
                stderr_tail_bytes=self._MAX_DIAGNOSTIC_BYTES,
            )
        except ProcessTimeoutError as exc:
            _report_diagnostic(name, _diagnostic(exc.stderr_tail))
            raise ToolExecutionError(
                f"tool capability {name!r} timed out after {self.spec.timeout_seconds:g}s"
            ) from exc
        except ProcessOutputLimitError as exc:
            _report_diagnostic(name, _diagnostic(exc.stderr_tail))
            detail = str(exc).removeprefix("process ")
            raise ToolExecutionError(f"tool capability {name!r} {detail}") from exc
        except ProcessError as exc:
            raise ToolExecutionError(f"tool capability {name!r} failed to start: {exc}") from exc
        if completed.returncode != 0:
            detail = _diagnostic(completed.stderr_tail)
            _report_diagnostic(name, detail)
            raise ToolExecutionError(f"tool capability {name!r} exited {completed.returncode}")
        try:
            text = completed.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolExecutionError(
                f"tool capability {name!r} stdout was not valid UTF-8"
            ) from exc
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(
                f"tool capability {name!r} stdout was not exactly one JSON value: {exc}"
            ) from exc


def _diagnostic(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").strip()


def _report_diagnostic(name: str, detail: str) -> None:
    """Keep unstructured tool stderr interactive and out of persisted traces."""
    if detail:
        print(f"tool capability {name!r} stderr: {detail}", file=sys.stderr)


def load_tool_config(path: Path) -> dict[str, tuple[Capability, SubprocessToolProvider]]:
    """Load and validate the version-1 CLI tool-provider configuration."""
    try:
        raw = load_json_config(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolConfigError(f"failed to load tool config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ToolConfigError("tool config root must be an object")
    unknown_root = set(raw) - {"version", "capabilities"}
    if unknown_root:
        raise ToolConfigError(f"unknown tool config keys: {sorted(unknown_root)}")
    if raw.get("version") != 1:
        raise ToolConfigError("tool config version must be 1")
    capabilities = raw.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ToolConfigError("tool config capabilities must be an object")
    return {
        name: _configured_capability(name, value)
        for name, value in capabilities.items()
        if _require_capability_name(name)
    }


def _require_capability_name(name: object) -> bool:
    if not isinstance(name, str) or not name:
        raise ToolConfigError("tool capability names must be non-empty strings")
    return True


def _configured_timeout(name: str, timeout: object) -> float:
    if not is_positive_number(timeout, finite=True, maximum=MAX_TIMEOUT_SECONDS):
        raise ToolConfigError(
            f"tool capability {name!r} timeout_seconds must be finite, positive, "
            f"and at most {MAX_TIMEOUT_SECONDS:g}"
        )
    return float(cast("int | float", timeout))


def _configured_capability(name: str, value: object) -> tuple[Capability, SubprocessToolProvider]:
    if not isinstance(value, dict):
        raise ToolConfigError(f"tool capability {name!r} config must be an object")
    config = cast("dict[str, Any]", value)
    allowed = {
        "command",
        "timeout_seconds",
        "redact_args",
        "redact_result",
        "remediation",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ToolConfigError(f"tool capability {name!r} has unknown keys: {sorted(unknown)}")
    command = config.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise ToolConfigError(f"tool capability {name!r} command must be non-empty strings")
    timeout = _configured_timeout(name, config.get("timeout_seconds", 60.0))
    redact_args = config.get("redact_args", [])
    if not isinstance(redact_args, list) or any(not isinstance(item, str) for item in redact_args):
        raise ToolConfigError(f"tool capability {name!r} redact_args must be strings")
    redact_result = config.get("redact_result", False)
    if not isinstance(redact_result, bool):
        raise ToolConfigError(f"tool capability {name!r} redact_result must be boolean")
    remediation = config.get("remediation")
    if remediation is not None and not isinstance(remediation, str):
        raise ToolConfigError(f"tool capability {name!r} remediation must be a string")
    spec = SubprocessToolSpec(
        command=tuple(command),
        timeout_seconds=timeout,
        redact_args=tuple(redact_args),
        redact_result=redact_result,
        remediation=remediation,
    )
    capability = Capability(
        name=name,
        provider="subprocess",
        redact_args=spec.redact_args,
        redact_result=spec.redact_result,
        remediation=spec.remediation or f"configure an executable command for {name}",
    )
    return capability, SubprocessToolProvider(spec)


class CapabilityProvider(Protocol):
    def available(self, name: str) -> bool: ...

    def invoke(self, name: str, /, **kwargs: Any) -> Any: ...


class ToolRegistry:
    """Registry of declared capabilities to provider implementations."""

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}
        self._providers: dict[str, CapabilityProvider] = {}

    def register(self, capability: Capability, provider: CapabilityProvider) -> None:
        self._capabilities[capability.name] = capability
        self._providers[capability.name] = provider

    def known(self) -> tuple[str, ...]:
        return tuple(sorted(self._capabilities))

    def get_capability(self, name: str) -> Capability:
        if name not in self._capabilities:
            raise ToolUnavailableError(f"unknown tool capability {name!r}")
        return self._capabilities[name]

    def get_provider(self, name: str) -> CapabilityProvider:
        self.get_capability(name)
        return self._providers[name]

    def available(self, name: str) -> bool:
        return name in self._capabilities and self._providers[name].available(name)

    def invoke(self, name: str, /, **kwargs: Any) -> Any:
        capability = self.get_capability(name)
        provider = self.get_provider(name)
        record = self._start_record(capability, _normalized_tool_kwargs(name, kwargs))
        started = time.time()
        try:
            if not provider.available(name):
                raise ToolUnavailableError(f"tool capability {name!r} is unavailable")
            result = _normalize_standard_tool_result(
                name,
                kwargs,
                provider.invoke(name, **kwargs),
            )
        except Exception as exc:
            _finish_record(record, started, error=f"{type(exc).__name__}: {exc}")
            raise
        _finish_record(record, started, result=_recorded_result(capability, result))
        return result

    def _start_record(
        self,
        capability: Capability,
        kwargs: dict[str, Any],
    ) -> trace.ToolCallRecord | None:
        ctx = trace.current()
        if ctx is None or not ctx.record_stack:
            return None
        record = trace.ToolCallRecord(
            tool=capability.name,
            kwargs=trace.redact(kwargs, capability.redact_args),
            result=None,
            error=None,
            duration_seconds=0.0,
        )
        ctx.record_stack[-1].tool_calls.append(record)
        return record


def _normalize_standard_tool_result(  # noqa: C901 - normalizes paths for Grep and Glob results
    name: str,
    kwargs: dict[str, Any],
    result: Any,
) -> Any:
    if name == "Grep":
        # The built-in search boundary is line-oriented text. Custom providers
        # may expose typed results under the same capability name; preserve
        # those values because no path grammar can be proven for them.
        return _normalize_grep_result(kwargs, result) if isinstance(result, str) else result
    if name != "Glob":
        return result
    root = Path(str(kwargs.get("path", "."))).resolve()
    if isinstance(result, str):
        values: list[object] = result.splitlines()
    elif isinstance(result, list | tuple):
        values = list(result)
    else:
        raise ToolExecutionError(
            "Glob provider result must be a string or sequence of path strings"
        )
    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ToolExecutionError("Glob provider result paths must be strings")
        rendered = value.strip()
        if not rendered:
            continue
        path = Path(rendered)
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ToolExecutionError(
                f"Glob provider result escapes declared path root: {rendered!r}"
            ) from exc
        normalized.add(relative.as_posix())
    return sorted(normalized)


_GREP_LINE_RE = re.compile(r"^(.*):([1-9][0-9]*):(.*)$")


def _normalize_grep_result(kwargs: dict[str, Any], result: Any) -> str:
    if not isinstance(result, str):  # pragma: no cover - guarded by the caller
        raise ToolExecutionError("Grep provider result must be a string")
    root = Path(str(kwargs.get("path", "."))).resolve()
    normalized: list[str] = []
    for line in result.splitlines():
        if not line.strip():
            continue
        match = _GREP_LINE_RE.fullmatch(line)
        if match is None:
            raise ToolExecutionError(f"Grep provider returned a malformed result line: {line!r}")
        source, line_number, text = match.groups()
        syntax = PathSyntax.inspect(source, normalize_backslashes=True)
        path = syntax.native
        if path.is_absolute():
            try:
                rendered = path.resolve().relative_to(root).as_posix()
            except ValueError as exc:
                raise ToolExecutionError(
                    f"Grep provider result escapes declared path root: {source!r}"
                ) from exc
        else:
            pure = syntax.posix
            if pure.is_absolute() or ".." in pure.parts or pure.as_posix() in {"", "."}:
                raise ToolExecutionError(
                    f"Grep provider result has an unsafe relative path: {source!r}"
                )
            rendered = pure.as_posix()
        normalized.append(f"{rendered}:{line_number}:{text}")
    return "\n".join(normalized)


def _normalized_tool_kwargs(name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(kwargs)
    if name in {"Glob", "Grep"} and "path" in normalized:
        path = Path(str(normalized["path"]))
        normalized["path"] = "." if path.is_absolute() else path.as_posix()
    return normalized


def _recorded_result(capability: Capability, result: Any) -> Any:
    if capability.redact_result:
        return trace._REDACTED_PLACEHOLDER
    return trace.redact(result)


def _finish_record(
    record: trace.ToolCallRecord | None,
    started: float,
    *,
    result: Any | None = None,
    error: str | None = None,
) -> None:
    if record is None:
        return
    record.result = result
    record.error = error
    record.duration_seconds = time.time() - started


class NullProvider:
    """Stand-in provider for a declared capability that has no configured implementation."""

    def available(self, name: str) -> bool:
        return False

    def invoke(self, name: str, /, **kwargs: Any) -> Any:
        raise ToolUnavailableError(f"tool capability {name!r} is unavailable")


def configured_tool_registry(capabilities: Sequence[str], config_path: Path | None) -> ToolRegistry:
    """Build a run's tool registry from an optional tool config and the declared capabilities.

    A trusted child runs with its parent's registry but enforces its own declared
    capabilities, so every configured provider is registered, not only the ones the
    parent declares. A declared capability without a provider gets a stand-in that
    reports itself unavailable.
    """
    registry = ToolRegistry()
    configured = {} if config_path is None else load_tool_config(config_path)
    for capability, provider in configured.values():
        registry.register(capability, provider)
    for name in capabilities:
        if name not in configured:
            remediation = f"configure a provider for {name}"
            registry.register(Capability(name, "null", remediation=remediation), NullProvider())
    return registry


def tool_preflight(registry: ToolRegistry, names: Sequence[str]) -> trace.ToolPreflightReport:
    """Report each named capability that no available provider serves."""
    missing = tuple(
        trace.CapabilityMiss(name=name, remediation=registry.get_capability(name).remediation)
        for name in names
        if not registry.available(name)
    )
    return trace.ToolPreflightReport(missing=missing)


def render_tool_preflight(report: trace.ToolPreflightReport) -> str:
    if report.ok:
        return "tool preflight: ok"
    lines = ["tool preflight: missing capabilities:"]
    for miss in report.missing:
        suffix = f" ({miss.remediation})" if miss.remediation else ""
        lines.append(f"- {miss.name}{suffix}")
    return "\n".join(lines)


def tool(name: str, /, **kwargs: Any) -> Any:
    """Invoke a declared, host-configured capability with keyword inputs.

    Requires an active recipe step, including the traced recipe body, and a tool
    registry. The host provider validates inputs, pre-approvals and dry-run policy.
    """
    ctx = trace.current()
    if ctx is None or ctx.tool_registry is None:
        raise ToolUnavailableError(f"no tool registry for capability {name!r}")
    if ctx.allowed_capabilities is not None and name not in ctx.allowed_capabilities:
        raise ToolUnavailableError(
            f"tool capability {name!r} is not declared by the active recipe; "
            f"allowed: {sorted(ctx.allowed_capabilities)}"
        )
    return ctx.tool_registry.invoke(name, **kwargs)
