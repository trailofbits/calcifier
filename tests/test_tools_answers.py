"""Tool-provider known answers."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from calcifier import (
    recipe,
    step,
    tool,
    trace,
)
from calcifier.subprocesses import MAX_TIMEOUT_SECONDS
from calcifier.tools import (
    Capability,
    SubprocessToolProvider,
    SubprocessToolSpec,
    ToolConfigError,
    ToolRegistry,
    ToolUnavailableError,
    load_tool_config,
)


def _elapsed(seconds: float) -> bool:
    return 0.0 <= seconds < 60.0


# -- tool capabilities ----------------------------------------------------------

_ECHO_STDIN = "import json, sys; json.dump(sys.stdin.read(), sys.stdout)"
_TIMEOUT_ERROR = (
    "tool capability 'Echo' timeout_seconds must be finite, positive, and at most 86400"
)


class _RecordingProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def available(self, name: str) -> bool:
        return name == "Echo"

    def invoke(self, name: str, /, **kwargs: Any) -> Any:
        del name
        self.calls.append(kwargs)
        return kwargs.get("index")


def _tool_config(tmp_path: Path, capabilities: dict[str, Any]) -> Path:
    path = tmp_path / "tools.json"
    path.write_text(json.dumps({"version": 1, "capabilities": capabilities}))
    return path


def _load_echo(tmp_path: Path, config: dict[str, Any]) -> tuple[Capability, SubprocessToolProvider]:
    return load_tool_config(_tool_config(tmp_path, {"Echo": config}))["Echo"]


def _configured_registry(tmp_path: Path, config: dict[str, Any]) -> ToolRegistry:
    registry = ToolRegistry()
    capability, provider = _load_echo(tmp_path, config)
    registry.register(capability, provider)
    return registry


def test_tool_config_defaults_are_exact(tmp_path: Path) -> None:
    command = [sys.executable, "-c", _ECHO_STDIN]

    capability, provider = _load_echo(tmp_path, {"command": command})

    assert capability == Capability(
        name="Echo",
        provider="subprocess",
        redact_args=(),
        redact_result=False,
        remediation="configure an executable command for Echo",
    )
    assert provider.spec == SubprocessToolSpec(
        command=tuple(command),
        timeout_seconds=60.0,
        redact_args=(),
        redact_result=False,
        remediation=None,
    )


@pytest.mark.parametrize("timeout", [5e-324, 0.5, 1, 60, MAX_TIMEOUT_SECONDS])
def test_tool_config_accepts_timeouts_inside_the_inclusive_bound(
    tmp_path: Path, timeout: float
) -> None:
    _, provider = _load_echo(tmp_path, {"command": ["tool"], "timeout_seconds": timeout})

    assert provider.spec.timeout_seconds == float(timeout)


@pytest.mark.parametrize(
    "timeout",
    [
        0,
        0.0,
        -0.5,
        -1,
        math.nextafter(MAX_TIMEOUT_SECONDS, math.inf),
        MAX_TIMEOUT_SECONDS + 1,
        math.inf,
        True,
        "5",
        None,
    ],
)
def test_tool_config_rejects_timeouts_outside_the_bound(tmp_path: Path, timeout: Any) -> None:
    with pytest.raises(ToolConfigError) as raised:
        _load_echo(tmp_path, {"command": ["tool"], "timeout_seconds": timeout})

    assert str(raised.value) == _TIMEOUT_ERROR


@pytest.mark.parametrize("command", [["tool", ""], ["tool", 1], [""], [], "tool", None])
def test_tool_config_rejects_each_invalid_command(tmp_path: Path, command: Any) -> None:
    with pytest.raises(ToolConfigError) as raised:
        _load_echo(tmp_path, {"command": command})

    assert str(raised.value) == "tool capability 'Echo' command must be non-empty strings"


@pytest.mark.parametrize("redact_args", ["api_key", [1], ["api_key", None]])
def test_tool_config_rejects_each_invalid_redaction_list(tmp_path: Path, redact_args: Any) -> None:
    with pytest.raises(ToolConfigError) as raised:
        _load_echo(tmp_path, {"command": ["tool"], "redact_args": redact_args})

    assert str(raised.value) == "tool capability 'Echo' redact_args must be strings"


def test_tool_config_rejects_an_empty_capability_name(tmp_path: Path) -> None:
    with pytest.raises(ToolConfigError) as raised:
        load_tool_config(_tool_config(tmp_path, {"": {"command": ["tool"]}}))

    assert str(raised.value) == "tool capability names must be non-empty strings"


def test_tool_sends_a_sorted_compact_payload_and_records_elapsed_time(tmp_path: Path) -> None:
    registry = _configured_registry(tmp_path, {"command": [sys.executable, "-c", _ECHO_STDIN]})

    @step
    def call() -> str:
        return tool("Echo", zeta=2, alpha={"y": 1, "x": 0})

    @recipe(capabilities=("Echo",))
    def run() -> str:
        return call()

    result = run.run(tool_registry=registry)

    payload = '{"arguments":{"alpha":{"x":0,"y":1},"zeta":2},"capability":"Echo"}'
    assert result.error is None
    assert result.output == payload
    [record] = result.records[0].tool_calls
    assert (record.tool, record.kwargs, record.result, record.error) == (
        "Echo",
        {"zeta": 2, "alpha": {"y": 1, "x": 0}},
        payload,
        None,
    )
    assert _elapsed(record.duration_seconds)


def test_tool_entry_points_without_a_registry_raise_the_exact_error() -> None:
    message = "no tool registry for capability 'Echo'"
    with pytest.raises(ToolUnavailableError) as outside:
        tool("Echo")
    assert str(outside.value) == message
    with (
        trace.active_run(allowed_capabilities=frozenset({"Echo"})),
        pytest.raises(ToolUnavailableError) as inside,
    ):
        tool("Echo")
    assert str(inside.value) == message
