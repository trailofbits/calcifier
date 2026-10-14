"""Tests for declarative tool capability plumbing."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from calcifier import trace
from calcifier.tools import (
    Capability,
    SubprocessToolProvider,
    SubprocessToolSpec,
    ToolConfigError,
    ToolExecutionError,
    ToolRegistry,
    ToolUnavailableError,
    load_tool_config,
    tool,
)


class StubToolProvider:
    def __init__(self, *, available: bool = True, result: Any = "ok") -> None:
        self.is_available = available
        self.result = result
        self.seen: list[tuple[str, dict[str, Any]]] = []

    def available(self, name: str) -> bool:
        return self.is_available

    def invoke(self, name: str, /, **kwargs: Any) -> Any:
        self.seen.append((name, kwargs))
        return self.result


def _registry(
    capability: Capability | None = None,
    provider: StubToolProvider | None = None,
) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(capability or Capability("Read", "stub"), provider or StubToolProvider())
    return reg


def test_capability_registry_register_and_lookup():
    provider = StubToolProvider()
    capability = Capability("Read", "stub")
    reg = _registry(capability, provider)
    assert reg.known() == ("Read",)
    assert reg.get_capability("Read") == capability
    assert reg.get_provider("Read") is provider
    assert reg.available("Read") is True


def test_capability_registry_invoke_dispatches_to_provider():
    provider = StubToolProvider(result={"ok": True})
    reg = _registry(provider=provider)
    assert reg.invoke("Read", path="x") == {"ok": True}
    assert provider.seen == [("Read", {"path": "x"})]


@pytest.mark.parametrize(
    "provider_result",
    [
        lambda root: [str(root / "zeta.txt"), "alpha.txt", str(root / "alpha.txt")],
        lambda root: f"{root / 'zeta.txt'}\nalpha.txt\n{root / 'alpha.txt'}\n",
    ],
)
def test_glob_results_are_canonical_relative_paths_across_provider_shapes(
    tmp_path: Path,
    provider_result: Callable[[Path], object],
):
    root = tmp_path / "target"
    root.mkdir()
    provider = StubToolProvider(result=provider_result(root))
    registry = _registry(Capability("Glob", "stub"), provider)

    discovered = registry.invoke("Glob", path=str(root), pattern="**/*")

    assert discovered == ["alpha.txt", "zeta.txt"]
    assert all(not path.startswith("/") for path in discovered)


@pytest.mark.parametrize(
    "provider_result",
    [["../outside.txt"], [42], {"path": "record.txt"}],
)
def test_glob_result_rejects_escaping_or_unsupported_provider_values(
    tmp_path: Path,
    provider_result: object,
):
    root = tmp_path / "target"
    root.mkdir()
    registry = _registry(
        Capability("Glob", "stub"),
        StubToolProvider(result=provider_result),
    )

    with pytest.raises(ToolExecutionError, match="Glob provider"):
        registry.invoke("Glob", path=str(root), pattern="**/*")


def test_glob_result_rejects_absolute_path_outside_declared_root(tmp_path: Path):
    root = tmp_path / "target"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    registry = _registry(
        Capability("Glob", "stub"),
        StubToolProvider(result=[str(outside)]),
    )

    with pytest.raises(ToolExecutionError, match="escapes declared path root"):
        registry.invoke("Glob", path=str(root), pattern="**/*")


def test_glob_result_rejects_symlink_escape(tmp_path: Path):
    root = tmp_path / "target"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    registry = _registry(
        Capability("Glob", "stub"),
        StubToolProvider(result=["link/record.txt"]),
    )

    with pytest.raises(ToolExecutionError, match="escapes declared path root"):
        registry.invoke("Glob", path=str(root), pattern="**/*")


def test_glob_trace_records_the_canonical_result(tmp_path: Path):
    root = tmp_path / "target"
    root.mkdir()
    registry = _registry(
        Capability("Glob", "stub"),
        StubToolProvider(result=[str(root / "record.txt")]),
    )

    with trace.active_run(tool_registry=registry):
        with trace.step_frame("discover", list[str], {}):
            assert tool("Glob", path=str(root), pattern="**/*") == ["record.txt"]
        context = trace.current()
        assert context is not None
        call = context.records[0].tool_calls[0]

    assert call.result == ["record.txt"]


def test_grep_results_and_trace_paths_are_canonical_at_the_search_boundary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "target"
    root.mkdir()
    source = root / "src" / "alpha.py"
    provider = StubToolProvider(result=f"{source}:7:value = call()\n")
    registry = _registry(Capability("Grep", "stub"), provider)

    with trace.active_run(tool_registry=registry):
        with trace.step_frame("search", str, {}):
            result = tool("Grep", path=str(root), pattern="call")
        context = trace.current()
        assert context is not None
        call = context.records[0].tool_calls[0]

    assert result == "src/alpha.py:7:value = call()"
    assert call.kwargs == {"path": ".", "pattern": "call"}
    assert call.result == "src/alpha.py:7:value = call()"
    assert str(tmp_path) not in repr(call.kwargs)
    assert str(tmp_path) not in repr(call.result)


@pytest.mark.parametrize(
    "result",
    ["../outside.py:1:value", "not-a-grep-line"],
)
def test_grep_normalization_rejects_unsafe_or_malformed_results(
    tmp_path: Path, result: object
) -> None:
    registry = _registry(Capability("Grep", "stub"), StubToolProvider(result=result))

    with pytest.raises(ToolExecutionError, match="Grep provider"):
        registry.invoke("Grep", path=str(tmp_path), pattern="value")


def test_grep_custom_structured_result_is_preserved() -> None:
    result = [{"source": "alpha.py", "line": 1}]
    registry = _registry(Capability("Grep", "stub"), StubToolProvider(result=result))

    assert registry.invoke("Grep", pattern="value") == result


def test_capability_registry_invoke_records_to_trace():
    reg = _registry(provider=StubToolProvider(result="contents"))
    with trace.active_run(tool_registry=reg):
        with trace.step_frame("s", bool, {}):
            assert tool("Read", path="x") == "contents"
        ctx = trace.current()
        assert ctx is not None
        calls = ctx.records[0].tool_calls
    assert len(calls) == 1
    assert calls[0].tool == "Read"
    assert calls[0].kwargs == {"path": "x"}
    assert calls[0].result == "contents"
    assert calls[0].error is None


def test_capability_redacts_marked_arguments():
    capability = Capability("Read", "stub", redact_args=("token",))
    reg = _registry(capability=capability)
    with trace.active_run(tool_registry=reg):
        with trace.step_frame("s", bool, {}):
            tool("Read", path="x", token="secret")  # noqa: S106
        ctx = trace.current()
        assert ctx is not None
        call = ctx.records[0].tool_calls[0]
    assert call.kwargs == {"path": "x", "token": "<redacted>"}


def test_capability_redacts_result_when_marked():
    capability = Capability("Read", "stub", redact_result=True)
    reg = _registry(capability=capability, provider=StubToolProvider(result={"secret": "x"}))
    with trace.active_run(tool_registry=reg):
        with trace.step_frame("s", bool, {}):
            assert tool("Read") == {"secret": "x"}
        ctx = trace.current()
        assert ctx is not None
        call = ctx.records[0].tool_calls[0]
    assert call.result == "<redacted>"


def test_tool_unavailable_error_raised_on_unknown_name():
    with trace.active_run(tool_registry=ToolRegistry()), pytest.raises(ToolUnavailableError):
        tool("DoesNotExist")


def test_active_recipe_capabilities_restrict_configured_tools():
    with (
        trace.active_run(
            tool_registry=_registry(),
            allowed_capabilities=frozenset(),
        ),
        pytest.raises(ToolUnavailableError, match="not declared by the active recipe"),
    ):
        tool("Read", path="README.md")


def test_tool_unavailable_error_raised_when_provider_unavailable():
    reg = _registry(provider=StubToolProvider(available=False))
    with trace.active_run(tool_registry=reg), pytest.raises(ToolUnavailableError):
        tool("Read")


def test_subprocess_tool_provider_json_wire_contract(tmp_path: Path):
    script = tmp_path / "provider.py"
    script.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "json.dump({'seen': request}, sys.stdout)\n"
    )
    provider = SubprocessToolProvider(SubprocessToolSpec((sys.executable, str(script))))

    assert provider.invoke("Read", path="README.md") == {
        "seen": {"capability": "Read", "arguments": {"path": "README.md"}}
    }


@pytest.mark.parametrize(
    "body, match",
    [
        ("import sys; sys.stdout.buffer.write(b'\\xff')", "valid UTF-8"),
        ("print('not-json')", "exactly one JSON value"),
        ("import sys; print('bad', file=sys.stderr); raise SystemExit(3)", "exited 3"),
    ],
)
def test_subprocess_tool_provider_reports_wire_failures(tmp_path: Path, body: str, match: str):
    script = tmp_path / "provider.py"
    script.write_text(body)
    provider = SubprocessToolProvider(SubprocessToolSpec((sys.executable, str(script))))

    with pytest.raises(ToolExecutionError, match=match):
        provider.invoke("Read")


def test_subprocess_tool_stderr_is_interactive_not_persisted(tmp_path: Path, capsys):
    script = tmp_path / "provider.py"
    script.write_text("import sys; print('secret-detail', file=sys.stderr); raise SystemExit(3)")
    provider = SubprocessToolProvider(SubprocessToolSpec((sys.executable, str(script))))

    with pytest.raises(ToolExecutionError) as excinfo:
        provider.invoke("Read")

    assert "secret-detail" not in str(excinfo.value)
    assert "secret-detail" in capsys.readouterr().err


def test_load_tool_config_registers_versioned_capability(tmp_path: Path):
    path = tmp_path / "tools.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "capabilities": {
                    "Read": {
                        "command": [sys.executable, "provider.py"],
                        "timeout_seconds": 5,
                        "redact_args": ["token"],
                        "redact_result": True,
                        "remediation": "install provider",
                    }
                },
            }
        )
    )

    capability, provider = load_tool_config(path)["Read"]

    assert capability == Capability(
        name="Read",
        provider="subprocess",
        redact_args=("token",),
        redact_result=True,
        remediation="install provider",
    )
    assert provider.spec.timeout_seconds == 5


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"version": 2, "capabilities": {}},
        {"version": 1, "capabilities": {"Read": {"command": []}}},
        {"version": 1, "capabilities": {"Read": {"command": ["x"], "extra": True}}},
    ],
)
def test_load_tool_config_rejects_invalid_contract(tmp_path: Path, payload: dict[str, Any]):
    path = tmp_path / "tools.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ToolConfigError):
        load_tool_config(path)


@pytest.mark.parametrize("timeout", [float("inf"), float("nan"), 86_401])
def test_load_tool_config_rejects_unbounded_timeout(tmp_path: Path, timeout: float):
    path = tmp_path / "tools.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "capabilities": {"Read": {"command": ["provider"], "timeout_seconds": timeout}},
            }
        )
    )

    with pytest.raises(ToolConfigError, match="timeout_seconds must be finite"):
        load_tool_config(path)


def _write_script(tmp_path: Path, name: str, body: str) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    script = scripts / name
    script.write_text(body)
    return script


def test_subprocess_tool_enforces_timeout(tmp_path: Path):
    script = tmp_path / "provider.py"
    script.write_text("import time; time.sleep(10)")
    provider = SubprocessToolProvider(
        SubprocessToolSpec((sys.executable, str(script)), timeout_seconds=0.05)
    )

    with pytest.raises(ToolExecutionError, match="timed out"):
        provider.invoke("Read")


def test_subprocess_tool_enforces_output_limit(tmp_path: Path):
    script = tmp_path / "provider.py"
    script.write_text("print('x' * 1100000)")
    provider = SubprocessToolProvider(SubprocessToolSpec((sys.executable, str(script))))

    with pytest.raises(ToolExecutionError, match="stdout exceeded"):
        provider.invoke("Read")


def test_subprocess_tool_reports_stderr_output_limit(tmp_path: Path):
    script = tmp_path / "provider.py"
    script.write_text("import sys; print('x' * 1100000, file=sys.stderr)")
    provider = SubprocessToolProvider(SubprocessToolSpec((sys.executable, str(script))))

    with pytest.raises(ToolExecutionError, match="stderr exceeded"):
        provider.invoke("Read")
