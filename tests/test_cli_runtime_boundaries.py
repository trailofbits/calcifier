"""Runner boundaries preserve inputs and fail before undispatched work."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from calcifier import cli, load_recipe, recipe, step
from calcifier.metadata import attach_metadata
from calcifier.recipe import read_recipe_snapshot
from calcifier.registry import Registry
from calcifier.subprocesses import ProcessResult
from tests.support.agents import StubProvider


def write_node(root: Path, name: str, source: str, children=None) -> Path:
    skill = root / "SKILL.md"
    skill.write_text("# Generic planner worker fixture\n")
    target = root / f"{name}.py"
    target.write_text(attach_metadata(source, skill, recipe_path=target, subrecipes=children))
    return target


def test_direct_cli_delivers_scalar_types_and_unparsed_text(tmp_path, capsys):
    received = []

    @recipe
    def inspect_inputs(label: str, count: int, ratio: float, target: Path, extra: object) -> bool:
        received.append((label, count, ratio, target, extra))
        return True

    assert (
        inspect_inputs.cli(
            [
                "--label",
                "001",
                "--count",
                "21",
                "--ratio",
                "2.5",
                "--target",
                str(tmp_path),
                "--extra",
                "[not json",
            ]
        )
        == 0
    )
    assert received == [("001", 21, 2.5, tmp_path, "[not json")]
    assert tuple(type(value) for value in received[0]) == (str, int, float, type(tmp_path), str)
    assert " :: OK ===" in capsys.readouterr().out


@pytest.mark.parametrize(
    "raw", ["[unfinished", "{invalid", "true", "false", "null", "007", "1e3", '["json"]']
)
def test_exec_preserves_invalid_json_and_lowercase_bool_text(tmp_path, capsys, raw):
    source = tmp_path / "recipe.py"
    source.write_text(
        "from calcifier import recipe\n@recipe\ndef echo(text: str) -> str:\n    return text\n",
        encoding="utf-8",
    )
    trace = tmp_path / "trace.json"
    assert (
        cli.main(
            ["exec", str(source), "--trust", "--input", f"text={raw}", "--trace-out", str(trace)]
        )
        == 0
    )
    assert json.loads(trace.read_bytes())["output"] == raw
    assert " :: OK ===" in capsys.readouterr().out


@pytest.mark.parametrize("direct", [False, True], ids=["exec", "direct"])
@pytest.mark.parametrize("choice", ["red", "green", "blue", "purple"])
def test_cli_literal_inputs_report_allowed_values(tmp_path, capsys, direct, choice):
    source = tmp_path / "recipe.py"
    source.write_bytes(
        b"from typing import Literal\n"
        b"from calcifier import recipe\n"
        b"@recipe\n"
        b"def choose(choice: Literal['red', 'green', 'blue']) -> str:\n"
        b"    return choice\n"
    )
    if direct:
        loaded = load_recipe(source)
        if choice == "purple":
            with pytest.raises(SystemExit) as caught:
                loaded.cli(["--choice", choice])
            assert caught.value.code == 2
        else:
            assert loaded.cli(["--choice", choice]) == 0
    else:
        assert cli.main(["exec", str(source), "--trust", "--input", f"choice={choice}"]) == (
            2 if choice == "purple" else 0
        )

    captured = capsys.readouterr()
    if choice == "purple":
        assert "input coercion failed" in captured.err
        assert "Input should be 'red', 'green' or 'blue'" in captured.err
        assert captured.out == ""
    else:
        assert " :: OK ===" in captured.out
        assert choice in captured.out


def test_step_factory_redacts_secret_arguments_but_runs_with_original_value():
    @step(redacted_kwargs=("payload",))
    def check_private(payload: str) -> bool:
        return payload == "private-input"

    @recipe
    def invoke() -> bool:
        return check_private(payload="private-input")

    result = invoke.run()
    assert result.ok and result.output is True
    assert "private-input" not in repr(result.records)
    assert len(result.records) == 1
    assert result.records[0].inputs == {"payload": "<redacted>"}


@pytest.mark.parametrize("missing", ["role", "capability"])
def test_transitive_preflight_failure_returns_failed_trace_before_execution(tmp_path, missing):
    worker = write_node(
        tmp_path,
        "worker",
        "from calcifier import recipe, Budget, tool\n"
        f"@recipe(agents={('reviewer',) if missing == 'role' else ()!r}, "
        f"capabilities={('Read',) if missing == 'capability' else ()!r}, budget=Budget(1, 30))\n"
        "def worker() -> str:\n"
        "    raise AssertionError('must not dispatch')\n"
        + ('    return str(tool("Read", path="input.txt"))\n' if missing == "capability" else ""),
    )
    parent = write_node(
        tmp_path,
        "parent",
        "from calcifier import recipe, call_recipe, Budget\n"
        "@recipe(budget=Budget(1, 30))\n"
        "def parent() -> str:\n"
        "    return call_recipe('worker.py', {}).output\n",
        {"worker": str(worker)},
    )
    trace = tmp_path / "trace.json"
    result = load_recipe(parent).run(trace_path=trace)
    assert result.state == "failed" and not result.ok
    assert result.records == []
    assert result.error is not None
    assert "graph preflight missing" in result.error
    assert ("reviewer" if missing == "role" else "Read") in result.error
    assert "must not dispatch" not in result.error
    saved = json.loads(trace.read_bytes())
    assert not saved["ok"] and saved["error"] == result.error


def test_transitive_agent_requires_parent_shared_budget_before_dispatch(tmp_path):
    worker = write_node(
        tmp_path,
        "worker",
        "from calcifier import recipe, Budget\n"
        "@recipe(agents=('reviewer',), budget=Budget(1, 30))\n"
        "def worker() -> str:\n"
        "    raise AssertionError('must not dispatch')\n",
    )
    parent = write_node(
        tmp_path,
        "parent",
        "from calcifier import recipe, call_recipe\n"
        "@recipe\n"
        "def parent() -> str:\n"
        "    return call_recipe('worker.py', {}).output\n",
        {"worker": str(worker)},
    )
    registry = Registry()
    provider = StubProvider()
    provider.name = "reviewer"
    registry.register(provider)
    result = load_recipe(parent).run(registry=registry)
    assert not result.ok
    assert result.error == "agent-backed recipe has no declared Budget"
    assert result.records == []


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("data", b"unreviewed replacement"), ("sha256", "0" * 64), ("path", Path("other.py"))],
)
def test_pinned_recipe_bytes_cannot_be_replaced_after_snapshot(tmp_path, field, replacement):
    path = tmp_path / "recipe.py"
    path.write_bytes(b"pinned bytes")
    snapshot = read_recipe_snapshot(path)
    with pytest.raises(FrozenInstanceError):
        setattr(snapshot, field, replacement)
    assert snapshot.data == b"pinned bytes"


@pytest.mark.parametrize("stderr", [b"", b"specific stderr diagnostic"])
def test_provider_feature_failure_reports_stderr_or_stdout(monkeypatch, stderr):
    result = ProcessResult(("provider", "--help"), 1, b"stdout diagnostic", stderr)
    monkeypatch.setattr(cli, "run_process", lambda *args, **kwargs: result)
    with pytest.raises(SystemExit) as caught:
        cli._run_agent_feature_check(["provider", "--help"], ("--needed",), context="host")
    assert str(caught.value).endswith((stderr or result.stdout).decode())


@pytest.mark.parametrize("missing", [False, True])
def test_provider_features_include_both_output_streams(monkeypatch, missing):
    result = ProcessResult(("provider", "--help"), 0, b"--one", b"--two")
    monkeypatch.setattr(cli, "run_process", lambda *args, **kwargs: result)
    required = ("--one", "--two", "--missing") if missing else ("--one", "--two")
    if missing:
        with pytest.raises(SystemExit, match="does not support --missing$"):
            cli._run_agent_feature_check(["provider", "--help"], required, context="host")
    else:
        cli._run_agent_feature_check(["provider", "--help"], required, context="host")


@pytest.mark.parametrize("seconds", [None, "123"])
def test_successful_graph_build_prints_published_catalog_and_exits_zero(
    monkeypatch, capsys, seconds
):
    from calcifier import graph_build

    registry = Registry()
    calls = []
    requested_roles = []

    def configure(args, providers):
        requested_roles.append(providers)
        return registry

    monkeypatch.setattr(cli, "_registry_for_args", configure)

    def build(
        spec, supplied_registry, *, compilation_timeout, max_compiler_calls, runtime_wall_seconds
    ):
        assert runtime_wall_seconds == (None if seconds is None else 123)
        assert compilation_timeout is None
        assert max_compiler_calls is None
        calls.append((spec, supplied_registry))
        return Path("published/catalog.json")

    monkeypatch.setattr(graph_build, "build_graph", build)
    flags = [] if seconds is None else ["--max-wall-seconds", seconds]
    assert cli.main(["build-graph", "build.json", *flags]) == 0
    assert calls == [(Path("build.json"), registry)]
    assert requested_roles == [("compiler",)]
    assert capsys.readouterr().out.strip() == str(Path("published/catalog.json"))


def test_preflight_only_does_not_execute_recipe(tmp_path, capsys):
    path = tmp_path / "recipe.py"
    path.write_text(
        "from calcifier import recipe\n"
        "@recipe\n"
        "def invoke() -> bool:\n"
        "    raise AssertionError('must not execute')\n",
        encoding="utf-8",
    )
    assert cli.main(["exec", str(path), "--trust", "--preflight-only"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "output_directory": None,
        "agents": [],
        "capabilities": [],
        "unresolved": [],
    }
