"""Tests for persistent recipe trace export."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import calcifier.agent_events as agent_events_module
from calcifier import cli
from calcifier.agent_providers import ClaudeCli
from calcifier.recipe import RecipeRun
from calcifier.registry import Registry
from calcifier.side_effects import SideEffect
from calcifier.subprocesses import ProcessResult
from calcifier.trace import (
    AgentCallRecord,
    ArtifactRecord,
    CapabilityMiss,
    StepRecord,
    ToolCallRecord,
    ToolPreflightReport,
)
from calcifier.trace_export import (
    TRACE_SCHEMA_VERSION,
    serialize_run,
    serialize_step_record,
    validate_trace,
    write_trace,
)
from calcifier.trace_profile import render_trace_profile
from tests.support.agents import StubProvider


@dataclass
class SecretResult:
    password: str
    public: str


@dataclass
class Verdict:
    approved: bool
    reasoning: str


class Opaque:
    def __repr__(self) -> str:
        return "<Opaque value>"


def _stub_registry() -> Registry:
    reg = Registry()
    provider = StubProvider(default=json.dumps({"text": "hello"}))
    provider.name = "worker"
    reg.register(provider)
    return reg


def _walk_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step in steps:
        out.append(step)
        out.extend(_walk_steps(step["children"]))  # type: ignore[arg-type]
    return out


def _snapshot_run() -> RecipeRun:
    child = StepRecord(
        name="child",
        inputs={"value": 1},
        output="done",
        duration_seconds=0.25,
    )
    parent = StepRecord(
        name="parent",
        inputs={"target": "demo"},
        output={"approved": True},
        subrecipe={"name": "child-recipe", "hash": "def"},
        children=[child],
        duration_seconds=1.0,
        tool_calls=[
            ToolCallRecord(
                tool="Read",
                kwargs={"path": "README.md"},
                result="contents",
                duration_seconds=0.1,
            )
        ],
        side_effects=[
            SideEffect(
                kind="file_write",
                target="report.md",
                idempotency_key="effect-1",
                payload={"bytes": 12},
                reversible=True,
                completed=False,
                error="dry-run",
            )
        ],
    )
    return RecipeRun(
        name="trace-snapshot",
        inputs={"target": "demo", "api_key": "secret"},
        output={"status": "ok"},
        duration_seconds=2.5,
        records=[parent],
        artifacts=[ArtifactRecord(name="report", value={"path": "report.md"})],
        tool_preflight=ToolPreflightReport(
            missing=(CapabilityMiss(name="Slack.postMessage", remediation="configure it"),)
        ),
        recipe_hash="recipe-sha",
    )


def test_trace_v3_serialization_and_json_schema():
    actual = serialize_run(_snapshot_run())
    assert actual["version"] == "1"
    assert actual["state"] == "running"
    assert actual["tool_preflight"] == {
        "missing": [{"name": "Slack.postMessage", "remediation": "configure it"}],
        "ok": False,
    }
    validate_trace(actual)


def test_trace_profile_aggregates_nested_calls_without_reading_payloads():
    slow = AgentCallRecord(
        role="auditor",
        target_type="builtin",
        target_name="codex",
        runtime={"mode": "host"},
        attempts=2,
        duration_seconds=9.0,
        prompt_bytes=[1000, 1100],
        schema_bytes=[50, 50],
    )
    slow.usage.update(input_tokens=100, output_tokens=20, cache_read_tokens=80)
    slow.usage_reported_attempts.update(input_tokens=2, output_tokens=2, cache_read_tokens=2)
    fast = AgentCallRecord(
        role="verifier",
        target_type="builtin",
        target_name="codex",
        runtime={"mode": "host"},
        attempts=1,
        duration_seconds=3.0,
        prompt_bytes=[500],
        schema_bytes=[30],
    )
    fast.usage.update(input_tokens=50, output_tokens=10, cache_read_tokens=0)
    fast.usage_reported_attempts.update(input_tokens=1, output_tokens=1, cache_read_tokens=1)
    run = RecipeRun(
        name="profiled",
        inputs={"password": "never print"},
        duration_seconds=10.0,
        records=[
            StepRecord(
                name="scan",
                inputs={},
                children=[StepRecord(name="audit", inputs={}, agent_calls=[slow, fast])],
            )
        ],
    )

    rendered = render_trace_profile(serialize_run(run))

    assert "wall: 10.0s" in rendered
    assert "agent calls: 2, attempts: 3, extra attempts: 1, summed agent wall: 12.0s" in rendered
    assert "prompt bytes: 2,600; schema bytes: 130" in rendered
    assert "input_tokens=150" in rendered
    assert "input_tokens=100, output_tokens=20" in rendered
    assert rendered.index("9.0s auditor @ scan/audit") < rendered.index(
        "3.0s verifier @ scan/audit"
    )
    assert "never print" not in rendered


def test_cli_profile_reads_trace_and_rejects_invalid_input(tmp_path: Path, capsys) -> None:
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(serialize_run(_snapshot_run())))

    assert cli.main(["profile", str(path)]) == 0
    assert "recipe: trace-snapshot" in capsys.readouterr().out

    path.write_text("{}")
    assert cli.main(["profile", str(path)]) == 1
    assert "trace profile failed: unsupported trace version" in capsys.readouterr().err


def test_serialize_run_basic_shape():
    run = RecipeRun(
        name="basic",
        inputs={"target": "x"},
        output=True,
        duration_seconds=0.1,
        records=[StepRecord(name="only", inputs={"target": "x"}, output=True)],
    )
    data = json.loads(json.dumps(serialize_run(run)))
    assert list(data) == [
        "version",
        "state",
        "recipe",
        "inputs",
        "output",
        "output_directory",
        "error",
        "ok",
        "duration_seconds",
        "budget",
        "agent_runtime",
        "agent_preflight",
        "tool_preflight",
        "agent_calls",
        "steps",
        "artifacts",
        "agent_transport",
    ]
    assert data["agent_transport"] == []
    assert data["steps"][0]["name"] == "only"


def test_serialize_run_includes_recipe_hash():
    data = serialize_run(RecipeRun(name="hashed", inputs={}, recipe_hash="abc"))
    assert data["recipe"]["hash"] == "abc"


def test_serialize_run_failed_run():
    run = RecipeRun(name="failed", inputs={}, error="boom")
    data = serialize_run(run)
    assert data["ok"] is False
    assert data["error"] == "boom"
    assert data["output"] is None


def test_serialize_run_nested_step_records():
    child = StepRecord(name="child", inputs={}, output="ok")
    parent = StepRecord(name="parent", inputs={}, children=[child])
    data = serialize_run(RecipeRun(name="nested", inputs={}, records=[parent]))
    assert data["steps"][0]["children"][0]["name"] == "child"
    assert data["steps"][0]["children"][0]["children"] == []


def test_serialize_step_record_includes_subrecipe_marker():
    plain = serialize_step_record(StepRecord(name="plain", inputs={}))
    marked = serialize_step_record(
        StepRecord(
            name="invoke",
            inputs={},
            subrecipe={"name": "child", "hash": "abc123"},
        )
    )

    assert plain["subrecipe"] is None
    assert marked["subrecipe"] == {"name": "child", "hash": "abc123"}


def test_serialize_run_redacts_recipe_level_inputs():
    run = RecipeRun(name="redacted", inputs={"api_key": "shhh", "target": "x"}, output=True)
    data = serialize_run(run)
    assert data["inputs"] == {"api_key": "<redacted>", "target": "x"}


def test_serialize_run_redacts_structured_secret_suffixes():
    run = RecipeRun(
        name="redacted",
        inputs={"github_token": "secret", "session-id": "secret", "hockey": "public"},
        output=True,
    )

    data = serialize_run(run)

    assert data["inputs"] == {
        "github_token": "<redacted>",
        "session-id": "<redacted>",
        "hockey": "public",
    }


def test_serialize_run_does_not_redact_nonsecret_key_suffix():
    run = RecipeRun(name="public", inputs={"sort_key": "created_at"}, output=True)

    data = serialize_run(run)

    assert data["inputs"] == {"sort_key": "created_at"}


def test_serialize_run_redacts_recipe_output_dataclass():
    run = RecipeRun(
        name="redacted-output",
        inputs={},
        output=SecretResult(password="shhh", public="ok"),  # noqa: S106
    )
    data = serialize_run(run)
    assert data["output"] == {"password": "<redacted>", "public": "ok"}


def test_serialize_run_dataclass_in_output_becomes_dict():
    run = RecipeRun(name="dataclass-output", inputs={}, output=Verdict(True, "ok"))
    data = serialize_run(run)
    assert data["output"] == {"approved": True, "reasoning": "ok"}


def test_serialize_run_path_and_bytes_handled():
    run = RecipeRun(
        name="bytes",
        inputs={},
        output={"path": Path("/tmp/x"), "payload": b"abc"},  # noqa: S108
    )
    data = serialize_run(run)
    assert data["output"]["path"] == "/tmp/x"  # noqa: S108
    assert data["output"]["payload"] == {"__bytes_b64__": base64.b64encode(b"abc").decode("ascii")}


@pytest.mark.parametrize(
    "value,restore",
    [
        (date(2026, 1, 2), date.fromisoformat),
        (datetime(2026, 1, 2, 3, 4, 5, 678901, tzinfo=UTC), datetime.fromisoformat),
        (Decimal("12345678901234567890.00000000000000000001"), Decimal),
    ],
    ids=["date", "aware-datetime", "decimal-precision"],
)
def test_persisted_trace_preserves_dates_and_decimal_precision(tmp_path: Path, value, restore):
    payload = {"values": [value]}
    run = RecipeRun(
        name="data-preparation",
        inputs=payload,
        output=payload,
        records=[StepRecord(name="prepare", inputs=payload, output=payload)],
        artifacts=[ArtifactRecord(name="prepared", value=payload)],
    )
    path = tmp_path / "trace.json"
    write_trace(run, path)
    data = json.loads(path.read_bytes())
    validate_trace(data)

    for recorded in (
        data["inputs"],
        data["output"],
        data["steps"][0]["inputs"],
        data["steps"][0]["output"],
        data["artifacts"][0]["value"],
    ):
        (serialized,) = recorded["values"]
        assert isinstance(serialized, str)
        assert restore(serialized) == value


def test_serialize_run_uses_portable_path_separators():
    data = serialize_run(RecipeRun(name="path", inputs={}, output=Path("reports/result.json")))

    assert data["output"] == "reports/result.json"


def test_serialize_run_unknown_object_falls_back_to_repr():
    data = serialize_run(RecipeRun(name="opaque", inputs={}, output=Opaque()))
    assert data["output"] == {"__repr__": "<Opaque value>"}
    json.dumps(data)


def test_serialize_run_includes_tool_calls():
    run = RecipeRun(
        name="tools",
        inputs={},
        records=[
            StepRecord(
                name="only",
                inputs={},
                tool_calls=[
                    ToolCallRecord(
                        tool="Read",
                        kwargs={"path": "x"},
                        result="contents",
                        error=None,
                        duration_seconds=0.01,
                    )
                ],
            )
        ],
    )
    data = serialize_run(run)
    call = data["steps"][0]["tool_calls"][0]
    assert call["tool"] == "Read"
    assert call["kwargs"] == {"path": "x"}
    assert call["result"] == "contents"


def test_serialize_run_includes_side_effects():
    run = RecipeRun(
        name="effects",
        inputs={},
        records=[
            StepRecord(
                name="only",
                inputs={},
                side_effects=[
                    SideEffect(
                        kind="file_write",
                        target="file:/tmp/report.md",
                        idempotency_key="abc123",
                        payload={"bytes": 12},
                        reversible=True,
                        completed=False,
                        error="blocked",
                    )
                ],
            )
        ],
    )
    data = serialize_run(run)
    effect = data["steps"][0]["side_effects"][0]
    assert effect == {
        "kind": "file_write",
        "target": "file:/tmp/report.md",
        "idempotency_key": "abc123",
        "payload": {"bytes": 12},
        "reversible": True,
        "completed": False,
        "error": "blocked",
    }


def test_write_trace_atomic_to_disk(tmp_path: Path):
    out = tmp_path / "trace.json"
    run = RecipeRun(name="disk", inputs={}, output=True)
    write_trace(run, out)
    data = json.loads(out.read_text())
    assert data["version"] == TRACE_SCHEMA_VERSION
    assert list(tmp_path.iterdir()) == [out]


def test_checkpoint_trace_is_compact_and_still_schema_valid(tmp_path: Path):
    out = tmp_path / "trace.json"
    run = RecipeRun(name="disk", inputs={}, output=True)
    write_trace(run, out, checkpoint=True)
    text = out.read_text()
    assert "\n" not in text
    validate_trace(json.loads(text))


def test_cli_exec_writes_trace_when_flag_set(tmp_path: Path, monkeypatch):
    out = tmp_path / "trace.json"
    fixture = Path(__file__).parent / "fixtures" / "trivial_recipe.py"
    recipe_path = tmp_path / "trivial_recipe.py"
    recipe_path.write_text(fixture.read_text())
    monkeypatch.setattr(cli, "_registry_for_args", lambda args, providers: _stub_registry())
    rc = cli.main(["exec", str(recipe_path), "--trust", "--trace-out", str(out)])
    data = json.loads(out.read_text())
    assert rc == 0
    assert data["version"] == "1"
    assert data["recipe"]["name"] == "trivial"
    filesystem = data["steps"][0]["agent_calls"][0]["filesystem"]
    assert sorted(filesystem) == ["mode", "readable_paths", "working_directory", "writable_paths"]
    assert filesystem["writable_paths"] == []
    assert filesystem["mode"] == "read-only"


def _claude_stream(structured_output: object) -> bytes:
    """A completed Claude stream-json turn, as the CLI prints it, with two models."""
    result = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "structured_output": structured_output,
        "total_cost_usd": 0.41,
        "usage": {
            "input_tokens": 4,
            "output_tokens": 30,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 10,
            "output_tokens_details": {"thinking_tokens": 12},
        },
        "modelUsage": {
            "primary-model[1m]": {
                "inputTokens": 4,
                "outputTokens": 30,
                "cacheReadInputTokens": 20,
                "cacheCreationInputTokens": 10,
                "thinkingTokens": 12,
            },
            "auxiliary-model": {"inputTokens": 80, "outputTokens": 3},
        },
    }
    events = [{"type": "system", "subtype": "init"}, result]
    return "\n".join(json.dumps(event) for event in events).encode()


def test_cli_exec_with_claude_writes_a_schema_valid_trace(tmp_path: Path, monkeypatch):
    out = tmp_path / "trace.json"
    fixture = Path(__file__).parent / "fixtures" / "trivial_recipe.py"
    recipe_path = tmp_path / "trivial_recipe.py"
    recipe_path.write_text(fixture.read_text())
    registry = Registry()
    registry.register(ClaudeCli(role="worker", default_timeout=30))
    stream = _claude_stream({"text": "hello"})

    def claude_process(argv, **kwargs):  # noqa: ANN001, ANN003
        kwargs["stdout_consumer"](stream)
        return ProcessResult(tuple(argv), 0, b"", b"")

    monkeypatch.setattr(agent_events_module, "run_process", claude_process)
    monkeypatch.setattr(cli, "_registry_for_args", lambda args, providers: registry)
    monkeypatch.chdir(tmp_path)

    rc = cli.main(["exec", str(recipe_path), "--trust", "--trace-out", str(out)])

    data = json.loads(out.read_text())
    assert rc == 0
    validate_trace(data)
    (call,) = [call for step in _walk_steps(data["steps"]) for call in step["agent_calls"]]
    assert call["usage"] == {
        "input_tokens": 84,
        "output_tokens": 33,
        "cache_read_tokens": 20,
        "cache_write_tokens": 10,
        "nested_model_tokens": 12,
        "cost_usd": 0.41,
    }


def test_cli_exec_trace_export_failure_does_not_mask_run_status(
    tmp_path: Path, capsys, monkeypatch
):
    fixture = Path(__file__).parent / "fixtures" / "trivial_recipe.py"
    recipe_path = tmp_path / "trivial_recipe.py"
    recipe_path.write_text(fixture.read_text())
    monkeypatch.setattr(cli, "_registry_for_args", lambda args, providers: _stub_registry())
    rc = cli.main(["exec", str(recipe_path), "--trust", "--trace-out", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "trace export failed" in captured.err
