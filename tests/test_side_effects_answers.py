"""Side-effect and command known answers."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from calcifier import (
    CommandResult,
    RecipeError,
    ensure_directory,
    recipe,
    run_command,
    step,
    verify_artifact,
    write_json,
    write_text,
)
from calcifier.side_effects import DryRunMode


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- step cache -----------------------------------------------------------------


# -- declared filesystem mutations and commands ---------------------------------


def _effect(kind: str, target: str, payload: dict[str, Any], *, completed: bool) -> dict[str, Any]:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "kind": kind,
        "target": target,
        "idempotency_key": _sha(f"{kind}|{target}|{canonical}".encode()),
        "payload": payload,
        "reversible": False,
        "completed": completed,
        "error": None,
    }


def _trace_side_effects(trace_path: Path) -> list[dict[str, Any]]:
    return json.loads(trace_path.read_text())["steps"][0]["side_effects"]


def _observe_mkdir(
    monkeypatch: pytest.MonkeyPatch, target: Path, trace_path: Path
) -> list[list[dict[str, Any]]]:
    """Capture the on-disk trace each time the mutation first creates ``target``."""
    observed: list[list[dict[str, Any]]] = []
    real_mkdir = Path.mkdir

    def observing_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == target and kwargs.get("parents") is True:
            observed.append(_trace_side_effects(trace_path))
        real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", observing_mkdir)
    return observed


def test_ensure_directory_is_pending_during_creation_then_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    trace_path = tmp_path / "trace.json"
    target = out.resolve() / "a" / "b"
    observed = _observe_mkdir(monkeypatch, target, trace_path)

    @step
    def create() -> list[str]:
        return [str(ensure_directory("a/b")), str(ensure_directory("a/b"))]

    @recipe(output_root=out)
    def run() -> list[str]:
        return create()

    result = run.run(trace_path=trace_path)

    pending = _effect("file_write", str(target), {"operation": "mkdir"}, completed=False)
    completed = _effect("file_write", str(target), {"operation": "mkdir"}, completed=True)
    assert result.error is None
    assert result.output == [str(target), str(target)]
    assert target.is_dir()
    assert observed == [[pending], [completed]]
    [effect] = result.records[0].side_effects
    assert (effect.completed, effect.error) == (True, None)


def test_ensure_directory_failure_leaves_the_effect_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    target = out.resolve() / "blocked"
    real_mkdir = Path.mkdir

    def refusing_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == target:
            raise OSError("read only")
        real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", refusing_mkdir)

    @step
    def create() -> str:
        return str(ensure_directory("blocked"))

    @recipe(output_root=out)
    def run() -> str:
        return create()

    result = run.run()

    assert result.error == f"RecipeError: failed to create artifact directory {target}: read only"
    [effect] = result.records[0].side_effects
    assert (effect.completed, effect.error) == (False, "read only")


def test_soft_dry_run_mutations_return_exact_targets_without_touching_disk(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    root = out.resolve()

    @step
    def plan() -> list[str]:
        return [
            str(ensure_directory("reports")),
            str(write_text("reports/summary.txt", "summary")),
            str(write_json("reports/state.json", {"done": True})),
            str(verify_artifact("reports/absent.txt")),
        ]

    @recipe(output_root=out)
    def run() -> list[str]:
        return plan()

    result = run.run(dry_run=DryRunMode.SOFT)

    assert result.error is None
    assert result.output == [
        str(root / "reports"),
        str(root / "reports" / "summary.txt"),
        str(root / "reports" / "state.json"),
        str(root / "reports" / "absent.txt"),
    ]
    assert not out.exists()
    effects = result.records[0].side_effects
    assert [(item.payload["operation"], item.completed) for item in effects] == [
        ("mkdir", False),
        ("write_text", False),
        ("write_text", False),
    ]


def test_text_and_json_writes_are_pending_during_io_and_return_their_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    trace_path = tmp_path / "trace.json"
    reports = out.resolve() / "reports"
    observed = _observe_mkdir(monkeypatch, reports, trace_path)
    summary = b"all clear\n"
    state = b'{\n  "alpha": [\n    2,\n    1\n  ],\n  "zeta": 1\n}\n'

    @step
    def persist() -> list[str]:
        return [
            str(write_text("reports/summary.txt", summary.decode())),
            str(write_json("reports/state.json", {"zeta": 1, "alpha": [2, 1]})),
            str(verify_artifact("reports/state.json", sha256=_sha(state))),
        ]

    @recipe(output_root=out)
    def run() -> list[str]:
        return persist()

    result = run.run(trace_path=trace_path)

    text_target, json_target = str(reports / "summary.txt"), str(reports / "state.json")
    text_payload = {"operation": "write_text", "sha256": _sha(summary)}
    json_payload = {"operation": "write_text", "sha256": _sha(state)}
    assert result.error is None
    assert result.output == [text_target, json_target, json_target]
    assert (reports / "state.json").read_bytes() == state
    assert (reports / "summary.txt").read_bytes() == summary
    assert observed == [
        [_effect("file_write", text_target, text_payload, completed=False)],
        [
            _effect("file_write", text_target, text_payload, completed=True),
            _effect("file_write", json_target, json_payload, completed=False),
        ],
    ]
    assert [item.completed for item in result.records[0].side_effects] == [True, True]
    assert [(item.name, item.value) for item in result.artifacts] == [
        ("verified-artifact", {"path": json_target, "sha256": _sha(state), "size": len(state)})
    ]


@pytest.mark.parametrize("argv", [[], [""], ["tool", ""], [1]])
def test_run_command_rejects_each_invalid_argv_shape(argv: list[Any]) -> None:
    with pytest.raises(RecipeError) as raised:
        run_command(argv, timeout_seconds=1)

    assert str(raised.value) == "run_command() argv must contain non-empty strings"


@pytest.mark.parametrize("size", [9, 8 * 1024 * 1024])
def test_read_only_command_returns_output_up_to_eight_mebibytes(tmp_path: Path, size: int) -> None:
    argv = [sys.executable, "-c", f"import sys; sys.stdout.buffer.write(b'x' * {size})"]

    result = run_command(argv, timeout_seconds=60, working_directory=tmp_path, mutating=False)

    assert result == CommandResult(argv=argv, exit_code=0, stdout="x" * size, stderr="")


def test_read_only_command_output_one_byte_over_eight_mebibytes_is_rejected(
    tmp_path: Path,
) -> None:
    size = 8 * 1024 * 1024 + 1
    argv = [sys.executable, "-c", f"import sys; sys.stdout.buffer.write(b'x' * {size})"]

    with pytest.raises(RecipeError) as raised:
        run_command(argv, timeout_seconds=60, working_directory=tmp_path, mutating=False)

    assert str(raised.value) == (
        "command failed to execute: ProcessOutputLimitError: process stdout exceeded 8388608 bytes"
    )


def test_mutating_command_is_pending_while_running_and_completed_after(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    probe = (
        "import json, sys; "
        "print(json.dumps(json.load(open(sys.argv[1]))['steps'][0]['side_effects']))"
    )
    argv = [sys.executable, "-c", probe, str(trace_path)]

    @step
    def command() -> str:
        return run_command(argv, timeout_seconds=60, working_directory=tmp_path).stdout

    @recipe
    def run() -> str:
        return command()

    result = run.run(trace_path=trace_path)

    payload = {"operation": "run_command", "argv": argv, "cwd": str(tmp_path.resolve())}
    assert result.error is None
    assert isinstance(result.output, str)
    assert json.loads(result.output) == [
        _effect("external_object", sys.executable, payload, completed=False)
    ]
    [effect] = result.records[0].side_effects
    assert (effect.completed, effect.error) == (True, None)
