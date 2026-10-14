from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

from calcifier import (
    Budget,
    RecipeError,
    agent,
    ensure_directory,
    parallel,
    recipe,
    step,
    verify_artifact,
    write_json,
    write_text,
)
from calcifier.registry import Registry
from calcifier.side_effects import DryRunMode
from calcifier.trace import active_run
from tests.support.agents import StubProvider


class CountingProvider(StubProvider):
    name = "worker"

    def __init__(self, response: str) -> None:
        super().__init__(default=response)
        self.invocations = 0
        self._lock = threading.Lock()

    def invoke(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        with self._lock:
            self.invocations += 1
        return super().invoke(*args, **kwargs)


def _registry(provider: CountingProvider) -> Registry:
    registry = Registry()
    registry.register(provider)
    return registry


def test_declared_atomic_mutations_are_verified_and_traced(tmp_path):
    @step
    def produce() -> bool:
        write_json("state.json", {"status": "complete"})
        verify_artifact("state.json")
        return True

    @recipe(output_root=tmp_path)
    def run() -> bool:
        return produce()

    result = run.run()

    assert result.ok
    assert json.loads((tmp_path / "state.json").read_text()) == {"status": "complete"}
    assert result.records[0].side_effects[0].completed
    assert result.artifacts[0].value["path"] == str(tmp_path / "state.json")


def test_relative_output_root_and_requested_path_are_rooted_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @step
    def produce(output_path: str) -> bool:
        write_text(output_path + "/report.md", "done")
        return True

    @recipe(output_root=".")
    def run(output_path: str = "artifacts/run") -> bool:
        return produce(output_path)

    result = run.run()

    assert result.ok
    assert (tmp_path / "artifacts" / "run" / "report.md").read_text() == "done"


@pytest.mark.parametrize("operation", ["text", "json", "directory"])
def test_preexisting_symlink_escape_is_rejected(tmp_path, operation):
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    try:
        (output / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")

    @step
    def produce() -> bool:
        if operation == "directory":
            ensure_directory("link/nested")
        elif operation == "json":
            write_json("link/keep.txt", {"replaced": True})
        else:
            write_text("link/keep.txt", "replaced")
        return True

    @recipe(output_root=output)
    def run() -> bool:
        return produce()

    result = run.run()
    assert not result.ok
    assert "escapes declared output root" in str(result.error)
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert list(outside.iterdir()) == [sentinel]
    assert all(not record.side_effects for record in result.records)


@pytest.mark.parametrize("entry", ["run", "cli"])
def test_selected_output_directory_preserves_containment(tmp_path, monkeypatch, entry):
    monkeypatch.chdir(tmp_path)
    destination = Path("reports café") / "nested"

    @step
    def produce(path: str) -> str:
        return str(write_text(path, "Report complete."))

    @recipe(output_root="original")
    def run(path: str = "report.txt") -> str:
        return produce(path)

    if entry == "run":
        result = run.run(output_directory=destination)
        assert result.ok, result.error
        assert result.output_directory == str(destination.resolve())
        assert result.output == str((destination / "report.txt").resolve())
        failure = run.run({"path": "../escape.txt"}, output_directory=destination)
        assert not failure.ok
        assert "escapes declared output root" in (failure.error or "")
    else:
        assert run.cli(["--output-dir", str(destination)]) == 0
        assert run.cli(["--output-dir", str(destination), "--path", "../escape.txt"]) == 1
    assert (destination / "report.txt").read_text(encoding="utf-8") == "Report complete."
    assert not (tmp_path / "original").exists()
    assert not (destination.parent / "escape.txt").exists()


@pytest.mark.parametrize("invalid", ["file", "parent_file", "undeclared"])
def test_invalid_output_directory_fails_before_recipe_execution(tmp_path, invalid):
    observed = []
    destination = tmp_path / "destination"
    if invalid != "undeclared":
        destination.write_text("Keep this file.", encoding="utf-8")
    if invalid == "parent_file":
        destination /= "nested"

    @recipe(output_root=None if invalid == "undeclared" else tmp_path / "original")
    def run() -> bool:
        observed.append(True)
        return True

    error = "declared output_root" if invalid == "undeclared" else "not a directory"
    result = run.run(output_directory=destination)
    assert not result.ok
    assert error in (result.error or "")
    with pytest.raises(SystemExit) as exc:
        run.cli(["--output-dir", str(destination)])
    assert exc.value.code == 2
    assert observed == []


def test_selected_output_directory_is_blocked_in_hard_dry_run(tmp_path):
    observed = []

    @recipe(output_root=tmp_path / "original")
    def run() -> bool:
        observed.append(True)
        return True

    destination = tmp_path / "selected"
    result = run.run(output_directory=destination, dry_run=DryRunMode.HARD)
    assert not result.ok
    assert str(destination) in (result.error or "")
    assert not destination.exists()
    assert observed == []


@pytest.mark.parametrize("operation", ["text", "json", "directory"])
@pytest.mark.parametrize("path", [".git/config", "nested/.GiT/hooks", ".GIT", r".git\config"])
def test_git_writes_are_rejected_without_an_output_override(tmp_path, monkeypatch, operation, path):
    monkeypatch.chdir(tmp_path)

    @step
    def produce() -> bool:
        if operation == "text":
            write_text(path, "untrusted output")
        elif operation == "json":
            write_json(path, {"untrusted": True})
        else:
            ensure_directory(path)
        return True

    @recipe(output_root=".")
    def run() -> bool:
        return produce()

    result = run.run()
    assert not result.ok
    assert ".git paths" in str(result.error)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("path", [".git./config", "report.txt:stream", "NUL", "report. "])
def test_windows_reserved_output_names_are_rejected_on_every_platform(tmp_path, path):
    @step
    def produce() -> str:
        return str(write_text(path, "untrusted output"))

    @recipe(output_root=tmp_path)
    def run() -> str:
        return produce()

    result = run.run()
    assert not result.ok
    assert "Windows reserved name" in (result.error or "")
    assert list(tmp_path.iterdir()) == []


def test_git_symlink_alias_is_rejected(tmp_path):
    metadata = tmp_path / ".git"
    metadata.mkdir()
    sentinel = metadata / "config"
    sentinel.write_bytes(b"original")
    try:
        (tmp_path / "alias").symlink_to(metadata, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")

    @step
    def produce() -> str:
        return str(write_text("alias/config", "untrusted output"))

    @recipe(output_root=tmp_path)
    def run() -> str:
        return produce()

    result = run.run()
    assert not result.ok
    assert ".git paths" in (result.error or "")
    assert sentinel.read_bytes() == b"original"


@pytest.mark.parametrize("dry_run", [DryRunMode.OFF, DryRunMode.SOFT])
def test_output_override_rejects_existing_files(tmp_path, dry_run):
    sentinel = tmp_path / "app.py"
    sentinel.write_bytes(b"original")

    @step
    def produce() -> bool:
        write_text("app.py", "untrusted output")
        return True

    @recipe(output_root=tmp_path / "original")
    def run() -> bool:
        return produce()

    result = run.run(output_directory=tmp_path, dry_run=dry_run)
    assert not result.ok
    assert "refusing to replace an existing file" in str(result.error)
    assert sentinel.read_bytes() == b"original"
    assert all(not record.side_effects for record in result.records)


def test_output_override_can_update_its_own_files_but_not_a_previous_runs_files(tmp_path):
    @step
    def produce(value: str) -> str:
        return str(write_text("report.md", value))

    @recipe(output_root=tmp_path / "original")
    def run() -> str:
        produce("draft")
        return produce("final")

    first = run.run(output_directory=tmp_path)
    assert first.ok, first.error
    assert (tmp_path / "report.md").read_bytes() == b"final"
    second = run.run(output_directory=tmp_path)
    assert not second.ok
    assert "refusing to replace an existing file" in (second.error or "")
    assert (tmp_path / "report.md").read_bytes() == b"final"


def test_output_override_does_not_clobber_a_concurrently_created_file(tmp_path, monkeypatch):
    import calcifier.mutations as mutations

    atomic_write = mutations._atomic_write

    def race(path, data, **kwargs):  # noqa: ANN001, ANN003, ANN202
        path.write_bytes(b"concurrent writer")
        atomic_write(path, data, **kwargs)

    monkeypatch.setattr(mutations, "_atomic_write", race)

    @step
    def produce() -> bool:
        write_text("report.md", "untrusted output")
        return True

    @recipe(output_root=tmp_path / "original")
    def run() -> bool:
        return produce()

    result = run.run(output_directory=tmp_path)
    assert not result.ok
    assert "refusing to replace an existing file" in str(result.error)
    assert (tmp_path / "report.md").read_bytes() == b"concurrent writer"
    assert list(tmp_path.iterdir()) == [tmp_path / "report.md"]
    effect = result.records[0].side_effects[0]
    assert effect.completed is False and effect.error


def test_directory_text_and_soft_dry_run_primitives(tmp_path):
    @step
    def produce() -> bool:
        ensure_directory("nested")
        write_text("nested/report.txt", "report")
        verify_artifact("nested/report.txt")
        return True

    @recipe(output_root=tmp_path)
    def run() -> bool:
        return produce()

    result = run.run(dry_run=DryRunMode.SOFT)

    assert result.ok
    assert not (tmp_path / "nested").exists()
    assert len(result.records[0].side_effects) == 2


def test_mutation_primitives_require_context_step_and_valid_values(tmp_path):
    with pytest.raises(RecipeError, match="output_root"):
        write_text("x", "x")
    with active_run(output_root=tmp_path), pytest.raises(RecipeError, match="active @step"):
        write_text("x", "x")

    @step
    def invalid_text() -> bool:
        write_text("x", 1)  # ty: ignore[invalid-argument-type]
        return True

    @recipe(output_root=tmp_path)
    def text_recipe() -> bool:
        return invalid_text()

    assert "requires a string" in (text_recipe.run().error or "")

    @step
    def invalid_json() -> bool:
        write_json("x", object())
        return True

    @recipe(output_root=tmp_path)
    def json_recipe() -> bool:
        return invalid_json()

    assert "not JSON serializable" in (json_recipe.run().error or "")


def test_verify_artifact_checks_empty_and_hash(tmp_path):
    @step
    def empty() -> bool:
        write_text("empty", "")
        verify_artifact("empty")
        return True

    @recipe(output_root=tmp_path)
    def empty_recipe() -> bool:
        return empty()

    assert "required artifact is empty" in (empty_recipe.run().error or "")

    @step
    def wrong_hash() -> bool:
        write_text("value", "content")
        verify_artifact("value", sha256="0" * 64)
        return True

    @recipe(output_root=tmp_path)
    def hash_recipe() -> bool:
        return wrong_hash()

    assert "artifact hash mismatch" in (hash_recipe.run().error or "")


def test_mutations_cannot_escape_output_root(tmp_path):
    @step
    def escape() -> bool:
        write_text("../outside", "bad")
        return True

    @recipe(output_root=tmp_path / "out")
    def escaping() -> bool:
        return escape()

    assert "escapes declared output root" in (escaping.run().error or "")


def test_missing_required_artifact_is_a_failed_run(tmp_path):
    @step
    def initialize() -> bool:
        verify_artifact("audit_state.json")
        return True

    @recipe(output_root=tmp_path)
    def run() -> bool:
        return initialize()

    result = run.run()

    assert not result.ok
    assert "required artifact does not exist" in (result.error or "")


def test_hard_dry_run_of_mutating_recipe_invokes_no_agent(tmp_path):
    provider = CountingProvider("true")

    @step
    def model() -> bool:
        return agent(
            "worker",
            "review",
            working_directory=tmp_path,
            readable_paths=(tmp_path,),
        )

    @recipe(agents=("worker",), budget=Budget(2, 30), output_root=tmp_path / "out")
    def run() -> bool:
        return model()

    result = run.run(registry=_registry(provider), dry_run=DryRunMode.HARD)

    assert not result.ok
    assert "blocked by hard dry-run" in (result.error or "")
    assert provider.invocations == 0


def test_every_retry_is_charged_to_the_shared_budget(tmp_path):
    provider = CountingProvider("not-json")

    @step
    def model() -> bool:
        return agent(
            "worker",
            "review",
            working_directory=tmp_path,
            readable_paths=(tmp_path,),
        )

    @recipe(agents=("worker",), budget=Budget(1, 30))
    def run() -> bool:
        return model()

    result = run.run(registry=_registry(provider))

    assert not result.ok
    assert "agent-call budget exhausted" in (result.error or "")
    assert provider.invocations == 1
    assert result.records[0].agent_calls[0].attempts == 1


def test_parallel_workers_share_one_call_budget(tmp_path):
    provider = CountingProvider("true")

    @step
    def model(item: int) -> bool:
        return agent(
            "worker",
            "review {item}",
            working_directory=tmp_path,
            readable_paths=(tmp_path,),
            item=item,
        )

    @recipe(agents=("worker",), budget=Budget(2, 30))
    def run() -> bool:
        return all(parallel(lambda item: model(item), [1, 2, 3], max_workers=3))

    result = run.run(registry=_registry(provider))

    assert not result.ok
    assert "agent-call budget exhausted" in (result.error or "")
    assert provider.invocations == 2


def test_parallel_workers_cannot_mutate_shared_output_root(tmp_path):
    @step
    def write(item: int) -> bool:
        write_text(f"{item}.txt", "bad")
        return True

    @recipe(output_root=tmp_path)
    def run() -> bool:
        return all(parallel(write, [1, 2], max_workers=2))

    result = run.run()

    assert not result.ok
    assert "parallel workers cannot mutate shared state" in (result.error or "")
    assert list(tmp_path.iterdir()) == []


def test_partial_trace_survives_failed_parsing(tmp_path):
    provider = CountingProvider("not-json")
    trace_path = tmp_path / "trace.json"

    @step
    def model() -> bool:
        return agent(
            "worker",
            "review",
            working_directory=tmp_path,
            readable_paths=(tmp_path,),
        )

    @recipe(agents=("worker",), budget=Budget(2, 30))
    def run() -> bool:
        return model()

    result = run.run(registry=_registry(provider), trace_path=trace_path)
    trace = json.loads(trace_path.read_text())

    assert not result.ok
    assert trace["version"] == "1"
    assert trace["steps"][0]["agent_calls"][0]["attempts"] == 2
    assert trace["error"]


def test_interrupted_run_checkpoints_explicit_trace_state(tmp_path):
    trace_path = tmp_path / "trace.json"

    @step
    def interrupt() -> bool:
        raise KeyboardInterrupt

    @recipe
    def run() -> bool:
        return interrupt()

    with pytest.raises(KeyboardInterrupt):
        run.run(trace_path=trace_path)

    trace = json.loads(trace_path.read_text())
    assert trace["state"] == "interrupted"
    assert trace["error"] == "KeyboardInterrupt: interrupted"


def test_declared_command_is_bounded_and_returns_data(tmp_path):
    from calcifier import run_command

    @step
    def command() -> bool:
        result = run_command(
            [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
            timeout_seconds=5,
            working_directory=tmp_path,
            mutating=False,
        )
        return result.exit_code == 0 and result.stdout.strip() == "out" and result.stderr == "err\n"

    @recipe
    def run() -> bool:
        return command()

    assert run.run().ok


def test_mutation_failures_are_recorded_before_returning_success(tmp_path, monkeypatch):
    import calcifier.mutations as mutations

    @step
    def write() -> bool:
        write_text("result.txt", "value")
        return True

    @recipe(output_root=tmp_path)
    def run() -> bool:
        return write()

    def fail_atomic_write(path, data, **kwargs):  # noqa: ANN001, ANN003, ANN202, ARG001
        raise OSError("disk full")

    monkeypatch.setattr(mutations, "_atomic_write", fail_atomic_write)
    result = run.run()

    assert not result.ok
    assert "disk full" in (result.error or "")
    effect = result.records[0].side_effects[0]
    assert effect.completed is False
    assert effect.error == "disk full"


def test_ensure_directory_failure_is_recorded(tmp_path, monkeypatch):
    @step
    def create() -> bool:
        ensure_directory("result")
        return True

    @recipe(output_root=tmp_path)
    def run() -> bool:
        return create()

    def fail_mkdir(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202, ARG001
        raise OSError("read only")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    result = run.run()

    assert not result.ok
    assert "read only" in (result.error or "")
    assert result.records[0].side_effects[0].error == "read only"


def test_mutating_command_soft_dry_run_does_not_launch(tmp_path):
    from calcifier import run_command

    @step
    def command() -> bool:
        result = run_command(
            ["definitely-absent-command"],
            timeout_seconds=1,
            working_directory=tmp_path,
        )
        return result.exit_code == 0

    @recipe
    def run() -> bool:
        return command()

    result = run.run(dry_run=DryRunMode.SOFT)
    assert result.ok
    assert result.records[0].side_effects[0].completed is False


def test_declared_command_validates_inputs_and_records_failures(tmp_path):
    from calcifier import run_command

    with pytest.raises(RecipeError, match="argv"):
        run_command([], timeout_seconds=1)
    with pytest.raises(RecipeError, match="working directory"):
        run_command(["tool"], timeout_seconds=1, working_directory=tmp_path / "missing")

    @step
    def command() -> bool:
        run_command(
            [sys.executable, "-c", "import time; time.sleep(1)"],
            timeout_seconds=0.01,
            working_directory=tmp_path,
        )
        return True

    @recipe
    def run() -> bool:
        return command()

    result = run.run()
    assert not result.ok
    assert "ProcessTimeoutError" in (result.error or "")
    assert result.records[0].side_effects[0].completed is False
