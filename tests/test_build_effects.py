"""Analysis wrappers with implicit builds declare effects before process launch."""

from __future__ import annotations

import pytest

from calcifier.artifact import prepare_artifact
from calcifier.prompts import build_prompt
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from calcifier.trace import DryRunMode
from tests.support.agents import StubProvider
from tests.support.compiler import one_draft

_SOURCE = """import sys
from pathlib import Path
from calcifier import Budget, agent, fail, recipe, resource, run_command, step

@step
def inspect(target: Path, allow_build: bool, build_workspace: Path | None) -> bool:
    if allow_build:
        if build_workspace is None or not build_workspace.is_dir():
            fail("Target builds require a prepared isolated build workspace.")
        return run_command([sys.executable, str(resource("scripts/analyze.py"))],
                           working_directory=build_workspace, timeout_seconds=10,
                           mutating=True, output=bool)
    return agent("worker", "Analyze manually without target builds.",
                 working_directory=target, readable_paths=(target,),
                 required_capabilities=("read",))

@recipe(agents=("worker",), budget=Budget(2, 30), capabilities=("Bash",))
def workflow(target: Path = Path("."), allow_build: bool = False,
             build_workspace: Path | None = None) -> bool:
    return inspect(target, allow_build, build_workspace)

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""


@pytest.fixture
def compiled_wrapper(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(
        b"# Optional analysis\nAnalyze manually by default. Target builds require explicit opt-in "
        b"and an isolated environment prepared by the caller.\n"
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    # A neutral stand-in for an analyzer that invokes a target's build/configuration.
    (scripts / "analyze.py").write_bytes(
        b"from pathlib import Path\nPath('build-marker').write_bytes(b'built')\nprint('true')\n"
    )
    registry = Registry()
    registry.register(one_draft(_SOURCE))
    worker = StubProvider(default="true")
    worker.name = "worker"
    worker.capabilities = ("read",)
    registry.register(worker)
    artifact = prepare_artifact(skill, registry)
    assert artifact.outcome == "compiled"
    assert artifact.dry_run is not None and artifact.dry_run.status == "passed"
    artifact.path.parent.mkdir(parents=True)
    artifact.path.write_bytes(artifact.source.encode("utf-8"))
    return load_recipe(artifact.path), registry


@pytest.mark.parametrize("allow_build", [False, True])
def test_hard_dry_run_suppresses_builds_and_keeps_manual_route(
    tmp_path, monkeypatch, compiled_wrapper, allow_build
):
    workflow, registry = compiled_wrapper
    isolated = tmp_path / "isolated"
    isolated.mkdir()

    def unexpected_launch(*args, **kwargs):
        raise AssertionError("build reached the process runner during hard dry-run")

    monkeypatch.setattr("calcifier.mutations.run_process", unexpected_launch)
    result = workflow.run(
        registry=registry,
        dry_run=DryRunMode.HARD,
        inputs={"target": tmp_path, "allow_build": allow_build, "build_workspace": isolated},
    )

    assert result.ok is not allow_build
    assert not (isolated / "build-marker").exists()
    if allow_build:
        assert "side effect blocked by hard dry-run" in (result.error or "")
        # The command-context gate rejects hard mutations before recording or launching them.
        assert not result.records[0].side_effects
    else:
        assert result.output is True


def test_opted_in_build_runs_only_in_the_prepared_copy(tmp_path, compiled_wrapper):
    workflow, registry = compiled_wrapper
    original = tmp_path / "original"
    isolated = tmp_path / "isolated"
    original.mkdir()
    isolated.mkdir()

    result = workflow.run(
        registry=registry,
        inputs={"target": original, "allow_build": True, "build_workspace": isolated},
    )

    assert result.ok and result.output is True
    assert (isolated / "build-marker").read_bytes() == b"built"
    assert not (original / "build-marker").exists()
    assert result.records[0].side_effects[0].completed is True


def test_build_opt_in_requires_a_prepared_environment(tmp_path, compiled_wrapper):
    workflow, registry = compiled_wrapper

    result = workflow.run(registry=registry, inputs={"target": tmp_path, "allow_build": True})

    assert not result.ok
    assert "prepared isolated build workspace" in (result.error or "")
    assert not result.records[0].side_effects


def test_generation_receives_the_command_isolation_contract():
    prompt = build_prompt("Inspect a target with an optional analyzer.")

    assert "mutating=True does not isolate a process" in prompt
