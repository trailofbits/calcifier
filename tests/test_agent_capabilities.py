"""Configured worker tool support reaches generation, stub execution, and real dispatch."""

from __future__ import annotations

import json
import sys

import pytest

from calcifier import Budget, agent, recipe, step
from calcifier.agent_config import AgentConfigError, load_agent_config
from calcifier.agent_events import AgentActivity
from calcifier.agent_providers import ClaudeCli, CodexCli, claude_command
from calcifier.artifact import prepare_artifact
from calcifier.dry_run import dry_run_recipe
from calcifier.recipe import load_recipe
from calcifier.registry import Registry, configured_registry
from tests.support.agents import StubProvider
from tests.support.compiler import source_sequence


class Worker(StubProvider):
    name = "worker"

    def __init__(self, capabilities):
        super().__init__(default="true")
        self.capabilities = capabilities
        self.calls = 0

    def invoke(self, *args, **kwargs):
        self.calls += 1
        return super().invoke(*args, **kwargs)


def worker_recipe(required: str) -> str:
    return f"""from pathlib import Path
from calcifier import Budget, agent, recipe, step

@step
def inspect(target: Path) -> bool:
    return agent("worker", "Perform the requested check.", working_directory=target,
                 readable_paths=(target,), required_capabilities=({required!r},))

@recipe(agents=("worker",), budget=Budget(2, 30))
def workflow(target: Path = Path(".")) -> bool:
    return inspect(target)

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""


def configuration(tmp_path, *, builtin=None, capabilities=None):
    route = {
        "target": {"builtin": builtin} if builtin else {"command": [sys.executable, "adapter.py"]},
        "runtime": {"kind": "host"},
        "timeout_seconds": 30,
    }
    if capabilities is not None:
        route["capabilities"] = capabilities
    path = tmp_path / "agents.json"
    path.write_bytes(json.dumps({"version": 1, "agents": {"worker": route}}).encode("utf-8"))
    return load_agent_config(path)


def test_custom_capabilities_round_trip_and_affect_configuration_identity(tmp_path):
    config = configuration(tmp_path, capabilities=["research", "read", "read"])
    route = config.routes["worker"]
    assert route.capabilities == ("read", "research")
    assert route.as_dict()["capabilities"] == ["read", "research"]
    assert configured_registry(config, ("worker",)).get("worker").capabilities == (
        "read",
        "research",
    )
    assert config.fingerprint() != configuration(tmp_path, capabilities=["read"]).fingerprint()
    assert configuration(tmp_path).routes["worker"].capabilities is None


@pytest.mark.parametrize("capabilities", ["commands", ["unrecognized"], [1], {}])
def test_invalid_capability_declarations_fail_before_adapter_launch(tmp_path, capabilities):
    with pytest.raises(AgentConfigError, match="capabilities must be an array"):
        configuration(tmp_path, capabilities=capabilities)


@pytest.mark.parametrize("builtin", ["claude", "codex"])
def test_builtin_permissions_cannot_be_broadened_by_configuration(tmp_path, builtin):
    with pytest.raises(AgentConfigError, match="cannot be overridden"):
        configuration(tmp_path, builtin=builtin, capabilities=["research"])


def test_builtin_capability_descriptions_match_their_command_contracts():
    assert ClaudeCli().capabilities == ("read", "write")
    assert CodexCli().capabilities == ("read", "write", "commands")
    argv = claude_command("neutral-worker").argv
    assert argv[argv.index("--tools") + 1] == "Read,Glob,Grep"
    assert "Bash" not in argv


@pytest.mark.parametrize(
    "capabilities, required, allowed",
    [
        (None, "commands", False),
        (("read",), "commands", False),
        (("read", "commands"), "commands", True),
        (("read", "commands"), "research", False),
        (("read", "research"), "research", True),
    ],
)
def test_required_tools_are_checked_before_charging_or_invoking_worker(
    tmp_path, capabilities, required, allowed
):
    worker = Worker(capabilities)
    registry = Registry()
    registry.register(worker)

    @step
    def inspect() -> bool:
        return agent(
            "worker",
            "Check the target.",
            working_directory=tmp_path,
            readable_paths=(tmp_path,),
            required_capabilities=(required,),
        )

    @recipe(agents=("worker",), budget=Budget(2, 30))
    def workflow() -> bool:
        return inspect()

    result = workflow.run(registry=registry)
    assert result.ok is allowed
    assert worker.calls == int(allowed)
    if not allowed:
        assert "does not declare required capabilities" in (result.error or "")
        assert not result.records[0].agent_calls


@pytest.mark.parametrize("capabilities, status", [(("read",), "failed"), (("commands",), "passed")])
def test_complete_dry_run_uses_configured_worker_capabilities(tmp_path, capabilities, status):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Checker\nUse a command-capable worker to check the target.\n")
    outcome = dry_run_recipe(
        worker_recipe("commands"),
        recipe_path=tmp_path / "recipe.py",
        skill_path=skill,
        agent_capabilities={"worker": capabilities},
    )
    assert outcome.status == status
    if status == "failed":
        assert outcome.step == "inspect"
        assert "commands" in (outcome.error or "")


def test_compiler_repairs_worker_commands_into_host_execution(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(
        b"# Measurement\nRun scripts/check.py to gather a measurement; "
        b"have the worker interpret it.\n"
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "check.py").write_bytes(b"print(1)\n")
    source = """import sys
from pathlib import Path
from calcifier import Budget, agent, recipe, resource, run_command, step

@step
def inspect(target: Path) -> bool:
    measurement = run_command([sys.executable, str(resource("scripts/check.py"))],
                              timeout_seconds=10, mutating=False, output=int)
    return agent("worker", "Interpret measurement {measurement}.", measurement=measurement,
                 working_directory=target, readable_paths=(target,),
                 required_capabilities=("read",))

@recipe(agents=("worker",), budget=Budget(2, 30), capabilities=("Bash",))
def workflow(target: Path = Path(".")) -> bool:
    return inspect(target)

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    compiler = source_sequence(worker_recipe("commands"), source)
    worker = Worker(("read",))
    registry = Registry()
    registry.register(compiler)
    registry.register(worker)

    artifact = prepare_artifact(skill, registry)

    assert artifact.outcome == "compiled"
    assert artifact.dry_run is not None and artifact.dry_run.status == "passed"
    assert len(compiler.calls) == 2
    for prompt in compiler.all_prompts:
        assert '{"worker": ["read"]}' in prompt
        assert "not the host tool providers" in prompt
    assert "does not declare required capabilities" in compiler.all_prompts[1]
    artifact.path.parent.mkdir(parents=True)
    artifact.path.write_bytes(artifact.source.encode("utf-8"))
    result = load_recipe(artifact.path).run(registry=registry, inputs={"target": tmp_path})
    assert result.ok and result.output is True
    assert worker.calls == 1


def test_unavailable_research_is_not_reported_as_successfully_compiled(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Verify\nPerform current online verification before answering.\n")
    compiler = source_sequence(worker_recipe("research"))
    worker = Worker(("read", "commands"))
    registry = Registry()
    registry.register(compiler)
    registry.register(worker)
    activity = AgentActivity()

    artifact = prepare_artifact(skill, registry, max_compiler_calls=1, activity=activity)

    assert artifact.outcome == "baseline_retained"
    assert "research" in (activity.retained_reason or "")
    assert worker.calls == 0


def test_a_different_runtime_provider_must_satisfy_the_recipe_requirements(tmp_path):
    path = tmp_path / "recipe.py"
    path.write_bytes(worker_recipe("commands").encode("utf-8"))
    registry = Registry()
    worker = Worker(("read",))
    registry.register(worker)

    result = load_recipe(path).run(registry=registry, inputs={"target": tmp_path})

    assert not result.ok
    assert worker.calls == 0
    assert "commands" in (result.error or "")


@pytest.mark.parametrize(
    "required, writable, allowed",
    [
        (("write",), False, False),
        (("write",), True, True),
        (("undeclared-operation",), False, False),
    ],
)
def test_capability_requirements_do_not_grant_filesystem_writes(
    tmp_path, required, writable, allowed
):
    worker = Worker(("read", "write"))
    registry = Registry()
    registry.register(worker)

    @step
    def inspect() -> bool:
        return agent(
            "worker",
            "Check the target.",
            working_directory=tmp_path,
            readable_paths=(tmp_path,),
            writable_paths=(tmp_path,) if writable else (),
            required_capabilities=required,
        )

    @recipe(agents=("worker",), budget=Budget(2, 30))
    def workflow() -> bool:
        return inspect()

    result = workflow.run(registry=registry)

    assert result.ok is allowed
    assert worker.calls == int(allowed)
    if required == ("write",) and not writable:
        assert "needs declared writable_paths" in (result.error or "")
    elif required == ("undeclared-operation",):
        assert "unknown required agent capabilities" in (result.error or "")
