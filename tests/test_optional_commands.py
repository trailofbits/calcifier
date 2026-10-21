"""Optional commands retain their fallback environment without hiding fatal errors."""

from __future__ import annotations

import sys

import pytest

from calcifier import Budget, RecipeError, run_command, step
from calcifier.artifact import prepare_artifact
from calcifier.budget import BudgetExceeded, BudgetState
from calcifier.prompts import build_prompt
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from calcifier.subprocesses import ProcessError, ProcessOutputLimitError, ProcessTimeoutError
from calcifier.trace import active_run
from tests.support.compiler import registry_emitting, source_sequence

_ENVIRONMENT_RUNNER = b"""import sys
if sys.argv[1:] != ["--environment", "private", "neutral-tool", "inspect"]:
    raise SystemExit(4)
print("environment result")
"""


def workflow_source(preferred: list[str], fallback: str, *, timeout: float = 10) -> str:
    return f"""import sys
from calcifier import Budget, RecipeError, recipe, resource, run_command, step

@step
def inspect() -> str:
    commands = [{preferred!r}, {fallback}]
    for argv in commands:
        try:
            result = run_command(argv, timeout_seconds={timeout!r}, mutating=False)
        except RecipeError as error:
            if error.command_failure not in ("launch", "timeout"):
                raise
            continue
        if result.exit_code == 0:
            return result.stdout.strip()
    return "unavailable"

@recipe(capabilities=("Bash",), budget=Budget(None, 30))
def workflow() -> str:
    return inspect()

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""


@pytest.mark.parametrize("route", ["direct", "environment-only", "absent", "timeout"])
def test_compiler_and_runtime_preserve_optional_routes(tmp_path, route):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(
        b"# Optional inspection\nTry the direct tool, then the private environment runner.\n"
        b"If neither works, report unavailable. Keep the environment arguments intact.\n"
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "environment.py").write_bytes(_ENVIRONMENT_RUNNER)
    missing = [str(tmp_path / "missing-neutral-tool")]
    preferred = {
        "direct": [sys.executable, "-c", 'print("direct result")'],
        "environment-only": missing,
        "absent": missing,
        "timeout": [sys.executable, "-c", "import time; time.sleep(60)"],
    }[route]
    fallback = (
        '[sys.executable, str(resource("scripts/environment.py")), '
        '"--environment", "private", "neutral-tool", "inspect"]'
        if route != "absent"
        else repr([str(tmp_path / "missing-fallback-tool")])
    )
    source = workflow_source(preferred, fallback, timeout=1 if route == "timeout" else 10)

    artifact = prepare_artifact(skill, registry_emitting(source))

    assert artifact.outcome == "compiled"
    assert artifact.dry_run is not None and artifact.dry_run.status == "passed"
    artifact.path.parent.mkdir(parents=True)
    artifact.path.write_bytes(artifact.source.encode("utf-8"))
    result = load_recipe(artifact.path).run()
    assert result.ok, result.error
    assert (
        result.output
        == {
            "direct": "direct result",
            "environment-only": "environment result",
            "absent": "unavailable",
            "timeout": "environment result",
        }[route]
    )


def test_missing_executable_remains_a_recipe_error_with_launch_details(tmp_path):
    with pytest.raises(RecipeError) as caught:
        run_command([str(tmp_path / "absent")], timeout_seconds=1, mutating=False)

    assert caught.value.command_failure == "launch"
    assert isinstance(caught.value.__cause__, ProcessError)
    assert isinstance(caught.value.__cause__.__cause__, FileNotFoundError)


def test_repair_replaces_a_wrong_exception_handler_after_the_full_dry_run(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Inspect\nUse the optional tool or report unavailable.\n")
    source = workflow_source([str(tmp_path / "missing-tool")], '["missing-fallback-tool"]')
    wrong_handler = source.replace(
        "except RecipeError as error:\n"
        '            if error.command_failure not in ("launch", "timeout"):\n'
        "                raise\n            continue",
        "except FileNotFoundError:\n            continue",
    )
    assert wrong_handler != source
    compiler = source_sequence(wrong_handler, source)
    registry = Registry()
    registry.register(compiler)

    artifact = prepare_artifact(skill, registry)

    assert artifact.outcome == "compiled"
    assert artifact.dry_run is not None and artifact.dry_run.status == "passed"
    assert len(compiler.calls) == 2
    assert "command failed to execute" in compiler.all_prompts[1]
    assert 'set to "launch" or "timeout"' in compiler.all_prompts[1]


@pytest.mark.parametrize("exhausted", [False, True])
def test_local_timeout_and_exhausted_shared_budget_are_distinct(monkeypatch, exhausted):
    expired = False

    def remaining(self):
        return 0.0 if expired else 5.0

    timeouts = []

    def timed_out(argv, *, timeout_seconds, **kwargs):
        nonlocal expired
        timeouts.append(timeout_seconds)
        expired = exhausted
        raise ProcessTimeoutError("local timeout")

    monkeypatch.setattr(BudgetState, "remaining_wall_seconds", remaining)
    monkeypatch.setattr("calcifier.mutations.run_process", timed_out)
    expected = BudgetExceeded if exhausted else RecipeError
    with active_run(budget_state=BudgetState(Budget(None, 5))), pytest.raises(expected) as caught:
        run_command(["neutral-tool"], timeout_seconds=20, mutating=False)

    assert timeouts == [5.0]
    if not exhausted:
        assert isinstance(caught.value, RecipeError)
        assert caught.value.command_failure == "timeout"


def test_fallback_cannot_start_after_shared_budget_expires(tmp_path, monkeypatch):
    path = tmp_path / "recipe.py"
    path.write_bytes(workflow_source(["neutral-tool"], '["fallback-tool"]').encode("utf-8"))
    expired = False
    commands = []

    def remaining(self):
        return 0.0 if expired else 5.0

    def timed_out(argv, **kwargs):
        nonlocal expired
        commands.append(argv)
        expired = True
        raise ProcessTimeoutError("shared deadline elapsed")

    monkeypatch.setattr(BudgetState, "remaining_wall_seconds", remaining)
    monkeypatch.setattr("calcifier.mutations.run_process", timed_out)

    result = load_recipe(path).run()

    assert not result.ok
    assert "budget exhausted" in (result.error or "")
    assert commands == [["neutral-tool"]]


@pytest.mark.parametrize(
    "failure", [ProcessError("supervisor failure"), ProcessOutputLimitError("limit")]
)
def test_other_process_failures_are_not_tool_unavailability(monkeypatch, failure):
    def fail(argv, **kwargs):
        raise failure

    monkeypatch.setattr("calcifier.mutations.run_process", fail)
    with pytest.raises(RecipeError) as caught:
        run_command(["neutral-tool"], timeout_seconds=1, mutating=False)

    assert caught.value.command_failure is None


@pytest.mark.parametrize("program", ['print("not JSON")', "raise SystemExit(3)"])
def test_invalid_typed_command_results_are_not_tool_unavailability(program):
    with pytest.raises(RecipeError) as caught:
        run_command([sys.executable, "-c", program], timeout_seconds=10, mutating=False, output=int)

    assert caught.value.command_failure is None


def test_failed_mutating_launch_retains_effect_record(tmp_path):
    @step
    def inspect() -> None:
        run_command([str(tmp_path / "missing-tool")], timeout_seconds=1)

    with active_run() as context, pytest.raises(RecipeError) as caught:
        inspect()

    assert caught.value.command_failure == "launch"
    effect = context.records[0].side_effects[0]
    assert effect.completed is False
    assert effect.error == "ProcessStartError"


def test_generation_and_repair_receive_the_installed_command_error_contract():
    prompt = build_prompt("Use an optional checker before manual review.")

    assert 'set to "launch" or "timeout"' in prompt
    assert "re-raise other RecipeError" in prompt
    assert "Shared deadline exhaustion raises BudgetExceeded" in prompt
    assert "complete environment-specific argv prefix" in prompt
