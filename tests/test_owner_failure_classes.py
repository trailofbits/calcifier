"""Offline regressions for the general failure classes in owner report 1."""

from __future__ import annotations

from pathlib import Path

import pytest

from calcifier import prepare_context, recipe
from calcifier.dry_run import DryRunResult, dry_run_recipe
from calcifier.exceptions import RecipeError
from calcifier.gate import verify_source
from calcifier.trace import DryRunMode

pytestmark = pytest.mark.semantic_offline


def test_class_1_prepared_context_needs_no_output_root_or_step_and_works_in_parallel(tmp_path):
    source = """from calcifier import Budget, agent, parallel, prepare_context, recipe, step

@step
def inspect(task: str) -> str:
    context = prepare_context(task, byte_budget=1024)
    return agent("worker", "Read task.txt and interpret it.", context=context, output=str)

@recipe(agents=("worker",), budget=Budget(2, 60))
def workflow() -> list[str]:
    context = prepare_context("complete assignment")
    return parallel(inspect, [context.prompt, context.prompt])

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nInterpret the complete assignment.\n")
    verify_source(source)
    assert dry_run_recipe(
        source, recipe_path=tmp_path / "recipe.py", skill_path=skill
    ) == DryRunResult("passed")


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("mode", [DryRunMode.OFF, DryRunMode.SOFT, DryRunMode.HARD])
@pytest.mark.parametrize("fail", [False, True])
def test_class_1_scratch_preserves_complete_bytes_and_cleans_up(tmp_path, newline, mode, fail):
    task = ("full assignment café" + newline) * 100
    instructions = "retain" + newline + "every record"
    prior = "first" + newline + "last"
    expected = (
        f"Task:\n{task}\n\nInstructions:\n{instructions}\n\nPrior results:\n{prior}\n"
    ).encode()
    directories = []

    @recipe
    def workflow() -> str:
        context = prepare_context(
            task, instructions=instructions, prior_results=prior, byte_budget=1024
        )
        directory = Path(context.directory)
        directories.append(directory)
        assert not directory.is_relative_to(tmp_path)
        assert (directory / "task.txt").read_bytes() == expected
        assert len(context.prompt.encode("utf-8")) <= 1024
        assert task not in context.prompt
        if fail:
            raise ValueError("fixture failure")
        return context.directory

    for _ in range(2):
        run = workflow.run(dry_run=mode)
        assert run.ok is not fail
        assert not run.records
        assert run.output_directory is None
        assert not directories[-1].exists()
    assert directories[0] != directories[1]
    with pytest.raises(RecipeError, match="requires an active recipe run"):
        prepare_context("outside run")


def test_class_1_nested_runs_restore_the_parent_scratch_directory():
    @recipe
    def child() -> str:
        return prepare_context("same task").directory

    @recipe
    def parent() -> str:
        before = prepare_context("same task")
        nested = child.run()
        assert nested.ok
        assert isinstance(nested.output, str)
        assert nested.output != before.directory
        assert not Path(nested.output).exists()
        after = prepare_context("same task")
        assert before == after
        assert Path(before.directory).is_dir()
        return before.directory

    run = parent.run()
    assert run.ok
    assert isinstance(run.output, str)
    assert not Path(run.output).exists()


@pytest.mark.parametrize(
    ("imports", "command"),
    [
        ("from calcifier import run_command as command", "command"),
        ("import calcifier as runtime", "runtime.run_command"),
    ],
)
def test_class_2_run_command_counts_as_bash_use(tmp_path, imports, command):
    source = f"""import sys

from calcifier import recipe, step
{imports}

@step
def inspect() -> str:
    return {command}([sys.executable, "-V"], timeout_seconds=10, mutating=False).stdout

@recipe(capabilities=("Bash",))
def workflow() -> str:
    return inspect()

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    skill_text = "---\nallowed-tools: Bash\n---\nInspect the interpreter version.\n"
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(skill_text.encode())
    verify_source(source)
    assert dry_run_recipe(
        source, recipe_path=tmp_path / "recipe.py", skill_path=skill
    ) == DryRunResult("passed")
    unused = source.replace(
        f'{command}([sys.executable, "-V"], timeout_seconds=10, mutating=False).stdout', '"done"'
    )
    verify_source(unused, type_check=False)
    verify_source(source)


@pytest.mark.parametrize("annotation", ["list[str]", "tuple[str, ...]", "list[tuple[str, ...]]"])
def test_class_4_collection_inputs_execute_every_stage(tmp_path, annotation):
    source = f"""from calcifier import recipe, step, write_text

@step
def inspect(targets: {annotation}) -> str:
    if not targets:
        raise ValueError("targets must not be empty")
    return str(targets)

@step
def finish(summary: str) -> str:
    return str(write_text("report.txt", summary))

@recipe(output_root="out")
def workflow(targets: {annotation}) -> str:
    return finish(inspect(targets))

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nInspect all inputs and write the summary.\n")
    verify_source(source)
    assert dry_run_recipe(
        source, recipe_path=tmp_path / "recipe.py", skill_path=skill
    ) == DryRunResult("passed", writes=("<workspace>/out/report.txt",))


def test_class_4_text_validation_and_string_paths_reach_the_final_stage(tmp_path):
    source = """from pathlib import Path

from calcifier import fail, recipe, step, write_text

@step
def inspect(label: str, directory: str) -> str:
    if not label.isidentifier():
        fail("label must be an identifier")
    root = Path(directory)
    if not root.is_dir():
        fail("directory must exist")
    return label + root.joinpath("example.txt").read_text(encoding="utf-8")

@step
def finish(summary: str) -> str:
    return str(write_text("report.txt", summary))

@recipe(output_root="out")
def workflow(label: str, directory: str) -> str:
    return finish(inspect(label, directory))

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nInspect the named input directory and report.\n")
    verify_source(source)
    assert dry_run_recipe(
        source, recipe_path=tmp_path / "recipe.py", skill_path=skill
    ) == DryRunResult("passed", writes=("<workspace>/out/report.txt",))

    incompatible = source.replace("if not label.isidentifier():", 'if label != "custom-value":')
    gap = dry_run_recipe(incompatible, recipe_path=tmp_path / "recipe.py", skill_path=skill)
    assert gap.status == "unverified"
    assert gap.step == "inspect"
    assert "label must be an identifier" in (gap.error or "")
    assert gap.writes == ()

    defect = source.replace('return label + root.joinpath("example.txt")', "return label + [][1]")
    outcome = dry_run_recipe(defect, recipe_path=tmp_path / "recipe.py", skill_path=skill)
    assert outcome.status == "failed"
    assert "IndexError" in (outcome.error or "")
