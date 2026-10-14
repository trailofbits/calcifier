"""External JSON results remain usable throughout a compiler dry run."""

from __future__ import annotations

import sys
import textwrap

import pytest

from calcifier import RecipeError, output_type, run_command, step
from calcifier.dry_run import dry_run_recipe
from calcifier.outputs import OutputTypeError
from calcifier.recipe import load_recipe
from calcifier.side_effects import SideEffectBlocked
from calcifier.trace import DryRunMode, active_run


@output_type
class Item:
    name: str
    count: int


def test_invalid_output_type_does_not_record_a_pending_command():
    class Unsupported:
        pass

    @step
    def inspect() -> None:
        run_command(["unavailable-command"], timeout_seconds=10, output=Unsupported)

    with (
        active_run() as context,
        pytest.raises(OutputTypeError, match="unsupported annotation"),
    ):
        inspect()
    assert not context.records[0].side_effects


@pytest.mark.parametrize("dry_run", [DryRunMode.OFF, DryRunMode.SOFT])
def test_read_only_command_validates_real_output(dry_run):
    with active_run(dry_run=dry_run):
        result = run_command(
            [sys.executable, "-c", 'print(\'{"name":"item","count":3}\')'],
            timeout_seconds=10,
            mutating=False,
            output=Item,
        )
    assert result == Item(name="item", count=3)


@pytest.mark.parametrize(
    "payload",
    ['{"name":"item","count":"3"}', '{"name":"item","count":3,"extra":true}', "not JSON"],
)
@pytest.mark.parametrize("dry_run", [DryRunMode.OFF, DryRunMode.SOFT])
def test_malformed_command_results_are_not_coerced(payload, dry_run):
    with (
        active_run(dry_run=dry_run),
        pytest.raises(RecipeError, match="stdout does not match"),
    ):
        run_command(
            [sys.executable, "-c", f"print({payload!r})"],
            timeout_seconds=10,
            mutating=False,
            output=Item,
        )


def test_nonzero_exit_cannot_masquerade_as_valid_output(capsys):
    with pytest.raises(RecipeError, match="command exited 7") as error:
        run_command(
            [
                sys.executable,
                "-c",
                'import sys; print("detail", file=sys.stderr); print("[]"); raise SystemExit(7)',
            ],
            timeout_seconds=10,
            mutating=False,
            output=list[Item],
        )
    assert "detail" in capsys.readouterr().err
    assert "detail" not in str(error.value)


@pytest.mark.parametrize("parameters", ["", "unused: str", "unused: Path"])
def test_fixed_nonzero_command_is_a_failed_dry_run(tmp_path, parameters):
    command = [sys.executable, "-c", "print(1); raise SystemExit(7)"]
    source = f"""from pathlib import Path
from calcifier import recipe, run_command, step

@step
def inspect() -> int:
    return run_command({command!r}, timeout_seconds=10, mutating=False, output=int)

@recipe(capabilities=("Bash",))
def workflow({parameters}) -> int:
    return inspect()

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Inspect\nReturn the command result.\n")
    result = dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill)
    assert result.status == "failed", result
    assert not result.command_gap
    assert result.step == "inspect"
    assert result.error is not None and "command exited 7" in result.error


def test_soft_run_synthesizes_mutating_results_without_running_command(tmp_path):
    marker = tmp_path / "should-not-exist"

    @step
    def inspect() -> list[Item]:
        return run_command(
            [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
            timeout_seconds=10,
            output=list[Item],
        )

    with active_run(dry_run=DryRunMode.SOFT):
        result = inspect()
    assert result == [Item(name="stub", count=1)]
    assert not marker.exists()


@pytest.mark.parametrize(
    "annotation,mutating,expected", [("str", False, "failed"), ("Item", True, "unverified")]
)
def test_compiler_distinguishes_invalid_command_output_from_invalid_stub(
    tmp_path, annotation, mutating, expected
):
    command = [sys.executable, "-c", "print('plain text')"]
    source = f"""from uuid import UUID
from calcifier import output_type, recipe, run_command, step

@output_type
class Item:
    identifier: UUID

@step
def inspect() -> {annotation}:
    return run_command({command!r}, timeout_seconds=10, mutating={mutating}, output={annotation})

@recipe(capabilities=("Bash",))
def workflow() -> {annotation}:
    return inspect()

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Inspect\nRead a structured result.\n")
    result = dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill)
    assert result.status == expected, result


@pytest.mark.parametrize("parameter", ["data: Path", "data: str", 'data: Path = Path("missing")'])
@pytest.mark.parametrize("flagged", [False, True])
def test_compiler_reports_a_command_rejecting_synthetic_input_as_a_gap(
    tmp_path, parameter, flagged
):
    path_argument = "sys.argv[1].split('=', 1)[1]" if flagged else "sys.argv[1]"
    argument = '"--input=" + str(data)' if flagged else "str(data)"
    command = [
        sys.executable,
        "-c",
        "import json, sys; from pathlib import Path; "
        f"print(json.dumps(json.loads(Path({path_argument}).read_text(encoding='utf-8'))))",
    ]
    source = f"""from pathlib import Path
from calcifier import recipe, run_command, step

@step
def inspect(data: Path | str) -> dict[str, int]:
    return run_command(
        {command!r} + [{argument}],
        timeout_seconds=10, mutating=False, output=dict[str, int],
    )

@recipe(capabilities=("Bash",))
def workflow({parameter}) -> dict[str, int]:
    return inspect(data)

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Read data\nRead and return the supplied JSON data.\n")
    path = tmp_path / "recipe.py"
    path.write_bytes(source.encode())
    result = dry_run_recipe(source, recipe_path=path, skill_path=skill)
    assert result.status == "unverified", result
    assert result.step == "inspect"

    data = tmp_path / "data.json"
    data.write_bytes(b'{"count":3}\n')
    workflow = load_recipe(path)
    real = workflow.run({"data": str(data) if parameter == "data: str" else data})
    assert real.ok, real.error
    assert real.output == {"count": 3}
    assert not workflow.run({"data": tmp_path}, dry_run=DryRunMode.SOFT).ok


@pytest.mark.parametrize("fail_after", ["none", "python", "command"])
def test_command_gap_does_not_hide_later_writes_or_errors(tmp_path, fail_after):
    command = [
        sys.executable,
        "-c",
        "import sys; from pathlib import Path; print(Path(sys.argv[1]).read_text())",
    ]
    finish = {
        "none": "value",
        "python": "str(1 // 0)",
        "command": (
            f"run_command({[sys.executable, '-c', 'raise SystemExit(7)']!r}, "
            "timeout_seconds=10, mutating=False, output=str)"
        ),
    }[fail_after]
    source = f"""from pathlib import Path
from calcifier import recipe, run_command, step, write_text

@step
def inspect(data: Path) -> str:
    return run_command(
        {command!r} + [str(data)],
        timeout_seconds=10, mutating=False, output=str,
    )

@step
def finish(value: str) -> str:
    write_text("after.txt", value)
    return {finish}

@recipe(capabilities=("Bash",), output_root="artifacts")
def workflow(data: Path) -> str:
    return finish(inspect(data))

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Inspect\nInspect and save the result.\n")
    result = dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill)
    assert "<workspace>/artifacts/after.txt" in result.writes
    assert result.status == ("unverified" if fail_after == "none" else "failed"), result
    assert result.step == ("inspect" if fail_after == "none" else "finish")
    assert result.error is not None
    assert {"none": "command exited", "python": "division", "command": "command exited 7"}[
        fail_after
    ] in result.error


@pytest.mark.parametrize("root", ["out", "stub"])
def test_a_materialized_command_input_cannot_hide_a_failure(tmp_path, root):
    command = [sys.executable, "-c", "print(1); raise SystemExit(7)"]
    source = f"""from pathlib import Path
from calcifier import recipe, run_command, step, write_text

@step
def inspect() -> int:
    data = write_text("data.json", '{{"count":3}}')
    return run_command({command!r} + [str(data)], timeout_seconds=10, mutating=False, output=int)

@recipe(capabilities=("Bash",), output_root={root!r})
def workflow(unused: Path = Path({root!r})) -> int:
    return inspect()

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Inspect\nInspect prepared data.\n")
    result = dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill)
    assert result.status == "failed", result
    assert not result.command_gap
    assert result.error is not None and "command exited 7" in result.error
    assert any(path.endswith(f"{root}/data.json") for path in result.writes)


def test_rejected_fallback_stub_preserves_the_command_gap(tmp_path):
    command = [
        sys.executable,
        "-c",
        "import sys; from pathlib import Path; print(Path(sys.argv[1]).read_text())",
    ]
    source = f"""from pathlib import Path
from uuid import UUID
from calcifier import recipe, run_command, step

@step
def inspect(data: Path) -> UUID:
    return run_command(
        {command!r} + [str(data)],
        timeout_seconds=10, mutating=False, output=UUID,
    )

@recipe(capabilities=("Bash",))
def workflow(data: Path) -> UUID:
    return inspect(data)

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Inspect\nRead the identifier.\n")
    result = dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill)
    assert result.status == "unverified" and result.command_gap
    assert result.error is not None and "command exited" in result.error


def test_full_candidate_consumes_command_output_and_reaches_last_write(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Inventory\nInspect, interpret and save every item.\n")
    source = textwrap.dedent("""\
        from calcifier import (
            Budget, agent, output_type, parallel, recipe, run_command, step, write_json,
        )

        @output_type
        class Item:
            name: str
            count: int

        @step
        def describe(item: Item) -> str:
            return agent("worker", "Describe {item}", item=item, working_directory=".")

        @recipe(agents=("worker",), capabilities=("Bash",), budget=Budget(4, 30), output_root="out")
        def inventory() -> int:
            items = run_command(
                ["unavailable-inventory-service"], timeout_seconds=10, output=list[Item],
            )
            answers = parallel(describe, items, max_workers=2)
            write_json("complete.json", {"answers": answers})
            return len(answers)

        if __name__ == "__main__":
            raise SystemExit(inventory.cli())
    """)
    result = dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill)
    assert result.status == "passed", result
    assert any(path.endswith("complete.json") for path in result.writes)


def test_typed_result_does_not_bypass_hard_dry_run():
    @step
    def change() -> list[Item]:
        return run_command(["unavailable-service"], timeout_seconds=10, output=list[Item])

    with active_run(dry_run=DryRunMode.HARD), pytest.raises(SideEffectBlocked):
        change()
