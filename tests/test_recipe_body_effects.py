"""Class 3: recipe-body operations use a runtime frame without weakening write policy."""

from __future__ import annotations

import asyncio
import json

import pytest

from calcifier import recipe, step, write_text
from calcifier.dry_run import DryRunResult, dry_run_recipe
from calcifier.gate import verify_source
from calcifier.recipe import load_recipe, parallel
from calcifier.trace import DryRunMode, current
from calcifier.trace_export import serialize_run, validate_trace


@pytest.mark.parametrize("asynchronous", [False, True])
def test_class_3_recipe_body_writes_compile_execute_and_record_nested_steps(tmp_path, asynchronous):
    prefix, wait = ("async ", "await asyncio.sleep(0)\n    ") if asynchronous else ("", "")
    source = f"""import asyncio
from calcifier import ensure_directory, recipe, step, verify_artifact, write_json, write_text

@step
def inspect(value: str) -> str:
    return value.upper()

@recipe(output_root="out")
{prefix}def workflow(value: str = "ready") -> str:
    {wait}ensure_directory("nested")
    result = inspect(value)
    path = write_text("nested/report.txt", result)
    write_json("metadata.json", {{"result": result}})
    verify_artifact(path)
    return result

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    if not asynchronous:
        source = source.replace("import asyncio\n", "")
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral workflow\nInspect the value and save the result.\n")
    verify_source(source)
    path = tmp_path / "recipe.py"
    path.write_bytes(source.encode())
    dry = dry_run_recipe(source, recipe_path=path, skill_path=skill)
    assert dry == DryRunResult(
        "passed",
        writes=(
            "<workspace>/out/nested",
            "<workspace>/out/nested/report.txt",
            "<workspace>/out/metadata.json",
        ),
    )
    assert not (tmp_path / "out").exists()
    workflow = load_recipe(path)
    run = workflow.run(output_directory=tmp_path / "out")
    assert run.ok, run.error
    assert run.output == "READY"
    assert (tmp_path / "out/nested/report.txt").read_bytes() == b"READY"
    assert json.loads((tmp_path / "out/metadata.json").read_bytes()) == {"result": "READY"}
    [body] = run.records
    assert body.name == "workflow" and body.output == "READY"
    assert body.inputs == {"value": "ready"}
    assert [child.name for child in body.children] == ["inspect"]
    assert body.children[0].output == "READY"
    assert len(body.side_effects) == 3
    assert all(effect.completed for effect in body.side_effects)
    validate_trace(serialize_run(run))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_class_3_recipe_body_failure_records_effects_and_restores_context(
    tmp_path, asynchronous, error
):
    def execute() -> str:
        write_text("started.txt", "started")
        raise error("fixture failure")

    async def execute_async() -> str:
        await asyncio.sleep(0)
        return execute()

    workflow = recipe(output_root=str(tmp_path))(execute_async if asynchronous else execute)
    if error is KeyboardInterrupt:
        trace = tmp_path / "trace.json"
        with pytest.raises(KeyboardInterrupt):
            workflow.run(trace_path=trace)
        payload = json.loads(trace.read_bytes())
        assert payload["state"] == "interrupted"
        assert payload["steps"][0]["error"] == "KeyboardInterrupt: fixture failure"
        assert payload["steps"][0]["side_effects"][0]["completed"] is True
        validate_trace(payload)
    else:
        run = workflow.run()
        assert not run.ok and run.error_type == "ValueError"
        [body] = run.records
        assert body.error == "ValueError: fixture failure"
        assert body.side_effects[0].completed is True
        validate_trace(serialize_run(run))
    assert current() is None
    assert (tmp_path / "started.txt").read_bytes() == b"started"


def test_class_3_recipe_body_keeps_output_protection_and_soft_dry_run(tmp_path):
    target = tmp_path / "report.txt"
    target.write_bytes(b"original")

    @recipe(output_root=".")
    def workflow() -> str:
        return str(write_text("report.txt", "new"))

    for mode in (DryRunMode.OFF, DryRunMode.SOFT):
        run = workflow.run(output_directory=tmp_path, dry_run=mode)
        assert not run.ok
        assert "refusing to replace" in (run.error or "")
        assert target.read_bytes() == b"original"
    target.unlink()
    soft = workflow.run(output_directory=tmp_path, dry_run=DryRunMode.SOFT)
    assert soft.ok and not target.exists()
    assert not soft.records[0].side_effects[0].completed
    hard = workflow.run(output_directory=tmp_path, dry_run=DryRunMode.HARD)
    assert not hard.ok and "hard dry-run" in (hard.error or "")
    assert not target.exists()


def test_class_3_recipe_body_preserves_flat_step_only_traces():
    @step
    def inspect(value: int) -> int:
        return value + 1

    @recipe
    def workflow() -> list[int]:
        return parallel(inspect, [1, 2, 3])

    run = workflow.run()
    assert run.ok and run.output == [2, 3, 4]
    assert len(run.records) == 3
    assert all(record.name == "inspect" and not record.children for record in run.records)


def test_class_3_async_recipe_body_inside_an_active_event_loop(tmp_path):
    @step
    async def inspect(value: str) -> str:
        await asyncio.sleep(0)
        return value.upper()

    @recipe(output_root=str(tmp_path))
    async def workflow() -> str:
        result = await inspect("ready")
        write_text("report.txt", result)
        return result

    async def invoke():
        assert current() is None
        run = workflow.run()
        assert current() is None
        return run

    run = asyncio.run(invoke())
    assert run.ok and run.output == "READY"
    [body] = run.records
    assert [child.name for child in body.children] == ["inspect"]
    assert body.side_effects[0].completed is True
    assert (tmp_path / "report.txt").read_bytes() == b"READY"
    validate_trace(serialize_run(run))


def test_class_3_write_summary_preserves_body_child_body_execution_order(tmp_path):
    source = """from calcifier import recipe, step, write_text

@step
def middle() -> str:
    return str(write_text("middle.txt", "middle"))

@recipe(output_root="out")
def workflow() -> str:
    write_text("first.txt", "first")
    middle()
    write_text("first.txt", "first")
    return str(write_text("last.txt", "last"))

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nSave the first, middle, and last outputs.\n")
    verify_source(source)
    result = dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill)
    assert result == DryRunResult(
        "passed",
        writes=tuple(f"<workspace>/out/{name}.txt" for name in ("first", "middle", "last")),
    )
    path = tmp_path / "recipe.py"
    path.write_bytes(source.encode())
    run = load_recipe(path).run(output_directory=tmp_path / "out")
    assert run.ok
    assert [child.name for child in run.records[0].children] == ["middle"]
    payload = serialize_run(run)
    validate_trace(payload)
    assert "_recorded_at_ns" not in json.dumps(payload)
