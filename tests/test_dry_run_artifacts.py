"""Compiler dry runs can consume their own artifacts without changing host output."""

from __future__ import annotations

import json

import pytest

from calcifier.dry_run import dry_run_recipe
from calcifier.gate import verify_source

pytestmark = pytest.mark.semantic_offline


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("in_body", [False, True])
def test_dry_run_materializes_intermediate_artifacts_and_runs_the_consumer(
    tmp_path, newline, in_body
):
    data = ("first café" + newline + "last" + newline).encode()
    source = f"""from pathlib import Path
from calcifier import ensure_directory, recipe, step, verify_artifact, write_json, write_text

@step
def prepare() -> str:
    ensure_directory("nested")
    target = write_text("nested/input.txt", {data.decode()!r})
    write_json("metadata.json", {{"size": {len(data)}}})
    verify_artifact(target)
    return str(target)

@step
def consume(target: str) -> str:
    if Path(target).read_bytes() != {data!r}:
        raise ValueError("intermediate bytes changed")
    return str(write_text("finished.txt", "consumer ran"))

@recipe(output_root="out")
def workflow() -> str:
    return consume(prepare())

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    if in_body:
        source = source.replace(
            "    return consume(prepare())",
            f'    target = write_text("nested/input.txt", {data.decode()!r})\n'
            "    return consume(str(target))",
        )
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nPrepare the input, then consume it.\n")
    verify_source(source)
    result = dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill)
    assert result.status == "passed", result
    assert result.writes[-1] == "<workspace>/out/finished.txt"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("operation", ["write_text", "ensure_directory", "verify_artifact"])
def test_dry_run_never_materializes_outputs_outside_its_workspace(tmp_path, operation):
    root = tmp_path / "owner-output"
    root.mkdir()
    target = root / "report.txt"
    target.write_bytes(b"owner bytes")
    arguments = {
        "write_text": '"report.txt", "changed"',
        "ensure_directory": '"created"',
        "verify_artifact": '"report.txt"',
    }[operation]
    source = f"""from calcifier import recipe, step, {operation}

@step
def persist() -> str:
    return str({operation}({arguments}))

@recipe(output_root={str(root)!r})
def workflow() -> str:
    return persist()
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nSave the result.\n")
    result = dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill)
    if operation == "verify_artifact":
        assert result.status == "failed", result
        assert "required artifact does not exist" in (result.error or "")
    else:
        assert result.status == "passed", result
        assert result.writes[0].startswith("<workspace>/absolute-output/")
    assert target.read_bytes() == b"owner bytes"
    assert list(root.iterdir()) == [target]


def test_dry_run_scratch_writes_do_not_enable_mutating_commands(tmp_path):
    target = tmp_path / "command-output"
    command = f"from pathlib import Path; Path({str(target)!r}).write_text('changed')"
    source = f"""import sys
from calcifier import recipe, run_command, step, write_text

@step
def persist() -> str:
    run_command([sys.executable, "-c", {command!r}], timeout_seconds=10)
    return str(write_text("report.txt", "stub"))

@recipe(output_root="out")
def workflow() -> str:
    return persist()
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nSave the result.\n")
    result = dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill)
    assert result.status == "passed", json.dumps(result._asdict())
    assert not target.exists()
    assert not (tmp_path / "out").exists()
