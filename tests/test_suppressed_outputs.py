"""Declared external outputs remain absent and unverified under command suppression."""

from __future__ import annotations

import pytest

from calcifier.dry_run import dry_run_recipe
from calcifier.recipe import load_recipe

_SOURCE = """import sys
from calcifier import ensure_directory, recipe, run_command, step, verify_artifact, write_text

@step
def produce() -> str:
    destination = ensure_directory(output_files=OUTPUTS)
    COMMAND
    HOST_WRITE
    return str(verify_artifact(VERIFIED))

@recipe(output_root="artifacts", capabilities=("Bash",))
def workflow() -> str:
    return produce()

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
_COMMAND = 'run_command([sys.executable, "-c", "print(1)"], timeout_seconds=5, mutating=MUTATING)'


@pytest.mark.parametrize(
    "planned,command,mutating,host_write,verified,status",
    [
        (True, True, True, False, "report.md", "unverified"),
        (False, True, True, False, "report.md", "failed"),
        (True, False, True, False, "report.md", "failed"),
        (True, True, False, False, "report.md", "failed"),
        (True, True, True, False, "different.md", "failed"),
        (True, True, True, True, "report.md", "passed"),
    ],
)
def test_only_declared_outputs_after_suppressed_commands_are_coverage_gaps(
    tmp_path, planned, command, mutating, host_write, verified, status
):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Produce\nSave and independently verify report.md.\n")
    candidate = tmp_path / "recipe.py"
    source = (
        _SOURCE.replace("OUTPUTS", '("report.md",)' if planned else "()")
        .replace("COMMAND", _COMMAND.replace("MUTATING", str(mutating)) if command else "pass")
        .replace("HOST_WRITE", 'write_text("report.md", "complete\\n")' if host_write else "pass")
        .replace("VERIFIED", repr(verified))
    )

    outcome = dry_run_recipe(source, recipe_path=candidate, skill_path=skill)

    assert outcome.status == status
    if status == "unverified":
        assert "declared output is unavailable after a suppressed command" in (outcome.error or "")
        assert "downstream checks remain unverified" in (outcome.error or "")
    elif status == "failed":
        assert "required artifact does not exist" in (outcome.error or "")
    assert not (tmp_path / "artifacts").exists()


def test_actual_runtime_still_requires_the_file_after_a_successful_noop_command(tmp_path):
    source = (
        _SOURCE.replace("OUTPUTS", '("report.md",)')
        .replace("COMMAND", _COMMAND.replace("MUTATING", "True"))
        .replace("HOST_WRITE", "pass")
        .replace("VERIFIED", '"report.md"')
    )
    path = tmp_path / "recipe.py"
    path.write_bytes(source.encode("utf-8"))

    result = load_recipe(path).run(output_directory=tmp_path / "artifacts")

    assert not result.ok
    assert "required artifact does not exist" in (result.error or "")
    assert not (tmp_path / "artifacts" / "report.md").exists()


@pytest.mark.parametrize("empty", [False, True])
def test_materialized_content_failures_remain_failures_after_command_suppression(tmp_path, empty):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Produce\nSave a nonempty report with the expected content.\n")
    source = (
        _SOURCE.replace("OUTPUTS", '("report.md",)')
        .replace("COMMAND", _COMMAND.replace("MUTATING", "True"))
        .replace("HOST_WRITE", f'write_text("report.md", {"" if empty else "wrong content"!r})')
        .replace("VERIFIED", '"report.md", sha256=' + repr("0" * 64))
    )

    outcome = dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill)

    assert outcome.status == "failed"
    assert ("required artifact is empty" if empty else "artifact hash mismatch") in (
        outcome.error or ""
    )


@pytest.mark.parametrize("later_failure", [False, True])
def test_caught_artifact_gap_never_becomes_a_pass_or_hides_a_later_failure(tmp_path, later_failure):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(
        b"# Produce\nKeep a usable alternative if external delivery is unavailable.\n"
    )
    source = """import sys
from calcifier import (
    RecipeError, ensure_directory, recipe, run_command, verify_artifact, write_text,
)

@recipe(output_root="artifacts", capabilities=("Bash",))
def workflow() -> str:
    ensure_directory(output_files=("report.md",))
    run_command([sys.executable, "-c", "print(1)"], timeout_seconds=5, mutating=True)
    try:
        verify_artifact("report.md")
    except RecipeError:
        write_text("alternative.md", "Delivery remains incomplete.\\n")
    return RESULT

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
""".replace("RESULT", "str(1 // 0)" if later_failure else '"alternative"')

    outcome = dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill)

    assert outcome.status == ("failed" if later_failure else "unverified")
    assert any(path.endswith("/alternative.md") for path in outcome.writes)
    assert not outcome.command_gap
    assert ("ZeroDivisionError" if later_failure else "suppressed command") in (outcome.error or "")


def test_absolute_output_root_accepts_chained_helper_paths_in_compiler_scratch(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Produce\nSave and verify the complete report.\n")
    destination = tmp_path / "real-output"
    source = f"""from calcifier import ensure_directory, recipe, verify_artifact, write_text

@recipe(output_root={str(destination)!r})
def workflow() -> str:
    directory = ensure_directory(output_files=("report.md",))
    path = write_text(directory / "report.md", "complete\\n")
    return str(verify_artifact(path))

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""

    outcome = dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill)

    assert outcome.status == "passed"
    assert any(path.endswith("/report.md") for path in outcome.writes)
    assert not destination.exists()
