"""Review regressions distinguish suppressed effects from actual workflow defects."""

from __future__ import annotations

from pathlib import Path

import pytest

from calcifier.dry_run import dry_run_recipe
from calcifier.recipe import load_recipe


def run_candidate(tmp_path, source, **kwargs):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Report\nGather the selected inputs, save and verify the report.\n")
    return dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill, **kwargs)


@pytest.mark.parametrize("writable", [False, True])
def test_stub_worker_leaves_only_its_planned_writable_outputs_unverified(tmp_path, writable):
    source = """from calcifier import Budget, agent, ensure_directory, recipe, verify_artifact

@recipe(agents=("worker",), output_root="artifacts", budget=Budget(2, 30))
def workflow() -> str:
    destination = ensure_directory(output_files=("report.json",))
    agent("worker", "Save the report.", working_directory=destination, output=str,
          writable_paths=WRITABLE)
    return str(verify_artifact("report.json"))

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
""".replace("WRITABLE", "(destination,)" if writable else "()")

    outcome = run_candidate(tmp_path, source, agent_capabilities={"worker": ("read", "write")})

    assert outcome.status == ("unverified" if writable else "failed"), outcome
    assert "report.json" in (outcome.error or "")
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    ("output_root", "directory"),
    [
        ("calcifier-output", "reports"),
        ("input", "reports"),
        ("input/reports", "reports"),
        (".", "input"),
    ],
)
def test_required_path_input_does_not_enclose_relative_output_directory(
    tmp_path, output_root, directory
):
    source = f"""from pathlib import Path
from calcifier import ensure_directory, recipe

@recipe(output_root={output_root!r})
def workflow(project: Path) -> str:
    assert (project / "example.txt").is_file()
    return str(ensure_directory({directory!r}, output_files=("raw.json",),
                                exclude_roots=(project,)))

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""

    assert run_candidate(tmp_path, source).status == "passed"
    candidate = tmp_path / "recipe.py"
    candidate.write_bytes(source.encode("utf-8"))
    project = tmp_path / "project"
    project.mkdir()
    (project / "example.txt").write_bytes(b"example\n")
    output = tmp_path / "output"
    result = load_recipe(candidate).run({"project": project}, output_directory=output)
    assert result.ok
    assert isinstance(result.output, str)
    assert Path(result.output) == output / directory


@pytest.mark.parametrize("suppressed", [False, True])
@pytest.mark.parametrize("access", ["wrapped", "direct", "wrong-path"])
def test_wrapped_and_direct_reads_recognize_only_suppressed_planned_outputs(
    tmp_path, suppressed, access
):
    accesses = {
        "wrapped": '''try:
        verify_artifact("report.json")
    except RecipeError as exc:
        raise RecipeError(f"delivery failed: {exc}") from exc
    return "saved"''',
        "direct": 'return (destination / "report.json").read_text(encoding="utf-8")',
        "wrong-path": 'return (destination / "wrong.json").read_text(encoding="utf-8")',
    }
    source = """import sys
from calcifier import RecipeError, ensure_directory, recipe, run_command, verify_artifact

@recipe(output_root="artifacts", capabilities=("Bash",))
def workflow() -> str:
    destination = ensure_directory(output_files=("report.json",))
    COMMAND
    ACCESS

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
""".replace(
        "COMMAND",
        'run_command([sys.executable, "-c", "print(1)"], timeout_seconds=5, mutating=True)'
        if suppressed
        else "pass",
    ).replace("ACCESS", accesses[access])

    outcome = run_candidate(tmp_path, source)

    gap = suppressed and access != "wrong-path"
    assert outcome.status == ("unverified" if gap else "failed")
    assert ("wrong.json" if access == "wrong-path" else "report.json") in (outcome.error or "")


@pytest.mark.parametrize("prepared", [False, True])
def test_missing_default_input_retains_its_cause_through_context_gathering(tmp_path, prepared):
    source = """from pathlib import Path
from calcifier import prepare_context, recipe

@recipe
def workflow(target: Path = Path("project")) -> str:
    READ

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
""".replace(
        "READ",
        'return prepare_context("review", files=(target / "manifest.json",), '
        "readable_roots=(target,)).prompt"
        if prepared
        else 'return (target / "manifest.json").read_text(encoding="utf-8")',
    )

    outcome = run_candidate(tmp_path, source)

    assert outcome.status == "unverified"
    assert "absent default path" in (outcome.error or "")
