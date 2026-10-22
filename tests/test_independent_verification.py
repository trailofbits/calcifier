"""Worker completion claims cannot override independent required checker observations."""

from __future__ import annotations

import json

import pytest

from calcifier.artifact import prepare_artifact
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from tests.support.agents import StubProvider
from tests.support.compiler import one_draft

_SOURCE = """import sys
from pathlib import Path
from calcifier import Budget, agent, emit, fail, output_type, recipe, resource, run_command, step

@output_type
class Claim:
    complete: bool
    summary: str

@output_type
class Check:
    checked_files: int
    obligations: list[str]
    lint_failures: list[str]

@step
def inspect(project: Path) -> str:
    claim = agent("worker", "Review the complete target using the original instructions.",
                  working_directory=project, readable_paths=(project,),
                  required_capabilities=("read",), output=Claim)
    observed = run_command(
        [sys.executable, str(resource("scripts/check.py")), str(project)],
        working_directory=project, timeout_seconds=10, mutating=False, output=Check)
    emit(observed.model_dump(), name="independent-check")
    if observed.checked_files <= 0:
        fail("No source files inspected; completion remains unverified.")
    if observed.obligations or observed.lint_failures:
        fail("Independent required verification failed; work is incomplete.")
    if not claim.complete:
        fail("Worker has not completed the assigned reasoning.")
    return emit(claim.summary, name="review")

@recipe(agents=("worker",), capabilities=("Bash",), budget=Budget(2, 30))
def workflow(project: Path = Path(".")) -> str:
    return inspect(project)

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
_CHECKER = """import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
selected = sorted(root.glob("*.txt"))
resolved = [path.resolve(strict=True) for path in selected]
if any(not path.is_relative_to(root) or not path.is_file() for path in resolved):
    raise SystemExit("Selected source escapes the project or is not a regular file.")
obligations = []
lint_failures = []
for path in resolved:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        location = f"{path.name}:{number}"
        if "TODO" in line:
            obligations.append(location)
        if "LINT" in line:
            lint_failures.append(location)
# Exit zero transports the observations; it does not establish successful verification.
print(json.dumps({"checked_files": len(resolved), "obligations": obligations,
                  "lint_failures": lint_failures}))
"""


@pytest.fixture
def compiled_review(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(
        b"# Review\nUse an agent to interpret the target and explain its conclusions. "
        b"Then independently run scripts/check.py over all selected source files. "
        b"Completion requires inspected files, no open obligations, and no lint failures. "
        b"Keep checker observations in the result; an exit-zero checker is not enough.\n"
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "check.py").write_bytes(_CHECKER.encode("utf-8"))
    registry = Registry()
    registry.register(one_draft(_SOURCE))
    worker = StubProvider(
        default=json.dumps({"complete": True, "summary": "Worker claims completion."})
    )
    worker.name = "worker"
    worker.capabilities = ("read",)
    registry.register(worker)

    artifact = prepare_artifact(skill, registry)

    assert artifact.outcome == "compiled"
    assert artifact.dry_run is not None and artifact.dry_run.status == "unverified"
    assert "Worker has not completed" in (artifact.dry_run.error or "")
    artifact.path.parent.mkdir(parents=True)
    artifact.path.write_bytes(artifact.source.encode("utf-8"))
    return load_recipe(artifact.path), registry, worker


@pytest.mark.parametrize(
    "contents,obligations,lints",
    [(b"TODO unresolved obligation\n", 1, 0), (b"LINT required check failed\n", 0, 1)],
)
def test_claimed_completion_is_rejected_after_exit_zero_checker_reports_required_failures(
    tmp_path, compiled_review, contents, obligations, lints
):
    workflow, registry, _ = compiled_review
    project = tmp_path / "project"
    project.mkdir()
    (project / "source.txt").write_bytes(contents)

    result = workflow.run({"project": project}, registry=registry)

    assert not result.ok
    assert "Independent required verification failed" in (result.error or "")
    observed = next(item.value for item in result.artifacts if item.name == "independent-check")
    assert observed["checked_files"] == 1
    assert len(observed["obligations"]) == obligations
    assert len(observed["lint_failures"]) == lints
    assert not any(item.name == "review" for item in result.artifacts)


@pytest.mark.parametrize("worker_complete", [False, True])
def test_success_requires_the_independent_check_and_reasoning_completion(
    tmp_path, compiled_review, worker_complete
):
    workflow, registry, worker = compiled_review
    project = tmp_path / "project"
    project.mkdir()
    (project / "source.txt").write_bytes(b"complete checked source\n")
    worker._default = json.dumps({"complete": worker_complete, "summary": "Review conclusions."})

    result = workflow.run({"project": project}, registry=registry)

    observed = next(item.value for item in result.artifacts if item.name == "independent-check")
    assert observed == {"checked_files": 1, "obligations": [], "lint_failures": []}
    assert result.ok is worker_complete
    if worker_complete:
        assert result.output == "Review conclusions."
    else:
        assert "Worker has not completed" in (result.error or "")


def test_worker_cannot_claim_success_when_the_checker_inspected_no_files(tmp_path, compiled_review):
    workflow, registry, _ = compiled_review
    project = tmp_path / "empty"
    project.mkdir()

    result = workflow.run({"project": project}, registry=registry)

    assert not result.ok
    assert "No source files inspected" in (result.error or "")
    observed = next(item.value for item in result.artifacts if item.name == "independent-check")
    assert observed["checked_files"] == 0
