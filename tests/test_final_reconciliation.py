"""Final findings and coverage reflect validation and retries, with history retained."""

from __future__ import annotations

import json

import pytest

from calcifier.artifact import prepare_artifact
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from tests.support.agents import StubProvider
from tests.support.compiler import one_draft

_SOURCE = """from typing import Literal
from calcifier import (
    Budget, agent, emit, fail, output_type, recipe, step, verify_artifact, write_text,
)

@output_type
class Finding:
    id: str
    file: str
    disposition: Literal["candidate", "confirmed", "refuted"]

@output_type
class Outcome:
    findings: list[Finding]
    processed_files: list[str]
    blocked_files: list[str]

@output_type
class Phase:
    stage: str
    outcome: Outcome

@output_type
class Coverage:
    processed_files: list[str]
    unprocessed_files: list[str]

@output_type
class Report:
    state: Literal["complete", "incomplete"]
    findings: list[Finding]
    coverage: Coverage
    history: list[Phase]

@step
def review(files: list[str]) -> str:
    expected = set(files)
    blocked = set(files)
    findings: dict[str, Finding] = {}
    history: list[Phase] = []
    for stage in ("initial", "validation-and-retry"):
        outcome = agent(
            "worker", f"Perform {stage} on {sorted(expected)}. Prior results: {history}",
            working_directory=".", readable_paths=(".",), output=Outcome)
        processed, pending = set(outcome.processed_files), set(outcome.blocked_files)
        if (processed | pending) - expected or processed & pending:
            fail("Phase coverage conflicts with the selected inventory.")
        ids = [finding.id for finding in outcome.findings]
        if len(set(ids)) != len(ids):
            fail("Duplicate finding identity in one phase.")
        for finding in outcome.findings:
            if finding.file not in expected:
                fail("Finding is outside the selected inventory.")
            previous = findings.get(finding.id)
            if previous is not None and previous.file != finding.file:
                fail("Finding identity changed its source file.")
            findings[finding.id] = finding
        blocked.difference_update(processed)
        blocked.update(pending)
        history.append(Phase(stage, outcome))
    blocked.update(item.file for item in findings.values() if item.disposition == "candidate")
    final = sorted((item for item in findings.values() if item.disposition == "confirmed"),
                   key=lambda item: item.id)
    report = Report(
        "incomplete" if blocked else "complete", final,
        Coverage(sorted(expected - blocked), sorted(blocked)), history)
    text = report.model_dump_json()
    path = write_text("report.json", text)
    verify_artifact(path)
    return emit(text, name="report")

@recipe(agents=("worker",), budget=Budget(4, 30), output_root="artifacts")
def workflow(files: list[str]) -> str:
    return review(files)

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""


@pytest.fixture
def compiled_reconciliation(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(
        b"# Review inventory\nReview the supplied complete file inventory, then validate "
        b"candidate findings and retry blocked files once. Report final confirmed "
        b"findings and coverage.unprocessed_files after validation and retries. "
        b"Refuted candidates must not appear as final findings. Keep phase history "
        b"separately, and identify incomplete coverage explicitly.\n"
    )
    registry = Registry()
    registry.register(one_draft(_SOURCE))
    worker = StubProvider()
    worker.name = "worker"
    registry.register(worker)
    artifact = prepare_artifact(skill, registry)
    assert artifact.outcome == "compiled"
    assert artifact.dry_run is not None and artifact.dry_run.status == "unverified"
    assert artifact.dry_run.error == "Phase coverage conflicts with the selected inventory."
    artifact.path.parent.mkdir(parents=True)
    artifact.path.write_bytes(artifact.source.encode("utf-8"))
    return load_recipe(artifact.path), registry, worker


@pytest.mark.parametrize("still_blocked", [False, True])
def test_final_artifact_excludes_refuted_candidates_and_reconciles_retry_coverage(
    tmp_path, compiled_reconciliation, still_blocked
):
    workflow, registry, worker = compiled_reconciliation
    initial = {
        "findings": [
            {"id": "discarded", "file": "alpha.txt", "disposition": "candidate"},
            {"id": "retained", "file": "alpha.txt", "disposition": "candidate"},
        ],
        "processed_files": ["alpha.txt"],
        "blocked_files": ["beta.txt"],
    }
    validated = {
        "findings": [
            {"id": "discarded", "file": "alpha.txt", "disposition": "refuted"},
            {"id": "retained", "file": "alpha.txt", "disposition": "confirmed"},
        ],
        "processed_files": ["alpha.txt"] if still_blocked else ["alpha.txt", "beta.txt"],
        "blocked_files": ["beta.txt"] if still_blocked else [],
    }
    worker._responses = {
        "Perform initial": json.dumps(initial),
        "Perform validation-and-retry": json.dumps(validated),
    }
    output = tmp_path / "reports"

    result = workflow.run(
        {"files": ["alpha.txt", "beta.txt"]}, registry=registry, output_directory=output
    )

    assert result.ok
    saved = (output / "report.json").read_bytes()
    assert saved == result.output.encode("utf-8")
    report = json.loads(saved)
    assert report["state"] == ("incomplete" if still_blocked else "complete")
    assert report["findings"] == [
        {"id": "retained", "file": "alpha.txt", "disposition": "confirmed"}
    ]
    assert report["coverage"] == {
        "processed_files": ["alpha.txt"] if still_blocked else ["alpha.txt", "beta.txt"],
        "unprocessed_files": ["beta.txt"] if still_blocked else [],
    }
    assert report["history"] == [
        {"stage": "initial", "outcome": initial},
        {"stage": "validation-and-retry", "outcome": validated},
    ]
