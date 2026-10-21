"""Saved delivery is checked against trusted destinations and independently known scope."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calcifier.artifact import prepare_artifact
from calcifier.prompts import build_prompt
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from tests.support.agents import StubProvider
from tests.support.compiler import one_draft

_SOURCE = """import json
from calcifier import (
    Budget, agent, emit, ensure_directory, fail, output_type, recipe, step, verify_artifact,
)

@output_type
class Delivery:
    saved: bool
    path: str
    items: list[str]

@step
def deliver() -> str:
    expected = ["sample-a", "sample-b"]
    destination = ensure_directory(output_files=("report.json",))
    claim = agent("worker", "Save the complete report for {items} as report.json.",
                  items=expected, working_directory=destination,
                  readable_paths=(destination,), writable_paths=(destination,),
                  required_capabilities=("write",), output=Delivery)
    if not claim.saved or sorted(claim.items) != expected:
        fail("Worker delivery is incomplete; saved content remains unverified.")
    report = verify_artifact("report.json")
    rows = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or sorted(row["id"] for row in rows) != expected:
        fail("Saved report does not cover the full intended inventory.")
    return emit(str(report), name="saved-report")

@recipe(agents=("worker",), output_root="artifacts", budget=Budget(2, 30))
def workflow() -> str:
    return deliver()

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""


class SavedWorker(StubProvider):
    name = "worker"
    capabilities = ("read", "write")

    def __init__(self):
        super().__init__()
        self.contents = b""
        self.mode = "missing"
        self.invocations = 0

    def invoke(self, prompt, *, filesystem=None, **kwargs):
        assert filesystem is not None
        self.invocations += 1
        report = filesystem.working_directory / "report.json"
        if self.mode == "directory":
            report.mkdir()
        elif self.mode != "missing":
            report.write_bytes(self.contents)
        # The path is opaque text, deliberately different from the authorized destination.
        self._default = json.dumps(
            {
                "saved": True,
                "path": "/untrusted-worker-report.json",
                "items": ["sample-a", "sample-b"],
            }
        )
        return super().invoke(prompt, filesystem=filesystem, **kwargs)


@pytest.fixture
def compiled_delivery(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(
        b"# Delivery\nSave a JSON report containing exactly sample-a and sample-b. "
        b"Announce saved delivery only after independently verifying the complete report.\n"
    )
    registry = Registry()
    registry.register(one_draft(_SOURCE))
    worker = SavedWorker()
    registry.register(worker)
    artifact = prepare_artifact(skill, registry)
    assert artifact.outcome == "compiled"
    # Schema stubs cannot establish the meaning or saved content of a worker report.
    assert artifact.dry_run is not None and artifact.dry_run.status == "unverified"
    assert "Worker delivery is incomplete" in (artifact.dry_run.error or "")
    artifact.path.parent.mkdir(parents=True)
    artifact.path.write_bytes(artifact.source.encode("utf-8"))
    return load_recipe(artifact.path), registry, worker


@pytest.mark.parametrize(
    "mode,contents,error",
    [
        ("missing", b"", "required artifact does not exist"),
        ("file", b"", "required artifact is empty"),
        ("directory", b"", "required artifact does not exist"),
        ("file", b'[{"id":"sample-a"}]\n', "full intended inventory"),
        ("file", b'[{"id":"sample-a"},{"id":"sample-a"}]\n', "full intended inventory"),
    ],
)
def test_worker_saved_claim_cannot_hide_absent_empty_or_incomplete_delivery(
    tmp_path, compiled_delivery, mode, contents, error
):
    workflow, registry, worker = compiled_delivery
    worker.mode, worker.contents = mode, contents
    output = tmp_path / "reports"

    result = workflow.run(registry=registry, output_directory=output)

    assert not result.ok
    assert error in (result.error or "")
    assert worker.invocations == 1
    assert not any(artifact.name == "saved-report" for artifact in result.artifacts)


def test_complete_saved_report_is_announced_from_the_trusted_destination(
    tmp_path, compiled_delivery
):
    workflow, registry, worker = compiled_delivery
    worker.mode = "file"
    worker.contents = b'[{"id":"sample-b"},{"id":"sample-a"}]\r\n'
    output = tmp_path / "reports"

    result = workflow.run(registry=registry, output_directory=output)

    assert result.ok
    assert result.output == str((output / "report.json").resolve())
    assert Path(result.output).read_bytes() == worker.contents
    assert worker.invocations == 1
    assert any(artifact.name == "saved-report" for artifact in result.artifacts)


def test_generation_receives_saved_delivery_and_required_rendering_contracts():
    prompt = " ".join(build_prompt("Save the complete review.").split())
    assert "verify the trusted declared destination, never a worker-provided path" in prompt
    assert "compare its records with the full independently gathered inventory" in prompt
    assert "Preserve every source-required field through deterministic rendering" in prompt
