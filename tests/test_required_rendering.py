"""Grouping structured results preserves source-required semantics in the final report."""

from __future__ import annotations

import json

import pytest

from calcifier.artifact import prepare_artifact
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from tests.support.agents import StubProvider
from tests.support.compiler import one_draft

_SOURCE = """from typing import Literal
from calcifier import Budget, agent, emit, output_type, recipe, step, verify_artifact, write_text

@output_type
class Entry:
    name: str
    category: Literal["public", "internal"]
    conditions: list[str]
    caller: str

def cell(value: str) -> str:
    return value.replace("|", "&#124;").replace("\\n", "<br>")

def render(entries: list[Entry]) -> str:
    rows = ["| Name | Category | Conditions | Caller |", "| --- | --- | --- | --- |"]
    for item in sorted(entries, key=lambda item: (item.category, item.name)):
        values = (
            item.name, item.category, "; ".join(item.conditions) or "none", item.caller or "any",
        )
        rows.append("| " + " | ".join(cell(value) for value in values) + " |")
    return "\\n".join(rows) + "\\n"

@step
def inspect() -> str:
    entries = agent("worker", "Review entries with their conditions and callers.",
                    working_directory=".", readable_paths=(".",), output=list[Entry])
    report = render(entries)
    path = write_text("report.md", report)
    verify_artifact(path)
    return emit(report, name="report")

@recipe(agents=("worker",), budget=Budget(2, 30), output_root="artifacts")
def workflow() -> str:
    return inspect()

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""


@pytest.fixture
def compiled_renderer(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(
        b"# Entry review\nReport each entry's name, category, conditions and caller. "
        b"Group entries by category without dropping conditions or caller requirements.\n"
    )
    registry = Registry()
    registry.register(one_draft(_SOURCE))
    worker = StubProvider()
    worker.name = "worker"
    registry.register(worker)
    artifact = prepare_artifact(skill, registry)
    assert artifact.outcome == "compiled"
    assert artifact.dry_run is not None and artifact.dry_run.status == "passed"
    artifact.path.parent.mkdir(parents=True)
    artifact.path.write_bytes(artifact.source.encode("utf-8"))
    return load_recipe(artifact.path), registry, worker


@pytest.mark.parametrize("category", ["public", "internal"])
@pytest.mark.parametrize(
    "conditions,caller,rendered_conditions,rendered_caller",
    [
        (
            ["requires approval", "within schedule"],
            "maintenance team",
            "requires approval; within schedule",
            "maintenance team",
        ),
        ([], "", "none", "any"),
        (["mode A|B"], "caller\nwith context", "mode A&#124;B", "caller<br>with context"),
    ],
)
def test_required_fields_survive_every_category_in_the_final_artifact(
    tmp_path, compiled_renderer, category, conditions, caller, rendered_conditions, rendered_caller
):
    workflow, registry, worker = compiled_renderer
    worker._default = json.dumps(
        [{"name": "entry", "category": category, "conditions": conditions, "caller": caller}]
    )
    output = tmp_path / "reports"

    result = workflow.run(registry=registry, output_directory=output)

    assert result.ok
    expected = (
        "| Name | Category | Conditions | Caller |\n"
        "| --- | --- | --- | --- |\n"
        f"| entry | {category} | {rendered_conditions} | {rendered_caller} |\n"
    )
    assert result.output == expected
    assert (output / "report.md").read_bytes() == expected.encode("utf-8")
