"""Neutral end-to-end report contracts distinguish field values from Markdown spacing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calcifier.baseline import is_baseline
from calcifier.calcify import calcify_file
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from tests.support.agents import StubProvider
from tests.support.compiler import ScriptedCompiler

SOURCE = """from calcifier import Budget, agent, fail, output_type, recipe, write_text

@output_type
class Report:
    markdown: str

@output_type
class Counts:
    ready: int
    waiting: int

def counts(markdown: str) -> Counts:
    rows: dict[str, int] = {}
    for line in markdown.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 2 or cells[0] not in {"Ready", "Waiting"}:
            continue
        if cells[0] in rows or not cells[1].isdigit():
            fail("invalid count row")
        rows[cells[0]] = int(cells[1])
    if set(rows) != {"Ready", "Waiting"}:
        fail("required count row is missing")
    return Counts(ready=rows["Ready"], waiting=rows["Waiting"])

@recipe(agents=("worker",), budget=Budget(2, 30), output_root=".")
def run() -> Counts:
    report = agent("worker", "Return the complete inventory summary.",
                   working_directory=".", readable_paths=(".",), output=Report)
    result = counts(report.markdown)
    write_text("summary.md", report.markdown)
    return result

if __name__ == "__main__":
    raise SystemExit(run.cli())
"""


@pytest.fixture
def compiled_table_recipe(tmp_path: Path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "# Inventory summary\nReturn a Markdown table with Ready and Waiting rows and "
        "nonnegative integer counts. Save the complete report as summary.md.\n",
        encoding="utf-8",
    )
    compiler = ScriptedCompiler([SOURCE], wrap_sources=True)
    registry = Registry()
    registry.register(compiler)
    worker = StubProvider()
    worker.name = "worker"
    registry.register(worker)
    recipe = calcify_file(skill, registry)
    assert len(compiler.calls) == 1
    assert not is_baseline(recipe.read_text(encoding="utf-8"))
    return load_recipe(recipe)


@pytest.mark.parametrize("padding", ["", " ", "     ", "\t"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_presentation_whitespace_does_not_reject_valid_report_fields(
    compiled_table_recipe, tmp_path: Path, padding: str, newline: str
) -> None:
    markdown = newline.join(
        [
            "# Inventory",
            "| State | Count |",
            "| :--- | ---: |",
            f"|{padding}Ready{padding}|{padding}3{padding}|",
            f"|{padding}Waiting{padding}|{padding}0{padding}|",
        ]
    )
    worker = StubProvider(default=json.dumps({"markdown": markdown}))
    worker.name = "worker"
    registry = Registry()
    registry.register(worker)
    destination = tmp_path / "output"
    run = compiled_table_recipe.run({}, registry=registry, output_directory=destination)
    assert run.ok, run.error
    assert run.output.model_dump() == {"ready": 3, "waiting": 0}
    assert (destination / "summary.md").read_bytes() == markdown.encode("utf-8")


@pytest.mark.parametrize(
    "markdown",
    [
        "| Ready | 3 |",
        "| Ready | -1 |\n| Waiting | 0 |",
        "| Ready | 3 |\n| Ready | 4 |\n| Waiting | 0 |",
        "| Ready | three |\n| Waiting | 0 |",
    ],
)
def test_invalid_fields_still_fail_before_report_persistence(
    compiled_table_recipe, tmp_path: Path, markdown: str
) -> None:
    worker = StubProvider(default=json.dumps({"markdown": markdown}))
    worker.name = "worker"
    registry = Registry()
    registry.register(worker)
    destination = tmp_path / "output"
    run = compiled_table_recipe.run({}, registry=registry, output_directory=destination)
    assert not run.ok
    assert not (destination / "summary.md").exists()
