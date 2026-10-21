"""Complete inventories and observed execution support truthful completion states."""

from __future__ import annotations

import json

import pytest

from calcifier.artifact import prepare_artifact
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from tests.support.compiler import one_draft

_DATA = {
    "small": {"items": ["a", "b"]},
    "boundary": {"items": ["a", "b", "c", "d", "e"]},
    "truncated": {"items": ["a", "b", "c", "d", "e"], "cap": 2},
    "zero": {"items": ["a", "b"], "samples": 0},
    "uninspected": {"items": ["a", "b"], "inspect": False},
    "blocked": {"items": ["a", "b"], "blocked": True},
    "empty": {"items": []},
}
_SOURCE = """import sys
from typing import Literal
from calcifier import emit, fail, output_type, recipe, resource, run_command, step

@output_type
class Page:
    items: list[str]
    next_cursor: int | None
    total: int
    truncated: bool

@output_type
class Measurement:
    inspected: list[str]
    executed: int
    state: Literal["observed", "blocked"]

@output_type
class Result:
    state: Literal["measured", "empty"]
    items: list[str]
    pages: int
    executed: int

def command(mode: str, operation: str, cursor: int = 0) -> list[str]:
    return [sys.executable, str(resource("scripts/observe.py")),
            "--data", str(resource("references/data.json")), "--mode", mode,
            "--operation", operation, "--cursor", str(cursor)]

@step
def measure(mode: str) -> Result:
    items: list[str] = []
    seen: set[int] = set()
    cursor = 0
    total: int | None = None
    while True:
        if cursor in seen:
            fail("Inventory cursor did not progress; coverage is incomplete.")
        seen.add(cursor)
        page = run_command(command(mode, "page", cursor), timeout_seconds=10,
                           mutating=False, output=Page)
        if total is None:
            total = page.total
        if page.truncated or page.total != total:
            fail("Inventory is truncated or inconsistent; coverage is incomplete.")
        items.extend(page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    if len(items) != total or len(set(items)) != len(items):
        fail("Enumerated inventory does not cover the independently reported total.")
    if not items:
        return emit(Result(state="empty", items=[], pages=len(seen), executed=0))
    observed = run_command(command(mode, "measure"), timeout_seconds=10,
                           mutating=False, output=Measurement)
    if observed.state == "blocked":
        fail("Measurements are blocked; effectiveness remains unmeasured.")
    if sorted(observed.inspected) != sorted(items):
        fail("The full source inventory was not inspected.")
    if observed.executed <= 0:
        fail("No executed observations; effectiveness remains unmeasured.")
    return emit(Result(state="measured", items=items, pages=len(seen),
                       executed=observed.executed))

@recipe(capabilities=("Bash",))
def workflow(mode: Literal["small", "boundary", "truncated", "zero",
                           "uninspected", "blocked", "empty"] = "small") -> Result:
    return measure(mode)

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
_PROGRAM = """import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--mode", required=True)
parser.add_argument("--operation", choices=("page", "measure"), required=True)
parser.add_argument("--cursor", type=int, default=0)
args = parser.parse_args()
data = json.loads(args.data.read_bytes())[args.mode]
items = data["items"]
if args.operation == "page":
    visible = items[:data.get("cap", len(items))]
    end = args.cursor + 2
    result = {"items": visible[args.cursor:end], "total": len(items),
              "next_cursor": end if end < len(visible) else None,
              "truncated": len(visible) < len(items)}
else:
    inspected = items if data.get("inspect", True) else []
    executed = sum(len(list(range(data.get("samples", 1)))) for _ in inspected)
    result = {"inspected": inspected, "executed": executed,
              "state": "blocked" if data.get("blocked") else "observed"}
print(json.dumps(result))
"""


@pytest.fixture
def compiled_measurement(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(
        b"# Measure\nEnumerate every page of the selected inventory and inspect all items. "
        b"Claim measured completion only with executed observations. Report empty input "
        b"separately; truncated, blocked and unmeasured work cannot be called measured.\n"
    )
    references = tmp_path / "references"
    references.mkdir()
    (references / "data.json").write_bytes(json.dumps(_DATA).encode("utf-8"))
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "observe.py").write_bytes(_PROGRAM.encode("utf-8"))
    registry = Registry()
    registry.register(one_draft(_SOURCE))

    artifact = prepare_artifact(skill, registry)

    assert artifact.outcome == "compiled"
    assert artifact.dry_run is not None and artifact.dry_run.status == "passed"
    artifact.path.parent.mkdir(parents=True)
    artifact.path.write_bytes(artifact.source.encode("utf-8"))
    return load_recipe(artifact.path)


@pytest.mark.parametrize("mode,pages", [("small", 1), ("boundary", 3), ("empty", 1)])
def test_all_pages_are_counted_and_empty_scope_is_distinct_from_measurement(
    compiled_measurement, mode, pages
):
    result = compiled_measurement.run({"mode": mode})

    expected = _DATA[mode]["items"]
    assert isinstance(expected, list)
    assert result.ok
    assert result.output.items == expected
    assert result.output.pages == pages
    assert result.output.state == ("empty" if mode == "empty" else "measured")
    assert result.output.executed == len(expected)


@pytest.mark.parametrize(
    "mode,error",
    [
        ("truncated", "Inventory is truncated"),
        ("zero", "No executed observations"),
        ("uninspected", "full source inventory was not inspected"),
        ("blocked", "Measurements are blocked"),
    ],
)
def test_incomplete_or_unobserved_work_cannot_claim_measured_completion(
    compiled_measurement, mode, error
):
    result = compiled_measurement.run({"mode": mode})

    assert not result.ok
    assert error in (result.error or "")
    assert result.output is None
