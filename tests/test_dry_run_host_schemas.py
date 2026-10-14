"""Schema stubs synthesize valid host values before native transport normalization."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

import pytest

from calcifier.dry_run import _SchemaStub, dry_run_recipe
from calcifier.gate import verify_source

pytestmark = pytest.mark.semantic_offline


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (tuple[int, str], (1, "stub")),
        (list[tuple[int, str]], [(1, "stub")]),
        (date, date(2000, 1, 1)),
        (datetime, datetime.fromisoformat("2000-01-01T00:00:00Z")),
        (dict[str, tuple[int, str]], {"stub": (1, "stub")}),
        (dict[Literal["first", "second"], int], {"first": 1}),
        (dict[date, int], {date(2000, 1, 1): 1}),
        (dict[int, str], {1: "stub"}),
        (dict[str, dict[int, str]], {"stub": {1: "stub"}}),
        (list[dict[int, tuple[int, str]]], [{1: (1, "stub")}]),
    ],
)
def test_stub_values_pass_the_exact_host_annotation(annotation, expected):
    assert _SchemaStub("worker").structured("return a result", annotation=annotation) == expected


@pytest.mark.parametrize(
    "annotation, check",
    [
        ("tuple[int, str]", 'value == (1, "stub")'),
        ("date", 'value.isoformat() == "2000-01-01"'),
        ("datetime", 'value.date().isoformat() == "2000-01-01"'),
        ("dict[str, tuple[int, str]]", 'list(value.values()) == [(1, "stub")]'),
        ("dict[int, str]", 'value == {1: "stub"}'),
        ("dict[str, dict[int, str]]", 'value == {"stub": {1: "stub"}}'),
    ],
)
def test_supported_agent_outputs_execute_the_final_deterministic_stage(tmp_path, annotation, check):
    source = f"""from datetime import date, datetime
from calcifier import Budget, agent, recipe, step, write_text

@step
def inspect() -> {annotation}:
    return agent("worker", "Return a result.", working_directory=".", output={annotation})

@step
def finish(value: {annotation}) -> str:
    if not ({check}):
        raise ValueError("the synthetic value did not preserve its host type")
    return str(write_text("finished.txt", "final stage ran"))

@recipe(agents=("worker",), budget=Budget(2, 60), output_root="out")
def workflow() -> str:
    return finish(inspect())

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nInterpret the input and finish the report.\n")
    verify_source(source)
    result = dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill)
    assert result.status == "passed", result
    assert result.writes == ("<workspace>/out/finished.txt",)
