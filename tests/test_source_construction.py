"""Public-neutral source-phase reliability regressions (no native providers)."""

from __future__ import annotations

import ast

import pytest

from calcifier.agent_events import AgentActivity
from calcifier.calcify import (
    CalcifyError,
    calcify_source,
)
from calcifier.metadata import attach_metadata
from calcifier.portable import lint_portable
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from calcifier.source_quality import (
    format_and_lint_source,
    lint_source_quality,
)
from tests.support.agents import StubProvider
from tests.support.compiler import source_sequence
from tests.support.recipes import (
    _REPAIR_RECIPE,
    _REPAIR_SKILL,
    _STATIC_FAILURE_RECIPE,
    _distinct_failures,
)

pytestmark = pytest.mark.semantic_offline


@pytest.mark.parametrize(
    "first,second", [([], []), ([(1, 2)], []), ([], [(3, 4)]), ([(1, 2)], [(3, 4)])]
)
def test_b007_alpha_renaming_preserves_later_rebindings_and_empty_loops(tmp_path, first, second):
    source = """def collect(first, second):
    categories = "initial"
    _unused_categories = "keep"
    values = []
    for term, categories in first:
        values.append(term)
    for term, categories in second:
        values.append((term, categories))
    return values, categories, _unused_categories
"""
    formatted = format_and_lint_source(source, tmp_path / "recipe.py")
    assert lint_source_quality(formatted, tmp_path / "recipe.py") == []
    before, after = {}, {}
    exec(source, before)  # noqa: S102
    exec(formatted, after)  # noqa: S102
    assert after["collect"](first, second) == before["collect"](first, second)


def test_repaired_source_lints_loads_and_executes(tmp_path):
    compiler = source_sequence(
        "", _STATIC_FAILURE_RECIPE, _STATIC_FAILURE_RECIPE, "", _REPAIR_RECIPE
    )
    activity = AgentActivity()
    source = calcify_source(
        _REPAIR_SKILL, compiler, activity=activity, source_path=tmp_path / "recipe.py"
    )
    assert activity.attempts == 5
    assert activity.max_model_calls == 8
    assert activity.usage["input_tokens"] == 15
    assert "## Rejected candidate: repair 4" in compiler.source_prompts[-1]
    assert activity.events[-2]["phase"] == "source.repair_4"
    skill = tmp_path / "SKILL.md"
    skill.write_text(_REPAIR_SKILL)
    destination = tmp_path / "recipe.py"
    destination.write_text(
        format_and_lint_source(
            attach_metadata(source, skill, recipe_path=destination),
            destination,
        )
    )
    assert lint_portable(destination) == []
    registry = Registry()
    worker = StubProvider(default="true")
    worker.name = "worker"
    registry.register(worker)
    run = load_recipe(destination).run(registry=registry)
    assert run.ok is True, run.error
    assert run.output is True


def test_malformed_repairs_use_remaining_global_capacity(tmp_path):
    compiler = source_sequence(
        "",
        _STATIC_FAILURE_RECIPE,
        _STATIC_FAILURE_RECIPE,
        "",
        "from pathlib import import Path",
        _REPAIR_RECIPE,
    )
    activity = AgentActivity()

    source = calcify_source(
        _REPAIR_SKILL,
        compiler,
        activity=activity,
        source_path=tmp_path / "recipe.py",
    )

    assert "def writing_style" in source
    assert activity.attempts == 6
    assert any(
        event["phase"] == "source.repair_5" and event["status"] == "accepted"
        for event in activity.events
    )
    assert "from pathlib import import Path" in compiler.source_prompts[-1]


def test_fake_compiler_converges_with_one_complete_repaired_draft(tmp_path):
    malformed = _REPAIR_RECIPE.replace(
        "def inspect(target: str) -> bool:", "def inspect(target: str):"
    )
    activity = AgentActivity()
    source = calcify_source(
        _REPAIR_SKILL,
        source_sequence(malformed, _REPAIR_RECIPE),
        agent_roles=("worker",),
        activity=activity,
        source_path=tmp_path / "recipe.py",
    )

    assert activity.attempts == 2
    assert "def inspect(target: str) -> bool:" in source


def test_wrapped_adjacent_literals_pass_the_destination_line_limit(tmp_path):
    (tmp_path / "ruff.toml").write_text('line-length = 100\n[lint]\nextend-select = ["E501"]\n')
    segments = ["review {target} and explain "] + ["every finding in plain language "] * 4
    wrapped = "(\n" + "".join(f'            "{segment}"\n' for segment in segments) + "        )"
    assert len("".join(segments)) > 100
    draft = _REPAIR_RECIPE.replace('"review {target}"', wrapped)
    compiler = source_sequence(draft)

    source = calcify_source(_REPAIR_SKILL, compiler, source_path=tmp_path / "recipe.py")

    # Re-rendering the draft would join the literals into one line past the limit.
    assert len(compiler.source_prompts) == 1
    assert max(len(line) for line in source.splitlines()) <= 100


@pytest.mark.parametrize("prefix", ["", "r"], ids=["ordinary", "raw"])
def test_destination_line_length_findings_do_not_require_repair(tmp_path, prefix):
    (tmp_path / "ruff.toml").write_bytes(b'line-length = 100\n\n[lint]\nselect = ["E501"]\n')
    prompt = "review {target} " + "and explain every finding in plain language " * 3
    long_line = _REPAIR_RECIPE.replace('"review {target}"', f'{prefix}"{prompt}"')
    compiler = source_sequence(long_line, _REPAIR_RECIPE)
    destination = tmp_path / "recipe.py"

    source = calcify_source(_REPAIR_SKILL, compiler, source_path=destination)

    assert len(compiler.source_prompts) == 1
    assert any(
        isinstance(node, ast.Constant) and node.value == prompt
        for node in ast.walk(ast.parse(source))
    )
    assert lint_source_quality(source, destination) == []


def test_source_repairs_stop_at_the_eight_call_ceiling(tmp_path):
    compiler = source_sequence(*_distinct_failures(8))
    activity = AgentActivity()
    with pytest.raises(CalcifyError, match="exhausted the global compiler-call budget"):
        calcify_source(
            _REPAIR_SKILL, compiler, activity=activity, source_path=tmp_path / "recipe.py"
        )
    assert activity.attempts == 8
    assert len(compiler.all_prompts) == 8
    assert [
        event["phase"] for event in activity.events if event["status"] == "request_started"
    ] == ["source.initial", *(f"source.repair_{number}" for number in range(1, 8))]
