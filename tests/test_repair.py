"""Offline regressions for repair."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import cast

import pytest

from calcifier.agent_events import AgentActivity
from calcifier.calcify import (
    CalcifyError,
    calcify_file,
    calcify_source,
    render_compiler_activity,
    render_dry_run,
)
from calcifier.diagnostics import Diagnostic
from calcifier.registry import Registry
from calcifier.repair import (
    _failure_key,
    _normalize_uniformly_indented_recipe_draft,
)
from calcifier.type_check import type_check_source
from tests.support.agents import StubProvider
from tests.support.ast import assert_same_program as _assert_same_program
from tests.support.compiler import (
    draft_sequence,
    one_draft,
    source_sequence,
)
from tests.support.compiler_cases import (
    _BASH_TOOL_RECIPE,
    _COMMAND_RECIPE,
    _compile_file_drafts,
    _skill_allowing,
    _TwelveAndAHalfSecondTimer,
)
from tests.support.recipes import (
    _GOOD_RECIPE,
    _REPAIR_RECIPE,
    _REPAIR_SKILL,
    _STATIC_FAILURE_RECIPE,
)


@pytest.mark.parametrize(
    "rejected",
    [
        "",
        '{"version":"1","source":',
        json.dumps({"version": "1", "source": ""}),
        '{"version":"1","module":{"kind":"Module","fields":[]}}',
        json.dumps({"version": "1", "source": "def broken(:"}),
        json.dumps({"version": "1", "source": "import subprocess\n" + _REPAIR_RECIPE}),
    ],
)
def test_compact_draft_failure_is_repairable_and_cleans_up(rejected: str) -> None:
    compiler = draft_sequence(rejected, json.dumps({"version": "1", "source": _REPAIR_RECIPE}))
    generated = calcify_source(_REPAIR_SKILL, compiler, agent_roles=("worker",))
    _assert_same_program(generated, _REPAIR_RECIPE)
    assert len(compiler.source_prompts) == 2
    assert all(not directory.exists() for directory in compiler.compiler_directories)


def test_draft_repair_provider_failure_cleans_up_without_further_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calcifier.filesystem_contract import AgentError

    compiler = draft_sequence("", "unused")
    invoke = compiler.invoke

    def bounded_failure(*args, **kwargs):  # noqa: ANN002, ANN003
        if len(compiler.all_prompts) == 1:
            raise AgentError("native turn timed out")
        return invoke(*args, **kwargs)

    monkeypatch.setattr(compiler, "invoke", bounded_failure)
    with pytest.raises(
        CalcifyError, match="agent invocation failed: native turn timed out"
    ) as caught:
        calcify_source(_REPAIR_SKILL, compiler, agent_roles=("worker",))
    assert len(compiler.source_prompts) == 1
    assert caught.value.activity is not None
    assert caught.value.activity.attempts == 2
    assert all(not directory.exists() for directory in compiler.compiler_directories)


def test_uniformly_indented_recipe_draft_is_recovered_but_mixed_indent_is_not() -> None:
    shifted = (
        "    from calcifier import step\n\n    @step\n    def alpha() -> str:\n        return 'x'\n"
    )
    payload = json.dumps({"version": "1", "source": shifted})
    mixed = json.dumps({"version": "1", "source": shifted + "value = 1\n"})

    recovered = json.loads(
        _normalize_uniformly_indented_recipe_draft(payload, python_version="3.13")
    )

    assert recovered["source"].startswith("from calcifier import step")
    ast.parse(recovered["source"])
    assert _normalize_uniformly_indented_recipe_draft(mixed, python_version="3.13") == mixed


def test_calcify_repairs_the_newest_candidate_and_reports_telemetry(monkeypatch):
    rejected = "def broken(—):\n    pass"
    compiler = source_sequence(
        rejected,
        _STATIC_FAILURE_RECIPE,
        _STATIC_FAILURE_RECIPE,
        _REPAIR_RECIPE,
    )
    compiler.model = "gpt-5.4"
    compiler.reasoning_effort = "xhigh"
    activity = AgentActivity()
    # Fake only the activity timer: the global clock also drives subprocess timeouts.
    monkeypatch.setattr("calcifier.agent_events.SuspendAwareTimer", _TwelveAndAHalfSecondTimer)

    generated = calcify_source(
        _REPAIR_SKILL,
        compiler,
        agent_roles=("worker",),
        activity=activity,
    )

    _assert_same_program(generated, _REPAIR_RECIPE)
    assert len(compiler.source_prompts) == 4
    assert "## Rejected candidate: repair 3" in compiler.source_prompts[3]
    assert "arbitrary generated write" in compiler.source_prompts[3]
    assert compiler.responses == []
    assert activity.attempts == 4
    assert activity.max_model_calls == 8
    assert activity.usage["input_tokens"] == 10
    assert activity.usage["output_tokens"] == 4
    assert activity.duration_seconds == 12.5
    assert activity.model == "gpt-5.4"
    assert activity.reasoning_effort == "xhigh"
    telemetry = render_compiler_activity(activity)
    assert 'model="gpt-5.4"' in telemetry
    assert 'reasoning_effort="xhigh"' in telemetry


def test_the_same_failure_three_times_in_a_row_stops_the_repairs() -> None:
    compiler = source_sequence(*([_STATIC_FAILURE_RECIPE] * 3), _REPAIR_RECIPE)

    with pytest.raises(CalcifyError, match="made no progress") as caught:
        calcify_source(_REPAIR_SKILL, compiler, agent_roles=("worker",))

    assert len(compiler.source_prompts) == 3
    assert "arbitrary generated write" in str(caught.value)
    assert caught.value.draft is not None
    assert json.loads(caught.value.draft)["source"] == _STATIC_FAILURE_RECIPE
    assert caught.value.source is not None
    assert 'Path("report.md")' in caught.value.source


def test_three_different_parse_failures_do_not_stop_the_repairs() -> None:
    broken = ("x = (1,\n", "if True\n    pass\n", "value = = 1\n")
    compiler = source_sequence(*broken, _REPAIR_RECIPE)

    generated = calcify_source(_REPAIR_SKILL, compiler, agent_roles=("worker",))

    _assert_same_program(generated, _REPAIR_RECIPE)
    assert len(compiler.source_prompts) == 4


def test_failure_key_ignores_line_numbers_but_not_diagnostic_counts() -> None:
    def diagnostic(line: int) -> Diagnostic:
        return Diagnostic("ruff", "source", "f", line, f"E501 at line {line}", "", "")

    once = CalcifyError("x", diagnostics=(diagnostic(3),))
    moved = CalcifyError("x", diagnostics=(diagnostic(9),))
    twice = CalcifyError("x", diagnostics=(diagnostic(3), diagnostic(4)))

    assert _failure_key(once) == _failure_key(moved)
    assert _failure_key(once) != _failure_key(twice)
    assert _failure_key(CalcifyError("bad at line 2")) == _failure_key(
        CalcifyError("bad at line 5")
    )


@pytest.mark.parametrize("source", [_COMMAND_RECIPE, _BASH_TOOL_RECIPE], ids=["command", "tool"])
@pytest.mark.parametrize("tools", [None, "Read", "Bash(git *)"])
def test_tool_preapprovals_compile_without_a_repair(source: str, tools: str | None) -> None:
    compiler = source_sequence(source, _GOOD_RECIPE)

    skill = "# skill\n" if tools is None else _skill_allowing(tools)
    generated = calcify_source(skill, compiler, agent_roles=("worker",))

    _assert_same_program(generated, source)
    assert len(compiler.source_prompts) == 1


def test_calcify_keeps_repairing_while_the_call_budget_has_capacity():
    rejected = "def broken(—):\n    pass"
    source_after_hard_cap = _REPAIR_RECIPE
    compiler = source_sequence(
        rejected,
        _STATIC_FAILURE_RECIPE,
        _STATIC_FAILURE_RECIPE,
        rejected,
        rejected,
        source_after_hard_cap,
    )

    generated = calcify_source(_REPAIR_SKILL, compiler, agent_roles=("worker",))

    _assert_same_program(generated, _REPAIR_RECIPE)
    assert len(compiler.source_prompts) == 6
    assert compiler.responses == []


def test_calcify_usage_aggregation_includes_source_repair_call():
    rejected = "def broken(—):\n    pass"
    compiler = source_sequence(rejected, _REPAIR_RECIPE)
    activity = AgentActivity()

    calcify_source(
        _REPAIR_SKILL,
        compiler,
        agent_roles=("worker",),
        activity=activity,
    )

    assert activity.attempts == 2  # rejected source, repaired source
    assert activity.usage["input_tokens"] == 3
    assert activity.usage["output_tokens"] == 2
    assert activity.duration_seconds > 0


def test_draft_with_an_invalid_escape_sequence_is_repaired():
    warned = _REPAIR_RECIPE.replace(
        "def inspect(target: str) -> bool:\n",
        'def inspect(target: str) -> bool:\n    pattern = "\\("\n',
    )
    compiler = source_sequence(warned, _REPAIR_RECIPE)

    generated = calcify_source(_REPAIR_SKILL, compiler, agent_roles=("worker",))

    _assert_same_program(generated, _REPAIR_RECIPE)
    assert len(compiler.source_prompts) == 2
    assert "invalid escape sequence" in compiler.source_prompts[1]


def test_calcify_repairs_missing_step_return_annotation():
    missing = _REPAIR_RECIPE.replace(
        "def inspect(target: str) -> bool:",
        "def inspect(target: str):",
    )
    compiler = source_sequence(missing, _REPAIR_RECIPE)

    generated = calcify_source(_REPAIR_SKILL, compiler, agent_roles=("worker",))

    _assert_same_program(generated, _REPAIR_RECIPE)
    assert len(compiler.source_prompts) == 2
    assert "requires an explicit return annotation" in compiler.source_prompts[1]


def test_an_object_attribute_finding_does_not_cost_a_repair(tmp_path: Path) -> None:
    helper = "def mapping_value(value: object) -> str:\n    return value.get('target', '')\n\n\n"
    source = _REPAIR_RECIPE.replace("@recipe(", helper + "@recipe(", 1).replace(
        "return inspect(target=target)",
        'return inspect(target=mapping_value({"target": target}))',
    )
    assert type_check_source(source) == ()
    activity = AgentActivity()

    compiler, generated = _compile_file_drafts(tmp_path, source, activity=activity)

    assert len(compiler.source_prompts) == 1
    assert "def mapping_value" in generated
    assert render_dry_run(activity).startswith("dry run: passed")


def test_calcify_source_wraps_agent_error():
    class _Boom(StubProvider):
        name = "compiler"

        def invoke(self, prompt, **kwargs):  # noqa: ANN001, ANN003
            del prompt, kwargs
            from calcifier.filesystem_contract import AgentError

            raise AgentError("subprocess died")

    provider = _Boom()
    with pytest.raises(CalcifyError, match="agent invocation failed"):
        calcify_source("# skill", provider)


def test_calcify_source_deterministically_renders_the_draft():
    """The runtime renders equivalent code rather than returning model text."""

    rec = one_draft(_GOOD_RECIPE)
    out = calcify_source("# skill body", rec)
    _assert_same_program(out, _GOOD_RECIPE)
    assert "# skill body" in rec.all_prompts[0]


def test_successful_validation_telemetry_contains_only_bounded_metrics() -> None:
    compiler = source_sequence(_REPAIR_RECIPE)
    activity = AgentActivity()

    calcify_source(
        _REPAIR_SKILL,
        compiler,
        agent_roles=("worker",),
        activity=activity,
    )

    metrics = [event for event in activity.events if event["phase"] == "source.validation"]
    assert len(metrics) == 1
    assert metrics[0]["status"] == "metrics"
    detail_text = cast("str", metrics[0]["detail"])
    detail = json.loads(detail_text)
    assert detail["ast_nodes"] > 0
    assert detail["check_count"] == len(detail["tiers"])
    assert detail["outcome"] == "accepted"
    assert all(tier["status"] == "accepted" for tier in detail["tiers"])
    assert "prompt" not in detail_text.lower()
    assert _REPAIR_SKILL not in detail_text
    assert _REPAIR_RECIPE not in detail_text


@pytest.mark.parametrize("limit", [1, 2, 5])
def test_fixed_command_failure_cannot_be_accepted_after_unchanged_repair(tmp_path, limit):
    command = [sys.executable, "-c", "print(1); raise SystemExit(7)"]
    source = f"""from calcifier import recipe, run_command, step

@step
def inspect() -> int:
    return run_command({command!r}, timeout_seconds=10, mutating=False, output=int)

@recipe(capabilities=("Bash",))
def workflow() -> int:
    return inspect()

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    compiler = source_sequence(source, repeat_last=True)
    registry = Registry()
    registry.register(compiler)
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Inspect\nReturn the command result.\n")
    output = tmp_path / "recipe.py"
    with pytest.raises(CalcifyError, match="command exited 7") as caught:
        calcify_file(skill, registry, output=output, max_compiler_calls=limit)
    assert compiler.source_calls == min(limit, 3)
    assert not output.exists()
    assert caught.value.diagnostics[0].code == "dry_run"
