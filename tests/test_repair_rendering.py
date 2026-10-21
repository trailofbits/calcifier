"""Known-answer repair requests and validation telemetry."""

from __future__ import annotations

import json

import pytest

from calcifier.agent_events import AgentActivity
from calcifier.calcify import CalcifyError, calcify_source
from calcifier.recipe_draft import draft_response as _draft
from tests.support.ast import source_line
from tests.support.compiler import ScriptedCompiler as _ScriptedCompiler
from tests.support.recipes import _GOOD, _SCRIPT_HEADER, _SKILL, _UNANNOTATED_AND_WRITING, _WRITING


def _section(prompt: str, heading: str) -> object:
    """Parse the JSON section following one repair heading."""
    lines = prompt.splitlines()
    return json.loads(lines[lines.index(heading) + 1])


def _phases(activity: AgentActivity, status: str) -> list[object]:
    return [event["phase"] for event in activity.events if event["status"] == status]


# -- bounded source repair: exact prompt sections and error payloads ----------

# The accepted program stays byte-exact after the compiler-owned runner header.
_RENDERED_GOOD = _SCRIPT_HEADER + _GOOD
_REPAIR_HEADING = "## Rejected candidate: repair "
_ERROR_HEADING = "Validation error (JSON string):"
_DIAGNOSTICS_HEADING = "Structured diagnostics (JSON array):"
_REJECTED_HEADING = "Rejected source (JSON string):"


_ANNOTATION_LINE = source_line(_SCRIPT_HEADER + _UNANNOTATED_AND_WRITING, "def count_entries")
_EFFECT_LINE = source_line(_SCRIPT_HEADER + _UNANNOTATED_AND_WRITING, 'open("x.txt"')
_ANNOTATION_ERROR = (
    f"line {_ANNOTATION_LINE}: @step 'count_entries' requires an explicit return annotation"
)
_EFFECT_ERROR = f"line {_EFFECT_LINE}: arbitrary generated effect"
_BAD_DRAFT = "invalid RecipeDraft: invalid or incomplete recipe draft: "
_OBJECT_KEY_ERROR = "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"


def _diagnostic(code: str, node: str | None, line: int | None, actual: str) -> dict[str, object]:
    """One structured diagnostic, keyed in Diagnostic field order."""
    expected = {
        "step_annotations": "declare a loadable explicit return annotation",
    }.get(code, f"satisfy the deterministic {code} contract")
    fix = (
        "replace only the named step function annotation"
        if code == "step_annotations"
        else "repair only this contract and preserve all validated structure"
    )
    return {
        "code": code,
        "phase": "source.validation",
        "node": node,
        "line": line,
        "actual": actual.removeprefix(f"line {line}: "),
        "expected": expected,
        "safe_fix": fix,
    }


_ANNOTATION = _diagnostic("step_annotations", "count_entries", _ANNOTATION_LINE, _ANNOTATION_ERROR)
_EFFECT = _diagnostic("effect_contracts", "tally", _EFFECT_LINE, _EFFECT_ERROR)


def test_repair_requests_carry_exact_json_sections() -> None:
    compiler = _ScriptedCompiler(
        [_draft(_UNANNOTATED_AND_WRITING), _draft(_WRITING), _draft(_GOOD)]
    )
    activity = AgentActivity()

    source = calcify_source(_SKILL, compiler, agent_roles=(), activity=activity)

    assert source == _RENDERED_GOOD
    assert _phases(activity, "request_started") == [
        "source.initial",
        "source.repair_1",
        "source.repair_2",
    ]
    (initial,) = compiler.prompts("source")
    first, second = compiler.prompts("repair")
    assert first.startswith(initial)
    assert f"{_REPAIR_HEADING}1" in first.splitlines()
    assert _section(first, _ERROR_HEADING) == f"{_ANNOTATION_ERROR}; {_EFFECT_ERROR}"
    assert _section(first, _DIAGNOSTICS_HEADING) == [_ANNOTATION, _EFFECT]
    assert _section(first, _REJECTED_HEADING) == _SCRIPT_HEADER + _UNANNOTATED_AND_WRITING
    # The second request describes only the newest candidate, never earlier attempts.
    assert second.startswith(initial)
    assert f"{_REPAIR_HEADING}2" in second.splitlines()
    assert _section(second, _ERROR_HEADING) == _EFFECT_ERROR
    assert _section(second, _DIAGNOSTICS_HEADING) == [_EFFECT]
    rejected = _section(second, _REJECTED_HEADING)
    assert rejected == _SCRIPT_HEADER + _WRITING
    assert _ANNOTATION_ERROR not in second.removeprefix(initial)


def test_an_unparseable_candidate_is_shown_as_its_raw_response() -> None:
    compiler = _ScriptedCompiler([_draft(_WRITING), "{", "[", _draft(_GOOD)])
    activity = AgentActivity()

    source = calcify_source(_SKILL, compiler, agent_roles=(), activity=activity)

    assert source == _RENDERED_GOOD
    assert _phases(activity, "accepted") == ["source.repair_3", "compiler"]
    first, second, third = compiler.prompts("repair")
    assert _section(first, _ERROR_HEADING) == _EFFECT_ERROR
    assert _section(first, _DIAGNOSTICS_HEADING) == [_EFFECT]
    assert _section(first, _REJECTED_HEADING) == _SCRIPT_HEADER + _WRITING
    assert _section(second, _ERROR_HEADING) == _BAD_DRAFT + _OBJECT_KEY_ERROR
    assert _section(second, _DIAGNOSTICS_HEADING) == []
    assert _section(second, _REJECTED_HEADING) == "{"
    assert _section(third, _REJECTED_HEADING) == "["


def test_a_failure_repeated_three_times_stops_and_reports_the_latest_error() -> None:
    compiler = _ScriptedCompiler([_draft(_WRITING)] * 7)
    activity = AgentActivity()

    with pytest.raises(CalcifyError) as caught:
        calcify_source(_SKILL, compiler, agent_roles=(), activity=activity)

    assert str(caught.value) == (
        "source repair made no progress: the same diagnostics came back 3 times in a row, "
        "so a compiler transformation or a check the recipe cannot satisfy is the likely "
        f"cause; latest validation error: {_EFFECT_ERROR}"
    )
    assert [item.code for item in caught.value.diagnostics] == ["effect_contracts"]
    assert caught.value.draft == _draft(_WRITING)
    assert _phases(activity, "request_started") == [
        "source.initial",
        "source.repair_1",
        "source.repair_2",
    ]


def test_source_repairs_stop_exactly_at_the_global_call_ceiling() -> None:
    # Three kinds of failure in turn, so only the call ceiling stops the repairs.
    compiler = _ScriptedCompiler(["{", "[", _draft(_WRITING)] * 2 + ["{", "["])
    activity = AgentActivity()

    with pytest.raises(CalcifyError) as caught:
        calcify_source(_SKILL, compiler, agent_roles=(), activity=activity)

    assert str(caught.value) == (
        "source generation exhausted the global compiler-call budget; "
        f"latest validation error: {_BAD_DRAFT}Expecting value: line 1 column 2 (char 1)"
    )
    assert len(compiler.calls) == 8
    assert _phases(activity, "request_started") == [
        "source.initial",
        *(f"source.repair_{number}" for number in range(1, 8)),
    ]


# -- draft recovery and validation telemetry ---------------------------------


def test_uniformly_indented_draft_is_recovered_without_a_draft() -> None:
    indented = "".join(f"    {line}" if line.strip() else line for line in _GOOD.splitlines(True))
    compiler = _ScriptedCompiler([_draft(indented)])

    assert calcify_source(_SKILL, compiler, agent_roles=()) == _RENDERED_GOOD
    assert [kind for kind, _ in compiler.calls] == ["source"]


def test_mixed_tab_indentation_is_repaired_not_silently_recovered() -> None:
    tabbed = _GOOD.replace('    """Zählt die Einträge."""\n', '\t"""Zählt die Einträge."""\n')
    compiler = _ScriptedCompiler([_draft(tabbed), _draft(_GOOD)])
    activity = AgentActivity()

    assert calcify_source(_SKILL, compiler, agent_roles=(), activity=activity) == _RENDERED_GOOD
    assert [kind for kind, _ in compiler.calls] == ["source", "repair"]
    first_failure = next(
        event for event in activity.events if event["status"] == "validation_failed"
    )
    assert json.loads(str(first_failure["detail"]))["diagnostic"] == (
        "invalid RecipeDraft: invalid or incomplete recipe draft: unindent does not match any "
        "outer indentation level (<unknown>, line 7)"
    )


_VALIDATION_TIERS = [
    "step_annotations",
    "output_annotations",
    "agent_declarations",
    "tool_capabilities",
    "resource_usage",
    "model_derived_values",
    "subrecipe_calls",
    "sync_primitives",
    "async_steps",
    "effect_contracts",
    "configured_roles",
    "runtime_conformance",
]


def _expected_metrics(ast_nodes: int, failed: set[str]) -> dict[str, object]:
    return {
        "ast_nodes": ast_nodes,
        "check_count": len(_VALIDATION_TIERS),
        "outcome": "failed" if failed else "accepted",
        "tiers": [
            {
                "diagnostic_count": int(name in failed),
                "name": name,
                "status": "failed" if name in failed else "accepted",
            }
            for name in _VALIDATION_TIERS
        ],
    }


def test_validation_metrics_report_every_tier_with_sub_minute_durations() -> None:
    compiler = _ScriptedCompiler(
        [_draft(_UNANNOTATED_AND_WRITING), _draft(_WRITING), _draft(_GOOD)]
    )
    activity = AgentActivity()

    calcify_source(_SKILL, compiler, agent_roles=(), activity=activity)

    metrics = [
        json.loads(str(event["detail"]))
        for event in activity.events
        if event["status"] == "metrics"
    ]
    durations = [tier.pop("wall_seconds") for item in metrics for tier in item["tiers"]]
    assert metrics == [
        _expected_metrics(60, {"step_annotations", "effect_contracts"}),
        _expected_metrics(62, {"effect_contracts"}),
        _expected_metrics(56, set()),
    ]
    assert len(durations) == 3 * len(_VALIDATION_TIERS)
    assert all(0 <= duration < 60 for duration in durations)
