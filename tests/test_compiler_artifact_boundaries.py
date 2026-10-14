"""Compiler artifacts preserve metadata, configured limits and publication results."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError

import pytest

from calcifier.artifact import (
    _configured_runtime_wall_seconds,
    calcify_file,
    prepare_artifact,
    render_summary,
)
from calcifier.metadata import metadata_assignment, metadata_literal
from calcifier.registry import Registry
from tests.support.compiler import ScriptedCompiler
from tests.support.recipes import _GOOD_RECIPE


def test_metadata_round_trip_preserves_empty_nested_values_and_original_text():
    value = {
        "empty": {},
        "sequence": [(), [], {"flag": False, "count": 0, "unset": None}],
        "source": "café\r\nC:\\notes\\report.txt\n" * 12,
    }
    statement = ast.parse(metadata_assignment("METADATA", value)).body[0]
    assert isinstance(statement, ast.Assign)
    assert ast.literal_eval(statement.value) == value
    long_text = "source-reference-" * 300
    rendered = metadata_literal(long_text)
    assert ast.literal_eval(rendered) == long_text
    assert max(map(len, rendered.splitlines())) <= 100
    assert len(rendered) < 2 * len(long_text)


@pytest.mark.parametrize(
    ("configured", "override", "expected"),
    [
        (None, None, None),
        (0, None, None),
        (-1, None, None),
        (float("inf"), None, None),
        (float("nan"), None, None),
        (True, None, None),
        (0.25, None, 1),
        (1.25, None, 2),
        (20, 7, 7),
        (20, 0, None),
    ],
)
def test_runtime_limit_uses_only_finite_positive_values_and_explicit_override(
    configured, override, expected
):
    provider = ScriptedCompiler([])
    provider.default_timeout = configured
    assert _configured_runtime_wall_seconds(provider, override) == expected


def test_prepared_artifact_stays_unpublished_and_immutable_until_written(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("# Skill\nRead [notes](references/notes.md).\n", encoding="utf-8")
    reference = tmp_path / "references" / "notes.md"
    reference.parent.mkdir()
    reference.write_text("Retain uncertainty.\n", encoding="utf-8")
    destination = tmp_path / "generated" / "recipe.py"
    compiler = ScriptedCompiler([_GOOD_RECIPE], wrap_sources=True, repeat_last=True)
    registry = Registry()
    registry.register(compiler)
    prepared = prepare_artifact(
        skill, registry, output=destination, timeout=80, runtime_wall_seconds=123
    )
    assert not destination.exists()
    assert "use 123 seconds" in compiler.all_prompts[0]
    assert "references/notes.md" in compiler.all_prompts[0]
    assert "Retain uncertainty." in compiler.all_prompts[0]
    for field, changed in (("source", "invalid replacement"), ("path", tmp_path / "other.py")):
        with pytest.raises(FrozenInstanceError):
            setattr(prepared, field, changed)
    written = calcify_file(skill, registry, output=destination)
    assert written == destination
    assert destination.read_text(encoding="utf-8") == prepared.source
    summary = render_summary(destination)
    assert "name: emitted" in summary and "budget: none declared" in summary


def test_static_summary_keeps_unreadable_and_invalid_recipe_diagnostics(tmp_path):
    path = tmp_path / "absent.py"
    assert "could not statically summarise recipe" in render_summary(path)
    path.write_text("not python (", encoding="utf-8")
    assert "could not statically summarise recipe" in render_summary(path)


def test_successful_candidate_checks_final_bytes_once(tmp_path, monkeypatch):
    import calcifier.gate as gate_module
    import calcifier.repair as repair_module
    from calcifier.agent_events import AgentActivity

    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nCheck an input.\n")
    registry = Registry()
    registry.register(ScriptedCompiler([_GOOD_RECIPE], wrap_sources=True))
    checked = []
    typed = []
    original_gate = repair_module.verify_source
    original_ty = gate_module.type_check_source

    def gate(source, **kwargs):
        checked.append(source)
        return original_gate(source, **kwargs)

    def typecheck(source, **kwargs):
        typed.append(source)
        return original_ty(source, **kwargs)

    monkeypatch.setattr(repair_module, "verify_source", gate)
    monkeypatch.setattr(gate_module, "type_check_source", typecheck)
    activity = AgentActivity()
    prepared = prepare_artifact(skill, registry, activity=activity)
    assert prepared.outcome == "compiled"
    assert prepared.dry_run is not None and prepared.dry_run.status == "passed"
    assert activity.dry_run is prepared.dry_run
    assert checked == typed == [prepared.source]
    assert "_CALCIFIER_SKILL" in prepared.source
