"""Suitability is an advisory compiler assessment, separate from code validation."""

from __future__ import annotations

import json

import pytest

import calcifier.artifact as artifact_module
import calcifier.cli as cli
from calcifier.agent_events import AgentActivity
from calcifier.artifact import prepare_artifact
from calcifier.compiler_activity import (
    CompilationLimitError,
    compiler_activity_payload,
    render_compiler_activity,
)
from calcifier.diagnostics import CalcifyError
from calcifier.recipe import load_recipe
from calcifier.recipe_draft import RecipeDraftError, decode_recipe_draft, draft_response
from calcifier.registry import Registry
from tests.support.agents import StubProvider
from tests.support.compiler import draft_sequence
from tests.support.recipes import _GOOD_RECIPE, _REPAIR_RECIPE

_GUIDANCE = (
    "# Collaboration principles\r\n"
    "Apply these principles throughout ordinary work.\r\n"
    "Ask about uncertain assumptions and explain material tradeoffs.\r\n"
    "This guidance supplies no fixed sequence or standalone deliverable.\r\n"
)
_NOTE = (
    "Ongoing behavioral guidance supplies no standalone procedure; "
    "interpretation remains agent-owned."
)


def payload(source, note):
    return json.dumps({"version": "1", "source": source, "suitability_note": note})


@pytest.mark.parametrize("note", [None, "", "  ", _NOTE])
def test_draft_preserves_an_optional_assessment_without_changing_source(note):
    decoded = decode_recipe_draft(payload("value = 1", note))
    assert decoded.source == "value = 1\n"
    assert decoded.suitability_note == (note.strip() or None if note is not None else None)
    assert decode_recipe_draft(draft_response("value = 1")).suitability_note is None


def test_advisory_assessment_cannot_bypass_syntax_checks():
    with pytest.raises(RecipeDraftError, match="invalid or incomplete"):
        decode_recipe_draft(payload("def broken(", _NOTE))


@pytest.mark.parametrize("note", [23, [], {}])
def test_assessment_has_a_strict_text_contract(note):
    with pytest.raises(RecipeDraftError):
        decode_recipe_draft(payload("value = 1", note))


@pytest.mark.parametrize("control", ["\x1b[2K\x1b[1A", "\x00", "\x85", "\x07"])
def test_assessment_cannot_carry_terminal_controls(control):
    with pytest.raises(RecipeDraftError, match="control characters"):
        decode_recipe_draft(payload("value = 1", "Review " + control + " this candidate."))


@pytest.mark.parametrize("baseline", [False, True])
def test_post_compile_failure_clears_the_assessment(tmp_path, monkeypatch, baseline):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Normalize\nReturn whether the provided target is nonempty.\n")
    registry = Registry()
    registry.register(draft_sequence(payload(_GOOD_RECIPE, _NOTE)))
    if baseline:
        worker = StubProvider(default="true")
        worker.name = "worker"
        registry.register(worker)
    activity = AgentActivity()
    compile_source = artifact_module.compile_source
    remaining = artifact_module.CompilerDeadline.remaining
    accepted = False

    def compile(context):
        nonlocal accepted
        result = compile_source(context)
        assert activity.suitability_note == _NOTE
        accepted = True
        if baseline:
            skill.write_bytes(b"# Changed during compilation\n")
        return result

    def expire(deadline):
        if accepted:
            raise CompilationLimitError("exhausted after source acceptance")
        return remaining(deadline)

    monkeypatch.setattr(artifact_module, "compile_source", compile)
    monkeypatch.setattr(artifact_module.CompilerDeadline, "remaining", expire)
    with pytest.raises(CalcifyError, match="changed|exhausted"):
        prepare_artifact(skill, registry, activity=activity)

    assert accepted
    assert activity.suitability_note is None
    assert "owner review required" not in render_compiler_activity(activity)
    assert compiler_activity_payload(activity)["suitability_note"] is None


def registry_for(*drafts):
    registry = Registry()
    compiler = draft_sequence(*drafts)
    registry.register(compiler)

    class Worker(StubProvider):
        name = "worker"

        def __init__(self):
            super().__init__(default="true")
            self.prompts = []

        def invoke(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return super().invoke(prompt, **kwargs)

    worker = Worker()
    registry.register(worker)
    return registry, compiler, worker


def test_advisory_source_is_preserved_and_assessment_survives_full_compilation(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(_GUIDANCE.encode("utf-8"))
    registry, compiler, worker = registry_for(payload(_REPAIR_RECIPE, _NOTE))
    activity = AgentActivity()

    artifact = prepare_artifact(skill, registry, activity=activity)

    assert artifact.outcome == "compiled"
    assert artifact.dry_run is not None and artifact.dry_run.status == "passed"
    assert artifact.suitability_note == activity.suitability_note == _NOTE
    assert "owner review required (compiler assessment)" in render_compiler_activity(activity)
    assert compiler_activity_payload(activity)["suitability_note"] == _NOTE
    assert "without inventing a procedure" in " ".join(compiler.all_prompts[0].split())
    artifact.path.parent.mkdir(parents=True)
    artifact.path.write_bytes(artifact.source.encode("utf-8"))

    result = load_recipe(artifact.path).run(registry=registry)

    assert result.ok
    assert len(worker.prompts) == 1
    assert _GUIDANCE in worker.prompts[0]
    assert skill.read_bytes() == _GUIDANCE.encode("utf-8")


def test_only_the_validated_final_candidate_supplies_an_assessment(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Normalize\nReturn whether the provided target is nonempty.\n")
    invalid = _GOOD_RECIPE.replace("return bool(target)", "return undefined_result")
    registry, compiler, _ = registry_for(payload(invalid, _NOTE), payload(_GOOD_RECIPE, None))
    activity = AgentActivity()

    artifact = prepare_artifact(skill, registry, activity=activity)

    assert artifact.outcome == "compiled"
    assert artifact.dry_run is not None and artifact.dry_run.status == "passed"
    assert len(compiler.calls) == 2
    assert artifact.suitability_note is activity.suitability_note is None
    assert "suitability:" not in render_compiler_activity(activity)
    assert not any(event.get("detail") == _NOTE for event in activity.events)


def test_cli_surfaces_owner_review_and_retains_assessment_in_trace(tmp_path, monkeypatch, capsys):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(_GUIDANCE.encode("utf-8"))
    trace = tmp_path / "trace.json"
    registry, _, _ = registry_for(payload(_REPAIR_RECIPE, _NOTE))
    monkeypatch.setattr(cli, "_registry_for_args", lambda *_: registry)

    assert cli.main(["calcify", str(skill), "--compiler-trace", str(trace)]) == 0

    output = capsys.readouterr().out
    assert "owner review required (compiler assessment)" in output
    assert _NOTE in output
    recorded = json.loads(trace.read_bytes())
    assert recorded["outcome"] == "compiled"
    assert recorded["suitability_note"] == _NOTE
    assert any(event.get("detail") == _NOTE for event in recorded["events"])
