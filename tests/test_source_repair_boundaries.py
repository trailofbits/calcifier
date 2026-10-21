"""Source-only compilation and malformed drafts preserve the compiler boundary."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from calcifier.calcify import calcify_source
from calcifier.recipe_draft import draft_response
from tests.support.compiler import ScriptedCompiler
from tests.support.recipes import _GOOD_RECIPE, _SCRIPT_HEADER
from tests.support.skill_source import SKILL


def test_compiler_phases_cannot_replace_pinned_source_configuration(monkeypatch):
    import calcifier.repair as compiler_module

    original = compiler_module._generate_source

    def inspect_context(context, *, prompt, filesystem):
        for field, replacement in [
            ("skill_text", "Different source instructions"),
            ("agent_roles", ("undeclared",)),
            ("runtime_wall_seconds", 999),
        ]:
            with pytest.raises(FrozenInstanceError):
                setattr(context, field, replacement)
        assert context.skill_text == SKILL
        assert context.agent_roles == ("worker",)
        assert context.runtime_wall_seconds == 30
        return original(context, prompt=prompt, filesystem=filesystem)

    monkeypatch.setattr(compiler_module, "_generate_source", inspect_context)
    provider = ScriptedCompiler([_GOOD_RECIPE], wrap_sources=True)
    calcify_source(SKILL, provider, runtime_wall_seconds=30)
    assert len(provider.all_prompts) == 1


def test_source_only_profile_compilation_uses_destination_without_claiming_package_provenance(
    tmp_path,
):
    destination = tmp_path / "recipe.py"
    provider = ScriptedCompiler([_GOOD_RECIPE], wrap_sources=True)
    source = calcify_source(SKILL, provider, source_path=destination)
    assert "_CALCIFIER_SKILL" not in source
    assert not destination.exists()
    assert len(provider.all_prompts) == 1


@pytest.mark.parametrize("whitespace", ["", " \n\t", "\r\n", "\v", "\u2003", "\x1c"])
def test_invalid_whitespace_draft_gets_a_diagnosed_bounded_repair(whitespace):
    rejected = draft_response(whitespace)
    provider = ScriptedCompiler([rejected, draft_response(_GOOD_RECIPE)])
    source = calcify_source(SKILL, provider)
    assert source == _SCRIPT_HEADER + _GOOD_RECIPE + "\n"
    assert len(provider.all_prompts) == 2
    prompt = provider.all_prompts[1]
    assert "recipe draft source is empty" in prompt
    raw = prompt.split("Rejected source (JSON string):\n", 1)[1].strip()
    assert json.loads(raw) == rejected
