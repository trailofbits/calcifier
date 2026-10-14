"""Offline regressions for compiler prompts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calcifier.calcify import (
    calcify_file,
)
from calcifier.prompts import (
    build_prompt,
)
from calcifier.registry import Registry
from tests.support.compiler import (
    one_draft,
)
from tests.support.recipes import (
    _GOOD_RECIPE,
)


def test_build_prompt_states_the_destination_line_length():
    prompt = build_prompt("# skill", agent_roles=("worker",), line_length=100)
    unknown = build_prompt("# skill", agent_roles=("worker",))

    assert "keep every line within 100 characters" in prompt
    assert "parenthesized adjacent literals" in prompt
    assert "within the target repository's Ruff line length" in unknown


def test_build_prompt_lists_configured_agent_roles():
    prompt = build_prompt("# skill", agent_roles=("worker", "verifier"))

    assert "Roles available for this generation are: `verifier`, `worker`" in prompt


def test_calcify_prompt_includes_capabilities_section_from_frontmatter():
    prompt = build_prompt("---\nallowed-tools:\n  - Read\n  - Grep\n---\n# skill\n")
    assert "## Declared tool capabilities" in prompt
    assert '["Read","Grep"]' in prompt


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_calcify_prompt_lists_packaged_resources_section(tmp_path: Path, newline: str):
    skill = tmp_path / "SKILL.md"
    skill.write_text("# my skill\nRead references/notes.md before processing.\n")
    (tmp_path / "references").mkdir()
    (tmp_path / "scripts").mkdir()
    contents = f"café notes{newline}"
    (tmp_path / "references" / "notes.md").write_bytes(contents.encode("utf-8"))
    (tmp_path / "scripts" / "run.sh").write_text("run\n")
    reg = Registry()
    recorder = one_draft(_GOOD_RECIPE)
    reg.register(recorder)

    calcify_file(skill, reg, compiler="compiler")

    prompt = recorder.all_prompts[0]
    assert "## Packaged resources" in prompt
    assert "references/notes.md" in prompt
    assert "scripts/run.sh" in prompt
    assert f'"references/notes.md": {json.dumps(contents)}' in prompt


def test_calcify_passes_through_skill_with_braces_in_text():
    """Skills frequently contain ``{{ }}``-shaped examples; the calcify
    prompt builds with str.format so unguarded braces would explode."""
    skill = "Use {{ inputs.x }} or {raw}".replace("{raw}", "{raw_braces}")
    # build_prompt should not raise; it must format only its own placeholders.
    prompt = build_prompt(skill)
    assert "{{ inputs.x }}" in prompt or "{ inputs.x }" in prompt
