"""Tests for ``calcifier update``: prepare the recipe again and diff it, writing nothing."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from calcifier.calcify import calcify_file
from calcifier.portable import lint_portable
from calcifier.registry import Registry
from calcifier.update import update_skill
from tests.support.agents import StubProvider
from tests.support.compiler import ScriptedCompiler, one_draft

_GOOD_RECIPE = textwrap.dedent(
    """
    from __future__ import annotations
    from calcifier import recipe, step

    @step
    def noop(target: str) -> bool:
        return bool(target)

    @recipe(version="0.1")
    def emitted(target: str = "x") -> bool:
        return noop(target=target)

    if __name__ == "__main__":
        raise SystemExit(emitted.cli())
    """
).strip()
# Passes every static check, then raises ValueError when the dry run calls the step.
_CRASHING_RECIPE = _GOOD_RECIPE.replace("return bool(target)", "return bool(int(target))")
# Names a packaged file the skill does not have.
_UNPACKAGED_RECIPE = _GOOD_RECIPE.replace(
    "from calcifier import recipe, step", "from calcifier import recipe, resource, step"
).replace("return bool(target)", 'return bool(resource("references/absent.md").read_text())')


def _registry(compiler: StubProvider | ScriptedCompiler) -> Registry:
    registry = Registry()
    registry.register(compiler)
    worker = StubProvider(default="true")
    worker.name = "worker"
    registry.register(worker)
    return registry


def _compiler(source: str, skill_text: str = "# skill body\n") -> ScriptedCompiler:
    return ScriptedCompiler([source], wrap_sources=True, repeat_last=True)


def _calcified_skill(tmp_path: Path) -> tuple[Path, Path]:
    skill = tmp_path / "skill" / "SKILL.md"
    skill.parent.mkdir()
    skill.write_text("# skill body\n")
    recipe = calcify_file(skill, _registry(_compiler(_GOOD_RECIPE)))
    return skill, recipe


def _tree(root: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
    }


def test_update_is_empty_when_the_prepared_recipe_matches(tmp_path: Path) -> None:
    skill, recipe = _calcified_skill(tmp_path)

    diff = update_skill(skill, existing_recipe=recipe, registry=_registry(_compiler(_GOOD_RECIPE)))

    assert diff == ""


def test_update_preserves_the_explicit_compiler_call_limit(tmp_path: Path) -> None:
    from calcifier.agent_events import AgentActivity

    skill, recipe = _calcified_skill(tmp_path)
    before = _tree(tmp_path)
    activity = AgentActivity()
    diff = update_skill(
        skill,
        existing_recipe=recipe,
        registry=_registry(_compiler(_GOOD_RECIPE)),
        activity=activity,
        max_compiler_calls=6,
    )
    assert diff == ""
    assert activity.max_model_calls == 6
    assert activity.attempts == 1
    assert _tree(tmp_path) == before


def test_update_diffs_a_changed_recipe_and_writes_nothing(tmp_path: Path) -> None:
    skill, recipe = _calcified_skill(tmp_path)
    before = _tree(tmp_path)
    renamed = _GOOD_RECIPE.replace("emitted", "renamed")

    diff = update_skill(skill, existing_recipe=recipe, registry=_registry(_compiler(renamed)))

    assert "-def emitted(" in diff and "+def renamed(" in diff
    assert diff.startswith(f"--- {recipe}\n+++ {recipe} (prepared)\n")
    assert _tree(tmp_path) == before


def test_update_shows_a_changed_skill_in_the_provenance_diff(tmp_path: Path) -> None:
    skill, recipe = _calcified_skill(tmp_path)
    skill.write_text("# skill body, edited\n")

    diff = update_skill(
        skill,
        existing_recipe=recipe,
        registry=_registry(_compiler(_GOOD_RECIPE, "# skill body, edited\n")),
    )

    changed = [line for line in diff.splitlines() if line[:1] in "+-" and '"sha256":' in line]
    assert len(changed) == 2


def test_portable_lint_reports_a_new_packaged_resource(tmp_path: Path) -> None:
    skill, recipe = _calcified_skill(tmp_path)
    references = skill.parent / "references"
    references.mkdir()
    (references / "notes.md").write_text("new\n")

    messages = [item.actual for item in lint_portable(recipe)]

    assert any(
        "references/notes.md" in message and "not in the recipe's manifest" in message
        for message in messages
    )


def test_update_lists_the_configured_roles_to_the_compiler(tmp_path: Path) -> None:
    skill, recipe = _calcified_skill(tmp_path)
    compiler = _compiler(_GOOD_RECIPE)

    update_skill(skill, existing_recipe=recipe, registry=_registry(compiler))

    assert "Roles available for this generation are: `worker`" in compiler.all_prompts[0]


def test_update_compiles_a_recipe_that_was_compiled_from_source(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("# skill body\n", encoding="utf-8")
    recipe = calcify_file(skill, _registry(one_draft(_GOOD_RECIPE)))
    before = _tree(tmp_path)
    compiler = _compiler(_GOOD_RECIPE)

    diff = update_skill(skill, existing_recipe=recipe, registry=_registry(compiler))

    assert diff == ""
    assert "RecipeDraft" in compiler.all_prompts[0]
    assert len(compiler.calls) == 1
    assert _tree(tmp_path) == before


def test_update_requires_an_existing_recipe(tmp_path: Path) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text("# body\n")

    with pytest.raises(FileNotFoundError, match="no existing recipe"):
        update_skill(
            skill,
            existing_recipe=tmp_path / "missing.py",
            registry=_registry(_compiler(_GOOD_RECIPE)),
        )


@pytest.mark.parametrize(
    ("draft", "fragment"),
    [(_CRASHING_RECIPE, "dry run failed"), (_UNPACKAGED_RECIPE, "not in the skill package")],
)
def test_a_rejected_update_prepares_a_baseline_without_changing_the_destination(
    tmp_path: Path, draft: str, fragment: str, caplog
) -> None:
    skill, recipe = _calcified_skill(tmp_path)
    before = _tree(tmp_path)

    compiler = ScriptedCompiler([draft], wrap_sources=True, repeat_last=True)
    diff = update_skill(skill, existing_recipe=recipe, registry=_registry(compiler))

    assert "Calcifier baseline" in diff
    assert fragment in caplog.text
    assert _tree(tmp_path) == before


@pytest.mark.parametrize(
    ("draft", "fragment"),
    [(_CRASHING_RECIPE, "dry run failed"), (_UNPACKAGED_RECIPE, "not in the skill package")],
)
def test_a_rejected_compile_publishes_only_the_baseline(
    tmp_path: Path, draft: str, fragment: str, caplog
) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# skill body\n")

    compiler = ScriptedCompiler([draft], wrap_sources=True, repeat_last=True)
    path = calcify_file(skill, _registry(compiler))

    assert "Calcifier baseline" in path.read_text(encoding="utf-8")
    assert fragment in caplog.text
    assert set(_tree(tmp_path)) == {"SKILL.md", ".calcifier", ".calcifier/recipe.py"}
