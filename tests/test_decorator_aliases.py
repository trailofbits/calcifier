"""Imported decorator aliases retain all generated-source contracts."""

from __future__ import annotations

import pytest

from calcifier.artifact import render_summary
from calcifier.diagnostics import CalcifyError
from calcifier.dry_run import dry_run_recipe
from calcifier.gate import verify_source
from calcifier.metadata import attach_metadata
from calcifier.portable import lint_portable
from calcifier.prompts import _child_recipe_contract
from calcifier.recipe import load_recipe

pytestmark = pytest.mark.semantic_offline

SOURCE = """import calcifier as runtime

@runtime.output_type
class Result:
    text: str

@runtime.step
def save() -> Result:
    runtime.write_text("result.txt", "complete\\n")
    return Result(text="complete")

@runtime.recipe(version="0.2", capabilities=("Write",), output_root="out")
def workflow() -> Result:
    return save()

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""


@pytest.mark.parametrize("style", ["module", "imported"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_aliased_declarations_compile_run_and_describe_the_same_contract(
    tmp_path, monkeypatch, style, newline
):
    monkeypatch.chdir(tmp_path)
    source = SOURCE
    if style == "imported":
        source = source.replace(
            "import calcifier as runtime",
            "from calcifier import output_type as record, recipe as flow, "
            "step as traced, write_text",
        )
        for original, alias in [
            ("output_type", "record"),
            ("recipe", "flow"),
            ("step", "traced"),
            ("write_text", "write_text"),
        ]:
            source = source.replace("runtime." + original, alias)
    source = source.replace("\n", newline)
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"---\nallowed-tools: Write\n---\nPersist the complete result.\n")
    path = tmp_path / "recipe.py"
    verify_source(source)
    assert dry_run_recipe(source, recipe_path=path, skill_path=skill).status == "passed"
    path.write_bytes(attach_metadata(source, skill, recipe_path=path).encode())
    assert lint_portable(path) == []
    run = load_recipe(path).run()
    assert run.ok, run.error
    assert run.output is not None
    assert run.output.text == "complete"
    assert (tmp_path / "out/result.txt").read_bytes() == b"complete\n"
    assert "version: 0.2" in render_summary(path)
    contract = _child_recipe_contract(path)
    assert contract["capabilities"] == ("Write",)
    assert contract["output_types"] == {"Result": {"text": "str"}}


@pytest.mark.parametrize(
    "change, expected",
    [
        (lambda source: source.replace('output_root="out"', 'output_root=""'), "output_root"),
        (
            lambda source: source.replace('version="0.2"', 'version="0.2", agents=("codex",)'),
            "provider names",
        ),
        (
            lambda source: source.replace('version="0.2"', 'version="0.2", agents=("worker",)'),
            "requires Budget",
        ),
    ],
)
def test_aliased_recipe_preserves_declaration_checks(change, expected):
    with pytest.raises(CalcifyError, match=expected):
        verify_source(change(SOURCE))


def test_aliased_recipe_preapproved_tools_do_not_restrict_capabilities():
    verify_source(SOURCE.replace('capabilities=("Write",)', 'capabilities=("Bash",)'))


def test_decorator_from_another_module_is_not_a_recipe():
    source = SOURCE.replace("@runtime.recipe", "@record").replace(
        "import calcifier as runtime",
        "import calcifier as runtime\nfrom pathlib import Path as record",
    )
    with pytest.raises(CalcifyError, match="no @recipe"):
        verify_source(source, type_check=False)
