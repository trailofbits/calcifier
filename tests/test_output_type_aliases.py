"""Static schema checking accepts named Python aliases without running initializers."""

from __future__ import annotations

import ast

import pytest

from calcifier.dry_run import dry_run_recipe
from calcifier.gate import verify_source
from calcifier.metadata import attach_metadata
from calcifier.output_annotations import output_annotation_error
from calcifier.portable import lint_portable
from calcifier.recipe import load_recipe
from calcifier.recipe_ast import import_bindings

pytestmark = pytest.mark.semantic_offline

SOURCE = """from __future__ import annotations
from typing import Literal, TypeAlias
from calcifier import output_type, recipe, step, write_text

Choice: TypeAlias = Literal["ready", "hold"]
Choices = list[Choice]

@output_type
class Result:
    choices: Choices

ResultAlias = Result

@step
def choose() -> ResultAlias:
    return Result(choices=["ready", "hold"])

@recipe(output_root="out")
def workflow() -> str:
    result = choose()
    return str(write_text("result.json", result.model_dump_json()))

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_named_literal_aliases_execute_the_complete_recipe(tmp_path, monkeypatch, newline):
    monkeypatch.chdir(tmp_path)
    source = SOURCE.replace("\n", newline)
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral selection\nPersist both supported choices.\n")
    path = tmp_path / "recipe.py"
    verify_source(source)
    dry = dry_run_recipe(source, recipe_path=path, skill_path=skill)
    assert dry.status == "passed", dry
    assert dry.writes == ("<workspace>/out/result.json",)
    path.write_bytes(attach_metadata(source, skill, recipe_path=path).encode())
    assert lint_portable(path) == []
    run = load_recipe(path).run()
    assert run.ok, run.error
    assert (tmp_path / "out/result.json").read_bytes() == b'{"choices":["ready","hold"]}'


def test_an_alias_can_name_an_output_record_and_a_forward_annotation():
    module = ast.parse(SOURCE)
    annotations = [
        ast.parse(text, mode="eval").body for text in ("Choice", "ResultAlias", '"Choices"')
    ]
    assert output_annotation_error(module, annotations, import_bindings(module)) is None


@pytest.mark.parametrize("value", ["Alias", "Alias | None", "object()"])
def test_invalid_or_cyclic_aliases_are_rejected_without_executing_them(value):
    module = ast.parse(f"Alias = {value}\n")
    error = output_annotation_error(module, [ast.Name(id="Alias")], {})
    assert error is not None
    assert "unsupported output annotation" in error


@pytest.mark.parametrize(
    "annotation,expression", [("frozenset[str]", 'frozenset({"ready"})'), ("object", '"ready"')]
)
def test_supported_builtin_step_annotations_run_through_the_gate_and_dry_run(
    tmp_path, annotation, expression
):
    source = f"""from calcifier import recipe, step
@step
def choose() -> {annotation}:
    return {expression}
@recipe
def workflow() -> {annotation}:
    return choose()
if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nChoose a value.\n")
    verify_source(source)
    result = dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill)
    assert result.status == "passed", result
