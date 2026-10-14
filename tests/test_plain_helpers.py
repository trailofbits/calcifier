"""Natural Python declarations retain effect and authorization checks without forced tracing."""

from __future__ import annotations

import pytest

from calcifier.diagnostics import CalcifyError
from calcifier.dry_run import dry_run_recipe
from calcifier.gate import verify_source
from calcifier.metadata import attach_metadata
from calcifier.portable import lint_portable
from calcifier.recipe import load_recipe

pytestmark = pytest.mark.semantic_offline

SOURCE = """import re
from calcifier import recipe, write_text

PATTERN: re.Pattern[str] = re.compile("input")
LABELS = tuple(value.upper() for value in ("first", "second"))

def choose() -> re.Pattern[str]:
    return PATTERN

def save(value: str) -> str:
    return str(write_text("result.txt", value))

@recipe(output_root="out")
def workflow() -> str:
    if choose().search("input") is None:
        raise ValueError("computed constant changed")
    return save("\\n".join(LABELS))

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_plain_helpers_and_computed_constants_execute_the_complete_recipe(
    tmp_path, monkeypatch, newline
):
    monkeypatch.chdir(tmp_path)
    source = SOURCE.replace("\n", newline)
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral result\nCompute and persist the complete selection.\n")
    path = tmp_path / "recipe.py"
    verify_source(source)
    dry = dry_run_recipe(source, recipe_path=path, skill_path=skill)
    assert dry.status == "passed", dry
    assert dry.writes == ("<workspace>/out/result.txt",)
    path.write_bytes(attach_metadata(source, skill, recipe_path=path).encode())
    assert lint_portable(path) == []
    run = load_recipe(path).run()
    assert run.ok, run.error
    assert (tmp_path / "out/result.txt").read_bytes() == b"FIRST\nSECOND"
    assert len(run.records) == 1
    assert run.records[0].side_effects[0].completed is True


@pytest.mark.parametrize("location", ["initializer", "helper"])
def test_effect_checks_reject_computed_assignments_and_helper_bodies_before_loading(
    tmp_path, location
):
    marker = tmp_path / "unexpected.txt"
    mutation = f"Path({str(marker)!r}).write_text('unexpected')"
    source = SOURCE.replace("import re\n", "import re\nfrom pathlib import Path\n", 1)
    if location == "initializer":
        source = source.replace("LABELS = ", f"UNEXPECTED = {mutation}\nLABELS = ")
    else:
        source = source.replace("    return PATTERN", f"    {mutation}\n    return PATTERN")
    with pytest.raises(CalcifyError, match="arbitrary generated"):
        verify_source(source)
    assert not marker.exists()


def test_plain_helper_tool_calls_still_require_a_declaration():
    source = SOURCE.replace("recipe, write_text", "recipe, tool, write_text").replace(
        "    return PATTERN", '    tool("Read", path="input.txt")\n    return PATTERN'
    )
    with pytest.raises(CalcifyError, match="undeclared tool capability Read"):
        verify_source(source)


def test_plain_helper_agent_calls_still_require_a_role_and_budget():
    source = SOURCE.replace("recipe, write_text", "agent, recipe, write_text").replace(
        "    return PATTERN",
        '    agent("worker", "Inspect.", working_directory=".", output=str)\n    return PATTERN',
    )
    with pytest.raises(CalcifyError, match="undeclared agent role"):
        verify_source(source)


def test_agent_data_cannot_be_hidden_in_a_helper_to_choose_a_tool_path():
    source = (
        SOURCE.replace("recipe, write_text", "Budget, agent, recipe, tool, write_text")
        .replace(
            '@recipe(output_root="out")',
            '@recipe(agents=("worker",), capabilities=("Read",), '
            'budget=Budget(2, 60), output_root="out")',
        )
        .replace('write_text("result.txt", value)', 'tool("Read", path=value)')
        .replace(
            'return save("\\n".join(LABELS))',
            'return save(agent("worker", "Choose.", working_directory=".", output=str))',
        )
    )
    with pytest.raises(CalcifyError, match="model-derived"):
        verify_source(source)
