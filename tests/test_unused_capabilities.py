"""Capability declarations authorize calls; they do not oblige a recipe to make them."""

from __future__ import annotations

import ast
import json
import sys

import pytest

from calcifier import cli
from calcifier.diagnostics import CalcifyError
from calcifier.dry_run import dry_run_recipe
from calcifier.gate import verify_source
from calcifier.recipe import load_recipe
from calcifier.recipe_ast import tool_provider_capabilities

pytestmark = pytest.mark.semantic_offline

SOURCE = """from calcifier import recipe, step, write_text

@step
def save() -> str:
    return str(write_text("result.txt", "complete\\n"))

@recipe(capabilities=("Read", "Write", "Bash"), output_root="out")
def workflow() -> str:
    return save()

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""


@pytest.mark.parametrize("direct", [False, True])
def test_unused_authorizations_compile_and_execute_without_tool_configuration(
    tmp_path, monkeypatch, direct
):
    monkeypatch.chdir(tmp_path)
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"---\nallowed-tools: Read, Write, Bash\n---\nWrite the complete result.\n")
    path = tmp_path / "recipe.py"
    path.write_bytes(SOURCE.encode())
    verify_source(SOURCE, type_check=True)
    assert dry_run_recipe(SOURCE, recipe_path=path, skill_path=skill).status == "passed"
    recipe = load_recipe(path)
    assert recipe.capabilities == ("Read", "Write", "Bash")
    assert recipe._provider_capabilities == ()
    assert (recipe.cli([]) if direct else cli.main(["exec", str(path), "--trust"])) == 0
    assert (tmp_path / "out/result.txt").read_bytes() == b"complete\n"


def test_actual_tool_call_requires_only_its_provider(tmp_path, capsys):
    source = SOURCE.replace("step, write_text", "step, tool").replace(
        'str(write_text("result.txt", "complete\\n"))', 'str(tool("Read", path="input.txt"))'
    )
    verify_source(source)
    path = tmp_path / "recipe.py"
    path.write_bytes(source.encode())
    assert cli.main(["exec", str(path), "--trust"]) == 2
    error = capsys.readouterr().err
    assert "Read (configure a provider for Read)" in error
    assert "configure a provider for Write" not in error
    provider = "import json, sys; json.load(sys.stdin); print(json.dumps('read complete'))"
    config = tmp_path / "tools.json"
    config.write_bytes(
        json.dumps(
            {"version": 1, "capabilities": {"Read": {"command": [sys.executable, "-c", provider]}}}
        ).encode()
    )
    assert cli.main(["exec", str(path), "--tool-config", str(config)]) == 0
    assert "read complete" in capsys.readouterr().out


def test_actual_undeclared_tool_call_remains_rejected():
    source = SOURCE.replace("step, write_text", "step, tool").replace(
        'str(write_text("result.txt", "complete\\n"))', 'str(tool("Search", query="input"))'
    )
    with pytest.raises(CalcifyError, match="undeclared tool capability Search"):
        verify_source(source)


def test_tool_preapprovals_do_not_bound_unused_declarations():
    verify_source(SOURCE)


@pytest.mark.parametrize(
    "body",
    [
        'invoke = tool; return invoke("Read")',
        "return tool(name)",
        'return getattr(runtime, "tool")("Read")',
    ],
)
def test_indirect_dispatch_keeps_all_declared_provider_requirements(body):
    module = ast.parse(
        f"import calcifier as runtime\nfrom calcifier import tool\ndef helper(name):\n    {body}\n"
    )
    assert tool_provider_capabilities(module, ("Read", "Write")) == ("Read", "Write")
