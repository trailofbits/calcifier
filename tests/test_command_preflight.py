"""Bash authorization for host commands is distinct from external tool configuration."""

from __future__ import annotations

import ast
import json
import sys

import pytest

from calcifier import cli
from calcifier.graph import inspect_graph
from calcifier.metadata import attach_metadata
from calcifier.recipe import load_recipe
from calcifier.recipe_ast import tool_provider_capabilities

pytestmark = pytest.mark.semantic_offline


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    "imports, command",
    [
        ("from calcifier import run_command as command", "command"),
        ("import calcifier as runtime", "runtime.run_command"),
    ],
)
def test_host_commands_execute_through_both_cli_entries_without_a_bash_provider(
    tmp_path, capsys, direct, imports, command
):
    command_code = "import sys; sys.stdout.buffer.write(b'command ran\\n')"
    source = f"""import sys
from calcifier import recipe, step
{imports}

@step
def inspect() -> str:
    return {command}([sys.executable, "-c", {command_code!r}], timeout_seconds=10).stdout

@recipe(capabilities=("Bash",))
def workflow() -> str:
    return inspect()

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    path = tmp_path / "recipe.py"
    path.write_bytes(source.encode())
    if direct:
        assert load_recipe(path).cli([]) == 0
    else:
        trace = tmp_path / "trace.json"
        assert cli.main(["exec", str(path), "--trust", "--trace-out", str(trace)]) == 0
        payload = json.loads(trace.read_bytes())
        assert payload["output"] == "command ran\n"
        assert payload["tool_preflight"]["missing"] == []
        assert payload["steps"][0]["side_effects"][0]["completed"] is True
    output = capsys.readouterr()
    assert "command ran" in output.out
    assert "missing capabilities" not in output.err


@pytest.mark.parametrize("dynamic", [False, True])
def test_explicit_bash_tool_still_requires_its_provider(tmp_path, capsys, dynamic):
    argument = "name" if dynamic else '"Bash"'
    source = f"""import sys
from calcifier import recipe, run_command, tool

@recipe(capabilities=("Bash",))
def workflow(name: str = "Bash") -> str:
    run_command([sys.executable, "-V"], timeout_seconds=10, mutating=False)
    return str(tool({argument}, command="echo ready"))
"""
    path = tmp_path / "recipe.py"
    path.write_bytes(source.encode())
    assert cli.main(["exec", str(path), "--trust"]) == 2
    assert "Bash (configure a provider for Bash)" in capsys.readouterr().err


def test_source_without_tool_calls_does_not_require_unused_providers():
    module = ast.parse("from calcifier import recipe\n@recipe\ndef workflow():\n    return True\n")
    assert tool_provider_capabilities(module, ("Bash", "Read")) == ()


@pytest.mark.parametrize("external_bash", [False, True])
def test_graph_preflight_combines_host_and_external_bash_requirements(
    tmp_path, capsys, external_bash
):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nRun the command.\n")
    child = tmp_path / "child.py"
    body = (
        '    return str(tool("Bash", command="echo ready"))\n'
        if external_bash
        else '    return run_command([sys.executable, "-c", "print(\'child ran\')"], '
        "timeout_seconds=10, mutating=False).stdout\n"
    )
    source = (
        "import sys\nfrom calcifier import recipe, run_command, tool\n"
        '@recipe(capabilities=("Bash",))\ndef child() -> str:\n' + body
    )
    child.write_bytes(attach_metadata(source, skill, recipe_path=child).encode())
    parent = tmp_path / "parent.py"
    parent.write_bytes(
        attach_metadata(
            "from calcifier import call_recipe, recipe\n@recipe\ndef parent() -> str:\n"
            '    return call_recipe("child.py", {}).output\n',
            skill,
            recipe_path=parent,
            subrecipes={"child": str(child)},
        ).encode()
    )
    graph = inspect_graph(parent)
    assert graph.capabilities == ("Bash",)
    assert graph.provider_capabilities == (("Bash",) if external_bash else ())
    assert cli.main(["exec", str(parent), "--trust", "--subrecipe", str(child)]) == (
        2 if external_bash else 0
    )
    captured = capsys.readouterr()
    assert ("missing capabilities" in captured.err) is external_bash
    if not external_bash:
        assert "child ran" in captured.out


def test_direct_script_cli_loads_its_source_before_command_preflight(tmp_path):
    from calcifier.subprocesses import run_process

    path = tmp_path / "recipe.py"
    path.write_bytes(b"""import sys
from calcifier import recipe, run_command
@recipe(capabilities=("Bash",))
def workflow() -> str:
    return run_command([sys.executable, "-V"], timeout_seconds=10, mutating=False).stdout
if __name__ == "__main__":
    raise SystemExit(workflow.cli())
""")
    result = run_process([sys.executable, str(path)], timeout_seconds=20, max_output_bytes=10000)
    assert result.returncode == 0
    assert b"Python" in result.stdout


@pytest.mark.parametrize("dispatch", ["tool", "tool.__call__", 'getattr(runtime, "tool")'])
def test_tool_aliases_keep_provider_preflight_and_use_configured_bash(tmp_path, capsys, dispatch):
    source = f"""import sys
import calcifier as runtime
from calcifier import recipe, run_command, tool

@recipe(capabilities=("Bash",))
def workflow() -> str:
    run_command([sys.executable, "-V"], timeout_seconds=10, mutating=False)
    invoke = {dispatch}
    return str(invoke("Bash", command="echo ready"))
"""
    path = tmp_path / "recipe.py"
    path.write_bytes(source.encode())
    assert cli.main(["exec", str(path), "--trust"]) == 2
    assert "Bash (configure a provider for Bash)" in capsys.readouterr().err
    provider = (
        "import json, sys; request = json.load(sys.stdin); print(json.dumps(request['capability']))"
    )
    config = tmp_path / "tools.json"
    config.write_bytes(
        json.dumps(
            {
                "version": 1,
                "capabilities": {"Bash": {"command": [sys.executable, "-c", provider]}},
            }
        ).encode()
    )
    assert cli.main(["exec", str(path), "--tool-config", str(config)]) == 0
    assert "output: 'Bash'" in capsys.readouterr().out
