"""Trusted handwritten legacy tool imports still require providers before host effects."""

from __future__ import annotations

import json
import sys

import pytest

from calcifier import cli
from calcifier.recipe import load_recipe

pytestmark = pytest.mark.semantic_offline


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("capability", ["Bash", "Read"])
@pytest.mark.parametrize(
    "import_line, expression, escaped",
    [
        ("from calcifier.tools import tool", "tool", False),
        ("import calcifier.tools as tools", "tools.tool", False),
        ("import calcifier.tools as tools", "[tools][0].tool", True),
    ],
)
def test_legacy_tool_paths_preflight_before_host_mutation(
    tmp_path, capsys, direct, capability, import_line, expression, escaped
):
    marker = tmp_path / "command-ran.txt"
    command = f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')"
    path = tmp_path / "recipe.py"
    path.write_bytes(
        f"""import sys
from calcifier import recipe, run_command
{import_line}

@recipe(capabilities=("Bash", "Read"), output_root=".")
def workflow() -> str:
    run_command([sys.executable, "-c", {command!r}], timeout_seconds=10, mutating=True)
    return str({expression}({capability!r}, path="input.txt"))
""".encode()
    )
    recipe = load_recipe(path)
    assert recipe._provider_capabilities == (("Bash", "Read") if escaped else (capability,))
    result = recipe.cli([]) if direct else cli.main(["exec", str(path), "--trust"])
    assert result == 2
    assert not marker.exists()
    assert "missing capabilities" in capsys.readouterr().err
    provider = "import json,sys; r=json.load(sys.stdin); print(json.dumps(r['capability']))"
    config = tmp_path / "tools.json"
    config.write_bytes(
        json.dumps(
            {
                "version": 1,
                "capabilities": {
                    name: {"command": [sys.executable, "-c", provider]}
                    for name in recipe._provider_capabilities
                },
            }
        ).encode()
    )
    result = (
        recipe.cli(["--tool-config", str(config)])
        if direct
        else cli.main(["exec", str(path), "--tool-config", str(config)])
    )
    assert result == 0
    assert marker.read_bytes() == b"ran"
    assert capability in capsys.readouterr().out
