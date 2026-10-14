"""Offline wall limits stop deterministic command recipes through either CLI."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from calcifier import cli
from calcifier.recipe import load_recipe
from calcifier.trace_export import validate_trace


@pytest.mark.parametrize("entrypoint", ["exec", "recipe"])
@pytest.mark.parametrize(
    ("declared", "override"),
    [(None, 0.5), (60, 0.5), (0.5, 60)],
)
def test_wall_limit_stops_command_before_it_finishes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    entrypoint: str,
    declared: float | None,
    override: float,
) -> None:
    command = (
        "from pathlib import Path; import time; "
        "time.sleep(30); Path('finished').write_bytes(b'done')"
    )
    budget = "" if declared is None else f", budget=Budget(2, {declared})"
    path = tmp_path / "recipe.py"
    path.write_bytes(
        f'''from calcifier import Budget, recipe, run_command, step

@step
def wait() -> bool:
    run_command(
        {[sys.executable, "-c", command]!r}, timeout_seconds=60,
        working_directory={str(tmp_path)!r},
    )
    return True

@recipe(version="0.1"{budget})
def main() -> bool:
    return wait()
'''.encode()
    )
    trace = tmp_path / "trace.json"
    started = time.monotonic()
    if entrypoint == "exec":
        status = cli.main(
            [
                "exec",
                str(path),
                "--trust",
                "--max-wall-seconds",
                str(override),
                "--trace-out",
                str(trace),
            ]
        )
        payload = json.loads(trace.read_text(encoding="utf-8"))
        validate_trace(payload)
        assert payload["budget"] == {
            "max_agent_calls": None if declared is None else 2,
            "max_wall_seconds": 0.5,
            "attempted_agent_calls": 0,
        }
        assert payload["state"] == "failed"
        assert payload["steps"][0]["side_effects"][0]["completed"] is False
    else:
        status = load_recipe(path).cli(["--max-wall-seconds", str(override)])

    assert time.monotonic() - started < 10
    assert status == 1
    assert "wall-clock budget exhausted (0.5s)" in capsys.readouterr().out
    assert not (tmp_path / "finished").exists()
