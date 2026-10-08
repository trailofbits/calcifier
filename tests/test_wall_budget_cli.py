"""Offline wall limits stop deterministic command recipes through either CLI."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from calcifier import RecipeRun, cli
from calcifier.recipe import load_recipe
from calcifier.trace_export import validate_trace


@pytest.mark.parametrize("entrypoint", ["exec", "recipe"])
@pytest.mark.parametrize(("declared", "override"), [(None, 10), (60, 10), (10, 60)])
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["LF", "CRLF"])
def test_live_budget_query_observes_effective_cli_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    declared: float | None,
    override: float,
    newline: str,
) -> None:
    runs: list[RecipeRun] = []
    render = RecipeRun.render

    def capture_run(run: RecipeRun) -> str:
        runs.append(run)
        return render(run)

    monkeypatch.setattr(RecipeRun, "render", capture_run)
    budget = "" if declared is None else f", budget=Budget(None, {declared})"
    path = tmp_path / "recipe.py"
    source = f'''from calcifier import Budget, recipe, step

@step
def inspect() -> float:
    return Budget.remaining_wall_seconds()

@recipe(version="0.1"{budget})
def main() -> float:
    return inspect()
'''
    path.write_bytes(source.replace("\n", newline).encode("utf-8"))
    trace = tmp_path / "trace.json"
    args = ["--max-wall-seconds", str(override)]
    if entrypoint == "exec":
        status = cli.main(["exec", str(path), "--trust", *args, "--trace-out", str(trace)])
        validate_trace(json.loads(trace.read_text(encoding="utf-8")))
    else:
        status = load_recipe(path).cli(args)
    assert status == 0
    assert len(runs) == 1
    run = runs[0]
    assert run.state == "completed"
    assert run.budget is not None
    assert run.budget["max_wall_seconds"] == 10
    assert run.budget["attempted_agent_calls"] == 0
    assert isinstance(run.output, float)
    assert 0 < run.output <= 10
    assert run.records[0].output == run.output
    assert path.read_bytes() == source.replace("\n", newline).encode("utf-8")


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
