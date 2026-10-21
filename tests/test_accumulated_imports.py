"""Final combined evidence survives replace-by-default old and new dependency contracts."""

from __future__ import annotations

import json
import sys

import pytest

from calcifier.artifact import prepare_artifact
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from calcifier.subprocesses import run_process
from tests.support.compiler import one_draft

_INPUTS = [[{"id": "sample-a", "origin": "first"}], [{"id": "sample-b", "origin": "second"}]]
_SOURCE = """import json
import sys
from typing import Literal
from calcifier import (
    emit, ensure_directory, fail, output_type, recipe, resource, run_command, step,
    verify_artifact, write_json,
)

@output_type
class ImportCount:
    matched: int

@step
def combine(api: Literal["old", "new"]) -> list[dict[str, str]]:
    files = (resource("references/one.json"), resource("references/two.json"))
    expected = [row for path in files for row in json.loads(path.read_text(encoding="utf-8"))]
    destination = ensure_directory(output_files=("combined-input.json", "state.json"))
    data = write_json("combined-input.json", expected)
    counts = run_command(
        [sys.executable, str(resource("scripts/importer.py")), "--api", api,
         "--input", str(data), "--output", str(destination / "state.json")],
        timeout_seconds=10, mutating=True, output=ImportCount)
    emit(counts.model_dump(), name="import-counts")
    state = verify_artifact("state.json")
    actual = json.loads(state.read_text(encoding="utf-8"))
    def canonical(rows: list[dict[str, str]]) -> list[str]:
        return sorted(json.dumps(row, sort_keys=True) for row in rows)
    if canonical(actual) != canonical(expected):
        fail("Combined artifact does not retain the complete input evidence.")
    return emit(actual, name="combined-evidence")

@recipe(output_root="artifacts", capabilities=("Bash",))
def workflow(api: Literal["old", "new"] = "old") -> list[dict[str, str]]:
    return combine(api)

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
_IMPORTER = """import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--api", choices=("old", "new"), required=True)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--append", action="store_true")
args = parser.parse_args()
if args.append and args.api == "old":
    parser.error("this supported release has no append API")
data = json.loads(args.input.read_bytes())
matched = len(data)
if args.append and args.output.exists():
    data = json.loads(args.output.read_bytes()) + data
if DROP_FIRST and len(data) > 1:
    data = data[1:]
args.output.write_bytes(json.dumps(data).encode("utf-8"))
print(json.dumps({"matched": matched}))
"""


@pytest.mark.parametrize("api", ["old", "new"])
def test_neutral_dependency_replaces_by_default_while_each_import_reports_success(tmp_path, api):
    program = tmp_path / "importer.py"
    program.write_bytes(_IMPORTER.replace("DROP_FIRST", "False").encode("utf-8"))
    destination = tmp_path / "state.json"
    counts = []
    for index, rows in enumerate(_INPUTS):
        source = tmp_path / f"input-{index}.json"
        source.write_bytes(json.dumps(rows).encode("utf-8"))
        result = run_process(
            [
                sys.executable,
                str(program),
                "--api",
                api,
                "--input",
                str(source),
                "--output",
                str(destination),
            ],
            timeout_seconds=10,
            max_output_bytes=4096,
        )
        assert result.returncode == 0
        counts.append(json.loads(result.stdout))

    assert counts == [{"matched": 1}, {"matched": 1}]
    assert json.loads(destination.read_bytes()) == _INPUTS[-1]


@pytest.mark.parametrize("api", ["old", "new"])
@pytest.mark.parametrize("drops_first", [False, True])
def test_aggregate_import_and_final_union_check_support_both_dependency_contracts(
    tmp_path, api, drops_first
):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(
        b"# Combine evidence\nPreserve all evidence from references/one.json "
        b"and references/two.json "
        b"in the final imported state. Use supported dependency APIs. Verify the final union; "
        b"individual success counters do not establish accumulation.\n"
    )
    resources = tmp_path / "references"
    resources.mkdir()
    for name, rows in zip(("one", "two"), _INPUTS, strict=True):
        (resources / f"{name}.json").write_bytes(json.dumps(rows).encode("utf-8"))
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "importer.py").write_bytes(
        _IMPORTER.replace("DROP_FIRST", str(drops_first)).encode("utf-8")
    )
    registry = Registry()
    registry.register(one_draft(_SOURCE))

    artifact = prepare_artifact(skill, registry)

    assert artifact.outcome == "compiled"
    assert artifact.dry_run is not None and artifact.dry_run.status == "unverified"
    assert "declared output is unavailable after a suppressed command" in (
        artifact.dry_run.error or ""
    )
    artifact.path.parent.mkdir(parents=True)
    artifact.path.write_bytes(artifact.source.encode("utf-8"))
    workflow = load_recipe(artifact.path)
    output = tmp_path / "reports"

    result = workflow.run({"api": api}, output_directory=output)

    assert result.ok is not drops_first
    counts = next(item.value for item in result.artifacts if item.name == "import-counts")
    assert counts == {"matched": 2}
    if drops_first:
        assert "does not retain the complete input evidence" in (result.error or "")
        assert not any(item.name == "combined-evidence" for item in result.artifacts)
    else:
        expected = _INPUTS[0] + _INPUTS[1]
        assert result.output == expected
        assert json.loads((output / "state.json").read_bytes()) == expected
