"""Host-selected artifact paths remain usable after writing model-produced content."""

from __future__ import annotations

import json
from pathlib import Path

from calcifier.artifact import prepare_artifact
from calcifier.portable import lint_portable
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from calcifier.subprocesses import ProcessResult
from tests.support.agents import StubProvider
from tests.support.compiler import one_draft

SOURCE = """from calcifier import Budget, agent, recipe, run_command, step, verify_artifact
from calcifier import write_text as save_report

@step
def export(root: str) -> str:
    answer = agent("stub", "Summarize", working_directory=root, readable_paths=(root,))
    written = answer
    written = save_report("report.txt", written)
    artifact = verify_artifact(written)
    return run_command(
        ["fixture-reader", str(artifact)], timeout_seconds=1,
        working_directory=root, mutating=False,
    ).stdout

@recipe(agents=("stub",), budget=Budget(2, 30), output_root="reports")
def run(root: str) -> str:
    return export(root)

if __name__ == "__main__":
    raise SystemExit(run.cli())
"""


def test_portable_recipe_passes_the_saved_artifact_to_a_consumer(tmp_path, monkeypatch):
    path = tmp_path / "recipe.py"
    path.write_bytes(SOURCE.encode())
    assert lint_portable(path) == []
    monkeypatch.chdir(tmp_path)
    text = "Résumé — α\r\nUntrusted model content.\n"
    registry = Registry()
    registry.register(StubProvider(default=json.dumps(text)))
    calls = []

    def consume(argv, *, timeout_seconds, max_output_bytes, cwd):
        del timeout_seconds, max_output_bytes
        calls.append(argv)
        assert cwd == tmp_path
        assert argv == ["fixture-reader", str(tmp_path / "reports/report.txt")]
        assert Path(argv[1]).read_bytes() == text.encode()
        return ProcessResult(tuple(argv), 0, b"consumed", b"")

    monkeypatch.setattr("calcifier.mutations.run_process", consume)
    run = load_recipe(path).run({"root": str(tmp_path)}, registry=registry)
    assert run.ok, run.error
    assert run.output == "consumed"
    assert len(calls) == 1


def test_compiler_accepts_computed_report_keys_without_tainting_the_project(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Review\nSummarize the project, then run the supplied observation.\n")
    source = """import sys
from pathlib import Path
from calcifier import Budget, agent, recipe, run_command, step

@step
def inspect(root: Path) -> str:
    reports: dict[str, str] = {}
    reports[str(root)] = agent("stub", "Summarize the project.",
                               working_directory=root, readable_paths=(root,))
    run_command([sys.executable, "-c", "print(1)", str(root)],
                working_directory=root, timeout_seconds=5, mutating=False, output=int)
    return reports[str(root)]

@recipe(agents=("stub",), capabilities=("Bash",), budget=Budget(2, 30))
def workflow(root: Path = Path(".")) -> str:
    return inspect(root)

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    registry = Registry()
    registry.register(one_draft(source))
    registry.register(StubProvider(default=json.dumps("Complete project summary.")))

    artifact = prepare_artifact(skill, registry)

    assert artifact.outcome == "compiled"
    assert artifact.dry_run is not None and artifact.dry_run.status == "passed"
    artifact.path.parent.mkdir(parents=True)
    artifact.path.write_bytes(artifact.source.encode("utf-8"))
    run = load_recipe(artifact.path).run({"root": tmp_path}, registry=registry)
    assert run.ok, run.error
    assert run.output == "Complete project summary."
