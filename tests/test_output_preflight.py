"""Subprocess destinations are checked as a set before any artifact effects."""

from __future__ import annotations

import pytest

from calcifier import ensure_directory, recipe, write_text
from calcifier.artifact import prepare_artifact
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from calcifier.trace import DryRunMode, RunHooks
from tests.support.compiler import one_draft

_OUTPUTS = ("raw.json", "summary.md", "report.md")


@pytest.mark.parametrize("name", ["output_files", "exclude_roots"])
@pytest.mark.parametrize("value", ["out", b"out"])
def test_bare_path_collections_fail_before_creating_outputs(tmp_path, name, value):
    @recipe(output_root=str(tmp_path / "destination"))
    def workflow() -> str:
        return str(ensure_directory(**{name: value}))

    result = workflow.run()

    assert not result.ok
    assert f"{name} must be" in (result.error or "")
    assert not (tmp_path / "destination").exists()


_SOURCE = """import sys
from pathlib import Path
from calcifier import ensure_directory, recipe, resource, run_command, step

@step
def collect(project: Path) -> bool:
    destination = ensure_directory(".", output_files=("raw.json", "summary.md", "report.md"),
                                   exclude_roots=(project,))
    return run_command([sys.executable, str(resource("scripts/collect.py")), str(destination)],
                       working_directory=project, timeout_seconds=10, mutating=True, output=bool)

@recipe(output_root="../artifacts", capabilities=("Bash",))
def workflow(project: Path = Path(".")) -> bool:
    return collect(project)

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""


@pytest.fixture
def compiled_collector(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(
        b"# Collection\nSave raw.json, summary.md, and report.md outside the project. "
        b"Reject existing outputs before creating directories or running the collector.\n"
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "collect.py").write_bytes(
        b"import sys\nfrom pathlib import Path\nout = Path(sys.argv[1])\n"
        b"for name in ('raw.json', 'summary.md', 'report.md'):\n"
        b"    (out / name).write_bytes(b'complete\\n')\nprint('true')\n"
    )
    registry = Registry()
    registry.register(one_draft(_SOURCE))
    artifact = prepare_artifact(skill, registry)
    assert artifact.outcome == "compiled"
    assert artifact.dry_run is not None and artifact.dry_run.status == "passed"
    artifact.path.parent.mkdir(parents=True)
    artifact.path.write_bytes(artifact.source.encode("utf-8"))
    return load_recipe(artifact.path)


@pytest.mark.parametrize("existing", _OUTPUTS)
def test_existing_intermediate_or_final_output_stops_before_process_launch(
    tmp_path, monkeypatch, compiled_collector, existing
):
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / existing
    sentinel.write_bytes(b"preserve these bytes\r\n")

    def unexpected_launch(*args, **kwargs):
        raise AssertionError("collector reached the process boundary after preflight rejection")

    monkeypatch.setattr("calcifier.mutations.run_process", unexpected_launch)
    result = compiled_collector.run({"project": project}, output_directory=output)

    assert not result.ok
    assert "existing planned output" in (result.error or "")
    assert sentinel.read_bytes() == b"preserve these bytes\r\n"
    assert list(output.iterdir()) == [sentinel]
    assert all(not record.side_effects for record in result.records)


@pytest.mark.parametrize("dangling", [False, True])
def test_symlinked_intermediates_never_reach_a_subprocess(
    tmp_path, monkeypatch, compiled_collector, dangling
):
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    target = output / "keep.txt"
    if not dangling:
        target.write_bytes(b"keep\n")
    link = output / "raw.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    def unexpected_launch(*args, **kwargs):
        raise AssertionError("collector followed a symlinked output")

    monkeypatch.setattr("calcifier.mutations.run_process", unexpected_launch)
    result = compiled_collector.run({"project": project}, output_directory=output)

    assert not result.ok
    assert "symlink" in (result.error or "")
    assert link.is_symlink()
    assert target.exists() is not dangling
    if not dangling:
        assert target.read_bytes() == b"keep\n"
    assert not (output / "report.md").exists()
    assert all(not record.side_effects for record in result.records)


def test_project_contained_output_is_rejected_before_directory_creation(
    tmp_path, monkeypatch, compiled_collector
):
    project = tmp_path / "project"
    project.mkdir()
    destination = project / "new" / "outputs"

    def unexpected_launch(*args, **kwargs):
        raise AssertionError("collector launched for a rejected output root")

    monkeypatch.setattr("calcifier.mutations.run_process", unexpected_launch)
    result = compiled_collector.run({"project": project}, output_directory=destination)

    assert not result.ok
    assert "excluded root" in (result.error or "")
    assert not (project / "new").exists()
    assert all(not record.side_effects for record in result.records)


def test_clean_output_set_is_created_and_written(tmp_path, compiled_collector):
    project = tmp_path / "project"
    project.mkdir()
    destination = tmp_path / "new" / "outputs"

    result = compiled_collector.run({"project": project}, output_directory=destination)

    assert result.ok and result.output is True
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == dict.fromkeys(
        _OUTPUTS, b"complete\n"
    )
    assert list(project.iterdir()) == []


@pytest.mark.parametrize(
    "outputs,error",
    [
        (("../escape.txt",), "escapes the selected directory"),
        ((".git/config",), "cannot modify .git paths"),
        (("parent", "parent/child.txt"), "parent directory"),
        (("parent/child.txt",), "non-directory parent"),
    ],
)
def test_all_paths_are_validated_before_selected_directory_creation(tmp_path, outputs, error):
    destination = tmp_path / "selected"
    if outputs == ("parent/child.txt",):
        destination.mkdir()
        (destination / "parent").write_bytes(b"preserve\n")

    @recipe(output_root=tmp_path)
    def workflow() -> bool:
        ensure_directory("selected", output_files=outputs)
        return True

    result = workflow.run()
    assert not result.ok
    assert error in (result.error or "")
    assert all(not record.side_effects for record in result.records)
    if outputs == ("parent/child.txt",):
        assert (destination / "parent").read_bytes() == b"preserve\n"
    else:
        assert not destination.exists()


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_explicit_replacement_still_respects_selected_directory_policy(
    tmp_path, overwrite, override
):
    sentinel = tmp_path / "raw.json"
    sentinel.write_bytes(b"keep\n")

    @recipe(output_root=tmp_path)
    def workflow() -> bool:
        ensure_directory(output_files=("raw.json",), overwrite=overwrite)
        return True

    result = workflow.run(output_directory=tmp_path if override else None)
    assert result.ok is (overwrite and not override)
    assert sentinel.read_bytes() == b"keep\n"


def test_file_created_this_run_can_be_declared_for_explicit_replacement(tmp_path):
    @recipe(output_root=tmp_path)
    def workflow() -> bool:
        write_text("raw.json", "initial\n")
        ensure_directory(output_files=("raw.json",), overwrite=True)
        return True

    assert workflow.run(output_directory=tmp_path).ok


def test_compiler_scratch_redirection_does_not_hide_actual_output_collisions(tmp_path):
    destination = tmp_path / "real"
    destination.mkdir()
    sentinel = destination / "raw.json"
    sentinel.write_bytes(b"keep\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    @recipe(output_root=destination)
    def workflow() -> bool:
        ensure_directory(output_files=("raw.json",))
        return True

    result = workflow._execute(
        dry_run=DryRunMode.SOFT, _hooks=RunHooks(scratch_output_root=scratch)
    )
    assert not result.ok
    assert "existing planned output" in (result.error or "")
    assert sentinel.read_bytes() == b"keep\n"
    assert list(scratch.iterdir()) == []
