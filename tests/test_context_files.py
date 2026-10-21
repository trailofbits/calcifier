"""Context gathering confines every selected file before reading any of them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calcifier import RecipeError, prepare_context
from calcifier.artifact import prepare_artifact
from calcifier.prepared_context import prepared_context_run
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from tests.support.agents import StubProvider
from tests.support.compiler import one_draft


def symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")


def files_in(context) -> list[dict[str, str]]:
    contents = (Path(context.directory) / "task.txt").read_text(encoding="utf-8")
    return json.loads(contents.split("\nFiles:\n", 1)[1])


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_context_retains_complete_source_and_manifest_content(tmp_path, newline):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "module.txt"
    manifest = project / "manifest.json"
    contents = b"caf\xc3\xa9" + newline + b"last line" + newline
    source.write_bytes(contents)
    manifest.write_bytes(b'{"version":1}' + newline)

    with prepared_context_run():
        context = prepare_context(
            "Review the complete project.",
            files=(source, manifest),
            readable_roots=(project,),
        )
        assert files_in(context) == [
            {"path": str(source), "text": contents.decode("utf-8")},
            {"path": str(manifest), "text": (b'{"version":1}' + newline).decode("utf-8")},
        ]


def test_internal_symlink_and_explicit_external_input_are_supported(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "source.txt"
    source.write_bytes(b"internal content\n")
    alias = project / "alias.txt"
    symlink(alias, source)
    external = tmp_path / "authorized.txt"
    external.write_bytes(b"explicit external input\n")

    with prepared_context_run():
        context = prepare_context(
            "Review selected inputs.",
            files=(alias, external),
            readable_roots=(project, external),
        )
        assert files_in(context) == [
            {"path": str(alias), "text": "internal content\n"},
            {"path": str(external), "text": "explicit external input\n"},
        ]


@pytest.mark.parametrize("name", ["source.txt", "manifest.json"])
def test_escaping_symlink_is_rejected_before_any_contents_are_read(tmp_path, monkeypatch, name):
    project = tmp_path / "project"
    project.mkdir()
    ordinary = project / "ordinary.txt"
    ordinary.write_bytes(b"ordinary input\n")
    external = tmp_path / "outside.txt"
    external.write_bytes(b"synthetic outside content must not enter context\n")
    escaping = project / name
    symlink(escaping, external)
    reads = []
    original = Path.read_bytes

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with prepared_context_run(), pytest.raises(RecipeError, match="escapes readable_roots"):
        prepare_context("Review inputs.", files=(ordinary, escaping), readable_roots=(project,))

    assert reads == []


@pytest.mark.parametrize("case", ["no-roots", "directory", "missing"])
def test_context_inputs_require_explicit_roots_and_regular_files(tmp_path, case):
    source = tmp_path / "source.txt"
    source.write_bytes(b"ordinary input\n")
    selected = {"no-roots": source, "directory": tmp_path, "missing": tmp_path / "missing"}[case]
    expected = {
        "no-roots": "explicit readable_roots",
        "directory": "regular file",
        "missing": "cannot read",
    }

    with prepared_context_run(), pytest.raises(RecipeError, match=expected[case]):
        prepare_context(
            "Review inputs.",
            files=(selected,),
            readable_roots=() if case == "no-roots" else (tmp_path,),
        )


def test_pinned_resources_can_be_included_explicitly_in_gathered_context(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Review\nRead references/rules.txt and the selected project files.\n")
    resources = tmp_path / "references"
    resources.mkdir()
    (resources / "rules.txt").write_bytes(b"Preserve every relevant observation.\n")
    source = """from pathlib import Path
from calcifier import Budget, agent, prepare_context, recipe, resource, step

@step
def inspect(target: Path) -> bool:
    rules = resource("references/rules.txt")
    context = prepare_context("Review the complete selected input.",
        files=(*sorted(target.glob("*.txt")), rules), readable_roots=(target, rules))
    return agent("worker", context.prompt, context=context, working_directory=target,
                 readable_paths=(target,), required_capabilities=("read",))

@recipe(agents=("worker",), budget=Budget(2, 30))
def workflow(target: Path = Path(".")) -> bool:
    return inspect(target)

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    registry = Registry()
    registry.register(one_draft(source))

    class Worker(StubProvider):
        name = "worker"
        capabilities = ("read",)

        def __init__(self):
            super().__init__(default="true")
            self.prompts = []

        def invoke(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return super().invoke(prompt, **kwargs)

    worker = Worker()
    registry.register(worker)
    artifact = prepare_artifact(skill, registry)
    assert artifact.outcome == "compiled"
    assert artifact.dry_run is not None and artifact.dry_run.status == "passed"
    artifact.path.parent.mkdir(parents=True)
    artifact.path.write_bytes(artifact.source.encode("utf-8"))
    workflow = load_recipe(artifact.path)
    project = tmp_path / "project"
    project.mkdir()
    (project / "source.txt").write_bytes(b"complete source content\n")

    result = workflow.run({"target": project}, registry=registry)
    assert result.ok and result.output is True
    assert len(worker.prompts) == 1
    assert "complete source content" in worker.prompts[0]
    assert "Preserve every relevant observation" in worker.prompts[0]

    external = tmp_path / "outside.txt"
    external.write_bytes(b"synthetic excluded content\n")
    symlink(project / "manifest.txt", external)
    result = workflow.run({"target": project}, registry=registry)
    assert not result.ok
    assert "escapes readable_roots" in (result.error or "")
    assert len(worker.prompts) == 1
    assert "synthetic excluded content" not in worker.prompts[0]
