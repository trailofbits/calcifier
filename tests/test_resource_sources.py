"""Explicit build resources are pinned without altering the source skill."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calcifier.artifact import prepare_artifact
from calcifier.calcify import CalcifyError
from calcifier.graph_build import _read_spec
from calcifier.hashing import sha256_hex
from calcifier.recipe import RecipeError, load_recipe
from calcifier.registry import Registry
from calcifier.skill_package import (
    RESOURCE_FILE_SIZE_LIMIT,
    add_resource_sources,
    discover_resources,
)
from tests.support.compiler import one_draft
from tests.test_graph_build import _spec

SOURCE = """from calcifier import recipe, resource, step
@step
def read_template() -> str:
    return resource("report_templates/index.txt").read_text(encoding="utf-8")
@recipe
def example() -> str:
    return read_template()
if __name__ == "__main__":
    raise SystemExit(example.cli())
"""


def package(tmp_path: Path) -> tuple[Path, Path]:
    skill = tmp_path / "plugin" / "skills" / "exporter" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Export the pinned template\n", encoding="utf-8")
    template = tmp_path / "plugin" / "report_templates" / "index.txt"
    template.parent.mkdir()
    template.write_bytes("café 雪\r\n".encode())
    return skill, template


def test_explicit_resources_reach_compiler_snapshot_metadata_and_runtime(tmp_path, monkeypatch):
    skill, template = package(tmp_path)
    original = skill.read_bytes()
    compiler = one_draft(SOURCE)
    invoke = compiler.invoke

    def inspect(prompt, *, filesystem, **kwargs: Any):
        assert (
            filesystem.working_directory / "report_templates/index.txt"
        ).read_bytes() == template.read_bytes()
        return invoke(prompt, filesystem=filesystem, **kwargs)

    monkeypatch.setattr(compiler, "invoke", inspect)
    registry = Registry()
    registry.register(compiler)
    target = tmp_path / "recipe.py"
    artifact = prepare_artifact(
        skill,
        registry,
        output=target,
        resource_sources={"report_templates/index.txt": "../../report_templates/index.txt"},
    )
    target.write_text(artifact.source, encoding="utf-8")
    run = load_recipe(target).run()
    assert run.ok and run.output == "café 雪\n", run
    assert skill.read_bytes() == original
    assert sha256_hex(template.read_bytes()) in artifact.source
    assert "../../report_templates/index.txt" in artifact.source
    template.write_bytes(b"changed in place")
    with pytest.raises(RecipeError, match="packaged resource drift"):
        load_recipe(target).run()


@pytest.mark.parametrize(
    "alias",
    [
        "",
        ".",
        "/absolute",
        "C:/absolute",
        "C:relative",
        r"folder\file",
        "../file",
        "a/../file",
        "a//b",
        "bad\0file",
        "SKILL.md",
        "skill.md/child",
        "report.txt:stream",
        "CON",
        "NUL.txt",
        "name.",
        "name ",
        123,
    ],
)
def test_unsafe_aliases_fail_before_provider_calls(tmp_path, alias):
    skill, _ = package(tmp_path)
    compiler = one_draft(SOURCE)
    registry = Registry()
    registry.register(compiler)
    with pytest.raises(CalcifyError, match="explicit compiler resources"):
        prepare_artifact(
            skill,
            registry,
            resource_sources={alias: "../../report_templates/index.txt"},
        )
    assert compiler.calls == []


@pytest.mark.parametrize(
    "source",
    [
        "",
        ".",
        "missing",
        "/outside",
        "C:/outside",
        "C:relative",
        r"..\outside",
        "../../../outside.txt",
        123,
    ],
)
def test_missing_or_escaped_sources_are_rejected(tmp_path, source):
    skill, _ = package(tmp_path)
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    with pytest.raises(ValueError):
        add_resource_sources(skill, [], {"data.txt": source})


def test_oversize_and_symlink_escape_are_rejected(tmp_path):
    skill, template = package(tmp_path)
    template.write_bytes(b"x" * (RESOURCE_FILE_SIZE_LIMIT + 1))
    with pytest.raises(ValueError, match="oversized"):
        add_resource_sources(skill, [], {"data.txt": "../../report_templates/index.txt"})
    template.write_bytes(b"x" * RESOURCE_FILE_SIZE_LIMIT)
    assert (
        add_resource_sources(skill, [], {"data.txt": "../../report_templates/index.txt"})[0].size
        == RESOURCE_FILE_SIZE_LIMIT
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = skill.parent / "escape.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="escaped"):
        add_resource_sources(skill, [], {"data.txt": "escape.txt"})


@pytest.mark.parametrize(
    "alias", ["references/a.txt", "REFERENCES/A.TXT", "references/a.txt/child", "references"]
)
def test_aliases_cannot_override_discovered_files_or_directories(tmp_path, alias):
    skill, _ = package(tmp_path)
    guide = skill.parent / "references/a.txt"
    guide.parent.mkdir()
    guide.write_text("guide", encoding="utf-8")
    with pytest.raises(ValueError, match="collision"):
        add_resource_sources(
            skill, discover_resources(skill), {alias: "../../report_templates/index.txt"}
        )


@pytest.mark.parametrize("aliases", [("data.txt", "DATA.TXT"), ("a", "a/b"), ("a/b", "a")])
def test_explicit_alias_collisions_are_rejected_in_either_order(tmp_path, aliases):
    skill, _ = package(tmp_path)
    with pytest.raises(ValueError, match="collision"):
        add_resource_sources(skill, [], dict.fromkeys(aliases, "../../report_templates/index.txt"))


def test_manifest_resources_resolve_from_manifest_then_pin_relative_to_skill(tmp_path):
    skill, template = package(tmp_path)
    spec = _spec(tmp_path)
    payload = json.loads(spec.read_text())
    for node in payload["nodes"].values():
        node["skill"] = skill.relative_to(tmp_path).as_posix()
        node["resources"] = {
            "report_templates/index.txt": template.relative_to(tmp_path).as_posix()
        }
    spec.write_text(json.dumps(payload))
    parsed, _ = _read_spec(spec)
    for node in parsed["nodes"].values():
        assert node["resources"] == {
            "report_templates/index.txt": "../../report_templates/index.txt"
        }
    template.unlink()
    with pytest.raises(RecipeError, match="graph resource"):
        _read_spec(spec)


@pytest.mark.parametrize(
    "alias",
    ["", "/absolute", "C:/absolute", "C:relative", r"a\b.txt", "../file", "NUL.txt"],
)
def test_gate_trust_and_runtime_reject_the_same_nonportable_resource_paths(tmp_path, alias):
    from calcifier.gate import verify_source
    from calcifier.metadata import attach_metadata
    from calcifier.skill_package import ResourceEntry
    from calcifier.trust import TrustError
    from tests.support.trust import record_trust

    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nRead a packaged note.\n")
    target = tmp_path / "recipe.py"
    entries = [ResourceEntry(alias, "0" * 64, 0)]
    source = SOURCE.replace('resource("report_templates/index.txt")', f"resource({alias!r})")
    target.write_bytes(attach_metadata(source, skill, entries, recipe_path=target).encode("utf-8"))
    with pytest.raises(CalcifyError) as compiler_error:
        verify_source(source, type_check=False)
    assert "resource_usage" in {item.code for item in compiler_error.value.diagnostics}
    with pytest.raises(RecipeError) as runtime_error:
        load_recipe(target)
    with pytest.raises(TrustError) as trust_error:
        record_trust(target, recorded_by="tester")
    assert str(runtime_error.value) == str(trust_error.value)
