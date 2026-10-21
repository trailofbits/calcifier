"""A graph catalog is published only after every node compiles and its edges resolve."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from calcifier import RecipeError
from calcifier.artifact import PreparedArtifact
from calcifier.catalog import load_skill_catalog
from calcifier.graph import inspect_graph
from calcifier.graph_build import _read_spec, build_graph, build_order
from calcifier.metadata import attach_metadata, split_metadata
from calcifier.registry import Registry

_WORKER = """from calcifier import recipe


@recipe(name="worker")
def worker(scope: str) -> str:
    return scope
"""

_ENTRY = """from calcifier import call_recipe, recipe


@recipe
def entry() -> list[str]:
    return [call_recipe("worker.py", {"scope": scope}).output for scope in ("A", "B")]
"""


def graph_fixture(tmp_path: Path, entry: str = _ENTRY) -> None:
    (tmp_path / "SKILL.md").write_text("# Generic planner worker fixture\n")
    (tmp_path / "worker.py").write_text(_WORKER)
    (tmp_path / "entry.py").write_text(entry)


def _spec(tmp_path: Path) -> Path:
    graph_fixture(tmp_path)
    path = tmp_path / "build.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "entry": "entry",
                "output_directory": "built",
                "runtime_wall_seconds": 10800,
                "nodes": {
                    "worker": {
                        "skill": "SKILL.md",
                        "output": "worker.py",
                        "interface": "Preserve typed work context.",
                    },
                    "entry": {
                        "skill": "SKILL.md",
                        "output": "entry.py",
                        "dependencies": ["worker"],
                    },
                },
            }
        )
    )
    return path


def _compiler(
    monkeypatch,
    tmp_path,
    expected_resources=None,
    expected_compilation_timeout=None,
    expected_max_compiler_calls=None,
    expected_runtime_wall_seconds=10800,
    suitability_note=None,
):
    calls = []

    def prepare(
        skill: Path,
        registry: Registry,
        *,
        output: Path,
        skill_catalog: dict[str, str],
        interface_requirements: str,
        runtime_wall_seconds: int,
        compilation_timeout: int | None,
        max_compiler_calls: int | None,
        resource_sources: dict[str, str],
        optimize: bool,
    ) -> PreparedArtifact:
        assert registry.known() == []
        assert runtime_wall_seconds == expected_runtime_wall_seconds
        assert compilation_timeout == expected_compilation_timeout
        assert max_compiler_calls == expected_max_compiler_calls
        assert resource_sources == (expected_resources or {})
        assert optimize is True
        calls.append((output.name, interface_requirements))
        assert all(Path(path).exists() for path in skill_catalog.values())
        source, _ = split_metadata((tmp_path / output.name).read_text())
        return PreparedArtifact(
            output,
            attach_metadata(source, skill, recipe_path=output, subrecipes=skill_catalog),
            suitability_note=suitability_note,
        )

    monkeypatch.setattr("calcifier.graph_build.prepare_artifact", prepare)
    return calls


def test_build_reports_per_node_suitability_assessments(tmp_path, monkeypatch, caplog):
    spec = _spec(tmp_path)
    note = "Most reasoning remains agent-owned; review extraction usefulness."
    _compiler(monkeypatch, tmp_path, suitability_note=note)

    catalog = build_graph(spec, Registry())

    assert catalog.exists()
    assert f"worker: owner suitability review required (compiler assessment): {note}" in caplog.text
    assert f"entry: owner suitability review required (compiler assessment): {note}" in caplog.text


def test_build_compiles_dependencies_and_publishes_a_pinned_catalog(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    calls = _compiler(monkeypatch, tmp_path)
    ticks = iter([100.0, 103.0, 105.0, 110.0])
    monkeypatch.setattr(
        "calcifier.graph_build.time", SimpleNamespace(monotonic=lambda: next(ticks))
    )
    catalog = build_graph(spec, Registry())
    assert calls == [("worker.py", "Preserve typed work context."), ("entry.py", "")]
    assert catalog == tmp_path / "built" / "catalog.json"
    entries = load_skill_catalog(catalog)
    assert set(entries) == {"worker", "entry"}
    graph = inspect_graph(Path(entries["entry"]))
    assert graph.unresolved == ()
    assert set(graph.nodes) == {Path(entries["worker"]), Path(entries["entry"])}
    assert json.loads((catalog.parent / "compile-timings.json").read_text()) == {
        "worker": 3.0,
        "entry": 5.0,
    }


@pytest.mark.parametrize(
    ("specified", "override", "expected"),
    [(1200, None, 1200), (1200, 2400, 2400), (None, 1800, 1800)],
)
def test_build_passes_each_node_the_configured_shared_deadline(
    tmp_path, monkeypatch, specified, override, expected
):
    spec = _spec(tmp_path)
    payload = json.loads(spec.read_bytes())
    if specified is not None:
        payload["compilation_timeout"] = specified
        spec.write_bytes(json.dumps(payload).encode())
    calls = _compiler(monkeypatch, tmp_path, expected_compilation_timeout=expected)
    build_graph(spec, Registry(), compilation_timeout=override)
    assert len(calls) == 2


def test_build_forwards_explicit_call_budget_to_every_node(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    calls = _compiler(monkeypatch, tmp_path, expected_max_compiler_calls=6)
    build_graph(spec, Registry(), max_compiler_calls=6)
    assert len(calls) == 2


def test_unresolved_build_keeps_staging_and_publishes_no_catalog(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    graph_fixture(
        tmp_path, entry=_ENTRY.replace('call_recipe("worker.py", {"scope": scope}).output', "scope")
    )
    _compiler(monkeypatch, tmp_path)
    with pytest.raises(RecipeError, match="unresolved edges"):
        build_graph(spec, Registry())
    assert not (tmp_path / "built").exists()
    staged = list(tmp_path.glob(".calcifier-build-*/entry.py"))
    assert len(staged) == 1
    assert not list(tmp_path.glob(".calcifier-build-*/catalog.json"))


def test_build_forwards_explicit_resource_sources(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    payload = json.loads(spec.read_text())
    (tmp_path / "supplement.txt").write_text("pinned template", encoding="utf-8")
    resources = {"report_templates/index.txt": "supplement.txt"}
    for node in payload["nodes"].values():
        node["resources"] = resources
    spec.write_text(json.dumps(payload))
    calls = _compiler(monkeypatch, tmp_path, resources)
    build_graph(spec, Registry())
    assert len(calls) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"version": 2},
        {"nodes": {}},
        {"entry": "missing"},
        {"output_directory": "."},
    ],
)
def test_bad_build_specs_fail_before_compilation(tmp_path, change):
    spec = _spec(tmp_path)
    data = json.loads(spec.read_text())
    data.update(change)
    spec.write_text(json.dumps(data))
    with pytest.raises(RecipeError):
        _read_spec(spec)


def test_unknown_build_dependency_is_rejected():
    nodes: dict[str, Any] = {"entry": {"dependencies": ["absent"]}}
    with pytest.raises(RecipeError, match="unknown graph build dependency"):
        build_order(nodes)


@pytest.mark.parametrize("output", ["CON.py", "dir/NUL.py", "dir./worker.py", "aux/worker.py"])
def test_windows_reserved_outputs_fail_before_compilation(tmp_path, monkeypatch, output):
    spec = _spec(tmp_path)
    calls = _compiler(monkeypatch, tmp_path)
    payload = json.loads(spec.read_text())
    payload["nodes"]["worker"]["output"] = output
    spec.write_text(json.dumps(payload))
    with pytest.raises(RecipeError, match="output"):
        build_graph(spec, Registry())
    assert calls == []
    assert not (tmp_path / "built").exists()


def test_runtime_wall_override_reaches_every_compiled_graph_node(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    calls = _compiler(monkeypatch, tmp_path, expected_runtime_wall_seconds=123)
    catalog = build_graph(spec, Registry(), runtime_wall_seconds=123)
    assert catalog.exists()
    assert [name for name, _ in calls] == ["worker.py", "entry.py"]
