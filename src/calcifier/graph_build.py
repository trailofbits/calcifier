"""Compile a graph of skills in dependency order and publish its catalog."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .artifact import PreparedArtifact, prepare_artifact
from .exceptions import RecipeError
from .graph import inspect_graph
from .path_syntax import PathSyntax
from .registry import Registry
from .skill_package import add_resource_sources, discover_resources, resource_relative_path

logger = logging.getLogger(__name__)

_SPEC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["version", "entry", "output_directory", "nodes"],
    "properties": {
        "version": {"const": 1},
        "entry": {"type": "string", "minLength": 1},
        "output_directory": {"type": "string", "minLength": 1},
        "runtime_wall_seconds": {"type": "integer", "minimum": 1},
        "compilation_timeout": {"type": "integer", "minimum": 1},
        "nodes": {
            "type": "object",
            "minProperties": 1,
            "propertyNames": {"pattern": "^[a-zA-Z0-9][a-zA-Z0-9_.-]*$"},
            "additionalProperties": {
                "type": "object",
                "additionalProperties": False,
                "required": ["skill", "output"],
                "properties": {
                    "skill": {"type": "string", "minLength": 1},
                    "output": {"type": "string", "minLength": 1},
                    "interface": {"type": "string"},
                    "resources": {
                        "type": "object",
                        "additionalProperties": {"type": "string", "minLength": 1},
                    },
                    "dependencies": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    },
}


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecipeError(f"duplicate build field: {key}")
        result[key] = value
    return result


def build_order(nodes: dict[str, Any]) -> list[str]:
    pending: set[str] = set()
    complete: list[str] = []

    def visit(name: str) -> None:
        if name in pending:
            raise RecipeError(f"cyclic graph build dependency: {name}")
        if name not in nodes:
            raise RecipeError(f"unknown graph build dependency: {name}")
        if name in complete:
            return
        pending.add(name)
        for child in nodes[name].get("dependencies", []):
            visit(child)
        pending.remove(name)
        complete.append(name)

    for name in sorted(nodes):
        visit(name)
    return complete


def _read_spec(path: Path) -> tuple[dict[str, Any], Path]:
    from jsonschema import Draft202012Validator, ValidationError

    try:
        spec = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
        Draft202012Validator(_SPEC_SCHEMA).validate(spec)
    except (ValueError, ValidationError) as exc:
        raise RecipeError(f"invalid graph build specification: {exc}") from exc
    if spec.get("entry") not in spec["nodes"]:
        raise RecipeError("build entry is not a declared node")
    destination = (path.parent / spec["output_directory"]).resolve()
    if destination.exists():
        raise RecipeError("graph destination must not exist yet")
    outputs: set[tuple[str, ...]] = set()
    for node in spec["nodes"].values():
        skill = (path.parent / node["skill"]).resolve()
        target = (destination / node["output"]).resolve()
        syntax = PathSyntax.inspect(node["output"])
        parts = tuple(part.casefold() for part in syntax.posix.parts)
        if (
            not skill.is_file()
            or not target.is_relative_to(destination)
            or target.suffix != ".py"
            or syntax.windows.drive
            or syntax.windows.root
            or syntax.has_backslash
            or syntax.has_windows_reserved_name
            or ".." in parts
            or any(
                parts[: len(other)] == other or other[: len(parts)] == parts for other in outputs
            )
        ):
            raise RecipeError("missing graph source, escaped output or output collision")
        node["skill"] = str(skill)
        try:
            node["resources"] = {
                alias: Path(
                    os.path.relpath(
                        (path.parent / resource_relative_path(source, source=True)).resolve(),
                        skill.parent,
                    )
                ).as_posix()
                for alias, source in node.get("resources", {}).items()
            }
            add_resource_sources(skill, discover_resources(skill), node["resources"])
        except (OSError, ValueError) as exc:
            raise RecipeError(f"invalid graph resource: {exc}") from exc
        outputs.add(parts)
    build_order(spec["nodes"])
    return spec, destination


def _check_built_graph(spec: dict[str, Any], root: Path) -> None:
    entry = root / spec["nodes"][spec["entry"]]["output"]
    graph = inspect_graph(entry)
    expected = {name: (root / node["output"]).resolve() for name, node in spec["nodes"].items()}
    if set(graph.nodes) != set(expected.values()):
        raise RecipeError("generated graph omitted declared build nodes")
    for name, path in expected.items():
        declared = {str(expected[child]) for child in spec["nodes"][name].get("dependencies", [])}
        if set(graph.nodes[path]["children"]) != declared:
            raise RecipeError(f"generated dependency set differs from build specification: {name}")
    if graph.unresolved:
        raise RecipeError("generated graph has unresolved edges: " + "; ".join(graph.unresolved))


def build_graph(
    spec_path: Path,
    registry: Registry,
    *,
    compilation_timeout: int | None = None,
    max_compiler_calls: int | None = None,
    runtime_wall_seconds: int | None = None,
) -> Path:
    spec_path = spec_path.resolve()
    spec, destination = _read_spec(spec_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep failed staging trees for diagnosis; never publish a half-built catalog.
    staging = Path(tempfile.mkdtemp(prefix=".calcifier-build-", dir=destination.parent))
    built: dict[str, str] = {}
    timings = {}
    baseline_nodes: list[str] = []

    def compile_node(name: str, *, optimize: bool = True) -> PreparedArtifact:
        node = spec["nodes"][name]
        artifact = prepare_artifact(
            Path(node["skill"]),
            registry,
            output=staging / node["output"],
            skill_catalog={child: built[child] for child in node.get("dependencies", [])},
            interface_requirements=node.get("interface", ""),
            runtime_wall_seconds=runtime_wall_seconds
            if runtime_wall_seconds is not None
            else spec.get("runtime_wall_seconds"),
            compilation_timeout=(
                spec.get("compilation_timeout")
                if compilation_timeout is None
                else compilation_timeout
            ),
            max_compiler_calls=max_compiler_calls,
            resource_sources=node["resources"],
            optimize=optimize,
        )
        if artifact.suitability_note:
            logger.warning(
                "%s: owner suitability review required (compiler assessment): %s",
                name,
                artifact.suitability_note,
            )
        return artifact

    for name in build_order(spec["nodes"]):
        node = spec["nodes"][name]
        target = staging / node["output"]
        started = time.monotonic()
        try:
            artifact = compile_node(name, optimize=not baseline_nodes)
        finally:
            timings[name] = time.monotonic() - started
            (staging / "compile-timings.json").write_text(
                json.dumps(timings, indent=2) + "\n", encoding="utf-8", newline="\n"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(artifact.source, encoding="utf-8", newline="\n")
        built[name] = str(target)
        if artifact.outcome == "baseline_retained":
            baseline_nodes.append(name)
    if baseline_nodes:
        # Mixed generic and optimized interfaces cannot form a typed graph. Retain the
        # complete source workflow for every node without further model calls.
        for name in built:
            if name not in baseline_nodes:
                Path(built[name]).write_text(
                    compile_node(name, optimize=False).source, encoding="utf-8", newline="\n"
                )
        # Baseline nodes delegate their children to the agent, so typed edges do not apply.
        logger.warning("Graph compilation retained baseline workflows: %s", baseline_nodes)
    else:
        _check_built_graph(spec, staging)
    staging.rename(destination)
    return _write_catalog(spec, destination)


def _write_catalog(spec: dict[str, Any], destination: Path) -> Path:
    catalog = destination / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "version": 1,
                "recipes": {name: node["output"] for name, node in spec["nodes"].items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return catalog
