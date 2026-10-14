"""Closed child-recipe catalog loading and dependency validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from .diagnostics import CalcifyError
from .json_config import load_json_config
from .metadata import parse_metadata_namespace, split_metadata


def load_skill_catalog(path: Path) -> dict[str, str]:
    try:
        payload = load_json_config(path, object_pairs_hook=_unique_catalog_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise CalcifyError(f"failed to load skill catalog: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"version", "recipes"}:
        raise CalcifyError("skill catalog must contain exactly version and recipes")
    if payload["version"] != 1 or not isinstance(payload["recipes"], dict):
        raise CalcifyError("skill catalog version must be 1 and recipes must be an object")
    result: dict[str, str] = {}
    for name, raw_path in payload["recipes"].items():
        if not isinstance(name, str) or not name or not isinstance(raw_path, str):
            raise CalcifyError("skill catalog entries must map non-empty names to paths")
        recipe_path = (path.parent / raw_path).resolve()
        if not recipe_path.is_file():
            raise CalcifyError(f"skill catalog recipe does not exist: {name!r} -> {recipe_path}")
        result[name] = str(recipe_path)
    _validate_catalog_graph(result)
    return result


def discover_sibling_skill_catalog(skill_path: Path) -> dict[str, str]:
    """Discover already-calcified sibling skills as closed child recipes."""
    skills_root = next(
        (parent for parent in skill_path.resolve().parents if parent.name == "skills"), None
    )
    if skills_root is None:
        return {}
    current_recipe = (skill_path.parent / ".calcifier" / "recipe.py").resolve()
    result: dict[str, str] = {}
    for recipe_path in sorted(skills_root.glob("*/.calcifier/recipe.py")):
        resolved = recipe_path.resolve()
        if resolved == current_recipe:
            continue
        name = recipe_path.parent.parent.name
        if name in result and Path(result[name]).resolve() != resolved:
            raise CalcifyError(f"automatically discovered duplicate skill name {name!r}")
        result[name] = str(resolved)
    _validate_catalog_graph(result)
    return result


def _merge_skill_catalogs(
    discovered: dict[str, str],
    explicit: dict[str, str],
) -> dict[str, str]:
    result = dict(discovered)
    for name, path in explicit.items():
        previous = result.get(name)
        if previous is not None and Path(previous).resolve() != Path(path).resolve():
            raise CalcifyError(
                f"explicit skill catalog conflicts with discovered skill {name!r}: "
                f"{previous} != {path}"
            )
        result[name] = path
    _validate_catalog_graph(result)
    return result


def _unique_catalog_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CalcifyError(f"skill catalog contains duplicate key {key!r}")
        result[key] = value
    return result


def _validate_catalog_graph(catalog: dict[str, str]) -> None:
    """Reject ambiguous paths and dependency cycles before compiler invocation."""
    path_to_name: dict[Path, str] = {}
    for name, raw_path in catalog.items():
        path = Path(raw_path).resolve()
        if previous := path_to_name.get(path):
            raise CalcifyError(
                f"skill catalog is ambiguous: {previous!r} and {name!r} map to {path}"
            )
        path_to_name[path] = name
    edges = {
        name: {
            path_to_name[dependency]
            for dependency in _catalog_dependency_paths(Path(raw_path))
            if dependency in path_to_name
        }
        for name, raw_path in catalog.items()
    }
    visiting: list[str] = []
    complete: set[str] = set()

    def visit(name: str) -> None:
        if name in complete:
            return
        if name in visiting:
            start = visiting.index(name)
            cycle = visiting[start:] + [name]
            raise CalcifyError(f"cyclic skill catalog dependency: {' -> '.join(cycle)}")
        visiting.append(name)
        for dependency in sorted(edges[name]):
            visit(dependency)
        visiting.pop()
        complete.add(name)

    for name in sorted(catalog):
        visit(name)


def _catalog_dependency_paths(recipe_path: Path) -> set[Path]:
    try:
        _, metadata = split_metadata(recipe_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CalcifyError(f"failed to inspect catalog recipe {recipe_path}: {exc}") from exc
    raw = parse_metadata_namespace(metadata or "").get("_CALCIFIER_SUBRECIPES")
    if raw is None:
        return set()
    if not isinstance(raw, list):
        raise CalcifyError(f"invalid subrecipe metadata in catalog recipe {recipe_path}")
    dependencies: set[Path] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise CalcifyError(f"invalid subrecipe metadata in catalog recipe {recipe_path}")
        entry = cast("dict[str, object]", item)
        dependency_path = entry.get("path")
        if not isinstance(dependency_path, str):
            raise CalcifyError(f"invalid subrecipe metadata in catalog recipe {recipe_path}")
        dependencies.add((recipe_path.parent / dependency_path).resolve())
    return dependencies
