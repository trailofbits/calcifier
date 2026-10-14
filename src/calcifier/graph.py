"""Pinned recipe closure: every child a composed recipe calls, and what each one needs."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .exceptions import RecipeError
from .hashing import sha256_hex
from .metadata import parse_metadata_namespace, split_metadata
from .recipe_ast import (
    called_subrecipes,
    import_bindings,
    import_resolver,
    subrecipe_calls,
    tool_provider_capabilities,
)


@dataclass(frozen=True)
class Graph:
    entry: Path
    nodes: dict[Path, dict[str, Any]]
    unresolved: tuple[str, ...]
    provider_capabilities: tuple[str, ...] = ()

    @property
    def agents(self) -> tuple[str, ...]:
        return tuple(sorted({role for n in self.nodes.values() for role in n["agents"]}))

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(sorted({name for n in self.nodes.values() for name in n["capabilities"]}))


def _declarations(module: ast.Module) -> dict[str, list[str]]:
    """Read the literal ``agents=`` and ``capabilities=`` of a node's single ``@recipe``."""
    resolve = import_resolver(module)
    decorators = [
        decorator
        for node in module.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        for decorator in node.decorator_list
        if resolve(decorator.func if isinstance(decorator, ast.Call) else decorator)
        == "calcifier.recipe"
    ]
    if len(decorators) != 1:
        raise RecipeError("graph nodes must expose exactly one @recipe")
    keywords = (
        {k.arg: k.value for k in decorators[0].keywords}
        if isinstance(decorators[0], ast.Call)
        else {}
    )
    return {
        name: list(ast.literal_eval(keywords[name])) if name in keywords else []
        for name in ("agents", "capabilities")
    }


def inspect_graph(entry: Path, *, entry_bytes: bytes | None = None) -> Graph:
    """Walk the pinned child closure of ``entry``.

    Raises:
        RecipeError: A dependency is cyclic or no longer matches its pinned hash.
    """
    nodes: dict[Path, dict[str, Any]] = {}
    active: set[Path] = set()
    unresolved: list[str] = []
    provider_capabilities: set[str] = set()

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in active:
            raise RecipeError(f"cyclic recipe dependency: {path}")
        if path in nodes:
            return
        active.add(path)
        data = (
            entry_bytes
            if path == entry.resolve() and entry_bytes is not None
            else path.read_bytes()
        )
        module = ast.parse(data)
        _, tail = split_metadata(data.decode())
        metadata = cast("dict[str, Any]", parse_metadata_namespace(tail or ""))
        bindings = import_bindings(module)
        child_calls = subrecipe_calls(module, bindings)
        calls = [ast.unparse(call) for call in child_calls]
        diagnostics, called = called_subrecipes(path, module, child_calls, bindings)
        unresolved.extend(f"{path}: {item.actual}" for item in diagnostics)
        node: dict[str, Any] = {**_declarations(module), "calls": calls, "children": []}
        provider_capabilities.update(
            tool_provider_capabilities(module, tuple(node["capabilities"]))
        )
        dependencies = metadata.get("_CALCIFIER_SUBRECIPES", [])
        if calls and not dependencies:
            unresolved.append(f"{path}: child calls lack pinned dependencies")
        for dependency in dependencies:
            child = (path.parent / dependency["path"]).resolve()
            if sha256_hex(child.read_bytes()) != dependency["sha256"]:
                raise RecipeError(f"dependency drift: {child}")
            node["children"].append(str(child))
            visit(child)
        if called != {Path(child) for child in node["children"]}:
            unresolved.append(f"{path}: called children differ from pinned graph")
        nodes[path] = node
        active.remove(path)

    visit(entry)
    return Graph(entry.resolve(), nodes, tuple(unresolved), tuple(sorted(provider_capabilities)))
