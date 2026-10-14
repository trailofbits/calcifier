"""Shared native-recipe admission checks for resources, parallel effects and child calls.

A packaged resource path that names a directory or a file outside the package
manifest passes parsing, Ruff, and type checking, yet fails at runtime. Only the
manifest can reveal it, so compilation, portable lint, and recipe loading all
run this check.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import PurePosixPath

from .diagnostics import Diagnostic
from .recipe_ast import CallResolver, import_resolver
from .skill_package import resource_relative_path

_RESOURCE = "calcifier.resource"
_MANIFEST_PREVIEW = 20


def runtime_conformance_errors(
    module: ast.Module, packaged: frozenset[str] | None
) -> list[Diagnostic]:
    """Return known resource, parallel-effect and child-call violations in source order.

    Args:
        module: Generated recipe module.
        packaged: The skill package manifest paths, or ``None`` when unknown.

    Returns:
        Violations sorted by line.
    """
    errors = packaged_resource_errors(module, import_resolver(module), packaged)
    from .graph_edges import child_reference_errors
    from .parallel_contract import parallel_effect_errors

    errors.extend([*parallel_effect_errors(module), *child_reference_errors(module)])
    return sorted(errors, key=lambda item: item.line or 1)


def packaged_resource_errors(
    module: ast.Module,
    resolve_call: CallResolver,
    packaged: frozenset[str] | None,
) -> list[Diagnostic]:
    """Require literal packaged paths to use resource() and name one packaged file.

    Args:
        module: Generated recipe module.
        resolve_call: Maps call targets to qualified names via module imports.
        packaged: The skill package manifest paths, or ``None`` when unknown, in
            which case only the path shape is checked.

    Returns:
        Violations in source order.
    """
    errors: list[Diagnostic] = []
    for owner, node in _owned_nodes(module):
        if isinstance(node, ast.Call) and resolve_call(node.func) == _RESOURCE:
            path = _literal_path(node)
            if path is not None and (message := _packaged_path_error(path, packaged)) is not None:
                errors.append(Diagnostic(line=node.lineno, node=owner, actual=message))
        if packaged:
            for argument in _filesystem_path_arguments(node, resolve_call):
                if (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                    and PurePosixPath(argument.value).as_posix() in packaged
                ):
                    errors.append(
                        Diagnostic(
                            line=argument.lineno,
                            node=owner,
                            actual=f"packaged file {argument.value!r} ships with the recipe; use "
                            f"resource({argument.value!r}) instead of Path() or a filesystem join, "
                            "remove any recipe input for it, and never ask the user to supply it",
                        )
                    )
    return sorted(errors, key=lambda item: item.line or 1)


def _owned_nodes(module: ast.Module) -> Iterator[tuple[str | None, ast.AST]]:
    """Yield nodes with the name of their enclosing top-level function, if any."""
    for statement in module.body:
        owner = (
            statement.name
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            else None
        )
        for node in ast.walk(statement):
            yield owner, node


def _filesystem_path_arguments(node: ast.AST, resolve_call: CallResolver) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return [node.right]
    if isinstance(node, ast.Call) and (
        resolve_call(node.func) in {"pathlib.Path", "os.path.join"}
        or isinstance(node.func, ast.Attribute)
        and node.func.attr == "joinpath"
    ):
        return node.args
    return []


def _literal_path(call: ast.Call) -> str | None:
    path = next(
        (item.value for item in call.keywords if item.arg == "rel_path"),
        call.args[0] if call.args else None,
    )
    if isinstance(path, ast.Constant) and isinstance(path.value, str):
        return path.value
    return None


def _packaged_path_error(raw_path: str, packaged: frozenset[str] | None) -> str | None:
    path = PurePosixPath(raw_path)
    if not path.parts or raw_path.endswith("/"):
        return (
            f"packaged resource path {raw_path!r} names a directory; resource() returns one "
            "packaged file"
        )
    try:
        resource_relative_path(raw_path)
    except ValueError as exc:
        return str(exc)
    if packaged is None or path.as_posix() in packaged:
        return None
    preview = sorted(packaged)[:_MANIFEST_PREVIEW]
    return (
        f"packaged resource path {raw_path!r} is not in the skill package manifest; "
        f"packaged files include {preview}"
    )
