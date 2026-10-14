"""Child-call primitives must be direct calls, so composed recipes can be detected and pinned."""

from __future__ import annotations

import ast

from .diagnostics import Diagnostic
from .recipe_ast import CHILD_TARGETS, import_bindings, resolve_call_target


def child_reference_errors(module: ast.Module) -> list[Diagnostic]:
    """Keep composed entry detection reliable: child primitives are direct calls."""
    bindings = import_bindings(module)
    parents = {
        id(child): parent for parent in ast.walk(module) for child in ast.iter_child_nodes(parent)
    }
    errors = []
    for node in ast.walk(module):
        if (
            not isinstance(node, ast.Name | ast.Attribute)
            or resolve_call_target(node, bindings) not in CHILD_TARGETS
        ):
            continue
        parent = parents[id(node)]
        if not isinstance(parent, ast.Call) or parent.func is not node:
            errors.append(
                Diagnostic(
                    line=node.lineno,
                    actual=(
                        "child-call primitives must be invoked directly; aliases are unsupported"
                    ),
                )
            )
    return errors
