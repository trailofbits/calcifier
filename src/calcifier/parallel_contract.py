"""Reject provable shared effects in parallel callbacks, including helper calls."""

from __future__ import annotations

import ast

from .diagnostics import Diagnostic
from .recipe_ast import import_resolver, primitive_targets

_SHARED = primitive_targets(
    "write_text", "write_json", "ensure_directory", "verify_artifact", "record_side_effect"
)


def parallel_effect_errors(module: ast.Module) -> list[Diagnostic]:
    resolve = import_resolver(module)
    functions = {
        node.name: node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }

    def effects(node: ast.AST, seen: set[str]) -> list[str]:
        found = []
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            target = resolve(call.func)
            if target in _SHARED or (
                target in primitive_targets("run_command")
                and not any(
                    k.arg == "mutating"
                    and isinstance(k.value, ast.Constant)
                    and k.value.value is False
                    for k in call.keywords
                )
            ):
                found.append(f"{target} at line {call.lineno}")
            if (
                isinstance(call.func, ast.Name)
                and call.func.id in functions
                and call.func.id not in seen
            ):
                found.extend(effects(functions[call.func.id], seen | {call.func.id}))
        return found

    errors = []
    for call in (n for n in ast.walk(module) if isinstance(n, ast.Call)):
        if resolve(call.func) not in primitive_targets("parallel"):
            continue
        callback = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "fn"),
            call.args[0] if call.args else None,
        )
        target = functions.get(callback.id) if isinstance(callback, ast.Name) else callback
        if target is not None and (found := effects(target, set())):
            errors.append(
                Diagnostic(
                    line=call.lineno,
                    actual="parallel callback mutates shared state: " + ", ".join(found),
                )
            )
    return errors
