"""The output-annotation check: a step may return only a closed, schema-describable type."""

from __future__ import annotations

import ast
import functools
import typing
from collections.abc import Mapping
from types import MappingProxyType

from .outputs import BUILTIN_OUTPUT_TYPES
from .recipe_ast import assignment_name


@functools.cache
def known_output_annotations() -> Mapping[str, typing.Any]:
    """Return the closed name table step output annotations may reference."""
    from datetime import date, datetime
    from decimal import Decimal
    from pathlib import Path

    import calcifier

    return MappingProxyType(
        {
            **BUILTIN_OUTPUT_TYPES,
            "pathlib.Path": Path,
            "datetime.date": date,
            "datetime.datetime": datetime,
            "decimal.Decimal": Decimal,
            **{
                f"typing.{name}": getattr(typing, name)
                for name in (
                    "Any",
                    "Annotated",
                    "Literal",
                    "Union",
                    "Optional",
                    "List",
                    "Dict",
                    "Tuple",
                    "Set",
                    "FrozenSet",
                    "Sequence",
                    "Mapping",
                )
            },
            **{
                f"calcifier.{name}": getattr(calcifier, name)
                for name in ("CommandResult", "PreparedContext")
            },
        }
    )


def output_annotation_error(  # noqa: C901 - wraps a recursive nested annotation resolver
    module: ast.Module, annotations: list[ast.expr], bindings: Mapping[str, str]
) -> str | None:
    """Validate schemas using runtime's adapter without executing generated code."""
    from .outputs import OutputTypeError, output_type, schema_for, word_count

    known = dict(known_output_annotations())
    classes = {node.name: node for node in module.body if isinstance(node, ast.ClassDef)}
    aliases = {
        name: node.value
        for node in module.body
        if isinstance(node, ast.Assign | ast.AnnAssign)
        and node.value is not None
        and (name := assignment_name(node)) is not None
    }
    resolving: set[str] = set()

    def target(node: ast.AST) -> str:
        text = ast.unparse(node)
        root, separator, tail = text.partition(".")
        return bindings.get(root, root) + (separator + tail)

    def resolve(  # noqa: C901 - recursive case per output-annotation AST shape
        node: ast.AST,
    ) -> typing.Any:
        if isinstance(node, ast.Constant):
            if node.value is None:
                return type(None)
            if node.value is Ellipsis:
                return Ellipsis
            if isinstance(node.value, str):
                return resolve(ast.parse(node.value, mode="eval").body)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return resolve(node.left) | resolve(node.right)
        if isinstance(node, ast.Subscript):
            constructor = resolve(node.value)
            items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            if constructor is typing.Literal:
                args = tuple(ast.literal_eval(item) for item in items)
            elif constructor is typing.Annotated:
                args = (resolve(items[0]), *(metadata(item) for item in items[1:]))
            else:
                args = tuple(resolve(item) for item in items)
            return constructor[args[0] if len(args) == 1 else args]
        name = target(node)
        if name in aliases and name not in known and name not in resolving:
            resolving.add(name)
            known[name] = resolve(aliases[name])
            resolving.remove(name)
        if name in classes and name not in known and name not in resolving:
            definition = classes[name]
            if not any(
                target(item.func if isinstance(item, ast.Call) else item) == "calcifier.output_type"
                for item in definition.decorator_list
            ):
                raise ValueError(f"output type {name!r} must use @output_type")
            resolving.add(name)
            fields: dict[str, typing.Any] = {}
            defaults: dict[str, typing.Any] = {}
            for statement in definition.body:
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                    field_name = statement.target.id
                    fields[field_name] = resolve(statement.annotation)
                    if statement.value is not None:
                        defaults[field_name] = ast.literal_eval(statement.value)
                elif isinstance(statement, ast.FunctionDef) and statement.name in {
                    "model_dump",
                    "model_dump_json",
                }:
                    raise ValueError("@output_type reserves serialization methods")
            known[name] = output_type(type(name, (), {"__annotations__": fields, **defaults}))
            resolving.remove(name)
        if name in known:
            return known[name]
        raise ValueError(f"unsupported output annotation {ast.unparse(node)!r}")

    def metadata(node: ast.AST) -> typing.Any:
        if isinstance(node, ast.Call) and target(node.func) == "calcifier.word_count":
            return word_count(
                *(ast.literal_eval(item) for item in node.args),
                **{
                    item.arg: ast.literal_eval(item.value)
                    for item in node.keywords
                    if item.arg is not None
                },
            )
        raise ValueError("unsupported output annotation metadata")

    try:
        for annotation in annotations:
            schema_for(resolve(annotation))
    except (OutputTypeError, TypeError, ValueError, SyntaxError, KeyError) as exc:
        return f"unsupported step output schema: {exc}"
    return None
