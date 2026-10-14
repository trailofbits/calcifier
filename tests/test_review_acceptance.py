"""Review regressions through semantic policy and the offline compiler pipeline."""

from __future__ import annotations

import ast
from typing import Any

import pytest

from calcifier.output_annotations import output_annotation_error

pytestmark = pytest.mark.semantic_offline


def function(body: str) -> ast.FunctionDef:
    result = ast.parse(
        "def main():\n" + "\n".join("    " + line for line in body.splitlines())
    ).body[0]
    assert isinstance(result, ast.FunctionDef)
    return result


def schema_error(annotation: str, declarations: str = "") -> str | None:
    module = ast.parse(declarations + "\ndef result() -> " + annotation + ":\n    pass\n")
    result = module.body[-1]
    assert isinstance(result, ast.FunctionDef) and result.returns is not None
    bindings = {
        "Path": "pathlib.Path",
        "Iterator": "collections.abc.Iterator",
        "typing": "typing",
        "output_type": "calcifier.output_type",
        "word_count": "calcifier.word_count",
    }
    return output_annotation_error(module, [result.returns], bindings)


@pytest.mark.parametrize(
    "annotation",
    [
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "None",
        "object",
        "Path",
        "list[str]",
        "dict[str, list[int]]",
        "tuple[int, ...]",
        "set[str]",
        "frozenset[str]",
        "str | None",
        "'list[str]'",
        "typing.Any",
        "typing.Optional[int]",
        "typing.Literal['alpha', 'beta']",
        "typing.Annotated[str, word_count(1)]",
        "typing.Annotated[str, word_count(minimum=1, maximum=3)]",
    ],
)
def test_closed_schema_validation_accepts_runtime_supported_types(annotation: str) -> None:
    assert schema_error(annotation) is None


@pytest.mark.parametrize(
    "annotation,expected",
    [("None", type(None)), ("tuple[int, ...]", tuple[int, ...]), ("str | None", str | None)],
)
def test_static_schema_adapter_receives_the_declared_type(
    monkeypatch, annotation: str, expected: Any
) -> None:
    from calcifier import outputs

    observed = []
    real_schema_for = outputs.schema_for

    def observe(annotation: Any) -> dict[str, Any]:
        observed.append(annotation)
        return real_schema_for(annotation)

    monkeypatch.setattr(outputs, "schema_for", observe)
    assert schema_error(annotation) is None
    assert observed == [expected]


@pytest.mark.parametrize(
    "annotation",
    [
        "Iterator[str]",
        "list[Iterator[str]]",
        "Unknown",
        "1",
        "'['",
        "typing.Optional[str, int]",
        "typing.Annotated[str, dynamic(1)]",
        "typing.Annotated[str, dynamic()]",
        "typing.Annotated[str, word_count(-1)]",
        "typing.Literal[dynamic()]",
        "factory()",
    ],
)
def test_imported_names_do_not_prove_loadable_output_schemas(annotation: str) -> None:
    assert schema_error(annotation) is not None


def test_closed_output_records_validate_nested_fields_and_defaults() -> None:
    declarations = """@output_type
class Child:
    '''A neutral nested output record.'''
    value: str
@output_type()
class Parent:
    child: Child
    maybe: str | None = None
"""
    assert schema_error("list[Parent]", declarations) is None
    assert schema_error("tuple[Parent, Child]", declarations) is None
    assert (
        schema_error("Parent", declarations.replace("value: str", "value: Iterator[str]"))
        is not None
    )


@pytest.mark.parametrize(
    "declarations",
    [
        "class Result:\n    value: str",
        "@output_type\nclass Result:\n    value: Result",
        "@output_type\nclass Result:\n    value: str = dangerous()",
        "@output_type\nclass Result:\n    values: list[str] = []",
        "@output_type\nclass Result:\n    def model_dump(self):\n        pass",
        "@output_type\nclass Result:\n    model_dump_json: str",
    ],
)
def test_unloadable_records_are_rejected_without_evaluating_user_code(declarations: str) -> None:
    assert schema_error("Result", declarations) is not None
