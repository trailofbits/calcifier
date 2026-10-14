from __future__ import annotations

import ast
import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

import calcifier.recipe_draft as recipe_draft_module
from calcifier.recipe_draft import (
    RecipeDraftError,
    check_recipe_syntax,
    draft_response,
    parse_recipe_draft,
    recipe_draft_prompt_contract,
)

pytestmark = pytest.mark.semantic_offline


_PROGRAMS = (
    "x = 1\n",
    "x: int = 1\nx += 2\n",
    "x = {'a': [1, 2], 'b': {3}}\ny = (x['a'][0], *x['a'])\n",
    "x = [item * 2 for item in range(5) if item != 3]\n",
    "x = {key: value for key, value in [('a', 1)]}\n",
    "x = {item for item in range(3)}\ny = (item for item in range(3))\n",
    "if value and not other:\n    result = value or other\nelse:\n    result = None\n",
    "for item in values:\n    if item in allowed:\n        continue\n    break\n",
    "while ready:\n    ready = False\n",
    "try:\n    run()\nexcept ValueError as exc:\n    raise RuntimeError(str(exc)) from exc\n",
    "with open('x') as stream:\n    value = stream.read()\n",
    "async def run(items):\n    async for item in items:\n        await item()\n",
    "def run(value: str = 'x', *, flag: bool = False) -> str:\n    return f'{value}:{flag!r}'\n",
    "class Record:\n    value: str\n",
    "from calcifier import recipe as define_recipe\nimport json as json_module\n",
    "value = left & right\nother = left | right\nremaining = left - right\n",
    "value = lower < item <= upper\n",
    "value = data[1:5]\n",
    "callback = lambda item: item.name\n",
    "def required(*, value):\n    return value\n",
    "mapping = {**other}\n",
    "items: tuple[str, ...] = ()\n",
)


@pytest.mark.parametrize("source", _PROGRAMS)
def test_an_accepted_draft_is_returned_exactly_as_written(source: str) -> None:
    assert parse_recipe_draft(draft_response(source)) == source


def test_wrapped_literals_comments_and_spacing_survive() -> None:
    # Re-rendering would join these literals into one line past the destination's limit.
    source = 'PROMPT = (\n    "first segment "\n    "second segment"  # note\n)\nx   =  1\n'

    assert parse_recipe_draft(draft_response(source)) == source


def test_a_missing_final_newline_is_added() -> None:
    assert parse_recipe_draft(draft_response("x = 1")) == "x = 1\n"


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ("x: int = 'a'  # ty: ignore\n", "`# ty: ignore` is not allowed"),
        ("x: int = 'a'  #ty:ignore[invalid-assignment]\n", "`# ty: ignore` is not allowed"),
        (
            "from typing import no_type_check\n\n@no_type_check\ndef f() -> int:\n    return 'a'\n",
            "`no_type_check` is not allowed",
        ),
        (
            "import typing\n\n@typing.no_type_check\ndef f() -> int:\n    return 'a'\n",
            "`no_type_check` is not allowed",
        ),
    ],
)
def test_suppressions_ty_configuration_cannot_disable_are_rejected(
    source: str, reason: str
) -> None:
    with pytest.raises(RecipeDraftError, match=reason):
        parse_recipe_draft(draft_response(source))


def test_directive_text_inside_a_string_is_not_a_suppression() -> None:
    source = "NOTE = '# ty: ignore is text here'\n"

    assert parse_recipe_draft(draft_response(source)) == source


def test_draft_response_is_the_provider_wire_format() -> None:
    assert json.loads(draft_response("x = 1\n")) == {"version": "1", "source": "x = 1\n"}
    assert draft_response("label = 'Zählt'\n") == (
        '{"version": "1", "source": "label = \'Zählt\'\\n"}'
    )


def test_recipe_draft_prompt_requires_host_parsing_and_closed_nodes() -> None:
    contract = recipe_draft_prompt_contract()

    assert "RecipeDraft JSON object" in contract
    assert "FINAL structured response" in contract
    assert "Do not run Python-to-AST conversion commands" in contract
    assert "The host parses your source and checks it" in contract
    assert "other syntax remain forbidden" in contract
    for kind in ("Module", "FunctionDef", "ListComp", "FormattedValue"):
        assert kind in contract


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_recipe_draft_preserves_repeated_text_and_unicode(newline: str) -> None:
    source = newline.join(["# café", 'first = "repeated"', 'second = "repeated"', ""])
    assert parse_recipe_draft(draft_response(source)) == source


@pytest.mark.parametrize(
    "response",
    [
        "",
        '{"version":"1","source":',
        '{"version":"1","source":""}',
        '{"version":"1","source":"   "}',
        '{"version":"1","source":23}',
        '{"version":"2","source":"pass"}',
        '{"version":"1","source":"pass","extra":true}',
        'prefix {"version":"1","source":"pass"}',
        '{"version":"1","source":"pass"} trailing',
        '[{"version":"1","source":"pass"}]',
        '{"version":"1","source":"def broken(:"}',
        '{"version":"1","module":{"kind":"Module","fields":[]}}',
    ],
)
def test_recipe_draft_rejects_empty_incomplete_or_open_envelopes(response: str) -> None:
    with pytest.raises(RecipeDraftError):
        parse_recipe_draft(response)


@pytest.mark.parametrize(
    ("source", "kind"),
    [("del value", "Delete"), ("match x:\n    case 1: pass", "Match"), ("global x", "Global")],
)
def test_recipe_draft_cannot_expand_closed_language(source: str, kind: str) -> None:
    with pytest.raises(RecipeDraftError, match=f"unsupported Python syntax.*: {kind}"):
        parse_recipe_draft(draft_response(source))


@pytest.mark.parametrize("source", ["type Item = int", 'x = "\\d"'])
def test_recipe_draft_checks_target_syntax_and_warnings_before_unparse(source: str) -> None:
    with pytest.raises(RecipeDraftError, match="invalid or incomplete"):
        parse_recipe_draft(draft_response(source), python_version="3.11")


@pytest.mark.parametrize("limit", ["_MAX_DRAFT_BYTES", "_MAX_NODES", "_MAX_DEPTH"])
def test_recipe_draft_is_bounded(limit: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recipe_draft_module, limit, 1)
    with pytest.raises(RecipeDraftError, match="exceeds"):
        parse_recipe_draft(draft_response("value = 1"))


def test_recipe_draft_accepts_exact_size_node_and_depth_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = draft_response("value = 1")
    monkeypatch.setattr(recipe_draft_module, "_MAX_DRAFT_BYTES", len(payload.encode()))
    # Module -> Assign -> Name -> Store; plus the sibling Constant.
    monkeypatch.setattr(recipe_draft_module, "_MAX_NODES", 5)
    monkeypatch.setattr(recipe_draft_module, "_MAX_DEPTH", 3)
    assert parse_recipe_draft(payload) == "value = 1\n"


def test_syntax_check_walks_every_node_of_an_already_parsed_module() -> None:
    check_recipe_syntax(ast.parse("def run():\n    return [item for item in range(3)]\n"))
    with pytest.raises(RecipeDraftError, match="Delete"):
        check_recipe_syntax(ast.parse("def run(value):\n    del value\n"))


@given(value=st.text())
def test_recipe_draft_string_transport_round_trip(value: str) -> None:
    rendered = parse_recipe_draft(draft_response(f"value = {value!r}"))
    statement = ast.parse(rendered).body[0]
    assert isinstance(statement, ast.Assign)
    assert ast.literal_eval(statement.value) == value


@given(
    value=st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(),
    )
)
def test_constant_values_round_trip(value: object) -> None:
    rendered = parse_recipe_draft(draft_response(f"value = {value!r}\n"))

    statement = ast.parse(rendered).body[0]
    assert isinstance(statement, ast.Assign)
    assert ast.literal_eval(statement.value) == value
