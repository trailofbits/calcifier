"""Tests for outputs.py.

The schema-generation layer delegates to ``pydantic.TypeAdapter``, so these
tests exercise behaviour, not exact schema-shape equality. We check that:

* schemas validate the right values via ``Draft202012Validator``;
* ``parse_response`` extracts JSON from various LLM-shaped strings;
* the round-trip property holds for our supported annotation set.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Annotated, Literal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jsonschema import Draft202012Validator

from calcifier.outputs import (
    OutputParseError,
    OutputTypeError,
    output_type,
    parse_response,
    schema_for,
    strict_schema_for,
    word_count,
)

# -- helpers ------------------------------------------------------------------


def _to_plain(value):  # noqa: ANN001 — recursive
    if dataclasses.is_dataclass(value):
        return {f.name: _to_plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    return value


def _validates(value, schema):
    return Draft202012Validator(schema).is_valid(value)


# -- example dataclasses ------------------------------------------------------


@output_type
class Finding:
    file: str
    line: int
    severity: Literal["low", "medium", "high"]
    description: str


@output_type
class Verdict:
    approved: bool
    reasoning: str


@output_type
class Review:
    finding: Finding
    verdicts: list[Verdict]
    note: str | None = None


@output_type
class DefaultChild:
    note: str | None = None


@output_type
class DefaultParent:
    a: int
    b: str = "x"
    child: DefaultChild | None = None


@output_type
class WordBoundReport:
    message: Annotated[str, word_count(3, 5)]


# -- schema_for: behavioural checks -----------------------------------------


@pytest.mark.parametrize(
    "annotation, value",
    [
        (str, "hello"),
        (int, 42),
        (float, 3.14),
        (bool, True),
    ],
)
def test_schema_primitives_validate(annotation, value):
    assert _validates(value, schema_for(annotation))


def test_word_count_metadata_rejects_nonconforming_agent_output():
    accepted = parse_response('{"message": "one two three"}', WordBoundReport)

    assert accepted.message == "one two three"
    with pytest.raises(OutputParseError, match="at least 3 words"):
        parse_response('{"message": "too short"}', WordBoundReport)
    with pytest.raises(OutputParseError, match="at most 5 words"):
        parse_response(
            '{"message": "one two three four five six"}',
            WordBoundReport,
        )


def test_schema_list_of_primitives():
    schema = schema_for(list[str])
    assert _validates(["a", "b"], schema)
    assert not _validates([1, 2], schema)


def test_schema_dict_str_to_int():
    schema = schema_for(dict[str, int])
    assert _validates({"a": 1}, schema)
    assert not _validates({"a": "x"}, schema)


def test_schema_optional_string_accepts_null_and_string():
    schema = schema_for(str | None)
    assert _validates(None, schema)
    assert _validates("hi", schema)
    assert not _validates(42, schema)


def test_schema_literal_enum():
    schema = schema_for(Literal["a", "b", "c"])
    assert _validates("a", schema)
    assert not _validates("z", schema)


def test_schema_dataclass_validates_correct_payload():
    schema = schema_for(Finding)
    assert _validates({"file": "x", "line": 1, "severity": "low", "description": "d"}, schema)
    # Wrong severity literal.
    assert not _validates(
        {"file": "x", "line": 1, "severity": "critical", "description": "d"}, schema
    )
    # Missing required field.
    assert not _validates({"file": "x", "line": 1, "severity": "low"}, schema)


def test_schema_dataclass_with_default_field_not_required():
    @output_type
    class WithDefault:
        a: int
        b: str = "x"
        items: list[int] = field(default_factory=list)

    schema = schema_for(WithDefault)
    # Only the no-default field should be required by the schema.
    assert "a" in schema.get("required", [])
    assert "b" not in schema.get("required", [])


def test_strict_schema_requires_defaulted_properties_recursively():
    schema = strict_schema_for(DefaultParent)

    assert schema["required"] == ["a", "b", "child"]
    assert schema["$defs"]["DefaultChild"]["required"] == ["note"]
    assert '"default"' not in json.dumps(schema)
    assert _validates({"a": 1, "b": "x", "child": None}, schema)
    assert not _validates({"a": 1}, schema)
    assert not _validates({"a": 1, "b": "x", "child": {}}, schema)


def test_strict_schema_translates_fixed_tuple_to_provider_array_subset():
    @output_type
    class TupleResult:
        key: tuple[int, str]
        pairs: list[tuple[int, int]]

    schema = strict_schema_for(TupleResult)
    key = schema["properties"]["key"]
    pair = schema["properties"]["pairs"]["items"]

    assert "prefixItems" not in json.dumps(schema)
    assert key["minItems"] == 2
    assert key["maxItems"] == 2
    assert key["items"] == {"anyOf": [{"type": "integer"}, {"type": "string"}]}
    assert pair["items"] == {"type": "integer"}


def test_host_parser_retains_positional_tuple_validation_after_schema_translation():
    @output_type
    class TupleResult:
        key: tuple[int, str]

    with pytest.raises(OutputParseError, match="schema mismatch"):
        parse_response('{"key":["alpha",1]}', TupleResult)


def test_schema_rejects_unsupported_class():
    class NotADataclass:
        pass

    with pytest.raises(OutputTypeError):
        schema_for(NotADataclass)


def test_schema_rejects_open_standard_dataclass():
    @dataclass
    class OpenResult:
        ok: bool

    with pytest.raises(OutputTypeError, match="must use @output_type"):
        schema_for(OpenResult)


def test_schema_output_type_forbids_extra_properties():
    schema = schema_for(Verdict)
    assert schema["additionalProperties"] is False
    assert not _validates({"approved": True, "reasoning": "ok", "error": "failed"}, schema)


def test_output_type_serializes_nested_records_without_pydantic_imports():
    review = Review(
        finding=Finding(
            file="src/example.py",
            line=7,
            severity="high",
            description="example",
        ),
        verdicts=[Verdict(approved=True, reasoning="confirmed")],
    )

    expected = {
        "finding": {
            "file": "src/example.py",
            "line": 7,
            "severity": "high",
            "description": "example",
        },
        "verdicts": [{"approved": True, "reasoning": "confirmed"}],
        "note": None,
    }
    assert review.model_dump() == expected  # ty: ignore[unresolved-attribute]
    assert json.loads(review.model_dump_json(indent=2)) == expected  # ty: ignore[unresolved-attribute]


def test_output_type_reserves_serialization_methods():
    with pytest.raises(OutputTypeError, match="reserves serialization method.*model_dump"):

        @output_type
        class ConflictingOutput:
            value: str

            def model_dump(self) -> dict[str, str]:
                return {"value": self.value}

    with pytest.raises(OutputTypeError, match="reserves serialization method.*model_dump_json"):

        @output_type
        class AnnotationConflict:
            model_dump_json: str


# -- parse_response: examples -------------------------------------------------


def test_parse_primitive_int():
    assert parse_response("42", int) == 42


@pytest.mark.parametrize(
    "raw, annotation",
    [
        ('"3"', int),
        ('"true"', bool),
        ("1", bool),
        ("3.0", int),
        ("true", int),
        ("3", str),
    ],
)
def test_parse_rejects_primitive_coercion(raw, annotation):
    with pytest.raises(OutputParseError, match="schema mismatch"):
        parse_response(raw, annotation)


def test_parse_strict_json_allows_integer_for_float():
    # Pydantic treats both JSON integer and number tokens as valid floats even
    # in strict JSON mode; this is numeric compatibility, not string coercion.
    assert parse_response("3", float) == 3.0


def test_parse_primitive_with_fences():
    assert parse_response('```json\n{"a": 1}\n```', dict[str, int]) == {"a": 1}


def test_parse_top_level_array():
    raw = json.dumps([{"file": "x.py", "line": 1, "severity": "low", "description": "d"}])
    out = parse_response(raw, list[Finding])
    assert out == [Finding(file="x.py", line=1, severity="low", description="d")]


def test_parse_dataclass():
    raw = json.dumps({"approved": True, "reasoning": "looks fine"})
    out = parse_response(raw, Verdict)
    assert out == Verdict(approved=True, reasoning="looks fine")


def test_parse_rejects_unknown_output_field():
    raw = json.dumps({"approved": True, "reasoning": "looks fine", "error": "failed"})
    with pytest.raises(OutputParseError, match="Unexpected keyword argument"):
        parse_response(raw, Verdict)


def test_parse_rejects_unknown_nested_output_field():
    raw = json.dumps(
        {
            "finding": {
                "file": "x",
                "line": 2,
                "severity": "high",
                "description": "d",
                "error": "failed",
            },
            "verdicts": [],
        }
    )
    with pytest.raises(OutputParseError, match="Unexpected keyword argument"):
        parse_response(raw, Review)


def test_parse_dataclass_with_optional():
    raw = json.dumps(
        {
            "finding": {"file": "x", "line": 2, "severity": "high", "description": "d"},
            "verdicts": [{"approved": False, "reasoning": "no"}],
            "note": None,
        }
    )
    out = parse_response(raw, Review)
    assert out.note is None
    assert out.finding.severity == "high"


def test_parse_extracts_last_json_block_in_prose():
    text = (
        'Sure — here is some prose. {"approved": false, "reasoning": "first"}\n'
        "Wait, on reflection, the right answer is:\n"
        '{"approved": true, "reasoning": "final"}'
    )
    out = parse_response(text, Verdict)
    assert out.reasoning == "final"


def test_parse_rejects_missing_required_field():
    raw = json.dumps({"approved": True})
    with pytest.raises(OutputParseError, match="schema mismatch"):
        parse_response(raw, Verdict)


def test_parse_invalid_literal_value():
    raw = json.dumps({"file": "x", "line": 1, "severity": "critical", "description": "d"})
    with pytest.raises(OutputParseError, match="schema mismatch"):
        parse_response(raw, Finding)


def test_parse_no_json_at_all():
    with pytest.raises(OutputParseError, match="no JSON block"):
        parse_response("Sorry, I can't help with that.", Verdict)


# -- property-based round-trip ------------------------------------------------


@st.composite
def findings(draw):
    return Finding(
        file=draw(st.text(min_size=1, max_size=20)),
        line=draw(st.integers(min_value=0, max_value=10_000)),
        severity=draw(st.sampled_from(["low", "medium", "high"])),
        description=draw(st.text(min_size=0, max_size=80)),
    )


@st.composite
def verdicts(draw):
    return Verdict(
        approved=draw(st.booleans()),
        reasoning=draw(st.text(min_size=0, max_size=80)),
    )


@st.composite
def reviews(draw):
    return Review(
        finding=draw(findings()),
        verdicts=draw(st.lists(verdicts(), min_size=0, max_size=4)),
        note=draw(st.one_of(st.none(), st.text(min_size=0, max_size=40))),
    )


@settings(max_examples=120, deadline=None)
@given(findings())
def test_roundtrip_finding(value):
    raw = json.dumps(_to_plain(value))
    parsed = parse_response(raw, Finding)
    assert parsed == value


@settings(max_examples=120, deadline=None)
@given(st.lists(findings(), min_size=0, max_size=8))
def test_roundtrip_list_of_findings(values):
    raw = json.dumps([_to_plain(v) for v in values])
    parsed = parse_response(raw, list[Finding])
    assert parsed == values


@settings(max_examples=80, deadline=None)
@given(reviews())
def test_roundtrip_review(value):
    raw = json.dumps(_to_plain(value))
    parsed = parse_response(raw, Review)
    assert parsed == value


@given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=8))
def test_roundtrip_dict_str_int(value):
    raw = json.dumps(value)
    assert parse_response(raw, dict[str, int]) == value


@given(st.one_of(st.none(), st.text(min_size=0, max_size=30)))
def test_roundtrip_optional_string(value):
    raw = json.dumps(value)
    assert parse_response(raw, str | None) == value


@given(st.one_of(st.text(), st.booleans()))
def test_strict_float_rejects_non_numeric_json_primitives(value):
    with pytest.raises(OutputParseError, match="schema mismatch"):
        parse_response(json.dumps(value), float)


def test_type_adapters_are_reused_across_model_dump_calls() -> None:
    from calcifier import outputs

    @output_type
    class Cached:
        value: int

    outputs._cached_adapter.cache_clear()
    Cached(1).model_dump()  # ty: ignore[unresolved-attribute] - injected by @output_type
    Cached(2).model_dump_json()  # ty: ignore[unresolved-attribute] - injected by @output_type

    info = outputs._cached_adapter.cache_info()
    assert (info.hits, info.misses) == (1, 1)


def test_unhashable_annotations_still_build_adapters() -> None:
    from calcifier import outputs

    class Unhashable:
        __hash__ = None

        def __repr__(self) -> str:
            return "Unhashable()"

    with pytest.raises(OutputTypeError, match="unsupported annotation Unhashable"):
        outputs._adapter(Unhashable())


def test_native_tuple_schema_deduplicates_semantically_identical_item_schemas():
    from calcifier.outputs import _normalize_fixed_tuples

    schema = {
        "type": "array",
        "prefixItems": [
            {"type": "string", "minLength": 1},
            {"minLength": 1, "type": "string"},
        ],
        "minItems": 2,
        "maxItems": 2,
    }
    _normalize_fixed_tuples(schema)
    assert schema == {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "minItems": 2,
        "maxItems": 2,
    }
