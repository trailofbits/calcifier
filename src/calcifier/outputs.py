"""Derive output schemas and parse agent answers strictly through Pydantic.

JSON extraction tolerates prose and fences, choosing the last balanced block.
Unsupported types raise OutputTypeError; invalid answers raise OutputParseError.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import re
from collections.abc import Callable, Iterator
from contextlib import suppress
from typing import Any, Literal, cast, dataclass_transform, get_args, get_type_hints, overload

from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JsonSchemaValidationError
from pydantic import (
    ConfigDict,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    TypeAdapter,
    ValidationError,
)
from pydantic.errors import PydanticSchemaGenerationError, PydanticUserError
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

from .exceptions import CalcifierError, ErrorFamily


class OutputTypeError(CalcifierError, TypeError):
    """Raised when an annotation is not supported by ``TypeAdapter``."""

    family = ErrorFamily.SETUP


class OutputParseError(CalcifierError, ValueError):
    """Raised when an LLM response cannot be parsed into the annotation."""


_NOTHING = object()
_OUTPUT_TYPE_MARKER = "__calcifier_output_type__"


@dataclasses.dataclass(frozen=True)
class _WordCount:
    minimum: int
    maximum: int | None

    def validate(self, value: str) -> str:
        count = len(value.split())
        if count < self.minimum:
            raise ValueError(f"must contain at least {self.minimum} words")
        if self.maximum is not None and count > self.maximum:
            raise ValueError(f"must contain at most {self.maximum} words")
        return value

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(self.validate, handler(source_type))

    def __get_pydantic_json_schema__(
        self, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        result = handler(schema)
        rule = f"At least {self.minimum} words"
        if self.maximum is not None:
            rule += f"; at most {self.maximum} words"
        rule += "."
        description = result.get("description")
        result["description"] = f"{description} {rule}" if isinstance(description, str) else rule
        return result


def word_count(minimum: int, maximum: int | None = None) -> _WordCount:
    """Return ``Annotated`` metadata that validates a string's word count.

    The provider sees the bound in the JSON schema description before its first
    attempt, and host-side validation still enforces the exact word count.
    """
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
        raise ValueError("word_count() minimum must be a non-negative integer")
    if maximum is not None and (
        not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < minimum
    ):
        raise ValueError("word_count() maximum must be at least the minimum")

    return _WordCount(minimum, maximum)


@overload
def output_type[T](cls: type[T], *, frozen: bool = False) -> type[T]: ...


@overload
def output_type[T](cls: None = None, *, frozen: bool = False) -> Callable[[type[T]], type[T]]: ...


@dataclass_transform()
def output_type[T](
    cls: type[T] | None = None, *, frozen: bool = False
) -> type[T] | Callable[[type[T]], type[T]]:
    """Declare a closed, serializable dataclass-shaped structured output type.

    Standard-library dataclasses are open records under Pydantic: unexpected
    keys are silently discarded even when value coercion is strict.  Calcifier
    output records deliberately forbid extras so an agent cannot smuggle a
    failure indicator beside apparently successful required fields. Decorated
    records also expose ``model_dump()`` and ``model_dump_json()`` so portable
    recipes can serialize structured agent output without importing Pydantic.
    """

    def decorate(target: type[T]) -> type[T]:
        if dataclasses.is_dataclass(target):
            raise OutputTypeError("@output_type must replace @dataclass, not wrap it")
        reserved = {"model_dump", "model_dump_json"} & (
            target.__dict__.keys() | getattr(target, "__annotations__", {}).keys()
        )
        if reserved:
            names = ", ".join(sorted(reserved))
            raise OutputTypeError(f"@output_type reserves serialization method(s): {names}")

        def model_dump(
            self: object,
            *,
            mode: Literal["python", "json"] = "python",
        ) -> dict[str, Any]:
            value = _adapter(type(self)).dump_python(self, mode=mode)
            return cast("dict[str, Any]", value)

        def model_dump_json(self: object, *, indent: int | None = None) -> str:
            return _adapter(type(self)).dump_json(self, indent=indent).decode()

        # B010: setattr rather than attribute assignment because `target` is a
        # type[T] that declares none of these names, so ty rejects `target.x = ...`.
        setattr(target, "model_dump", model_dump)  # noqa: B010
        setattr(target, "model_dump_json", model_dump_json)  # noqa: B010
        setattr(target, "__pydantic_config__", ConfigDict(extra="forbid"))  # noqa: B010
        result = (
            dataclasses.dataclass(target, frozen=True) if frozen else dataclasses.dataclass(target)
        )
        setattr(result, _OUTPUT_TYPE_MARKER, True)
        return result

    return decorate if cls is None else decorate(cls)


def _require_closed_output_types(annotation: Any, seen: dict[int, Any] | None = None) -> None:
    """Reject open dataclasses anywhere inside an output annotation."""
    # Strong references prevent temporary aliases from reusing a visited ID.
    visited = {} if seen is None else seen
    marker = id(annotation)
    if marker in visited:
        return
    visited[marker] = annotation
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        if not getattr(annotation, _OUTPUT_TYPE_MARKER, False):
            raise OutputTypeError(
                f"output dataclass {annotation.__qualname__} must use @output_type "
                "so unexpected fields are forbidden"
            )
        try:
            hints = get_type_hints(annotation)
        except (NameError, TypeError) as exc:
            raise OutputTypeError(
                f"could not resolve fields for output type {annotation.__qualname__}: {exc}"
            ) from exc
        for field_annotation in hints.values():
            _require_closed_output_types(field_annotation, visited)
        return
    for argument in get_args(annotation):
        _require_closed_output_types(argument, visited)


def _adapter(annotation: Any) -> TypeAdapter:
    try:
        hash(annotation)
    except TypeError:
        return _build_adapter(annotation)
    return _cached_adapter(annotation)


# Building a TypeAdapter re-resolves type hints and regenerates the core schema;
# ``model_dump()`` and every schema/parse call previously paid that per call.
# Failures are not cached, so a forward reference resolved later is retried.
@functools.lru_cache(maxsize=512)
def _cached_adapter(annotation: Any) -> TypeAdapter:
    return _build_adapter(annotation)


def _build_adapter(annotation: Any) -> TypeAdapter:
    try:
        _require_closed_output_types(annotation)
        return TypeAdapter(annotation)
    except (PydanticSchemaGenerationError, PydanticUserError, TypeError) as exc:
        raise OutputTypeError(f"unsupported annotation {annotation!r}: {exc}") from exc


def schema_for(annotation: Any) -> dict[str, Any]:
    """Derive a JSON schema describing ``annotation``."""
    return _adapter(annotation).json_schema()


def strict_schema_for(annotation: Any) -> dict[str, Any]:
    """Prepare declared fields and tuples for strict native transport.

    OpenAI-compatible strict schemas require every declared object property to
    appear in ``required``, including fields for which the Python dataclass has
    a default.  Defaults remain useful when host code constructs the type, but
    an agent must emit every field explicitly.  Removing the ``default``
    annotation also keeps the provider schema within the supported strict JSON
    Schema subset. The provider transport separately wraps roots and encodes
    open mappings; strict host parsing still enforces the original annotation.
    """
    schema = schema_for(annotation)
    _require_all_properties(schema)
    _normalize_fixed_tuples(schema)
    return schema


def _require_all_properties(value: Any) -> None:
    if isinstance(value, dict):
        value.pop("default", None)
        properties = value.get("properties")
        if isinstance(properties, dict):
            value["required"] = list(properties)
        for child in _schema_children(value):
            _require_all_properties(child)


def _schema_children(value: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Visit schema positions, leaving field names and instance metadata intact."""
    for key in ("properties", "$defs", "definitions", "patternProperties", "dependentSchemas"):
        mapping = value.get(key)
        if isinstance(mapping, dict):
            yield from (child for child in mapping.values() if isinstance(child, dict))
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        sequence = value.get(key)
        if isinstance(sequence, list):
            yield from (child for child in sequence if isinstance(child, dict))
    for key in (
        "items",
        "additionalProperties",
        "propertyNames",
        "additionalItems",
        "contains",
        "not",
        "if",
        "then",
        "else",
        "unevaluatedItems",
        "unevaluatedProperties",
    ):
        child = value.get(key)
        if isinstance(child, dict):
            yield child


def _normalize_fixed_tuples(value: Any) -> None:
    """Translate positional tuples into the native-provider array subset.

    Pydantic emits JSON Schema 2020-12 ``prefixItems`` for heterogeneous
    tuples. OpenAI-compatible structured-output endpoints require every array
    schema to contain ``items`` and do not enforce positional item schemas.
    Keep the exact length and the union of permitted element types in the
    provider schema; strict host-side Pydantic parsing still enforces each
    tuple position before a value reaches recipe code.
    """
    if isinstance(value, dict):
        prefix_items = value.get("prefixItems")
        if value.get("type") == "array" and isinstance(prefix_items, list) and prefix_items:
            unique_items: list[Any] = []
            fingerprints: set[str] = set()
            for item in prefix_items:
                fingerprint = json.dumps(item, sort_keys=True, separators=(",", ":"))
                if fingerprint not in fingerprints:
                    fingerprints.add(fingerprint)
                    unique_items.append(item)
            value["items"] = unique_items[0] if len(unique_items) == 1 else {"anyOf": unique_items}
            del value["prefixItems"]
        for child in _schema_children(value):
            _normalize_fixed_tuples(child)


def parse_response(text: str, annotation: Any) -> Any:
    """Extract JSON from ``text`` and parse it strictly into ``annotation``.

    We round-trip through ``json.dumps(...) -> TypeAdapter.validate_json(...)``
    rather than ``validate_python(strict=True)``: the strict-from-JSON path
    rejects type coercion (``"3"`` does not become ``int 3``) while still
    constructing dataclasses from JSON object bodies. ``validate_python`` in
    strict mode would also reject the dataclass-from-dict construction we
    actively want.
    """
    raw = _extract_json(text)
    adapter = _adapter(annotation)
    try:
        return adapter.validate_json(json.dumps(raw), strict=True)
    except ValidationError as exc:
        raise OutputParseError(f"schema mismatch: {exc}") from exc


def _extract_json(text: str) -> Any:
    s = _strip_fence(text.strip())
    # First try: parse the whole stripped text. Handles bare scalars
    # (``42``, ``null``, ``"hi"``, ``true``) and clean object/array output.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Fallback: scan for embedded JSON object/array blocks; prefer the LAST
    # one (most recent answer if the LLM included prose / multiple drafts).
    decoder = json.JSONDecoder()
    last_value: Any = _NOTHING
    start = 0
    while start < len(s):
        if s[start] not in "{[":
            start += 1
            continue
        try:
            value, end = decoder.raw_decode(s, start)
        except json.JSONDecodeError:
            start += 1
            continue
        last_value = value
        start = end
    if last_value is _NOTHING:
        raise OutputParseError(f"no JSON block found in response: {text!r}")
    return last_value


def _strip_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    # gremlin: pardon[equivalent] outer startswith() guarantees `lines` is non-empty here
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


_WRAPPED_FIELD = "value"
_JSON_ENCODING_FIELD = "_calcifier_json"


class StructuredAttemptError(OutputParseError):
    """One rejected structured answer, with a content-free label for the failed stage."""

    def __init__(self, stage: str, error: OutputParseError) -> None:
        super().__init__(f"{stage}: {error}")
        self.stage = stage


def wire_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Give native providers a closed object root; host validation retains map keys."""
    if needs_json_transport(schema):
        value = {
            "type": "string",
            "description": "JSON-encode the complete value conforming to this schema: "
            + json.dumps(schema),
        }
        wrapped = object_root_schema(value)
        wrapped["properties"][_JSON_ENCODING_FIELD] = {"type": "boolean", "const": True}
        wrapped["required"].append(_JSON_ENCODING_FIELD)
        return wrapped
    return object_root_schema(schema)


def needs_json_transport(schema: dict[str, Any]) -> bool:
    if schema.get("type") == "object" and schema.get("additionalProperties") is not False:
        return True
    if not any(
        key in schema for key in ("type", "$ref", "anyOf", "oneOf", "allOf", "enum", "const")
    ):
        return True
    children = [
        *schema.get("properties", {}).values(),
        *schema.get("$defs", {}).values(),
        *(
            child
            for key in ("anyOf", "oneOf", "allOf", "prefixItems")
            for child in schema.get(key, [])
        ),
        *(
            schema[key]
            for key in ("items", "additionalProperties")
            if isinstance(schema.get(key), dict)
        ),
    ]
    return any(needs_json_transport(child) for child in children)


def object_root_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") == "object":
        return schema
    wrapped: dict[str, Any] = {
        "type": "object",
        "properties": {_WRAPPED_FIELD: {k: v for k, v in schema.items() if k != "$defs"}},
        "required": [_WRAPPED_FIELD],
        "additionalProperties": False,
    }
    if "$defs" in schema:
        wrapped["$defs"] = schema["$defs"]
    return wrapped


def validated_structured(text: str, annotation: Any, *, json_encoded: bool = False) -> Any:
    """Parse one answer, unwrapping a ``{"value": ...}`` envelope, and label any failure."""
    parse = _parse_explicit_response
    try:
        payload = _extract_json(text)
        if (
            json_encoded
            and isinstance(payload, dict)
            and set(payload) == {_WRAPPED_FIELD, _JSON_ENCODING_FIELD}
            and payload[_JSON_ENCODING_FIELD] is True
        ):
            if not isinstance(payload[_WRAPPED_FIELD], str):
                raise OutputParseError("JSON-encoded output must be a string")
            return parse(payload[_WRAPPED_FIELD], annotation)
        if (
            strict_schema_for(annotation).get("type") != "object"
            and isinstance(payload, dict)
            and set(payload) == {_WRAPPED_FIELD}
        ):
            with suppress(OutputParseError):
                return parse(json.dumps(payload[_WRAPPED_FIELD]), annotation)
        return parse(text, annotation)
    except OutputParseError as exc:
        raise StructuredAttemptError("did not match the declared output schema", exc) from exc


def _parse_explicit_response(text: str, annotation: Any) -> Any:
    """Require explicit fields in every agent reply before host defaults apply."""
    payload = _extract_json(text)
    try:
        Draft202012Validator(strict_schema_for(annotation)).validate(payload)
    except JsonSchemaValidationError as exc:
        raise OutputParseError(f"decoded strict schema mismatch: {exc}") from exc
    return parse_response(json.dumps(payload), annotation)


BUILTIN_OUTPUT_TYPES: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "bytes": bytes,
    "object": object,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "frozenset": frozenset,
}


STUB_TEXT = "stub"
STUB_REJECTED = "returned no valid structured response"
_MAX_SCHEMA_DEPTH = 32
_WORD_COUNT_DESCRIPTION = re.compile(r"At least (\d+) words(?:; at most (\d+) words)?\.$")


def synthesize(schema: dict[str, Any]) -> Any:
    """Return a minimal value satisfying a strict Calcifier output schema.

    Arrays get one element so loops over agent output execute.

    Args:
        schema: JSON Schema produced by ``strict_schema_for``.

    Returns:
        A JSON-compatible value.
    """
    return _instance(schema, schema.get("$defs", {}), 0)


def _instance(schema: dict[str, Any], defs: dict[str, Any], depth: int) -> Any:
    if depth > _MAX_SCHEMA_DEPTH:
        return None
    child = functools.partial(_instance, defs=defs, depth=depth + 1)
    if "$ref" in schema:
        return child(defs[schema["$ref"].rsplit("/", 1)[-1]])
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    options = schema.get("anyOf") or schema.get("oneOf")
    if options:
        concrete = [item for item in options if item.get("type") != "null"] or options
        return child(concrete[0])
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((item for item in kind if item != "null"), "null")
    return _typed_instance(kind, schema, child)


def _typed_instance(
    kind: object, schema: dict[str, Any], child: Callable[[dict[str, Any]], Any]
) -> Any:
    if kind == "object":
        additional = schema.get("additionalProperties")
        if not schema.get("properties") and (isinstance(additional, dict) or additional is True):
            key = child({"type": "string", **schema.get("propertyNames", {})})
            return {key: child(additional) if isinstance(additional, dict) else None}
        return {name: child(value) for name, value in schema.get("properties", {}).items()}
    if kind == "array":
        if "prefixItems" in schema:
            return [child(item) for item in schema["prefixItems"]]
        count = max(1, schema.get("minItems", 1))
        return [child(schema.get("items", {})) for _ in range(count)]
    if kind == "string":
        return _string_instance(schema)
    if kind in {"integer", "number"}:
        value = max(schema.get("minimum", 1), 1)
        return value if kind == "integer" else float(value)
    return False if kind == "boolean" else None


def _string_instance(schema: dict[str, Any]) -> str:
    """Return stub text that also satisfies a described word-count bound."""
    formats = {"date": "2000-01-01", "date-time": "2000-01-01T00:00:00Z"}
    text = formats.get(schema.get("format"), STUB_TEXT).ljust(schema.get("minLength", 0), "s")
    description = schema.get("description")
    match = _WORD_COUNT_DESCRIPTION.search(description) if isinstance(description, str) else None
    if match is None:
        return text
    minimum = int(match.group(1))
    maximum = None if match.group(2) is None else int(match.group(2))
    count = 0 if maximum == 0 else max(1, minimum)
    if count == 0:
        return ""
    words = ["s"] * count
    words[-1] += "s" * max(0, schema.get("minLength", 0) - (2 * count - 1))
    return " ".join(words)


class StubJsonSchema(GenerateJsonSchema):
    """Retain host key types that ordinary JSON object schemas cannot express."""

    def dict_schema(self, schema: core_schema.DictSchema) -> JsonSchemaValue:
        result = super().dict_schema(schema)
        if "keys_schema" in schema:
            result["propertyNames"] = self.generate_inner(schema["keys_schema"])
        return result
