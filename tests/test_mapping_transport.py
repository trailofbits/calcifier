"""Native transport preserves open mapping values under strict host validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from calcifier import agent_providers
from calcifier.agent_events import AgentResult, wire_schema
from calcifier.agent_providers import ClaudeCli, CodexCli
from calcifier.filesystem_contract import AgentError, FilesystemContract
from calcifier.outputs import output_type, strict_schema_for
from tests.support.agents import StubProvider


@output_type
class Report:
    counts: dict[str, int]
    note: str


@output_type
class Entry:
    value: str
    children: dict[str, Entry]


def assert_native_objects_closed(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])
        for child in schema.values():
            assert_native_objects_closed(child)
    elif isinstance(schema, list):
        for child in schema:
            assert_native_objects_closed(child)


@pytest.mark.parametrize("annotation", [dict[str, int], Report, Entry, list[Report], object])
def test_mapping_wire_schema_is_closed_without_mutating_the_host_schema(annotation):
    host = strict_schema_for(annotation)
    original = copy.deepcopy(host)
    wire = wire_schema(host)
    Draft202012Validator.check_schema(wire)
    assert_native_objects_closed(wire)
    assert host == original
    assert wire["properties"]["value"]["type"] == "string"
    assert json.dumps(host) in wire["properties"]["value"]["description"]


@pytest.mark.parametrize(
    "annotation, value, expected",
    [
        (dict[str, int], {"first": 1, "second": 2}, {"first": 1, "second": 2}),
        (dict[str, str], {"value": '{"a":"b"}'}, {"value": '{"a":"b"}'}),
        (Report, {"counts": {"first": 1}, "note": "ready"}, Report({"first": 1}, "ready")),
        (
            Entry,
            {"value": "parent", "children": {"child": {"value": "leaf", "children": {}}}},
            Entry("parent", {"child": Entry("leaf", {})}),
        ),
        (
            dict[str, Any],
            {"first": [1, True, None, {"child": "ready"}]},
            {"first": [1, True, None, {"child": "ready"}]},
        ),
        (dict[str, int] | str, "ready", "ready"),
    ],
)
def test_encoded_outputs_round_trip_without_dropping_entries(annotation, value, expected):
    text = json.dumps({"value": json.dumps(value), "_calcifier_json": True})
    provider = StubProvider(default=text)
    assert provider.structured("return the value", annotation=annotation) == expected


@pytest.mark.parametrize(
    "annotation, value",
    [
        (dict[str, int], {"first": "1"}),
        (Report, {"counts": {"first": 1}, "note": "ready", "extra": "private extra"}),
    ],
)
def test_encoded_values_still_require_strict_host_types_and_closed_records(annotation, value):
    provider = StubProvider(
        default=json.dumps({"value": json.dumps(value), "_calcifier_json": True})
    )
    with pytest.raises(AgentError, match="returned no valid structured response"):
        provider.structured("return the value", annotation=annotation)


def test_legacy_unwrapped_mapping_responses_remain_supported():
    provider = StubProvider(default='{"first":1,"second":2}')
    assert provider.structured("return the value", annotation=dict[str, int]) == {
        "first": 1,
        "second": 2,
    }


def test_legacy_value_key_is_preserved_when_its_text_contains_json():
    value = {"value": '{"a":"b"}'}
    provider = StubProvider(default=json.dumps(value))
    assert provider.structured("return the value", annotation=dict[str, str]) == value


@pytest.mark.parametrize("value", ["not JSON", 1])
def test_invalid_encoded_unconstrained_answers_cannot_fall_back_to_the_envelope(value):
    provider = StubProvider(default=json.dumps({"value": value, "_calcifier_json": True}))
    with pytest.raises(AgentError, match="returned no valid structured response"):
        provider.structured("return the value", annotation=dict[str, Any])


@pytest.mark.parametrize("provider_class", [ClaudeCli, CodexCli])
def test_native_adapters_receive_the_closed_transport_schema(monkeypatch, tmp_path, provider_class):
    observed = []
    encoded = json.dumps({"value": json.dumps({"first": 1, "second": 2}), "_calcifier_json": True})

    def fake_run(command, stdin, **kwargs):
        if provider_class is CodexCli:
            schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_bytes())
        else:
            schema = json.loads(command[command.index("--json-schema") + 1])
        observed.append(schema)
        assert_native_objects_closed(schema)
        Draft202012Validator(schema).validate(json.loads(encoded))
        return AgentResult(encoded, encoded, "", 0, "worker")

    monkeypatch.setattr(agent_providers, "_run_cli", fake_run)
    provider = provider_class(role="worker", default_timeout=5)
    result = provider.structured(
        "return the counts",
        annotation=dict[str, int],
        filesystem=FilesystemContract.build(tmp_path, (tmp_path,)),
    )
    assert result == {"first": 1, "second": 2}
    assert len(observed) == 1
