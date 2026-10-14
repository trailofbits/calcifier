"""Schema transformations preserve field names that also happen to be JSON keywords."""

from __future__ import annotations

import copy
import json

import pytest
from jsonschema import Draft202012Validator

from calcifier.agent_events import AgentProvider, AgentResult
from calcifier.dry_run import _SchemaStub
from calcifier.outputs import (
    _normalize_fixed_tuples,
    _require_all_properties,
    output_type,
    strict_schema_for,
)

pytestmark = pytest.mark.semantic_offline


class CompleteReply(AgentProvider):
    name = "worker"
    default_timeout = 30

    def __init__(self, payload):
        self.payload = payload

    def invoke(self, prompt, **kwargs):
        del prompt, kwargs
        text = json.dumps({"value": json.dumps(self.payload), "_calcifier_json": True})
        return AgentResult(text, text, "", 0, self.name)


@pytest.mark.parametrize(
    "field",
    ["default", "properties", "required", "items", "prefixItems", "type", "additionalProperties"],
)
def test_keyword_field_names_work_in_strict_native_outputs_and_schema_stubs(field):
    record = output_type(
        type("KeywordRecord", (), {"__annotations__": {field: str, "labels": dict[str, int]}})
    )
    schema = strict_schema_for(record)
    assert set(schema["properties"]) == {field, "labels"}
    Draft202012Validator.check_schema(schema)
    payload = {field: "supplied", "labels": {"x": 1}}
    Draft202012Validator(schema).validate(payload)
    result = CompleteReply(payload).structured("Return both fields.", annotation=record)
    assert getattr(result, field) == "supplied"
    stub = _SchemaStub("worker").structured("Return both fields.", annotation=record)
    assert getattr(stub, field) == "stub"
    assert stub.labels == {"stub": 1}


def test_schema_walkers_do_not_treat_enum_instance_data_as_schemas():
    instance = {"default": "keep", "type": "array", "prefixItems": [{"properties": "data"}]}
    schema = {"type": "object", "properties": {"example": {"enum": [instance]}}}
    expected = copy.deepcopy(instance)
    _require_all_properties(schema)
    _normalize_fixed_tuples(schema)
    assert schema["properties"]["example"]["enum"] == [expected]
