"""Providers always receive an object-root schema; any other declared type is unwrapped."""

from __future__ import annotations

import json
from typing import Any

import pytest

from calcifier import output_type
from calcifier.agent_events import AgentProvider, AgentResult
from calcifier.dry_run import _SchemaStub
from calcifier.outputs import strict_schema_for


@output_type
class Candidate:
    name: str
    score: int


@output_type
class Boxed:
    value: int


class Recording(AgentProvider):
    name = "worker"
    default_timeout = 30

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.schemas: list[dict[str, Any] | None] = []

    def invoke(self, prompt, *, timeout=None, filesystem=None, schema=None) -> AgentResult:
        del prompt, timeout, filesystem
        self.schemas.append(schema)
        text = self.answers.pop(0)
        return AgentResult(text, text, "", 0, self.name)


@pytest.mark.parametrize(
    ("annotation", "answer"),
    [(list[Candidate], []), (str, "text"), (bool, True), (Candidate | None, None)],
)
def test_non_object_outputs_reach_the_provider_with_an_object_root(annotation, answer):
    declared = strict_schema_for(annotation)
    provider = Recording(json.dumps({"value": answer}))
    assert provider.structured("Answer", annotation=annotation) == answer
    sent = provider.schemas[0]
    assert sent is not None
    assert sent["type"] == "object"
    assert sent["required"] == ["value"]
    assert sent["additionalProperties"] is False
    inner = sent["properties"]["value"]
    assert "$defs" not in inner
    assert inner == {key: value for key, value in declared.items() if key != "$defs"}
    assert sent.get("$defs") == declared.get("$defs")


def test_object_outputs_are_sent_unchanged():
    provider = Recording(json.dumps({"name": "a", "score": 1}))
    assert provider.structured("Answer", annotation=Candidate) == Candidate("a", 1)
    assert provider.schemas == [strict_schema_for(Candidate)]


@pytest.mark.parametrize(
    "answer",
    [
        {"value": [{"name": "a", "score": 1}]},
        [{"name": "a", "score": 1}],
    ],
)
def test_wrapped_and_bare_list_answers_parse_to_the_declared_type(answer):
    provider = Recording(json.dumps(answer))
    assert provider.structured("Answer", annotation=list[Candidate]) == [Candidate("a", 1)]
    assert len(provider.schemas) == 1


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"value": {"value": 3}}, Boxed(3)),
        ({"value": 3}, Boxed(3)),
        ({"value": None}, None),
    ],
)
def test_a_record_with_a_value_field_is_not_mistaken_for_the_envelope(answer, expected):
    provider = Recording(json.dumps(answer))
    assert provider.structured("Answer", annotation=Boxed | None) == expected


def test_dry_run_stubs_answer_through_the_same_envelope():
    result = _SchemaStub("worker").structured("Answer", annotation=list[Candidate])
    assert len(result) == 1 and isinstance(result[0], Candidate)
