"""Encoded native outputs must supply every record field before defaults are applied."""

from __future__ import annotations

import json

import pytest

from calcifier.agent_events import AgentProvider, AgentResult
from calcifier.filesystem_contract import AgentError
from calcifier.outputs import output_type, parse_response

pytestmark = pytest.mark.semantic_offline


@output_type
class Decision:
    counts: dict[str, int]
    approved: bool = True


@output_type
class Nested:
    decisions: list[Decision]
    reviewed: bool = True


@output_type
class Alternative:
    label: str


@output_type
class ClosedDecision:
    reason: str
    approved: bool = True


@output_type
class ClosedHolder:
    value: ClosedDecision


class Replies(AgentProvider):
    name = "worker"
    default_timeout = 30

    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def invoke(self, prompt, **kwargs):
        del kwargs
        self.prompts.append(prompt)
        text = next(self.replies)
        return AgentResult(text, text, "", 0, self.name)


def encoded(value):
    return json.dumps({"value": json.dumps(value), "_calcifier_json": True})


@pytest.mark.parametrize(
    "annotation, value",
    [
        (Decision, {"counts": {"x": 1}}),
        (Nested, {"decisions": [{"counts": {"x": 1}}], "reviewed": False}),
        (dict[str, Decision], {"entry": {"counts": {"x": 1}}}),
        (tuple[Decision, str], [{"counts": {"x": 1}}, "complete"]),
    ],
)
@pytest.mark.parametrize("transport", [encoded, json.dumps])
def test_missing_defaulted_record_fields_reject_both_attempts(annotation, value, transport):
    reply = transport(value)
    provider = Replies([reply, reply])
    with pytest.raises(AgentError, match="one retry"):
        provider.structured("Return the complete decision.", annotation=annotation)
    assert len(provider.prompts) == 2


def test_retry_can_supply_an_explicit_false_instead_of_an_implicit_approval():
    provider = Replies(
        [
            encoded({"counts": {"x": 1}}),
            encoded({"counts": {"x": 1}, "approved": False}),
        ]
    )
    assert provider.structured("Return the decision.", annotation=Decision) == Decision(
        {"x": 1}, False
    )
    assert len(provider.prompts) == 2


def test_complete_nested_records_and_a_different_union_branch_are_valid():
    value = {"decisions": [{"counts": {"x": 1}, "approved": False}], "reviewed": False}
    provider = Replies([encoded(value)])
    assert provider.structured("Return complete decisions.", annotation=Nested) == Nested(
        [Decision({"x": 1}, False)], False
    )
    provider = Replies([encoded({"label": "complete"})])
    assert provider.structured(
        "Return either result.", annotation=Decision | Alternative
    ) == Alternative("complete")


def test_host_construction_and_public_parsing_keep_python_defaults():
    assert Decision({"x": 1}).approved is True
    assert parse_response('{"counts":{"x":1}}', Decision).approved is True
    assert parse_response('{"reason":"complete"}', ClosedDecision).approved is True


@pytest.mark.parametrize(
    "annotation,payload",
    [
        (ClosedDecision, {"reason": "needs work"}),
        (ClosedHolder, {"value": {"reason": "needs work"}}),
        (list[ClosedDecision], [{"reason": "needs work"}]),
        (list[ClosedDecision], {"value": [{"reason": "needs work"}]}),
    ],
)
def test_closed_native_and_legacy_outputs_cannot_apply_missing_defaults(annotation, payload):
    reply = json.dumps(payload)
    provider = Replies([reply, reply])
    with pytest.raises(AgentError, match="one retry"):
        provider.structured("Return every field explicitly.", annotation=annotation)
    assert len(provider.prompts) == 2


def test_closed_output_retry_can_explicitly_decline_instead_of_defaulting_to_approval():
    provider = Replies(
        [
            '{"reason":"needs work"}',
            '{"reason":"needs work","approved":false}',
        ]
    )
    result = provider.structured("Return every field.", annotation=ClosedDecision)
    assert result == ClosedDecision("needs work", False)
    assert len(provider.prompts) == 2
