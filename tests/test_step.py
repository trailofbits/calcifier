"""Tests for the @step / fail / RecipeFailure machinery."""

from __future__ import annotations

import json
from typing import Annotated

import pytest

from calcifier import agent, fail, output_type, step, word_count
from calcifier.agent_events import AgentResult
from calcifier.filesystem_contract import AgentError
from calcifier.outputs import OutputTypeError
from calcifier.recipe import (
    RecipeFailure,
)
from calcifier.registry import Registry
from calcifier.trace import active_run, current_frame
from tests.support.agents import StubProvider

# Module-level dataclasses; with `from __future__ import annotations`, the
# @step decorator resolves return-type hints via ``typing.get_type_hints``
# against the function's ``__globals__`` — which means dataclasses must be
# in the same module scope as any decorated function that returns them.


@output_type
class _Verdict:
    approved: bool
    reasoning: str


class _Unsupported:
    pass


@output_type
class _Plain:
    a: int


@output_type
class _RedactedOut:
    token: str
    public: str


@output_type
class _IntCarrier:
    n: int


@output_type
class _WordBoundResponse:
    message: Annotated[str, word_count(3)]


# -- decoration-time validation ----------------------------------------------


def test_step_requires_return_annotation():
    with pytest.raises(OutputTypeError, match="requires a return-type annotation"):

        @step
        def no_annotation(x: str):  # pragma: no cover - decoration fails
            return x


def test_step_rejects_unsupported_return_annotation():
    # Pydantic supports a wide annotation set (set[T], int | str,
    # dict[int, str] all work) so v0.1's bespoke "not in allowlist"
    # rejection is gone. The remaining hard error is "TypeAdapter can't
    # resolve a forward reference", which we surface as OutputTypeError.
    with pytest.raises(OutputTypeError):

        @step
        def returns_unsupported(x: str) -> _Unsupported:
            raise NotImplementedError


# -- runtime behavior --------------------------------------------------------


@output_type
class _Verdict:
    approved: bool
    reasoning: str


def test_step_pushes_frame_with_annotation():
    @step
    def judge_inner(reason: str) -> _Verdict:
        frame = current_frame()
        assert frame is not None
        assert frame.name == "judge_inner"
        assert frame.annotation is _Verdict
        return _Verdict(approved=True, reasoning=reason)

    with active_run() as ctx:
        out = judge_inner(reason="ok")
    assert out == _Verdict(approved=True, reasoning="ok")
    assert len(ctx.records) == 1
    assert ctx.records[0].name == "judge_inner"
    # Trace converts dataclasses to dicts (see test_step_redacts_input_kwargs_
    # and_dataclass_fields for redaction semantics).
    assert ctx.records[0].output == {"approved": True, "reasoning": "ok"}


def test_step_records_failure_on_recipefailure():
    @step
    def gate(must_be_true: bool) -> _Verdict:
        if not must_be_true:
            fail("gate failed")
        return _Verdict(approved=True, reasoning="ok")

    with active_run() as ctx, pytest.raises(RecipeFailure, match="gate failed"):
        gate(must_be_true=False)
    assert ctx.records[0].error == "gate failed"
    assert ctx.records[0].output is None


def test_step_redacts_input_kwargs_and_dataclass_fields():
    @step
    def login(api_key: str, username: str) -> _RedactedOut:
        return _RedactedOut(token="returned-token", public="hello-" + username)  # noqa: S106

    with active_run() as ctx:
        result = login(api_key="SECRET", username="alice")

    # The function body still receives the real values.
    assert result.token == "returned-token"  # noqa: S105
    record = ctx.records[0]
    # `api_key` is a default redacted kwarg name.
    assert record.inputs["api_key"] == "<redacted>"
    assert record.inputs["username"] == "alice"
    # Dataclass output is converted to a dict in the trace and dataclass
    # field names matching the redacted set are masked. The runtime still
    # hands the original dataclass to downstream code (the assertion on
    # `result.token` above proves it).
    assert record.output["token"] == "<redacted>"  # noqa: S105
    assert record.output["public"] == "hello-alice"


# -- Agent.__call__ inside @step body ----------------------------------------


def _stub_registry(default_text: str) -> Registry:
    reg = Registry()
    provider = StubProvider(default=default_text)
    provider.name = "worker"
    reg.register(provider)
    return reg


def test_agent_call_inside_step_uses_return_annotation():
    @step
    def greet(name: str) -> _Verdict:
        return agent(
            "worker",
            "Greet {name}",
            working_directory=".",
            readable_paths=(".",),
            name=name,
        )

    reg = _stub_registry(json.dumps({"approved": True, "reasoning": "hello alice"}))
    with active_run(registry=reg):
        result = greet(name="alice")

    assert result == _Verdict(approved=True, reasoning="hello alice")


def test_agent_call_outside_step_raises():
    reg = _stub_registry(json.dumps({"approved": True, "reasoning": "x"}))
    with active_run(registry=reg), pytest.raises(AgentError, match="outside a @step body"):
        agent(
            "worker",
            "hi",
            working_directory=".",
            readable_paths=(".",),
            output=None,
        )


def test_agent_call_with_explicit_output_overrides_step_annotation():
    @step
    def root(x: int) -> _Verdict:
        # Intermediate call returning a primitive — explicit override.
        intermediate: str = agent(
            "worker",
            "intermediate {x}",
            working_directory=".",
            readable_paths=(".",),
            x=x,
            output=str,
        )
        return _Verdict(approved=True, reasoning=intermediate)

    reg = Registry()
    provider = StubProvider(
        responses={
            "intermediate": '"intermediate result"',
        },
        default=json.dumps({"approved": True, "reasoning": "fallback"}),
    )
    provider.name = "worker"
    reg.register(provider)
    with active_run(registry=reg):
        result = root(x=1)

    assert result.reasoning == "intermediate result"


def test_agent_call_outside_active_run_raises():
    with pytest.raises(AgentError, match="outside an active recipe run"):
        agent(
            "worker",
            "hi",
            working_directory=".",
            readable_paths=(".",),
            output=str,
        )


# -- structured-output retry --------------------------------------------------


def test_agent_structured_retries_once_on_parse_failure():
    class _FlakyProvider(StubProvider):
        name = "worker"

        def __init__(self):
            super().__init__()
            self.calls = 0

        def invoke(self, prompt, *, timeout=None, **kwargs):
            del timeout, kwargs
            self.calls += 1
            text = "not json" if self.calls == 1 else json.dumps({"n": 42})
            return AgentResult(
                text=text, raw_stdout=text, raw_stderr="", exit_code=0, provider=self.name
            )

    flaky = _FlakyProvider()
    reg = Registry()
    reg.register(flaky)

    @step
    def fetch(x: int) -> _IntCarrier:
        return agent(
            "worker",
            "get {x}",
            working_directory=".",
            readable_paths=(".",),
            x=x,
        )

    with active_run(registry=reg):
        out = fetch(x=1)

    assert out == _IntCarrier(n=42)
    assert flaky.calls == 2


def test_word_constraint_retries_short_model_output_within_existing_bound():
    class _ShortThenValid(StubProvider):
        name = "worker"

        def __init__(self):
            super().__init__()
            self.prompts: list[str] = []

        def invoke(self, prompt, *, timeout=None, **kwargs):
            del timeout, kwargs
            self.prompts.append(prompt)
            message = "too short" if len(self.prompts) == 1 else "now long enough"
            text = json.dumps({"message": message})
            return AgentResult(
                text=text, raw_stdout=text, raw_stderr="", exit_code=0, provider=self.name
            )

    provider = _ShortThenValid()
    reg = Registry()
    reg.register(provider)

    @step
    def request() -> _WordBoundResponse:
        return agent(
            "worker",
            "Return a response whose message has at least 3 words.",
            working_directory=".",
            readable_paths=(".",),
        )

    with active_run(registry=reg):
        result = request()

    assert result.message == "now long enough"
    assert len(provider.prompts) == 2
    assert "must contain at least 3 words" in provider.prompts[1]


def test_word_constraint_stops_after_existing_single_retry():
    class _AlwaysShort(StubProvider):
        name = "worker"

        def __init__(self):
            super().__init__()
            self.calls = 0

        def invoke(self, prompt, *, timeout=None, **kwargs):
            del prompt, timeout, kwargs
            self.calls += 1
            text = json.dumps({"message": "too short"})
            return AgentResult(
                text=text, raw_stdout=text, raw_stderr="", exit_code=0, provider=self.name
            )

    provider = _AlwaysShort()
    reg = Registry()
    reg.register(provider)

    @step
    def request_short() -> _WordBoundResponse:
        return agent(
            "worker",
            "Return a response whose message has at least 3 words.",
            working_directory=".",
            readable_paths=(".",),
        )

    with active_run(registry=reg), pytest.raises(AgentError, match="after one retry"):
        request_short()

    assert provider.calls == 2


def test_agent_structured_gives_up_after_one_retry():
    class _AlwaysBad(StubProvider):
        name = "worker"

        def invoke(self, prompt, *, timeout=None, **kwargs):
            del timeout, kwargs
            text = "still not json"
            return AgentResult(
                text=text, raw_stdout=text, raw_stderr="", exit_code=0, provider=self.name
            )

    reg = Registry()
    reg.register(_AlwaysBad())

    @step
    def fetch_bad(x: int) -> _IntCarrier:
        return agent(
            "worker",
            "get {x}",
            working_directory=".",
            readable_paths=(".",),
            x=x,
        )

    schema_miss = "did not match the declared output schema"
    expected = f"first attempt {schema_miss}; second attempt {schema_miss}"
    with active_run(registry=reg), pytest.raises(AgentError, match=expected):
        fetch_bad(x=1)
