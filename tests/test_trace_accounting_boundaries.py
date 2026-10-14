"""Trace measurements and preflight decisions retain their original meaning."""

from __future__ import annotations

import pytest

from calcifier.trace import (
    active_run,
    agent_call_frame,
    record_agent_model,
    record_agent_usage,
    redact,
    step_frame,
)


def test_no_model_report_does_not_erase_a_known_provider_identity():
    record_agent_model("outside-a-call")
    with active_run() as ctx:
        record_agent_model("outside-a-call")
        assert ctx.agent_calls == []
        with agent_call_frame(
            role="worker", target_type="builtin", target_name="codex", runtime={}
        ) as call:
            record_agent_model("resolved-model")
            for absent in (None, ""):
                record_agent_model(absent)
                assert call.provider_reported_model == "resolved-model"
                assert call.model_identity_source == "provider_reported"
        assert ctx.agent_calls == [call]


def test_partial_usage_is_not_mistaken_for_a_final_attempt_total():
    with (
        active_run(),
        agent_call_frame(
            role="worker", target_type="builtin", target_name="codex", runtime={}
        ) as call,
    ):
        record_agent_usage({"input_tokens": 5})
        assert call.usage["input_tokens"] == 5
        assert call.usage_reported_attempts == {}
        record_agent_usage({"output_tokens": 2}, final=True)
        assert call.usage["input_tokens"] == 5 and call.usage["output_tokens"] == 2
        assert call.usage_reported_attempts == {"output_tokens": 1}


def test_step_duration_is_elapsed_time_even_when_it_fails(monkeypatch):
    from calcifier import trace

    with active_run() as ctx:
        clock = iter([100.0, 103.5])
        monkeypatch.setattr(trace.time, "time", lambda: next(clock))
        with pytest.raises(RuntimeError, match="fixture failed"), step_frame("measure", None, {}):
            raise RuntimeError("fixture failed")
        assert ctx.records[0].duration_seconds == 3.5
        assert ctx.frames == ctx.record_stack == []


def test_nested_tuple_redaction_preserves_nonsecret_evidence_and_input_value():
    original = ({"token": "sensitive", "public": "evidence"}, ({"password": "secret"},))
    assert redact(original) == (
        {"token": "<redacted>", "public": "evidence"},
        ({"password": "<redacted>"},),
    )
    assert original == ({"token": "sensitive", "public": "evidence"}, ({"password": "secret"},))
