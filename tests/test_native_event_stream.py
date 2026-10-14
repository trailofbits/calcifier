"""Native event volume must not replace final answers or defeat resource bounds."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from calcifier import subprocesses
from calcifier.agent_events import AgentActivity, AgentProvider, _run_cli, capture_agent_activity
from calcifier.filesystem_contract import AgentError, TransientAgentError
from calcifier.native_events import (
    MAX_EVENT_BYTES,
    MAX_EVENT_TYPES,
    MAX_MESSAGE_BYTES,
    NativeEventStream,
)
from calcifier.subprocesses import ProcessError, ProcessOutputLimitError


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_large_auxiliary_stream_keeps_only_the_answer_and_usage(provider):
    # The answer is small; the auxiliary JSONL alone exceeds the old capture cap.
    source = """
import json, sys
provider = sys.argv[1]
for i in range(150):
    print(json.dumps({'type': 'tool.progress', 'padding': 'x' * 65536}))
answer = {'answer': 'complete'}
if provider == 'codex':
    print(json.dumps({'type': 'item.completed', 'item': {
        'type': 'agent_message', 'text': json.dumps(answer)}}))
    print(json.dumps({'type': 'turn.completed', 'model': 'test-model',
                     'usage': {'input_tokens': 11, 'output_tokens': 7}}))
else:
    print(json.dumps({'type': 'result', 'subtype': 'success',
                     'structured_output': answer, 'model': 'test-model',
                     'usage': {'input_tokens': 11, 'output_tokens': 7}}))
"""
    result = _run_cli(
        [sys.executable, "-c", source, provider],
        "",
        timeout=20,
        name=provider,
        native_format=provider,
    )
    assert json.loads(result.text) == {"answer": "complete"}
    assert result.usage["input_tokens"] == 11
    assert result.usage["output_tokens"] == 7
    assert result.reported_model == "test-model"
    assert len(result.raw_stdout) < 1024


def feed(stream, event, *, newline=True):
    stream.consume(json.dumps(event, ensure_ascii=False).encode() + (b"\n" if newline else b""))


@pytest.mark.parametrize("chunk_size", [1, 7, 65536])
def test_chunk_boundaries_preserve_unicode_crlf_and_an_unterminated_final_event(chunk_size):
    stream = NativeEventStream("codex")
    data = (
        json.dumps(
            {"type": "item.completed", "item": {"type": "agent_message", "text": "café 🌱"}},
            ensure_ascii=False,
        )
        + '\r\n{"type":"turn.completed","usage":{"input_tokens":2}}'
    ).encode()
    for offset in range(0, len(data), chunk_size):
        stream.consume(data[offset : offset + chunk_size])
    stream.finish()
    stream.require_completion()
    assert stream.text == "café 🌱"
    assert stream.usage["input_tokens"] == 2
    assert stream.event_count == 2
    assert not stream.pending
    assert stream.total_bytes == len(data)


@pytest.mark.parametrize("chunk_size", [1, 7, 65536])
def test_event_limit_applies_to_each_event_across_transport_chunks(monkeypatch, chunk_size):
    import calcifier.native_events as native

    event = json.dumps({"type": "tool.progress", "padding": "x" * 40}).encode("utf-8")
    monkeypatch.setattr(native, "MAX_EVENT_BYTES", len(event))
    data = (event + b"\n") * 2
    stream = NativeEventStream("codex")
    for offset in range(0, len(data), chunk_size):
        stream.consume(data[offset : offset + chunk_size])
    diagnostics = stream.diagnostics()
    assert diagnostics["largest_event_bytes"] == len(event)
    assert diagnostics["event_count"] == 2
    assert diagnostics["total_bytes"] == len(data)
    assert diagnostics["pending_event_bytes"] == 0
    with pytest.raises(ProcessOutputLimitError, match="native event exceeds"):
        stream.consume(event + b" \n")
    assert stream.event_count == 2
    assert stream.event_types == {"tool.progress": 2}


@pytest.mark.parametrize("provider", ["codex", "claude"])
@pytest.mark.parametrize(
    "kind,expected",
    [
        ("x" * 63, "x" * 63),
        ("x" * 64, "x" * 64),
        ("x" * 65, "<other>"),
        ("🌱" * 64, "🌱" * 64),
        ("🌱" * 65, "<other>"),
        (None, "<other>"),
        (3, "<other>"),
        ([], "<other>"),
        ({"unexpected": "shape"}, "<other>"),
    ],
)
def test_unknown_event_types_are_bounded_and_do_not_block_completion(provider, kind, expected):
    stream = NativeEventStream(provider)
    diagnostics = stream.diagnostics()
    assert not diagnostics["completion_observed"]
    assert not diagnostics["usage_observed"]
    feed(stream, {"type": kind, "usage": {"input_tokens": 2}})
    assert stream.diagnostics()["event_types"] == {expected: 1}
    assert stream.diagnostics()["usage_observed"]
    assert not stream.diagnostics()["completion_observed"]
    message = (
        "no successfully completed native turn"
        if provider == "codex"
        else "no completed native result"
    )
    with pytest.raises(AgentError, match=message):
        stream.require_completion()

    if provider == "codex":
        feed(stream, {"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}})
        feed(stream, {"type": "turn.completed"})
    else:
        feed(stream, {"type": "result", "result": "ok"})
    stream.require_completion()
    assert stream.text == "ok"
    assert stream.diagnostics()["completion_observed"]


@pytest.mark.parametrize("terminal", ["turn.failed", "error"])
def test_unknown_event_type_cannot_mask_a_codex_failure(terminal):
    stream = NativeEventStream("codex")
    feed(stream, {"type": terminal})
    feed(stream, {"type": []})
    assert stream.diagnostics()["terminal"] == terminal
    with pytest.raises(AgentError, match="no successfully completed native turn"):
        stream.require_completion()


@pytest.mark.parametrize("kind", ["event", "message", "events"])
def test_native_limits_reap_the_process_and_preserve_partial_usage(monkeypatch, kind):
    import calcifier.native_events as native

    started: list[subprocess.Popen[bytes]] = []
    real_start = subprocesses._start_process

    def capture_start(*args, **kwargs):
        process = real_start(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(subprocesses, "_start_process", capture_start)
    monkeypatch.setattr(native, "MAX_EVENTS", 8)
    source = """
import json, sys, time
print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 9}}), flush=True)
kind = sys.argv[1]
if kind == 'event':
    sys.stdout.write('x' * (int(sys.argv[2]) + 1)); sys.stdout.flush()
elif kind == 'message':
    print(json.dumps({'type': 'item.completed', 'item': {
        'type': 'agent_message', 'text': 'x' * (int(sys.argv[3]) + 1)}}), flush=True)
else:
    for i in range(20):
        print(json.dumps({'type': 'tool.progress'}), flush=True)
time.sleep(30)
"""
    activity = AgentActivity()
    with capture_agent_activity(activity), pytest.raises(AgentError) as rejected:
        _run_cli(
            [sys.executable, "-c", source, kind, str(MAX_EVENT_BYTES), str(MAX_MESSAGE_BYTES)],
            "",
            timeout=15,
            name="codex",
            native_format="codex",
        )
    assert len(started) == 1 and started[0].poll() is not None
    assert activity.usage["input_tokens"] == 9
    assert activity.usage_reported_attempts == {}
    event = next(event for event in activity.events if event["phase"] == "native_transport")
    assert event["status"] == "failed"
    diagnostics = json.loads(str(event["detail"]))
    assert diagnostics["completion_observed"] and diagnostics["usage_observed"]
    assert diagnostics["event_count"] <= 9
    assert diagnostics["retained_final_message_bytes"] == 0
    assert diagnostics["pending_event_bytes"] <= MAX_EVENT_BYTES
    assert "native" in str(rejected.value) and "exceeds" in str(rejected.value)
    assert len(str(rejected.value)) < 4096


def test_event_count_and_type_counts_are_bounded_without_retaining_auxiliary_data(monkeypatch):
    import calcifier.native_events as native

    monkeypatch.setattr(native, "MAX_EVENTS", 100)
    stream = NativeEventStream("codex")
    for index in range(100):
        feed(stream, {"type": f"auxiliary-{index}", "discard": "x" * 4096})
    assert len(stream.event_types) <= MAX_EVENT_TYPES
    assert not stream.pending and not stream.text
    assert sum(stream.event_types.values()) == 100
    with pytest.raises(ProcessOutputLimitError, match="100 events"):
        feed(stream, {"type": "tool.progress"})


@pytest.mark.parametrize(
    "data",
    [
        b"invalid\n",
        b"\xff\n",
        b"[]\n",
        b'{"type":"item.completed","item":{"type":"agent_message","text":"\\ud800"}}\n',
    ],
)
def test_invalid_native_events_fail_without_replacement_decoding(data):
    stream = NativeEventStream("codex")
    with pytest.raises(ProcessError):
        stream.consume(data)
    assert not stream.text


def test_a_new_turn_cannot_reuse_an_old_answer_or_accept_an_incomplete_item():
    stream = NativeEventStream("codex")
    feed(stream, {"type": "item.completed", "item": {"type": "agent_message", "text": "old"}})
    feed(stream, {"type": "turn.completed"})
    feed(stream, {"type": "turn.started"})
    feed(stream, {"type": "item.started", "item": {"type": "agent_message", "text": "partial"}})
    assert not stream.text
    with pytest.raises(AgentError, match="native"):
        stream.require_completion()


def test_malformed_completed_agent_message_cannot_reuse_an_earlier_answer():
    stream = NativeEventStream("codex")
    feed(stream, {"type": "item.completed", "item": {"type": "agent_message", "text": "old"}})
    with pytest.raises(ProcessError, match="no string text"):
        feed(stream, {"type": "item.completed", "item": {"type": "agent_message"}})


@pytest.mark.parametrize("final", [False, True])
def test_partial_usage_is_preserved_without_inventing_a_final_field_report(final):
    class NativeProvider(AgentProvider):
        def invoke(self, *args, **kwargs):
            del args, kwargs
            events = [
                {"type": "usage", "usage": {"input_tokens": 3}},
                {"type": "usage", "usage": {}},
                {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}},
            ]
            if final:
                events.append({"type": "turn.completed", "usage": {"output_tokens": 2}})
            return _run_cli(
                [
                    sys.executable,
                    "-c",
                    "import sys; print(sys.argv[1])",
                    "\n".join(map(json.dumps, events)),
                ],
                "",
                timeout=10,
                name="codex",
                native_format="codex",
            )

    activity = AgentActivity()
    with capture_agent_activity(activity):
        if final:
            result = NativeProvider().complete("task")
            assert result.usage["input_tokens"] is None
        else:
            with pytest.raises(AgentError, match="native"):
                NativeProvider().complete("task")
    assert activity.usage["input_tokens"] == 3
    assert activity.usage_reported_attempts.get("input_tokens", 0) == 0
    assert activity.usage_reported_attempts.get("output_tokens", 0) == int(final)


def test_claude_final_usage_supersedes_partial_usage_and_preserves_primary_model():
    stream = NativeEventStream("claude")
    feed(stream, {"type": "assistant", "message": {"usage": {"input_tokens": 3}}})
    feed(stream, {"type": "stream_event", "usage": {"input_tokens": 4}})
    assert stream.usage["input_tokens"] == 7
    feed(
        stream,
        {
            "type": "result",
            "structured_output": {"answer": "valid"},
            "usage": {"input_tokens": 11},
            "modelUsage": {"primary": {"inputTokens": 20, "outputTokens": 10}},
        },
    )
    stream.require_completion()
    assert stream.usage["input_tokens"] == 20
    assert stream.diagnostics()["reported_model"] == "primary"
    assert stream.usage_observed


def test_usage_in_a_rejected_claude_result_is_not_lost():
    stream = NativeEventStream("claude")
    with pytest.raises(ProcessOutputLimitError, match="agent message"):
        feed(
            stream,
            {
                "type": "result",
                "result": "x" * (MAX_MESSAGE_BYTES + 1),
                "usage": {"input_tokens": 17, "output_tokens": 19},
            },
        )
    assert stream.partial_usage["input_tokens"] == 17
    assert stream.partial_usage["output_tokens"] == 19
    assert stream.usage_observed and stream.completion_observed


def test_interruption_preserves_observed_native_usage(monkeypatch):
    def interrupted(argv, **kwargs):
        del argv
        kwargs["stdout_consumer"](b'{"type":"usage","usage":{"input_tokens":17}}\n')
        raise KeyboardInterrupt

    monkeypatch.setattr("calcifier.agent_events.run_process", interrupted)
    activity = AgentActivity()
    with capture_agent_activity(activity), pytest.raises(KeyboardInterrupt):
        _run_cli(["offline"], "", timeout=10, name="codex", native_format="codex")
    assert activity.usage["input_tokens"] == 17
    assert activity.usage_reported_attempts == {}
    event = next(event for event in activity.events if event["phase"] == "native_transport")
    assert event["status"] == "failed"
    diagnostics = json.loads(str(event["detail"]))
    assert diagnostics["usage_observed"] and not diagnostics["completion_observed"]


@pytest.mark.parametrize("provider", ["codex", "claude"])
@pytest.mark.parametrize(
    "kind,overflow",
    [("event", False), ("message", False), ("model", False), ("model", True)],
    ids=[
        "event-at-limit",
        "unicode-message-at-limit",
        "unicode-model-at-limit",
        "model-over-limit",
    ],
)
def test_native_size_boundaries_preserve_complete_answers(provider, kind, overflow):
    # Generate the payload inside the real process; large argv values are not
    # portable, and binary UTF-8 writes avoid Windows console encoding defaults.
    source = """
import json, sys
provider, kind, overflow = sys.argv[1:4]
event_limit, message_limit = map(int, sys.argv[4:6])
def emit(event):
    sys.stdout.buffer.write((json.dumps(event, ensure_ascii=False) + '\\n').encode('utf-8'))
if kind == 'event':
    event = {'type': 'tool.progress', 'padding': ''}
    event['padding'] = 'x' * (event_limit - len(json.dumps(event).encode('utf-8')))
    emit(event)
answer = chr(233) * (message_limit // 2) if kind == 'message' else 'complete'
model = chr(233) * (256 + int(overflow)) if kind == 'model' else 'test-model'
usage = {'input_tokens': 11, 'output_tokens': 7}
if provider == 'codex':
    emit({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': answer}})
    emit({'type': 'turn.completed', 'model': model, 'usage': usage})
else:
    emit({'type': 'result', 'subtype': 'success', 'result': answer,
          'model': model, 'usage': usage})
"""
    argv = [
        sys.executable,
        "-c",
        source,
        provider,
        kind,
        str(int(overflow)),
        str(MAX_EVENT_BYTES),
        str(MAX_MESSAGE_BYTES),
    ]
    if overflow:
        with pytest.raises(AgentError, match="native model identifier exceeds") as rejected:
            _run_cli(argv, "", timeout=15, name=provider, native_format=provider)
        assert not isinstance(rejected.value, TransientAgentError)
    else:
        result = _run_cli(argv, "", timeout=15, name=provider, native_format=provider)
        expected = "é" * (MAX_MESSAGE_BYTES // 2) if kind == "message" else "complete"
        assert result.text == expected
        assert result.reported_model == ("é" * 256 if kind == "model" else "test-model")
        assert result.usage["input_tokens"] == 11
        assert result.usage["output_tokens"] == 7


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_saved_text_and_chunked_streams_decode_the_same_final_result(provider):
    from calcifier.native_events import parse_native_events

    if provider == "codex":
        events = [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "old"}},
            {"type": "turn.completed"},
            {"type": "turn.started"},
            {"type": "item.started", "item": {"type": "agent_message", "text": "partial"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "café"}},
            {"type": "turn.completed", "usage": {"input_tokens": 8, "output_tokens": 3}},
        ]
    else:
        events = [
            {"type": "assistant", "message": {"content": [{"text": "partial"}]}},
            {"type": "tool.progress", "discard": "auxiliary"},
            {"type": "result", "result": "café", "usage": {"input_tokens": 8, "output_tokens": 3}},
        ]
    text = "\r\n".join(json.dumps(event, ensure_ascii=False) for event in events)
    data = text.encode("utf-8")
    stream = NativeEventStream(provider)
    for offset in range(0, len(data), 3):
        stream.consume(data[offset : offset + 3])
    stream.finish()
    stream.require_completion()
    saved = parse_native_events(text, provider)
    saved.require_completion()
    assert saved.text == stream.text == "café"
    assert saved.usage == stream.usage
    assert saved.usage["input_tokens"] == 8 and saved.usage["output_tokens"] == 3


def test_transport_diagnostics_are_distinct_from_emitted_artifacts():
    from calcifier import emit, recipe, step
    from calcifier.trace_export import serialize_run, validate_trace

    @step
    def inspect_transport() -> str:
        result = _run_cli(
            [sys.executable, "-c", 'print(\'{"type":"result","result":"done"}\')'],
            "",
            timeout=5,
            name="neutral",
            native_format="claude",
        )
        emit(result.text, name="report")
        return result.text

    @recipe
    def workflow() -> str:
        return inspect_transport()

    run = workflow.run()
    assert run.ok and run.output == "done"
    assert len(run.artifacts) == 1 and run.artifacts[0].name == "report"
    assert len(run.agent_transport) == 1 and run.agent_transport[0]["status"] == "completed"
    exported = serialize_run(run)
    validate_trace(exported)
    assert exported["agent_transport"] == run.agent_transport
