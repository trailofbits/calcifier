"""Incremental native JSONL decoding with bounded semantic retention."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal, cast

from .filesystem_contract import AgentError
from .subprocesses import ProcessError, ProcessOutputLimitError
from .usage import add_usage, is_usage_value, normalized_usage

# Individual tool events have room beyond the final answer. Scaffolding's
# existing 256 KiB response limit remains stricter than the generic answer cap.
MAX_EVENT_BYTES = 2 * 1024 * 1024
MAX_MESSAGE_BYTES = 1024 * 1024
MAX_EVENTS = 10_000
MAX_EVENT_TYPES = 32


class NativeEventStream:
    """Keep one final message, completion state, usage and bounded counters."""

    def __init__(self, provider: Literal["codex", "claude"]) -> None:
        self.provider = provider
        self.pending = bytearray()
        self.total_bytes = 0
        self.event_count = 0
        self.largest_event_bytes = 0
        self.largest_message_bytes = 0
        self.event_types: Counter[str] = Counter()
        self.text = ""
        self.usage: dict[str, float | int | None] = {}
        self.partial_usage: dict[str, float | int | None] = {}
        self.reported_model: str | None = None
        self._usage_model: str | None = None
        self.completion_observed = False
        self.usage_observed = False
        self.terminal: str | None = None

    def consume(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        offset = 0
        while offset < len(chunk):
            newline = chunk.find(b"\n", offset)
            end = len(chunk) if newline < 0 else newline
            size = len(self.pending) + end - offset
            self.largest_event_bytes = max(self.largest_event_bytes, size)
            if size > MAX_EVENT_BYTES:
                raise ProcessOutputLimitError(f"native event exceeds {MAX_EVENT_BYTES} bytes")
            self.pending.extend(chunk[offset:end])
            if newline < 0:
                break
            self._event(bytes(self.pending))
            self.pending.clear()
            offset = newline + 1

    def finish(self) -> None:
        if self.pending:
            self._event(bytes(self.pending))
            self.pending.clear()

    def _event(self, line: bytes) -> None:
        self.event_count += 1
        if self.event_count > MAX_EVENTS:
            raise ProcessOutputLimitError(f"native stream exceeds {MAX_EVENTS} events")
        if not line.strip():
            return
        try:
            event = json.loads(line.decode("utf-8"))
        except (UnicodeError, ValueError, RecursionError) as exc:
            raise ProcessError("native event is not a valid UTF-8 JSON object") from exc
        if not isinstance(event, dict):
            raise ProcessError("native event is not a JSON object")
        kind = event.get("type")
        name = kind if isinstance(kind, str) and len(kind) <= 64 else "<other>"
        if name not in self.event_types and len(self.event_types) >= MAX_EVENT_TYPES - 1:
            name = "<other>"
        self.event_types[name] += 1
        model = event.get("model")
        if isinstance(model, str):
            self.reported_model = self._model(model)
        try:
            if self.provider == "codex":
                self._codex(event)
            else:
                self._claude(event)
        finally:
            self.partial_usage.update(
                {name: value for name, value in self.usage.items() if value is not None}
            )

    @staticmethod
    def _model(model: str) -> str:
        if len(model) > 256:
            raise ProcessOutputLimitError("native model identifier exceeds 256 characters")
        return model

    def _message(self, text: str) -> None:
        try:
            size = len(text.encode("utf-8"))
        except UnicodeError as exc:
            raise ProcessError("native agent message contains invalid Unicode") from exc
        self.largest_message_bytes = max(self.largest_message_bytes, size)
        if size > MAX_MESSAGE_BYTES:
            raise ProcessOutputLimitError(f"native agent message exceeds {MAX_MESSAGE_BYTES} bytes")
        self.text = text

    def _codex(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        usage = event.get("usage")
        if isinstance(usage, Mapping):
            self.usage_observed = True
            self.usage = normalized_usage(usage)
        if isinstance(kind, str) and kind in {
            "turn.started",
            "turn.completed",
            "turn.failed",
            "error",
        }:
            self.terminal = kind
            if kind == "turn.started":
                self.text = ""
            if kind == "turn.completed":
                self.completion_observed = True
        item = event.get("item")
        if (
            kind == "item.completed"
            and isinstance(item, Mapping)
            and item.get("type") == "agent_message"
        ):
            text = item.get("text")
            if not isinstance(text, str):
                raise ProcessError("native agent message has no string text")
            self._message(text)

    def _claude(self, event: dict[str, Any]) -> None:
        if event.get("type") == "result":
            self.completion_observed = True
            self.terminal = (
                "success"
                if not event.get("is_error") and event.get("subtype", "success") == "success"
                else "failed"
            )
            text, usage = _claude_result(event)
            self.usage = usage
            self.usage_observed |= isinstance(event.get("usage"), Mapping)
            self.usage_observed |= isinstance(event.get("modelUsage"), Mapping)
            model = provider_reported_model(json.dumps(event))
            if model is not None:
                self._usage_model = self._model(model)
            self._message(text)
            return
        message = event.get("message")
        raw = message.get("usage") if isinstance(message, Mapping) else None
        if not isinstance(raw, Mapping):
            raw = event.get("usage")
        if isinstance(raw, Mapping):
            self.usage_observed = True
            add_usage(self.usage, normalized_usage(raw))

    def require_completion(self) -> None:
        if self.provider == "codex" and self.terminal != "turn.completed":
            raise AgentError("Codex CLI returned no successfully completed native turn")
        if self.provider == "claude" and self.terminal != "success":
            raise AgentError("Claude CLI returned no completed native result or its turn failed")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "event_count": self.event_count,
            "event_types": dict(self.event_types),
            "largest_event_bytes": self.largest_event_bytes,
            "largest_message_bytes": self.largest_message_bytes,
            "retained_final_message_bytes": len(self.text.encode("utf-8")),
            "pending_event_bytes": len(self.pending),
            "completion_observed": self.completion_observed,
            "usage_observed": self.usage_observed,
            "terminal": self.terminal,
            "reported_model": self.reported_model or self._usage_model,
            "format": self.provider,
        }


_CLAUDE_MODEL_USAGE = {
    "input_tokens": "inputTokens",
    "output_tokens": "outputTokens",
    "cache_read_tokens": "cacheReadInputTokens",
    "cache_write_tokens": "cacheCreationInputTokens",
    "nested_model_tokens": "thinkingTokens",
}


def provider_reported_model(stdout: str) -> str | None:
    """Extract a canonical provider model without conflating it with configuration."""
    events = _structured_events(stdout)
    direct = [event.get("model") for event in events if isinstance(event.get("model"), str)]
    if direct:
        return cast("str", direct[-1])
    results = [event for event in events if event.get("type") == "result"]
    if not results:
        return None
    usage = results[-1].get("modelUsage")
    if not isinstance(usage, Mapping):
        return None
    candidates = [
        (name, details)
        for name, details in usage.items()
        if isinstance(name, str) and isinstance(details, Mapping)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item[1].get("outputTokens", 0) if type(item[1].get("outputTokens", 0)) is int else 0,
            item[0],
        ),
    )[0]


def _structured_events(stdout: str) -> list[dict[str, Any]]:
    events = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _claude_model_usage(model_usage: object) -> dict[str, float | int | None]:
    """Sum Claude's per-model accounting into the trace's token fields.

    Sum a field only when every model reports it; otherwise keep terminal usage.
    ``cost_usd`` stays the provider's total, which already includes all models.
    """
    if not isinstance(model_usage, Mapping):
        return {}
    models = list(model_usage.values())
    totals: dict[str, float | int | None] = {}
    for normalized, name in _CLAUDE_MODEL_USAGE.items():
        # Incomplete per-model fields keep the terminal result's `usage` value.
        counts = [
            cast("int | float", model[name])
            for model in models
            if isinstance(model, Mapping) and is_usage_value(model.get(name))
        ]
        if counts and len(counts) == len(models):
            totals[normalized] = sum(counts)
    return totals


def parse_adapter_envelope(stdout: str) -> tuple[str, dict[str, float | int | None]]:
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise AgentError("custom adapter did not return a v1 JSON envelope") from exc
    if not isinstance(envelope, Mapping) or envelope.get("version") != 1:
        raise AgentError("custom adapter did not return protocol version 1")
    output = envelope.get("output")
    usage = envelope.get("usage", {})
    if not isinstance(output, str) or not isinstance(usage, Mapping):
        raise AgentError("custom adapter v1 envelope has invalid output or usage")
    return output, normalized_usage(cast("Mapping[str, Any]", usage), cost=usage.get("cost_usd"))


def _claude_result(event: dict[str, Any]) -> tuple[str, dict[str, float | int | None]]:
    structured = event.get("structured_output")
    text = json.dumps(structured) if structured is not None else event.get("result")
    raw_usage = event.get("usage")
    raw = raw_usage if isinstance(raw_usage, Mapping) else {}
    usage = normalized_usage(raw, cost=event.get("total_cost_usd"))
    details = raw.get("output_tokens_details")
    if isinstance(details, Mapping) and is_usage_value(details.get("thinking_tokens")):
        usage["nested_model_tokens"] = details["thinking_tokens"]
    usage.update(_claude_model_usage(event.get("modelUsage")))
    return (text if isinstance(text, str) else "", usage)


def parse_native_events(stdout: str, provider: Literal["claude", "codex"]) -> NativeEventStream:
    """Decode saved JSONL with exactly the incremental production parser and limits."""
    stream = NativeEventStream(provider)
    stream.consume(stdout.encode("utf-8"))
    stream.finish()
    return stream


def parse_claude_events(stdout: str) -> tuple[str, dict[str, float | int | None]]:
    stream = parse_native_events(stdout, "claude")
    return stream.text, stream.usage


def parse_codex_events(stdout: str) -> tuple[str, dict[str, float | int | None]]:
    stream = parse_native_events(stdout, "codex")
    return stream.text, stream.usage
