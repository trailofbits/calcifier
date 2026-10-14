"""Deterministic agent provider for offline tests."""

from __future__ import annotations

from typing import Any

from calcifier import agent_providers
from calcifier.agent_events import AgentProvider, AgentResult
from calcifier.filesystem_contract import FilesystemContract


class StubProvider(AgentProvider):
    """Return a canned response selected by prompt prefix."""

    name = "stub"

    def __init__(self, responses: dict[str, str] | None = None, default: str = "") -> None:
        self._responses = dict(responses or {})
        self._default = default
        self.default_timeout = 30

    def invoke(
        self,
        prompt: str,
        *,
        timeout: float | None = None,
        filesystem: FilesystemContract | None = None,
        schema: dict[str, Any] | None = None,
    ) -> AgentResult:
        del timeout, filesystem, schema
        for key, value in self._responses.items():
            if prompt.startswith(key):
                return self._make(value)
        return self._make(self._default)

    def _make(self, text: str) -> AgentResult:
        return AgentResult(
            text=text,
            raw_stdout=text,
            raw_stderr="",
            exit_code=0,
            provider=self.name,
        )


def capture_run(monkeypatch):
    calls = []

    def fake_run(cmd, stdin, *, timeout, name, **kwargs):  # noqa: ANN001
        del kwargs
        calls.append((cmd, stdin, timeout, name))
        return AgentResult("ok", "ok", "", 0, name)

    monkeypatch.setattr(agent_providers, "_run_cli", fake_run)
    return calls
