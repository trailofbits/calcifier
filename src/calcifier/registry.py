"""Configured provider lookup for recipe execution."""

from __future__ import annotations

from typing import Any

from .agent_config import BUILTIN_AGENTS, AgentConfigError, AgentConfiguration, AgentRoute
from .agent_events import AgentProvider
from .agent_providers import ClaudeCli, CodexCli, CommandCli
from .filesystem_contract import AgentError

# -- registry -----------------------------------------------------------------


class Registry:
    """Lookup table for available providers. A recipe references providers by name."""

    def __init__(self, *, configuration_hash: str = "") -> None:
        self._providers: dict[str, AgentProvider] = {}
        self.configuration_hash = configuration_hash

    def register(self, provider: AgentProvider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> AgentProvider:
        if name not in self._providers:
            raise AgentError(f"unknown agent {name!r}; registered: {sorted(self._providers)}")
        return self._providers[name]

    def known(self) -> list[str]:
        return sorted(self._providers)

    def runtime_summary(self, names: tuple[str, ...]) -> dict[str, Any]:
        if not self.configuration_hash:
            raise AgentError("runtime summary requires a configured registry")
        return {
            "mode": "configured",
            "configuration_hash": self.configuration_hash,
            "agents": {
                name: {
                    **self.get(name).trace_metadata(),
                    "timeout_seconds": self.get(name).default_timeout,
                }
                for name in sorted(set(names))
            },
        }


def configured_registry(
    configuration: AgentConfiguration,
    required_roles: tuple[str, ...],
) -> Registry:
    """Resolve configured roles without any implicit host fallback."""
    registry = Registry(configuration_hash=configuration.fingerprint())
    for route in configuration.routes.values():
        provider = _provider_for_route(route)
        provider.remediation = route.remediation
        registry.register(provider)
    for role in sorted(set(required_roles)):
        if role in configuration.routes:
            continue
        if role in BUILTIN_AGENTS:
            raise AgentConfigError(
                f"implicit builtin agent {role!r} is not allowed with --agent-config; "
                "declare a named role and configure its target and runtime"
            )
        raise AgentConfigError(f"agent role {role!r} is declared but missing from --agent-config")
    return registry


def _provider_for_route(route: AgentRoute) -> AgentProvider:
    if route.builtin == "claude":
        return ClaudeCli(role=route.role, default_timeout=route.timeout_seconds, model=route.model)
    if route.builtin == "codex":
        return CodexCli(
            role=route.role,
            default_timeout=route.timeout_seconds,
            model=route.model,
            reasoning_effort=route.reasoning_effort,
        )
    if route.command is None:  # pragma: no cover - parsed configurations forbid this
        raise AgentConfigError(f"agent {route.role!r} has no command")
    return CommandCli(
        route.role,
        route.command,
        default_timeout=route.timeout_seconds,
        capabilities=route.capabilities,
    )
