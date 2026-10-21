"""Strict role-to-provider configuration for host agents."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

from .exceptions import CalcifierError, ErrorFamily
from .hashing import sha256_hex
from .json_config import load_json_config
from .validation import is_positive_number


class AgentConfigError(CalcifierError, ValueError):
    """Raised when an agent configuration is invalid."""

    family = ErrorFamily.SETUP


ROLE_RE = re.compile(r"[A-Za-z][A-Za-z0-9._-]*")
_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*")
BUILTIN_AGENTS = frozenset({"claude", "codex"})
AGENT_CAPABILITIES = frozenset({"read", "write", "commands", "research"})
# Read/search tools are always enabled; writable calls add scoped Edit/Write separately.
_CLAUDE_READ_TOOLS = "Read,Glob,Grep"
_CODEX_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
)


@dataclass(frozen=True)
class AgentRuntime:
    kind: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind}


@dataclass(frozen=True)
class AgentRoute:
    role: str
    builtin: str | None
    command: tuple[str, ...] | None
    runtime: AgentRuntime
    timeout_seconds: int | None = None
    remediation: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    capabilities: tuple[str, ...] | None = None

    @property
    def target_type(self) -> str:
        return "builtin" if self.builtin is not None else "command"

    @property
    def target_name(self) -> str:
        if self.builtin is not None:
            return self.builtin
        if self.command is None:  # defensive; parser enforces exactly one target
            raise AgentConfigError(f"agent {self.role!r} has no target")
        return self.command[0]

    def as_dict(self) -> dict[str, Any]:
        target: dict[str, Any]
        if self.builtin is not None:
            target = {"builtin": self.builtin}
        else:
            target = {"command": list(self.command or ())}
        result: dict[str, Any] = {
            "target": target,
            "runtime": self.runtime.as_dict(),
        }
        if self.timeout_seconds is not None:
            result["timeout_seconds"] = self.timeout_seconds
        if self.remediation is not None:
            result["remediation"] = self.remediation
        if self.model is not None:
            result["model"] = self.model
        if self.reasoning_effort is not None:
            result["reasoning_effort"] = self.reasoning_effort
        if self.capabilities is not None:
            result["capabilities"] = list(self.capabilities)
        return result


@dataclass(frozen=True)
class AgentConfiguration:
    routes: Mapping[str, AgentRoute] = field(default_factory=dict)

    def fingerprint(self, roles: tuple[str, ...] | None = None) -> str:
        selected = sorted(self.routes) if roles is None else sorted(set(roles))
        payload = {
            "version": 1,
            "agents": {
                role: self.routes[role].as_dict() for role in selected if role in self.routes
            },
        }
        # gremlin: pardon[equivalent] the payload is already built in sorted, fixed order.
        material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256_hex(material.encode())


def override_agent_timeouts(
    configuration: AgentConfiguration,
    values: list[str] | tuple[str, ...] | None,
) -> AgentConfiguration:
    """Apply strict ``ROLE=SECONDS`` overrides to a version-1 configuration."""
    routes = dict(configuration.routes)
    for value in values or ():
        if "=" not in value:
            raise AgentConfigError("--agent-timeout expects ROLE=SECONDS")
        role, raw_seconds = value.split("=", 1)
        if role not in routes:
            raise AgentConfigError(f"--agent-timeout names unknown role {role!r}")
        try:
            seconds = int(raw_seconds)
        except ValueError as exc:
            raise AgentConfigError("--agent-timeout seconds must be an integer") from exc
        if not is_positive_number(seconds, integral=True):
            raise AgentConfigError("--agent-timeout seconds must be positive")
        routes[role] = replace(routes[role], timeout_seconds=seconds)
    return AgentConfiguration(routes=routes)


def load_agent_config(path: str | Path) -> AgentConfiguration:
    """Load a strict version-1 role-to-subprocess configuration."""
    config_path = Path(path)
    try:
        raw = load_json_config(config_path, object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentConfigError(f"failed to load agent config {config_path}: {exc}") from exc
    root = _mapping(raw, "agent config")
    _reject_unknown(root, {"version", "agents"}, "agent config")
    if root.get("version") != 1:
        raise AgentConfigError("agent config version must be 1")
    raw_agents = _mapping(root.get("agents"), "agent config agents")
    routes: dict[str, AgentRoute] = {}
    for role, value in raw_agents.items():
        if not isinstance(role, str) or ROLE_RE.fullmatch(role) is None:
            raise AgentConfigError(f"invalid agent role name: {role!r}")
        if role in BUILTIN_AGENTS:
            raise AgentConfigError(f"agent role {role!r} is reserved")
        routes[role] = _parse_route(role, value)
    return AgentConfiguration(routes=routes)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AgentConfigError(f"agent config contains duplicate key {key!r}")
        result[key] = value
    return result


def _parse_route(role: str, raw: object) -> AgentRoute:
    route = _mapping(raw, f"agent {role!r}")
    _reject_unknown(
        route,
        {
            "target",
            "runtime",
            "timeout_seconds",
            "remediation",
            "model",
            "reasoning_effort",
            "capabilities",
        },
        f"agent {role!r}",
    )
    target = _mapping(route.get("target"), f"agent {role!r} target")
    _reject_unknown(target, {"builtin", "command"}, f"agent {role!r} target")
    has_builtin = "builtin" in target
    has_command = "command" in target
    if has_builtin == has_command:
        raise AgentConfigError(
            f"agent {role!r} target must contain exactly one of builtin or command"
        )
    builtin: str | None = None
    command: tuple[str, ...] | None = None
    if has_builtin:
        builtin_value = target["builtin"]
        if builtin_value not in BUILTIN_AGENTS:
            raise AgentConfigError(
                f"agent {role!r} builtin must be one of {sorted(BUILTIN_AGENTS)}"
            )
        builtin = str(builtin_value)
    else:
        command_value = target["command"]
        if not isinstance(command_value, list) or not command_value:
            raise AgentConfigError(f"agent {role!r} command must be a non-empty argv array")
        if not all(isinstance(part, str) and part for part in command_value):
            raise AgentConfigError(f"agent {role!r} command entries must be non-empty strings")
        command = tuple(command_value)
    runtime = _parse_runtime(role, route.get("runtime"))
    timeout = _parse_timeout(role, route.get("timeout_seconds"))
    remediation = route.get("remediation")
    if remediation is not None and not isinstance(remediation, str):
        raise AgentConfigError(f"agent {role!r} remediation must be a string")
    model = _parse_model(role, route.get("model"), builtin)
    reasoning_effort = _parse_codex_reasoning_effort(role, route.get("reasoning_effort"), builtin)
    capabilities = _parse_capabilities(role, route.get("capabilities"), builtin)
    return AgentRoute(
        role,
        builtin,
        command,
        runtime,
        timeout,
        remediation,
        model,
        reasoning_effort,
        capabilities,
    )


def _parse_capabilities(role: str, value: object, builtin: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if builtin is not None:
        raise AgentConfigError(f"agent {role!r} builtin capabilities cannot be overridden")
    if not isinstance(value, list) or any(
        not isinstance(item, str) or item not in AGENT_CAPABILITIES for item in value
    ):
        raise AgentConfigError(
            f"agent {role!r} capabilities must be an array drawn from {sorted(AGENT_CAPABILITIES)}"
        )
    return tuple(sorted(set(value)))


def _parse_model(role: str, value: object, builtin: str | None) -> str | None:
    if value is None:
        return None
    if builtin not in {"claude", "codex"}:
        raise AgentConfigError(f"agent {role!r} model is only valid for builtin claude or codex")
    if not isinstance(value, str) or _MODEL_RE.fullmatch(value) is None:
        raise AgentConfigError(f"agent {role!r} model must be a non-empty model name")
    return value


def _parse_codex_reasoning_effort(role: str, value: object, builtin: str | None) -> str | None:
    if value is None:
        return None
    if builtin != "codex":
        raise AgentConfigError(f"agent {role!r} reasoning_effort is only valid for builtin codex")
    if value not in _CODEX_REASONING_EFFORTS:
        raise AgentConfigError(
            f"agent {role!r} reasoning_effort must be one of {sorted(_CODEX_REASONING_EFFORTS)}"
        )
    return str(value)


def _parse_timeout(role: str, value: object) -> int:
    if value is None:
        raise AgentConfigError(f"agent {role!r} requires timeout_seconds")
    if not is_positive_number(value, integral=True):
        raise AgentConfigError(f"agent {role!r} timeout_seconds must be a positive integer")
    return cast("int", value)


def _parse_runtime(role: str, raw: object) -> AgentRuntime:
    runtime = _mapping(raw, f"agent {role!r} runtime")
    if runtime.get("kind") != "host":
        raise AgentConfigError(f"agent {role!r} runtime kind must be host")
    if set(runtime) != {"kind"}:
        raise AgentConfigError(f"host runtime for agent {role!r} accepts only kind")
    return AgentRuntime(kind="host")


def _mapping(raw: object, context: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise AgentConfigError(f"{context} must be a JSON object")
    if not all(isinstance(key, str) for key in raw):
        raise AgentConfigError(f"{context} keys must be strings")
    return cast("Mapping[str, Any]", raw)


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise AgentConfigError(f"{context} has unknown fields: {unknown}")
