"""Host CLI providers and custom command adapters."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent_config import _CLAUDE_READ_TOOLS
from .agent_events import (
    AgentProvider,
    AgentResult,
    _run_cli,
)
from .filesystem_contract import AgentError, FilesystemContract
from .native_events import parse_adapter_envelope


@dataclass(frozen=True)
class HostCommand:
    argv: list[str]
    preflight_features: tuple[str, ...]


def claude_command(
    binary: str,
    *,
    model: str | None = None,
    filesystem: FilesystemContract | None = None,
    schema: dict[str, Any] | None = None,
) -> HostCommand:
    command = [
        binary,
        "-p",
        "--output-format=stream-json",
        "--verbose",
    ]
    writable = filesystem.writable_paths if filesystem is not None else ()
    tools = _CLAUDE_READ_TOOLS + (",Edit,Write" if writable else "")
    allowed = [
        _CLAUDE_READ_TOOLS,
        *(_claude_edit_rule(path) for path in writable),
    ]
    features: list[str] = []
    for flag in ("--no-session-persistence", "--safe-mode"):
        command.append(flag)
        features.append(flag)
    command.extend(
        (
            "--restricted",
            "--permission-prompts",
            "none",
            "--permission-mode",
            "dontAsk" if writable else "plan",
            "--strict-mcp-config",
        )
    )
    if writable:
        # Edit path rules also govern Write; protect directories and worktree .git files.
        command.extend(("--disallowedTools", "Edit(//**/.git/**)", "Edit(//**/.git)"))
    command.extend(("--tools", tools))
    features.append("--tools")
    command.extend(
        (
            "--allowedTools",
            *allowed,
            "--max-turns",
            "12",
        )
    )
    if model is not None:
        command.extend(("--model", model))
    if filesystem is not None:
        for path in filesystem.readable_paths:
            if path != filesystem.working_directory:
                command.extend(("--add-dir", str(path)))
    if schema is not None:
        command.extend(("--json-schema", json.dumps(schema, sort_keys=True)))
    return HostCommand(command, tuple(features))


def _claude_edit_rule(path: Path) -> str:
    # Claude uses gitignore patterns with // for filesystem-root absolute paths.
    pattern = path.as_posix()
    if path.drive and pattern[1:2] == ":":
        pattern = f"/{pattern[0].lower()}{pattern[2:]}"
    pattern = "".join("\\" + char if char in "*?[]\\" else char for char in pattern)
    return f"Edit(/{pattern.rstrip('/')}/**)"


def codex_command(
    binary: str,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    filesystem: FilesystemContract | None = None,
    schema_path: Path | None = None,
) -> HostCommand:
    command = [binary, "exec", "--color", "never"]
    writable = filesystem.writable_paths if filesystem is not None else ()
    features: list[str] = []
    command.append("--ephemeral")
    features.append("--ephemeral")
    command.append("--json")
    for flag in ("--ignore-user-config", "--ignore-rules"):
        command.append(flag)
        features.append(flag)
    command.extend(
        (
            "--sandbox",
            "workspace-write" if writable else "read-only",
            "--skip-git-repo-check",
            "--disable",
            "multi_agent",
            "--disable",
            "plugins",
            "--disable",
            "apps",
            "--disable",
            "browser_use",
            "--disable",
            "computer_use",
            "-c",
            "project_doc_max_bytes=0",
        )
    )
    if model is not None:
        command.extend(("--model", model))
    if reasoning_effort is not None:
        command.extend(("-c", f"model_reasoning_effort={json.dumps(reasoning_effort)}"))
    if filesystem is not None:
        command.extend(("-C" if writable else "--cd", str(filesystem.working_directory)))
        additional = [
            path for path in writable if not filesystem.working_directory.is_relative_to(path)
        ]
        for path in additional:
            command.extend(("--add-dir", str(path)))
        if writable:
            command.extend(("-c", "sandbox_workspace_write.exclude_slash_tmp=true"))
    if schema_path is not None:
        command.extend(("--output-schema", str(schema_path)))
        features.append("--output-schema")
    command.append("-")
    return HostCommand(command, tuple(features))


def host_provider_check(provider: str) -> tuple[list[str], tuple[str, ...]]:
    if provider == "claude":
        return [provider, "--help"], claude_command(provider).preflight_features
    if provider == "codex":
        return [provider, "exec", "--help"], codex_command(
            provider, schema_path=Path("output-schema.json")
        ).preflight_features
    raise ValueError(f"unsupported agent provider: {provider}")


def command_adapter_environment(
    filesystem: FilesystemContract | None,
    schema: dict[str, Any] | None,
) -> dict[str, str]:
    contract = {
        "version": 1,
        "filesystem": None if filesystem is None else filesystem.as_dict(),
        "schema": schema,
    }
    environment = dict(os.environ)
    environment["CALCIFIER_AGENT_PROTOCOL"] = "1"
    environment["CALCIFIER_AGENT_CONTRACT"] = json.dumps(contract, sort_keys=True)
    return environment


# -- providers ----------------------------------------------------------------


class ClaudeCli(AgentProvider):
    """Headless Claude via the ``claude -p`` CLI."""

    name = "claude"

    target_name = "claude"

    def __init__(
        self,
        binary: str = "claude",
        *,
        role: str = "claude",
        default_timeout: float | None = None,
        model: str | None = None,
    ) -> None:
        self.name = role
        self.binary = binary
        self.default_timeout = default_timeout
        self.model = model

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    def invoke(
        self,
        prompt: str,
        *,
        timeout: float | None = None,
        filesystem: FilesystemContract | None = None,
        schema: dict[str, Any] | None = None,
    ) -> AgentResult:
        command = claude_command(
            self.binary, model=self.model, filesystem=filesystem, schema=schema
        )
        return _run_cli(
            command.argv,
            stdin=prompt,
            timeout=self.effective_timeout(timeout),
            name=self.name,
            filesystem=filesystem,
            native_format="claude",
        )


class CodexCli(AgentProvider):
    """Headless Codex via the ``codex exec`` CLI."""

    name = "codex"
    native_structured_output = True

    target_name = "codex"

    def __init__(
        self,
        binary: str = "codex",
        *,
        role: str = "codex",
        default_timeout: float | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.name = role
        self.binary = binary
        self.default_timeout = default_timeout
        self.model = model
        self.reasoning_effort = reasoning_effort

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    def invoke(
        self,
        prompt: str,
        *,
        timeout: float | None = None,
        filesystem: FilesystemContract | None = None,
        schema: dict[str, Any] | None = None,
    ) -> AgentResult:
        with tempfile.TemporaryDirectory(prefix="calcifier-codex-home-") as config_home:
            schema_path: Path | None = None
            if schema is not None:
                schema_path = Path(config_home) / "output-schema.json"
                schema_path.write_text(
                    json.dumps(schema, sort_keys=True), encoding="utf-8", newline="\n"
                )
            command = codex_command(
                self.binary,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                filesystem=filesystem,
                schema_path=schema_path,
            )
            _seed_codex_auth(Path(config_home))
            clean_env = {
                key: value
                for key, value in os.environ.items()
                if key in {"PATH", "OPENAI_API_KEY", "SSL_CERT_FILE", "SSL_CERT_DIR", "TMPDIR"}
            }
            clean_env["CODEX_HOME"] = config_home
            return _run_cli(
                command.argv,
                stdin=prompt,
                timeout=self.effective_timeout(timeout),
                name=self.name,
                filesystem=filesystem,
                env=clean_env,
                native_format="codex",
            )


class CommandCli(AgentProvider):
    """An explicitly configured argv-only subprocess provider."""

    target_type = "command"

    def __init__(
        self,
        role: str,
        command: tuple[str, ...],
        *,
        default_timeout: float | None = None,
    ) -> None:
        self.name = role
        self.command = command
        self.target_name = command[0]
        self.default_timeout = default_timeout

    def is_available(self) -> bool:
        return shutil.which(self.command[0]) is not None

    def invoke(
        self,
        prompt: str,
        *,
        timeout: float | None = None,
        filesystem: FilesystemContract | None = None,
        schema: dict[str, Any] | None = None,
    ) -> AgentResult:
        adapter_env = command_adapter_environment(filesystem, schema)
        return _run_cli(
            list(self.command),
            stdin=prompt,
            timeout=self.effective_timeout(timeout),
            name=self.name,
            filesystem=filesystem,
            env=adapter_env,
            event_parser=parse_adapter_envelope,
        )


def _seed_codex_auth(config_home: Path) -> None:
    original_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    source = original_home / "auth.json"
    if not source.is_file():
        return
    destination = config_home / "auth.json"
    try:
        shutil.copyfile(source, destination)
        destination.chmod(0o600)
    except OSError as exc:
        raise AgentError(f"could not prepare isolated Codex authentication: {exc}") from exc
