"""Copy only Codex authentication into a private, temporary configuration home."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tomllib
from importlib import import_module
from pathlib import Path

from .filesystem_contract import AgentError


def seed_codex_auth(config_home: Path) -> None:
    original_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    mode = _credential_store_mode(original_home)
    if mode == "ephemeral":
        return
    try:
        data = _load_codex_auth(original_home, mode)
        if data is not None:
            destination = config_home / "auth.json"
            # The containing TemporaryDirectory is already private (0700 on POSIX).
            destination.write_bytes(data)
            destination.chmod(0o600)
    except OSError:
        raise AgentError("could not prepare isolated Codex authentication") from None


def _credential_store_mode(original_home: Path) -> str:
    try:
        with (original_home / "config.toml").open("rb") as source:
            config = tomllib.load(source)
    except FileNotFoundError:
        return "file"
    except (OSError, tomllib.TOMLDecodeError):
        raise AgentError(
            "could not read Codex credential storage setting from config.toml"
        ) from None
    mode = config.get("cli_auth_credentials_store", "file")
    if mode not in ("file", "keyring", "auto", "ephemeral"):
        raise AgentError("unsupported Codex cli_auth_credentials_store setting")
    return mode


def _load_codex_auth(original_home: Path, mode: str) -> bytes | None:
    source = original_home / "auth.json"
    if mode in {"keyring", "auto"}:
        try:
            value = _read_keyring_auth(original_home)
        except Exception:
            # Backend failures may contain secret material. Do not persist their text.
            if mode == "keyring" or not source.is_file():
                raise AgentError(
                    "could not read Codex authentication from the OS keyring; "
                    "unlock the credential store and check `codex login status`"
                ) from None
            value = None
        if value is not None:
            return _keyring_auth_bytes(value)
        if mode == "keyring":
            raise AgentError("no Codex authentication found in the OS keyring; run `codex login`")
    try:
        return source.read_bytes()
    except FileNotFoundError:
        return None


def _keyring_auth_bytes(value: str) -> bytes:
    try:
        parsed = json.loads(value)
    except ValueError:
        raise AgentError("invalid Codex authentication in the OS keyring") from None
    if not isinstance(parsed, dict):
        raise AgentError("invalid Codex authentication in the OS keyring")
    return value.encode("utf-8")


def _codex_store_key(original_home: Path) -> str:
    # Match Codex's compute_store_key, including canonicalization of existing homes:
    # https://github.com/openai/codex/blob/main/codex-rs/login/src/auth/storage.rs
    try:
        canonical = str(original_home.resolve(strict=True))
    except OSError:
        canonical = str(original_home)
    else:
        # Rust canonicalize retains Windows' extended path prefix; pathlib removes it.
        if sys.platform == "win32" and not canonical.startswith("\\\\?\\"):
            canonical = (
                "\\\\?\\UNC\\" + canonical[2:]
                if canonical.startswith("\\\\")
                else "\\\\?\\" + canonical
            )
    digest = hashlib.sha256(canonical.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"cli|{digest}"


def _read_keyring_auth(original_home: Path) -> str | None:
    account = _codex_store_key(original_home)
    service = "Codex Auth"
    if sys.platform == "win32":
        # keyring-rs uses username.service as the Windows Credential Manager target.
        backend = import_module("keyring.backends.Windows").WinVaultKeyring()
        return backend.get_password(f"{account}.{service}", account)
    module = "macOS" if sys.platform == "darwin" else "SecretService"
    backend = import_module(f"keyring.backends.{module}").Keyring()
    return backend.get_password(service, account)
