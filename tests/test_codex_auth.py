"""Codex credentials survive configuration isolation without touching real keyrings."""

import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from calcifier import agent_providers, codex_auth
from calcifier.agent_events import AgentResult
from calcifier.agent_providers import CodexCli
from calcifier.filesystem_contract import AgentError


@pytest.mark.parametrize("mode", ["keyring", "auto"])
@pytest.mark.parametrize("stale_file", [False, True])
def test_codex_uses_the_original_homes_keyring(tmp_path, monkeypatch, mode, stale_file):
    original = tmp_path / "codex-home"
    original.mkdir()
    (original / "config.toml").write_text(
        f'cli_auth_credentials_store = "{mode}"\nmodel = "unwanted-model"\n',
        encoding="utf-8",
    )
    if stale_file:
        (original / "auth.json").write_text('{"token":"stale"}', encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(original))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    credentials = '{"OPENAI_API_KEY":null,"tokens":{"access_token":"fixture"}}'
    lookups = []

    def get_password(service, account):
        lookups.append((service, account))
        return credentials

    backend = SimpleNamespace(
        Keyring=lambda: SimpleNamespace(get_password=get_password),
        WinVaultKeyring=lambda: SimpleNamespace(get_password=get_password),
    )
    for module in ("macOS", "SecretService", "Windows"):
        monkeypatch.setitem(sys.modules, f"keyring.backends.{module}", backend)
    homes = []

    def fake_run(cmd, stdin, *, timeout, name, env, **kwargs):
        home = Path(env["CODEX_HOME"])
        homes.append(home)
        assert home != original
        assert json.loads((home / "auth.json").read_text(encoding="utf-8")) == json.loads(
            credentials
        )
        assert not (home / "config.toml").exists()
        assert 'cli_auth_credentials_store="file"' in cmd
        if sys.platform != "win32":
            assert (home / "auth.json").stat().st_mode & 0o777 == 0o600
        return AgentResult("ok", "ok", "", 0, "codex")

    monkeypatch.setattr(agent_providers, "_run_cli", fake_run)

    CodexCli().invoke("prompt", timeout=3)

    assert len(lookups) == 1
    account = codex_auth._codex_store_key(original)
    service = f"{account}.Codex Auth" if sys.platform == "win32" else "Codex Auth"
    assert lookups == [(service, account)]
    assert not homes[0].exists()
    assert (original / "config.toml").is_file()
    assert (original / "auth.json").exists() == stale_file
    if stale_file:
        assert (original / "auth.json").read_text(encoding="utf-8") == '{"token":"stale"}'


@pytest.fixture
def auth_homes(tmp_path, monkeypatch):
    original = tmp_path / "codex-home"
    isolated = tmp_path / "isolated-home"
    original.mkdir()
    isolated.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(original))

    def no_keyring(home):
        raise AssertionError("this credential mode must not access the keyring")

    monkeypatch.setattr(codex_auth, "_read_keyring_auth", no_keyring)
    return original, isolated


@pytest.mark.parametrize("mode", [None, "file", "auto"])
def test_file_auth_is_preserved_byte_for_byte(auth_homes, monkeypatch, mode):
    original, isolated = auth_homes
    credentials = b'{"OPENAI_API_KEY":"fixture"}\r\n'
    (original / "auth.json").write_bytes(credentials)
    if mode:
        (original / "config.toml").write_text(
            f'cli_auth_credentials_store = "{mode}"\n', encoding="utf-8"
        )
    if mode == "auto":
        monkeypatch.setattr(codex_auth, "_read_keyring_auth", lambda home: None)

    codex_auth.seed_codex_auth(isolated)

    assert (isolated / "auth.json").read_bytes() == credentials
    assert (original / "auth.json").read_bytes() == credentials


def test_auto_falls_back_to_file_when_keyring_is_unavailable(auth_homes, monkeypatch):
    original, isolated = auth_homes
    (original / "config.toml").write_text('cli_auth_credentials_store = "auto"\n', encoding="utf-8")
    (original / "auth.json").write_bytes(b'{"OPENAI_API_KEY":"fixture"}')

    def unavailable(home):
        raise RuntimeError("locked: secret-backend-detail")

    monkeypatch.setattr(codex_auth, "_read_keyring_auth", unavailable)
    codex_auth.seed_codex_auth(isolated)

    assert (isolated / "auth.json").read_bytes() == (original / "auth.json").read_bytes()


@pytest.mark.parametrize("mode", ["keyring", "auto"])
@pytest.mark.parametrize("error", [None, RuntimeError("secret-backend-detail")])
def test_keyring_failure_is_actionable_and_redacted(auth_homes, monkeypatch, mode, error):
    original, isolated = auth_homes
    (original / "config.toml").write_text(
        f'cli_auth_credentials_store = "{mode}"\n', encoding="utf-8"
    )
    if mode == "keyring":
        # Explicit keyring mode must never select an old fallback file.
        (original / "auth.json").write_bytes(b'{"OPENAI_API_KEY":"stale"}')

    def lookup(home):
        if error:
            raise error
        return

    monkeypatch.setattr(codex_auth, "_read_keyring_auth", lookup)
    if mode == "auto" and error is None:
        codex_auth.seed_codex_auth(isolated)
    else:
        with pytest.raises(AgentError, match="keyring") as caught:
            codex_auth.seed_codex_auth(isolated)
        assert "secret-backend-detail" not in "".join(traceback.format_exception(caught.value))
    assert not (isolated / "auth.json").exists()


@pytest.mark.parametrize("value", ["not-json: secret-token", "[]", "null", '"secret-token"'])
def test_invalid_keyring_data_never_reaches_the_child(auth_homes, monkeypatch, value):
    original, isolated = auth_homes
    (original / "config.toml").write_text(
        'cli_auth_credentials_store = "keyring"\n', encoding="utf-8"
    )
    monkeypatch.setattr(codex_auth, "_read_keyring_auth", lambda home: value)

    with pytest.raises(AgentError, match="invalid Codex authentication") as caught:
        codex_auth.seed_codex_auth(isolated)

    assert "secret-token" not in "".join(traceback.format_exception(caught.value))
    assert not (isolated / "auth.json").exists()


@pytest.mark.parametrize("setting", ['"unknown"', "[]", "true", '"unterminated'])
def test_invalid_storage_configuration_fails_visibly(auth_homes, setting):
    original, isolated = auth_homes
    (original / "config.toml").write_text(
        f"cli_auth_credentials_store = {setting}\n", encoding="utf-8"
    )

    with pytest.raises(AgentError, match="credential"):
        codex_auth.seed_codex_auth(isolated)


@pytest.mark.parametrize("mode", [None, "ephemeral"])
def test_absent_or_ephemeral_auth_is_not_materialized(auth_homes, mode):
    original, isolated = auth_homes
    if mode:
        (original / "config.toml").write_text(
            f'cli_auth_credentials_store = "{mode}"\n', encoding="utf-8"
        )
        (original / "auth.json").write_bytes(b'{"OPENAI_API_KEY":"stale"}')

    codex_auth.seed_codex_auth(isolated)

    assert not (isolated / "auth.json").exists()


@pytest.mark.parametrize("variable", ["CODEX_API_KEY", "OPENAI_API_KEY"])
def test_api_key_environment_is_preserved(auth_homes, monkeypatch, variable):
    original, isolated = auth_homes
    monkeypatch.setenv(variable, "fixture-api-key")
    observed = {}

    def fake_run(cmd, stdin, **kwargs):
        observed.update(kwargs["env"])
        assert not (Path(observed["CODEX_HOME"]) / "auth.json").exists()
        return AgentResult("ok", "ok", "", 0, "codex")

    monkeypatch.setattr(agent_providers, "_run_cli", fake_run)

    CodexCli().invoke("prompt")

    assert observed[variable] == "fixture-api-key"
    assert not (isolated / "auth.json").exists()


@pytest.mark.parametrize("mode", ["file", "keyring", "auto"])
@pytest.mark.parametrize("api_key", ["fixture-codex-api-key", "  fixture-codex-api-key  "])
def test_codex_api_key_override_keeps_storage_and_configuration_isolated(
    auth_homes, monkeypatch, mode, api_key
):
    original, _ = auth_homes
    config = f'cli_auth_credentials_store = "{mode}"\nmodel = "unwanted-model"\n'.encode()
    credentials = b'{"tokens":{"access_token":"saved-login"}}\r\n'
    (original / "config.toml").write_bytes(config)
    (original / "auth.json").write_bytes(credentials)
    monkeypatch.setenv("CODEX_API_KEY", api_key)
    monkeypatch.setenv("OPENAI_API_KEY", "separate-openai-key")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-reach-codex")
    homes = []

    def fake_run(cmd, stdin, **kwargs):
        env = kwargs["env"]
        home = Path(env["CODEX_HOME"])
        homes.append(home)
        assert home != original
        assert not (home / "config.toml").exists()
        assert not (home / "auth.json").exists()
        assert env["CODEX_API_KEY"] == api_key
        assert env["OPENAI_API_KEY"] == "separate-openai-key"
        assert "UNRELATED_SECRET" not in env
        return AgentResult("ok", "ok", "", 0, "codex")

    monkeypatch.setattr(agent_providers, "_run_cli", fake_run)

    CodexCli().invoke("prompt")

    assert len(homes) == 1 and not homes[0].exists()
    assert (original / "config.toml").read_bytes() == config
    assert (original / "auth.json").read_bytes() == credentials


@pytest.mark.parametrize("api_key", ["", " \t\n"])
def test_blank_codex_api_key_keeps_stored_authentication(auth_homes, monkeypatch, api_key):
    original, isolated = auth_homes
    (original / "config.toml").write_bytes(b'cli_auth_credentials_store = "keyring"\n')
    credentials = '{"tokens":{"access_token":"saved-login"}}'
    lookups = []

    def lookup(home):
        lookups.append(home)
        return credentials

    monkeypatch.setenv("CODEX_API_KEY", api_key)
    monkeypatch.setattr(codex_auth, "_read_keyring_auth", lookup)

    codex_auth.seed_codex_auth(isolated)

    assert lookups == [original]
    assert (isolated / "auth.json").read_bytes() == credentials.encode("utf-8")


@pytest.mark.parametrize(
    "platform,module", [("darwin", "macOS"), ("linux", "SecretService"), ("win32", "Windows")]
)
def test_native_backend_uses_codex_rust_keyring_names(tmp_path, monkeypatch, platform, module):
    original = tmp_path / "codex-home"
    original.mkdir()
    account = "cli|fixture-home"
    monkeypatch.setattr(codex_auth, "_codex_store_key", lambda home: account)
    calls = []

    def get_password(service, user):
        calls.append((service, user))
        return "fixture"

    backend = SimpleNamespace(
        Keyring=lambda: SimpleNamespace(get_password=get_password),
        WinVaultKeyring=lambda: SimpleNamespace(get_password=get_password),
    )
    monkeypatch.setitem(sys.modules, f"keyring.backends.{module}", backend)
    monkeypatch.setattr(sys, "platform", platform)

    assert codex_auth._read_keyring_auth(original) == "fixture"

    service = f"{account}.Codex Auth" if platform == "win32" else "Codex Auth"
    assert calls == [(service, account)]


@pytest.mark.parametrize(
    "platform,canonical,expected",
    [
        ("linux", "/codex-fixture", "cli|9733caae77f006bf"),
        ("win32", r"C:\Users\Alice\.codex", "cli|621273a21b1aa29a"),
        ("win32", r"\\?\C:\Users\Alice\.codex", "cli|621273a21b1aa29a"),
        ("win32", r"\\server\Users\Alice\.codex", "cli|74b0879314fa2613"),
    ],
)
def test_store_key_matches_rust_canonical_paths(monkeypatch, platform, canonical, expected):
    home = Mock(spec=Path)
    home.resolve.return_value = canonical
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", platform)
        actual = codex_auth._codex_store_key(home)

    assert actual == expected
    home.resolve.assert_called_once_with(strict=True)


def test_store_key_uses_the_resolved_home(tmp_path):
    original = tmp_path / "codex-home"
    original.mkdir()
    indirect = original / ".." / original.name

    assert codex_auth._codex_store_key(indirect) == codex_auth._codex_store_key(original)


def test_store_key_uses_the_literal_path_if_canonicalization_fails():
    home = Path("calcifier-missing-auth-home-fixture")
    assert not home.exists()

    assert codex_auth._codex_store_key(home) == "cli|703fc9fb4ecfa6dd"


def test_private_auth_is_removed_after_child_failure(auth_homes, monkeypatch):
    original, isolated = auth_homes
    (original / "auth.json").write_bytes(b'{"OPENAI_API_KEY":"fixture"}')
    homes = []

    def failed_run(*args, **kwargs):
        home = Path(kwargs["env"]["CODEX_HOME"])
        homes.append(home)
        assert (home / "auth.json").is_file()
        raise AgentError("child failed")

    monkeypatch.setattr(agent_providers, "_run_cli", failed_run)

    with pytest.raises(AgentError, match="child failed"):
        CodexCli().invoke("prompt")

    assert not homes[0].exists()
    assert (original / "auth.json").is_file()


def test_auth_write_failure_is_redacted(auth_homes):
    original, isolated = auth_homes
    (original / "auth.json").write_bytes(b'{"OPENAI_API_KEY":"fixture"}')
    (isolated / "auth.json").mkdir()

    with pytest.raises(AgentError, match="could not prepare isolated Codex authentication"):
        codex_auth.seed_codex_auth(isolated)
