"""Tests for trust-on-first-use consent for recipe execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from calcifier import cli
from calcifier import trust as trust_module
from calcifier.recipe import load_recipe
from calcifier.trust import (
    TrustError,
    TrustStatus,
    compute_sha256,
    render_untrusted_message,
    trust_path_for,
)
from tests.support.trust import check_trust, record_trust

_MINIMAL_RECIPE = """\
from calcifier import recipe

@recipe(version="0.1")
def smoke(target: str = "x") -> bool:
    return bool(target)
"""


def _write_recipe(tmp_path: Path, body: str = _MINIMAL_RECIPE) -> Path:
    p = tmp_path / "recipe.py"
    p.write_text(body)
    return p


def test_check_trust_returns_untrusted_new_when_no_user_record(tmp_path):
    recipe = _write_recipe(tmp_path)
    result = check_trust(recipe)
    assert result.status is TrustStatus.UNTRUSTED_NEW
    assert result.current_sha == compute_sha256(recipe)
    assert result.trust_path == trust_path_for()
    assert not result.trust_path.exists()


def test_record_trust_creates_user_store_with_entry(tmp_path):
    recipe = _write_recipe(tmp_path)
    record_trust(recipe, recorded_by="alice@host")
    assert trust_path_for().exists()
    payload = json.loads(trust_path_for().read_text())
    assert payload["version"] == "1"
    sha = compute_sha256(recipe)
    entry = payload["entries"][str(recipe.resolve())][sha]
    assert entry["path"] == str(recipe.resolve())
    assert entry["recorded_by"] == "alice@host"
    assert entry["trusted_at"]


def test_check_trust_returns_trusted_after_record(tmp_path):
    recipe = _write_recipe(tmp_path)
    record_trust(recipe, recorded_by="alice@host")
    result = check_trust(recipe)
    assert result.status is TrustStatus.TRUSTED


def test_check_trust_returns_untrusted_changed_when_sha_changes(tmp_path):
    recipe = _write_recipe(tmp_path)
    record_trust(recipe, recorded_by="alice@host")
    recipe.write_text(_MINIMAL_RECIPE.replace("smoke", "altered"))
    result = check_trust(recipe)
    assert result.status is TrustStatus.UNTRUSTED_CHANGED
    assert result.recorded_entries is not None
    assert len(result.recorded_entries) == 1


def test_record_trust_appends_new_entry_when_hash_changes(tmp_path):
    recipe = _write_recipe(tmp_path)
    record_trust(recipe, recorded_by="alice@host")
    recipe.write_text(_MINIMAL_RECIPE.replace("smoke", "altered"))
    record_trust(recipe, recorded_by="alice@host")
    payload = json.loads(trust_path_for().read_text())
    assert len(payload["entries"][str(recipe.resolve())]) == 2


def test_check_trust_rejects_empty_store(tmp_path):
    recipe = _write_recipe(tmp_path)
    trust_path_for().parent.mkdir(parents=True)
    trust_path_for().write_text("")
    trust_path_for().chmod(0o600)
    with pytest.raises(TrustError, match="empty or corrupt"):
        check_trust(recipe)


def test_adjacent_legacy_sidecar_cannot_grant_trust(tmp_path):
    recipe = _write_recipe(tmp_path)
    recipe.with_name(recipe.name + ".trust.json").write_text(
        json.dumps({"version": "0.1", "entries": {compute_sha256(recipe): {}}})
    )

    result = check_trust(recipe)

    assert result.status is TrustStatus.UNTRUSTED_NEW


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({"version": "0.1", "entries": {}}),
        json.dumps({"version": "1", "entries": []}),
    ],
)
def test_check_trust_rejects_malformed_store(tmp_path, payload):
    recipe = _write_recipe(tmp_path)
    store = trust_path_for()
    store.parent.mkdir(parents=True)
    store.write_text(payload)
    store.chmod(0o600)

    with pytest.raises(TrustError):
        check_trust(recipe)


def test_check_trust_rejects_malformed_entry_even_when_digest_does_not_match(tmp_path):
    recipe = _write_recipe(tmp_path)
    store = trust_path_for()
    store.parent.mkdir(parents=True)
    store.write_text(
        json.dumps(
            {
                "version": "1",
                "entries": {str(recipe.resolve()): {"0" * 64: {}}},
            }
        )
    )
    store.chmod(0o600)

    with pytest.raises(TrustError, match="invalid trust entry"):
        check_trust(recipe)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_record_trust_uses_private_permissions(tmp_path):
    recipe = _write_recipe(tmp_path)
    record_trust(recipe, recorded_by="alice@host")

    assert trust_path_for().stat().st_mode & 0o777 == 0o600
    assert trust_path_for().parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_check_trust_rejects_shared_store_permissions(tmp_path):
    recipe = _write_recipe(tmp_path)
    record_trust(recipe, recorded_by="alice@host")
    trust_path_for().chmod(0o666)

    with pytest.raises(TrustError, match="permissions must be 0600"):
        check_trust(recipe)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symbolic links")
def test_check_trust_rejects_symlinked_store(tmp_path, monkeypatch):
    recipe = _write_recipe(tmp_path)
    target = tmp_path / "target.json"
    target.write_text(json.dumps({"version": "1", "entries": {}}))
    target.chmod(0o600)
    store = tmp_path / "store.json"
    store.symlink_to(target)
    monkeypatch.setenv("CALCIFIER_TRUST_STORE", str(store))

    with pytest.raises(TrustError, match="symbolic link"):
        check_trust(recipe)


# -- CLI integration ---------------------------------------------------------


def test_cli_exec_refuses_untrusted_recipe(tmp_path, capsys):
    recipe = _write_recipe(tmp_path)
    rc = cli.main(["exec", str(recipe)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "not yet trusted" in err


def test_cli_exec_with_trust_records_and_runs(tmp_path, capsys, monkeypatch):
    recipe = _write_recipe(tmp_path)
    rc = cli.main(["exec", str(recipe), "--trust", "--input", "target=go"])
    assert trust_path_for().exists()
    assert rc == 0
    captured = capsys.readouterr()
    assert "smoke" in captured.out
    assert captured.err == (
        f"recorded trust for {recipe.resolve()}:\n"
        f"  current sha256:  {compute_sha256(recipe)}\n"
        "  declared capabilities: []\n"
        "  agent roles: []\n"
    )


@pytest.mark.parametrize("granting", [False, True], ids=["review", "grant"])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n"], ids=["lf", "crlf"])
def test_cli_trust_shows_snapshot_capabilities_and_roles(
    tmp_path, capsys, monkeypatch, granting, newline
):
    recipe = tmp_path / "recipe.py"
    original = (
        b"from calcifier import Budget, recipe as compiled\n"
        b"@compiled(capabilities=('Bash', 'Read'), agents=('reviewer',), budget=Budget(1, 30))\n"
        b"def inspect() -> bool:\n"
        b"    raise AssertionError('must not execute while showing trust')\n"
    ).replace(b"\n", newline)
    recipe.write_bytes(original)
    original_check = cli.check_trust_bytes

    def swap_after_check(path, data):
        result = original_check(path, data)
        recipe.write_bytes(_MINIMAL_RECIPE.encode("utf-8"))
        return result

    monkeypatch.setattr(cli, "check_trust_bytes", swap_after_check)
    status, snapshot = cli._enforce_trust(recipe, granting=granting)

    captured = capsys.readouterr()
    assert status == (0 if granting else 2)
    assert "  declared capabilities: ['Bash', 'Read']\n" in captured.err
    assert "  agent roles: ['reviewer']\n" in captured.err
    assert captured.out == ""
    if granting:
        assert snapshot is not None and snapshot.data == original
        assert trust_module.check_trust_bytes(recipe, original).status is TrustStatus.TRUSTED
    else:
        assert snapshot is None
        assert not trust_path_for().exists()


def test_changed_recipe_trust_shows_current_declarations(tmp_path):
    recipe = _write_recipe(tmp_path)
    record_trust(recipe, recorded_by="tester")
    recipe.write_bytes(
        b"import calcifier as runtime\n"
        b"@runtime.recipe(capabilities=('Read',))\n"
        b"def inspect() -> bool:\n"
        b"    return True\n"
    )

    result = check_trust(recipe)

    assert result.status is TrustStatus.UNTRUSTED_CHANGED
    assert "declared capabilities: ['Read']" in render_untrusted_message(result)


def test_trust_reports_unavailable_declarations_without_executing_code(tmp_path):
    marker = tmp_path / "executed"
    recipe = _write_recipe(
        tmp_path,
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
    )

    result = check_trust(recipe)

    assert not marker.exists()
    assert "declared capabilities: (unavailable)" in render_untrusted_message(result)
    assert "agent roles: (unavailable)" in render_untrusted_message(result)


def test_cli_trust_pins_declared_subrecipe(tmp_path, capsys, monkeypatch):
    recipe = _write_recipe(tmp_path)
    child = tmp_path / "child.py"
    child.write_text(_MINIMAL_RECIPE.replace("smoke", "child"))

    rc = cli.main(["exec", str(recipe), "--trust", "--subrecipe", str(child)])
    payload = json.loads(trust_path_for().read_text())
    entry = payload["entries"][str(recipe.resolve())][compute_sha256(recipe)]

    assert rc == 0
    assert entry["dependencies"] == [
        {"path": str(child.resolve()), "sha256": compute_sha256(child)}
    ]
    capsys.readouterr()


def test_cli_subrecipe_requires_trust(tmp_path, capsys):
    recipe = _write_recipe(tmp_path)
    child = tmp_path / "child.py"
    child.write_text(_MINIMAL_RECIPE)

    rc = cli.main(["exec", str(recipe), "--subrecipe", str(child)])

    assert rc == 2
    assert "--subrecipe requires --trust" in capsys.readouterr().err


def test_cli_exec_changed_recipe_refuses_until_re_trusted(tmp_path, capsys):
    recipe = _write_recipe(tmp_path)
    record_trust(recipe, recorded_by="alice@host")
    recipe.write_text(_MINIMAL_RECIPE.replace("smoke", "altered"))
    rc = cli.main(["exec", str(recipe)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "hash changed" in err


def test_cli_executes_same_snapshot_checked_for_trust(tmp_path, capsys, monkeypatch):
    marker = tmp_path / "substituted"
    recipe = _write_recipe(tmp_path)
    record_trust(recipe, recorded_by="alice@host")
    original_check = cli.check_trust_bytes

    def swap_after_check(path, data):
        result = original_check(path, data)
        recipe.write_text(f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')\n")
        return result

    monkeypatch.setattr(cli, "check_trust_bytes", swap_after_check)

    rc = cli.main(["exec", str(recipe)])

    assert rc == 0
    assert "smoke" in capsys.readouterr().out
    assert not marker.exists()


def test_resource_drift_invalidates_trust(tmp_path):
    skill = tmp_path / "SKILL.md"
    resource = tmp_path / "scripts" / "helper.py"
    resource.parent.mkdir()
    skill.write_text("# skill\n")
    resource.write_text("print('safe')\n")
    resource_sha = compute_sha256(resource)
    body = (
        _MINIMAL_RECIPE
        + "\n# --- calcifier metadata (skill provenance + sha256) ---\n"
        + f"_CALCIFIER_SKILL = {{'path': {str(skill)!r}, 'sha256': 'skill'}}\n"
        + "_CALCIFIER_RESOURCES = "
        + repr([{"path": "scripts/helper.py", "sha256": resource_sha}])
        + "\n"
    )
    recipe = _write_recipe(tmp_path, body)
    record_trust(recipe, recorded_by="alice@host")

    resource.write_text("print('changed')\n")
    result = check_trust(recipe)

    assert result.status is TrustStatus.TRUSTED_WITH_DRIFT
    assert result.dependency_drift[0].path == "scripts/helper.py"


def test_external_aliased_resource_is_verified_at_its_source(tmp_path):
    skill_root = tmp_path / "plugins" / "demo" / "skills" / "audit"
    skill_root.mkdir(parents=True)
    skill = skill_root / "SKILL.md"
    skill.write_text("# skill\n")
    resource = tmp_path / "plugins" / "demo" / "scripts" / "helper.py"
    resource.parent.mkdir()
    resource.write_text("print('safe')\n")
    resource_sha = compute_sha256(resource)
    recipe_dir = skill_root / ".calcifier"
    recipe_dir.mkdir()
    body = (
        _MINIMAL_RECIPE
        + "\n# --- calcifier metadata (skill provenance + sha256) ---\n"
        + f"_CALCIFIER_SKILL = {{'path': {str(skill)!r}, 'sha256': 'skill'}}\n"
        + "_CALCIFIER_RESOURCES = "
        + repr(
            [
                {
                    "path": "scripts/helper.py",
                    "sha256": resource_sha,
                    "source": "../../scripts/helper.py",
                }
            ]
        )
        + "\n"
    )
    recipe = recipe_dir / "recipe.py"
    recipe.write_text(body)

    record_trust(recipe, recorded_by="alice@host")

    assert check_trust(recipe).status is TrustStatus.TRUSTED
    resource.write_text("print('changed')\n")
    result = check_trust(recipe)
    assert result.status is TrustStatus.TRUSTED_WITH_DRIFT
    assert result.dependency_drift[0].path == "scripts/helper.py"


def test_resource_snapshot_resists_swap_after_read(tmp_path, monkeypatch):
    skill = tmp_path / "SKILL.md"
    resource = tmp_path / "scripts" / "data.txt"
    resource.parent.mkdir()
    skill.write_text("# skill\n")
    resource.write_text("safe")
    expected_sha = compute_sha256(resource)
    body = f"""
from calcifier import recipe, resource

@recipe
def read_resource() -> str:
    return resource("scripts/data.txt").read_text()

# --- calcifier metadata (skill provenance + sha256) ---
_CALCIFIER_SKILL = {{'path': {str(skill)!r}, 'sha256': {compute_sha256(skill)!r}}}
_CALCIFIER_RESOURCES = [{{'path': 'scripts/data.txt', 'sha256': {expected_sha!r}}}]
"""
    recipe_path = _write_recipe(tmp_path, body)
    recipe_obj = load_recipe(recipe_path)
    original_read_bytes = Path.read_bytes
    swapped = False

    def swap_after_read(path: Path) -> bytes:
        nonlocal swapped
        data = original_read_bytes(path)
        if path == resource and not swapped:
            swapped = True
            resource.write_text("changed")
        return data

    monkeypatch.setattr(Path, "read_bytes", swap_after_read)

    run = recipe_obj.run()

    assert not run.ok
    assert run.error is not None
    assert "authoritative instruction source changed" in run.error
    assert run.output is None
    assert resource.read_text() == "changed"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_check_trust_rejects_group_writable_store_directory(tmp_path):
    recipe = _write_recipe(tmp_path)
    record_trust(recipe, recorded_by="alice@host")
    trust_path_for().parent.chmod(0o770)

    with pytest.raises(TrustError, match="must not be group/world-writable"):
        check_trust(recipe)


def test_record_trust_reports_unwritable_store_without_leaving_temp_files(tmp_path, monkeypatch):
    recipe = _write_recipe(tmp_path)

    def refuse(source: str, destination: object) -> None:
        raise PermissionError("read-only")

    monkeypatch.setattr(trust_module.os, "replace", refuse)

    with pytest.raises(TrustError, match="could not write trust store"):
        record_trust(recipe, recorded_by="alice@host")
    assert list(trust_path_for().parent.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX path branches; Windows CI covers its own")
@pytest.mark.parametrize(
    ("platform", "env", "expected"),
    [
        ("darwin", {}, ("Library", "Application Support")),
        ("linux", {"XDG_CONFIG_HOME": "xdg"}, ("xdg",)),
        ("linux", {}, (".config",)),
    ],
)
def test_trust_store_defaults_follow_platform_conventions(
    monkeypatch, tmp_path, platform, env, expected
):
    monkeypatch.delenv("CALCIFIER_TRUST_STORE")
    monkeypatch.setattr(trust_module.sys, "platform", platform)
    monkeypatch.setattr(trust_module.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, str(tmp_path / value))

    path = trust_path_for()

    assert path.parts[-2:] == ("calcifier", "trust.json")
    assert path.parent.parent == tmp_path.joinpath(*expected)


@pytest.mark.parametrize(
    ("local_app_data", "expected"),
    [(None, ("AppData", "Local")), ("custom-local", ("custom-local",))],
)
def test_windows_trust_store_uses_local_app_data_or_home_fallback(
    monkeypatch, tmp_path, local_app_data, expected
):
    environment = {"LOCALAPPDATA": str(tmp_path / local_app_data)} if local_app_data else {}
    monkeypatch.setattr(trust_module, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(trust_module, "os", SimpleNamespace(name="nt", environ=environment))
    monkeypatch.setattr(trust_module.Path, "home", lambda: tmp_path)

    path = trust_path_for()

    assert path == tmp_path.joinpath(*expected, "calcifier", "trust.json")
