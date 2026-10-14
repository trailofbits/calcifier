"""Trust-store known answers."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from calcifier.trust import (
    DependencyDrift,
    TrustError,
    TrustResult,
    TrustStatus,
    render_untrusted_message,
    trust_path_for,
    trusted_entry,
)
from tests.support.recipes import metadata_recipe as _metadata_recipe
from tests.support.trust import check_trust, record_trust

_ZERO_SHA = "0" * 64


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# -- trust store ----------------------------------------------------------------

_TRUST_RECIPE = """from calcifier import recipe


@recipe
def trusted() -> bool:
    return True
"""


def _plain_recipe(tmp_path: Path) -> Path:
    path = tmp_path / "recipe.py"
    path.write_text(_TRUST_RECIPE)
    return path


def _rewrite_store(update: Callable[[dict[str, Any]], None]) -> None:
    store = trust_path_for()
    payload = json.loads(store.read_text())
    update(payload)
    store.write_text(json.dumps(payload))


def test_record_trust_returns_the_exact_result_and_writes_a_sorted_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "config" / "calcifier" / "trust.json"
    monkeypatch.setenv("CALCIFIER_TRUST_STORE", str(store))
    recipe_path = _plain_recipe(tmp_path)
    (tmp_path / "dep.txt").write_bytes(b"dependency\n")
    dep_sha = _sha(b"dependency\n")

    result = record_trust(recipe_path, recorded_by="tester", dependencies=[("dep.txt", dep_sha)])

    key, sha = str(recipe_path.resolve()), _sha(recipe_path.read_bytes())
    trusted_at = json.loads(store.read_text())["entries"][key][sha]["trusted_at"]
    entry = {
        "dependencies": [{"path": "dep.txt", "sha256": dep_sha}],
        "path": key,
        "recorded_by": "tester",
        "resources": [],
        "trusted_at": trusted_at,
    }
    assert result == TrustResult(
        status=TrustStatus.TRUSTED,
        current_sha=sha,
        recipe_path=recipe_path.resolve(),
        trust_path=store,
        recorded_entries={sha: entry},
    )
    expected = {"entries": {key: {sha: entry}}, "version": "1"}
    assert store.read_text() == json.dumps(expected, indent=2) + "\n"
    assert trusted_entry(recipe_path) == entry
    if os.name != "nt":
        assert (_mode(store), _mode(store.parent)) == (0o600, 0o700)
    assert record_trust(recipe_path, recorded_by="tester").status is TrustStatus.TRUSTED


def test_relative_dependency_drift_is_reported_exactly(tmp_path: Path) -> None:
    recipe_path = _plain_recipe(tmp_path)
    dependency = tmp_path / "dep.txt"
    dependency.write_bytes(b"v1")
    record_trust(recipe_path, recorded_by="tester", dependencies=[("dep.txt", _sha(b"v1"))])
    assert check_trust(recipe_path).status is TrustStatus.TRUSTED

    dependency.write_bytes(b"v2")
    result = check_trust(recipe_path)

    old, new = _sha(b"v1"), _sha(b"v2")
    assert result.status is TrustStatus.TRUSTED_WITH_DRIFT
    assert result.dependency_drift == (DependencyDrift("dep.txt", old, new),)
    assert trusted_entry(recipe_path) is None
    assert render_untrusted_message(result) == (
        f"refusing to execute {recipe_path.resolve()}: dependency or resource drift detected.\n"
        f"  current sha256:  {result.current_sha}\n"
        f"  user trust store: {trust_path_for()}\n"
        "  declared capabilities: []\n"
        "  agent roles: []\n"
        f"  drifted files:   {[f'dep.txt {old} -> {new}']}\n"
        "  Review the drift, then re-run with --trust to update consent."
    )


def test_changed_recipe_message_lists_the_recorded_digests(tmp_path: Path) -> None:
    recipe_path = _plain_recipe(tmp_path)
    record_trust(recipe_path, recorded_by="tester")
    old = _sha(recipe_path.read_bytes())
    recipe_path.write_text(_TRUST_RECIPE + "\n# changed\n")

    result = check_trust(recipe_path)

    assert result.status is TrustStatus.UNTRUSTED_CHANGED
    assert trusted_entry(recipe_path) is None
    assert render_untrusted_message(result) == (
        f"refusing to execute {recipe_path.resolve()}: hash changed since last trust record.\n"
        f"  current sha256:  {_sha(recipe_path.read_bytes())}\n"
        f"  user trust store: {trust_path_for()}\n"
        "  declared capabilities: []\n"
        "  agent roles: []\n"
        f"  recorded shas:   {[old]}\n"
        "  Review the diff, then re-run with --trust to update consent."
    )


def _set_path_entries(value: object) -> Callable[[dict[str, Any], str, str], None]:
    def apply(payload: dict[str, Any], key: str, sha: str) -> None:
        del sha
        payload["entries"][key] = value

    return apply


def _rename_digest(payload: dict[str, Any], key: str, sha: str) -> None:
    entries = payload["entries"][key]
    entries["not-a-digest"] = entries.pop(sha)


def _set_entry_field(name: str, value: object) -> Callable[[dict[str, Any], str, str], None]:
    def apply(payload: dict[str, Any], key: str, sha: str) -> None:
        payload["entries"][key][sha][name] = value

    return apply


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (_set_path_entries([]), "trust store {store} has invalid path entries"),
        (_rename_digest, "trust store {store} has an invalid recipe digest"),
        (_set_entry_field("path", 1), "trust store {store} has an invalid trust entry"),
        (_set_entry_field("recorded_by", ""), "trust store {store} has an invalid trust entry"),
        (
            _set_entry_field("dependencies", [{"path": 1, "sha256": _ZERO_SHA}]),
            "trust entry dependencies are invalid",
        ),
        (
            _set_entry_field("dependencies", [{"path": "dep.txt", "sha256": 1}]),
            "trust entry dependencies are invalid",
        ),
        (
            _set_entry_field("resources", [{"path": 1, "sha256": _ZERO_SHA}]),
            "trust entry resources are invalid",
        ),
    ],
)
def test_tampered_trust_store_is_rejected_with_the_exact_error(
    tmp_path: Path, tamper: Callable[[dict[str, Any], str, str], None], message: str
) -> None:
    recipe_path = _plain_recipe(tmp_path)
    record_trust(recipe_path, recorded_by="tester")
    key, sha = str(recipe_path.resolve()), _sha(recipe_path.read_bytes())
    _rewrite_store(lambda payload: tamper(payload, key, sha))

    with pytest.raises(TrustError) as raised:
        check_trust(recipe_path)

    assert str(raised.value) == message.format(store=trust_path_for())


_RESOURCE = "references/a.txt"
# "/abs/a.txt" is not absolute on Windows; anchor it so every platform rejects it alike.
_ABSOLUTE = Path(Path.cwd().anchor, "abs", "a.txt").as_posix()


@pytest.mark.parametrize(
    ("resource", "message"),
    [
        ({"path": 1, "sha256": _ZERO_SHA}, "recipe resource manifest is invalid"),
        ({"path": _RESOURCE, "sha256": 1}, "recipe resource manifest is invalid"),
        (
            {"path": _ABSOLUTE, "sha256": _ZERO_SHA},
            f"resource path must be relative, got absolute path: {_ABSOLUTE!r}",
        ),
        (
            {"path": "../a.txt", "sha256": _ZERO_SHA},
            "resource path escapes skill package: '../a.txt'",
        ),
        ({"path": "", "sha256": _ZERO_SHA}, "resource path must be a non-empty relative path"),
        (
            {"path": _RESOURCE, "sha256": _ZERO_SHA, "source": _ABSOLUTE},
            f"resource path must be relative, got absolute path: {_ABSOLUTE!r}",
        ),
        (
            {"path": _RESOURCE, "sha256": _ZERO_SHA, "source": ""},
            "resource path must be a non-empty relative path",
        ),
        (
            {"path": _RESOURCE, "sha256": _ZERO_SHA, "source": "references\\a.txt"},
            "resource path must use forward slashes: 'references\\\\a.txt'",
        ),
    ],
)
def test_record_trust_rejects_each_invalid_resource_declaration(
    tmp_path: Path, resource: dict[str, Any], message: str
) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text("# skill\n")
    recipe_path = _metadata_recipe(tmp_path / "recipe.py", str(skill), [resource])

    with pytest.raises(TrustError) as raised:
        record_trust(recipe_path, recorded_by="tester")

    assert str(raised.value) == message
    assert not trust_path_for().exists()


def test_relative_skill_path_resolves_beside_the_recipe_not_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill_dir = tmp_path / "package" / "skill"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# skill\n")
    (skill_dir / _RESOURCE).write_bytes(b"reference\n")
    resources = [{"path": _RESOURCE, "sha256": _sha(b"reference\n")}]
    recipe_path = _metadata_recipe(skill_dir.parent / "recipe.py", "skill/SKILL.md", resources)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = record_trust(recipe_path, recorded_by="tester")

    assert result.recorded_entries is not None
    assert result.recorded_entries[result.current_sha]["resources"] == resources
    assert check_trust(recipe_path).status is TrustStatus.TRUSTED


@pytest.mark.parametrize("calcifier_directory", [False, True])
def test_stored_resources_use_the_runtime_skill_root(
    tmp_path: Path,
    calcifier_directory: bool,
) -> None:
    directory = tmp_path / ".calcifier" if calcifier_directory else tmp_path
    directory.mkdir(exist_ok=True)
    recipe_path = _plain_recipe(directory)
    resource = tmp_path / _RESOURCE
    resource.parent.mkdir()
    resource.write_bytes(b"v1")
    record_trust(recipe_path, recorded_by="tester")
    key, sha = str(recipe_path.resolve()), _sha(recipe_path.read_bytes())
    stored = [{"path": _RESOURCE, "sha256": _sha(b"v1")}]
    _rewrite_store(lambda payload: payload["entries"][key][sha].update(resources=stored))
    if not calcifier_directory:
        with pytest.raises(TrustError, match="requires resolvable skill metadata"):
            check_trust(recipe_path)
        return
    assert check_trust(recipe_path).status is TrustStatus.TRUSTED

    resource.write_bytes(b"v2")

    assert check_trust(recipe_path).dependency_drift == (
        DependencyDrift(_RESOURCE, _sha(b"v1"), _sha(b"v2")),
    )
