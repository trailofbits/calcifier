"""Per-user trust-on-first-use consent for recipe execution."""

from __future__ import annotations

import ast
import enum
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from .atomic_io import atomic_write
from .diagnostics import CalcifyError
from .exceptions import CalcifierError, ErrorFamily
from .hashing import sha256_hex
from .metadata import parse_metadata_namespace, skill_root_from_metadata, split_metadata
from .recipe_ast import (
    collect_structure,
    import_bindings,
    recipe_agents,
    recipe_capabilities,
    require_recipe,
)
from .skill_package import resolve_resource_source, resource_relative_path

_TRUST_FILE_VERSION = "1"
_TRUST_STORE_ENV = "CALCIFIER_TRUST_STORE"


class TrustError(CalcifierError, RuntimeError):
    """Raised when the trust store or a pinned resource is invalid."""

    family = ErrorFamily.SETUP


class TrustStatus(enum.Enum):
    TRUSTED = "trusted"
    TRUSTED_WITH_DRIFT = "trusted-with-drift"
    UNTRUSTED_NEW = "untrusted-new"
    UNTRUSTED_CHANGED = "untrusted-changed"


@dataclass(frozen=True)
class DependencyDrift:
    path: str
    expected_sha: str
    current_sha: str | None


@dataclass(frozen=True)
class TrustResult:
    status: TrustStatus
    current_sha: str
    recipe_path: Path
    trust_path: Path
    recorded_entries: dict[str, dict[str, Any]] | None = None
    dependency_drift: tuple[DependencyDrift, ...] = ()
    capabilities: tuple[str, ...] | None = ()
    agent_roles: tuple[str, ...] | None = ()


def _declared_access(data: bytes) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    """Read declarations without executing the recipe or reopening its file."""
    try:
        module = ast.parse(data)
        recipes, _, _ = collect_structure(module)
        node = require_recipe(module, recipes)
        bindings = import_bindings(module)
        return recipe_capabilities(node, bindings), recipe_agents(node, bindings) or ()
    except (SyntaxError, ValueError, CalcifyError):
        return None, None


def compute_sha256_bytes(data: bytes) -> str:
    """Hash the complete recipe snapshot, including provenance metadata."""
    return sha256_hex(data)


def compute_sha256(path: Path) -> str:
    return compute_sha256_bytes(path.read_bytes())


def trust_path_for() -> Path:
    """Return the per-user trust store path."""
    override = os.environ.get(_TRUST_STORE_ENV)
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "calcifier" / "trust.json"


def check_trust_bytes(
    recipe_path: Path,
    data: bytes,
    *,
    verify_dependencies: bool = True,
) -> TrustResult:
    """Check consent for the exact recipe bytes supplied by the caller."""
    path = recipe_path.resolve()
    sha = compute_sha256_bytes(data)
    store_path = trust_path_for()
    payload = _load_store(store_path)
    path_entries = _path_entries(payload, path)
    entry = path_entries.get(sha)
    capabilities, agent_roles = _declared_access(data)
    if entry is not None:
        drift = _entry_drift(path, entry, data) if verify_dependencies else ()
        status = TrustStatus.TRUSTED if not drift else TrustStatus.TRUSTED_WITH_DRIFT
        return TrustResult(
            status=status,
            current_sha=sha,
            recipe_path=path,
            trust_path=store_path,
            recorded_entries=path_entries,
            dependency_drift=drift,
            capabilities=capabilities,
            agent_roles=agent_roles,
        )
    status = TrustStatus.UNTRUSTED_CHANGED if path_entries else TrustStatus.UNTRUSTED_NEW
    return TrustResult(
        status=status,
        current_sha=sha,
        recipe_path=path,
        trust_path=store_path,
        recorded_entries=path_entries or None,
        capabilities=capabilities,
        agent_roles=agent_roles,
    )


def record_trust_bytes(
    recipe_path: Path,
    data: bytes,
    *,
    recorded_by: str,
    dependencies: Iterable[tuple[str | Path, str]] = (),
) -> TrustResult:
    """Record consent for an already-read recipe snapshot."""
    path = recipe_path.resolve()
    sha = compute_sha256_bytes(data)
    resources = _resource_payload(path, data)
    capabilities, agent_roles = _declared_access(data)
    store_path = trust_path_for()
    payload = _load_store(store_path)
    entries = cast("dict[str, Any]", payload.setdefault("entries", {}))
    path_entries = cast("dict[str, Any]", entries.setdefault(str(path), {}))
    path_entries[sha] = {
        "path": str(path),
        "trusted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "recorded_by": recorded_by,
        "dependencies": _dependency_payload(dependencies),
        "resources": resources,
    }
    _write_store(store_path, payload)
    return TrustResult(
        status=TrustStatus.TRUSTED,
        current_sha=sha,
        recipe_path=path,
        trust_path=store_path,
        recorded_entries=cast("dict[str, dict[str, Any]]", path_entries),
        capabilities=capabilities,
        agent_roles=agent_roles,
    )


def trusted_entry(
    recipe_path: Path,
    data: bytes | None = None,
    *,
    verify_dependencies: bool = True,
) -> Mapping[str, Any] | None:
    """Return the trusted entry for exact bytes, or ``None`` when untrusted."""
    snapshot = recipe_path.read_bytes() if data is None else data
    result = check_trust_bytes(
        recipe_path,
        snapshot,
        verify_dependencies=verify_dependencies,
    )
    if result.status is not TrustStatus.TRUSTED or result.recorded_entries is None:
        return None
    entry = result.recorded_entries.get(result.current_sha)
    return entry if isinstance(entry, Mapping) else None


def _empty_store() -> dict[str, Any]:
    return {"version": _TRUST_FILE_VERSION, "entries": {}}


def _load_store(path: Path) -> dict[str, Any]:
    _validate_store_permissions(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _empty_store()
    except OSError as exc:
        raise TrustError(f"could not read trust store {path}: {exc}") from exc
    if not raw.strip():
        raise TrustError(f"trust store {path} is empty or corrupt")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TrustError(f"trust store {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != _TRUST_FILE_VERSION:
        raise TrustError(f"trust store {path} has an unsupported format")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise TrustError(f"trust store {path} has invalid entries")
    _validate_store_entries(entries, path)
    return cast("dict[str, Any]", payload)


def _validate_store_entries(entries: dict[object, object], store_path: Path) -> None:
    for recipe_path, raw_entries in entries.items():
        if not isinstance(recipe_path, str) or not isinstance(raw_entries, dict):
            raise TrustError(f"trust store {store_path} has invalid path entries")
        for sha, raw_entry in raw_entries.items():
            if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
                raise TrustError(f"trust store {store_path} has an invalid recipe digest")
            if not isinstance(raw_entry, Mapping):
                raise TrustError(f"trust store {store_path} has an invalid trust entry")
            entry = cast("Mapping[str, Any]", raw_entry)
            if not all(
                isinstance(entry.get(name), str) and entry.get(name)
                for name in ("path", "trusted_at", "recorded_by")
            ):
                raise TrustError(f"trust store {store_path} has an invalid trust entry")
            _entry_items(entry, "dependencies")
            _entry_items(entry, "resources")


def _validate_store_permissions(path: Path) -> None:
    if os.name == "nt" or not path.exists():
        return
    if path.is_symlink():
        raise TrustError(f"trust store {path} must not be a symbolic link")
    try:
        file_stat = path.stat()
        parent_stat = path.parent.stat()
    except OSError as exc:
        raise TrustError(f"could not inspect trust store permissions: {exc}") from exc
    if file_stat.st_uid != os.getuid() or parent_stat.st_uid != os.getuid():
        raise TrustError(f"trust store {path} must be owned by the current user")
    if file_stat.st_mode & 0o077:
        raise TrustError(f"trust store {path} permissions must be 0600")
    if parent_stat.st_mode & 0o022:
        raise TrustError(f"trust store directory {path.parent} must not be group/world-writable")


def _write_store(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with suppress(OSError):
        path.parent.chmod(0o700)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        atomic_write(path, encoded, mode=0o600)
        os.chmod(path, 0o600)
    except OSError as exc:
        raise TrustError(f"could not write trust store {path}: {exc}") from exc


def _path_entries(payload: Mapping[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    entries = payload.get("entries")
    if not isinstance(entries, Mapping):
        raise TrustError("trust store entries are invalid")
    raw = entries.get(str(path), {})
    if not isinstance(raw, Mapping):
        raise TrustError(f"trust entries for {path} are invalid")
    result: dict[str, dict[str, Any]] = {}
    for sha, entry in raw.items():
        if not isinstance(sha, str) or not isinstance(entry, Mapping):
            raise TrustError(f"trust entries for {path} are invalid")
        result[sha] = dict(entry)
    return result


def _dependency_payload(
    dependencies: Iterable[tuple[str | Path, str]],
) -> list[dict[str, str]]:
    return [{"path": str(path), "sha256": sha256} for path, sha256 in dependencies]


def _resource_payload(recipe_path: Path, data: bytes) -> list[dict[str, str]]:
    try:
        _, metadata = split_metadata(data.decode("utf-8"))
        namespace = parse_metadata_namespace(metadata or "")
    except (UnicodeDecodeError, SyntaxError, ValueError) as exc:
        raise TrustError(f"recipe {recipe_path} has invalid metadata: {exc}") from exc
    skill = namespace.get("_CALCIFIER_SKILL")
    resources = namespace.get("_CALCIFIER_RESOURCES", [])
    if resources and not isinstance(skill, Mapping):
        raise TrustError("recipe resource manifest is missing skill metadata")
    skill_root = _skill_root(recipe_path, skill)
    if not isinstance(resources, list):
        raise TrustError("recipe resource manifest is invalid")
    result: list[dict[str, str]] = []
    for raw in resources:
        if not isinstance(raw, Mapping):
            raise TrustError("recipe resource manifest is invalid")
        rel, source, expected, current = _resource_manifest_hash(
            skill_root, cast("Mapping[str, object]", raw)
        )
        if current != expected:
            raise TrustError(f"resource drift detected before trust: {rel}")
        item = {"path": rel, "sha256": expected}
        if source is not None:
            item["source"] = source
        result.append(item)
    return result


def _resource_manifest_hash(
    skill_root: Path | None, resource: Mapping[str, object]
) -> tuple[str, str | None, str, str | None]:
    rel = resource.get("path")
    expected = resource.get("sha256")
    source = resource.get("source")
    if not isinstance(rel, str) or not isinstance(expected, str):
        raise TrustError("recipe resource manifest is invalid")
    _validate_resource_alias(rel)
    if source is not None and not isinstance(source, str):
        raise TrustError("recipe resource manifest source is invalid")
    path = (
        _resolve_resource_source(skill_root, source)
        if source is not None
        else _resolve_relative(skill_root, rel)
    )
    return rel, source, expected, _file_sha(path)


def _skill_root(recipe_path: Path, skill: object) -> Path | None:
    return skill_root_from_metadata(recipe_path, skill)


def _resolve_relative(root: Path | None, rel: str) -> Path:
    path = _validate_resource_alias(rel)
    if root is None:
        raise TrustError("resource manifest requires resolvable skill metadata")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise TrustError(f"resource path escapes skill package: {rel!r}")
    return resolved


def _validate_resource_alias(rel: str) -> Path:
    try:
        return resource_relative_path(rel)
    except ValueError as exc:
        raise TrustError(str(exc)) from exc


def _resolve_resource_source(root: Path | None, rel: str) -> Path:
    if root is None:
        raise TrustError("resource manifest requires resolvable skill metadata")
    try:
        return resolve_resource_source(root, rel)
    except ValueError as exc:
        raise TrustError(str(exc)) from exc


def _entry_drift(
    recipe_path: Path,
    entry: Mapping[str, Any],
    recipe_data: bytes,
) -> tuple[DependencyDrift, ...]:
    drifts: list[DependencyDrift] = []
    for dep in _entry_items(entry, "dependencies"):
        current = _file_sha(_dependency_path(recipe_path, dep["path"]))
        if current != dep["sha256"]:
            drifts.append(DependencyDrift(dep["path"], dep["sha256"], current))
    resource_root = _resource_root_from_recipe(recipe_path, recipe_data)
    for resource in _entry_items(entry, "resources"):
        source = resource.get("source")
        path = (
            _resolve_resource_source(resource_root, source)
            if source is not None
            else _resolve_relative(resource_root, resource["path"])
        )
        current = _file_sha(path)
        if current != resource["sha256"]:
            drifts.append(DependencyDrift(resource["path"], resource["sha256"], current))
    return tuple(drifts)


def _entry_items(entry: Mapping[str, Any], name: str) -> list[dict[str, str]]:
    raw = entry.get(name, [])
    if not isinstance(raw, list):
        raise TrustError(f"trust entry {name} are invalid")
    items: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise TrustError(f"trust entry {name} are invalid")
        path = item.get("path")
        sha = item.get("sha256")
        source = item.get("source")
        if not isinstance(path, str) or not isinstance(sha, str):
            raise TrustError(f"trust entry {name} are invalid")
        if source is not None and not isinstance(source, str):
            raise TrustError(f"trust entry {name} are invalid")
        normalized = {"path": path, "sha256": sha}
        if source is not None:
            normalized["source"] = source
        items.append(normalized)
    return items


def _resource_root_from_recipe(recipe_path: Path, data: bytes) -> Path | None:
    try:
        _, metadata = split_metadata(data.decode("utf-8"))
        namespace = parse_metadata_namespace(metadata or "")
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as exc:
        raise TrustError(f"could not read recipe metadata for {recipe_path}: {exc}") from exc
    return _skill_root(recipe_path, namespace.get("_CALCIFIER_SKILL"))


def _dependency_path(recipe_path: Path, dependency_path: str) -> Path:
    path = Path(dependency_path)
    return path if path.is_absolute() else recipe_path.parent / path


def _file_sha(path: Path) -> str | None:
    try:
        return compute_sha256_bytes(path.read_bytes())
    except (FileNotFoundError, IsADirectoryError, OSError):
        return None


def _render_declared_access(result: TrustResult) -> str:
    capabilities = list(result.capabilities) if result.capabilities is not None else "(unavailable)"
    roles = list(result.agent_roles) if result.agent_roles is not None else "(unavailable)"
    return f"  declared capabilities: {capabilities}\n  agent roles: {roles}\n"


def render_trusted_message(result: TrustResult) -> str:
    return (
        f"recorded trust for {result.recipe_path}:\n"
        f"  current sha256:  {result.current_sha}\n"
        f"{_render_declared_access(result)}"
    ).rstrip("\n")


def render_untrusted_message(result: TrustResult) -> str:
    if result.status is TrustStatus.UNTRUSTED_NEW:
        return (
            f"refusing to execute {result.recipe_path}: not yet trusted.\n"
            f"  current sha256:  {result.current_sha}\n"
            f"  user trust store: {result.trust_path}\n"
            f"{_render_declared_access(result)}"
            "  Review the recipe before granting trust. Re-run with --trust\n"
            "  to record consent for this exact file and its resources."
        )
    if result.status is TrustStatus.TRUSTED_WITH_DRIFT:
        return (
            f"refusing to execute {result.recipe_path}: dependency or resource drift detected.\n"
            f"  current sha256:  {result.current_sha}\n"
            f"  user trust store: {result.trust_path}\n"
            f"{_render_declared_access(result)}"
            f"  drifted files:   {[_render_drift(d) for d in result.dependency_drift]}\n"
            "  Review the drift, then re-run with --trust to update consent."
        )
    return (
        f"refusing to execute {result.recipe_path}: hash changed since last trust record.\n"
        f"  current sha256:  {result.current_sha}\n"
        f"  user trust store: {result.trust_path}\n"
        f"{_render_declared_access(result)}"
        f"  recorded shas:   {sorted((result.recorded_entries or {}).keys())}\n"
        "  Review the diff, then re-run with --trust to update consent."
    )


def _render_drift(drift: DependencyDrift) -> str:
    current = drift.current_sha if drift.current_sha is not None else "<missing>"
    return f"{drift.path} {drift.expected_sha} -> {current}"
