"""Skill package resource discovery for calcified recipes."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .hashing import sha256_hex
from .path_syntax import PathSyntax

RESOURCE_DIRS = ("references", "scripts", "templates", "resources", "workflows")
RESOURCE_FILE_SIZE_LIMIT = 256 * 1024
GENERATION_CONTEXT_FILE_SIZE_LIMIT = 64 * 1024
GENERATION_CONTEXT_SUFFIXES = frozenset(
    {".csv", ".json", ".md", ".py", ".rst", ".sh", ".toml", ".txt", ".yaml", ".yml"}
)

logger = logging.getLogger(__name__)

_BASE_DIR_RESOURCE_RE = re.compile(r"\{baseDir\}/(?P<path>[A-Za-z0-9_./-]+)")
_MARKDOWN_RESOURCE_RE = re.compile(r"\]\((?P<path>[A-Za-z0-9_./-]+)\)")


@dataclass(frozen=True)
class ResourceEntry:
    path: str
    sha256: str
    size: int
    source: str | None = None


def discover_resources(skill_path: Path) -> list[ResourceEntry]:
    """Return package resources beside ``skill_path`` in deterministic order."""
    root = skill_path.parent
    root_resolved = root.resolve()
    entries: list[ResourceEntry] = []
    for resource_dir in RESOURCE_DIRS:
        base = root / resource_dir
        if not base.exists():
            continue
        entries.extend(_entries_under(base, root, root_resolved))
    entries.extend(_referenced_local_resources(skill_path, entries))
    entries.extend(_referenced_external_resources(skill_path, entries))
    return sorted(entries, key=lambda entry: entry.path)


def resource_containment_root(skill_root: Path) -> Path:
    """Bound external resource references to their containing skill package."""
    resolved = skill_root.resolve()
    skills_parent = next(
        (ancestor for ancestor in (resolved, *resolved.parents) if ancestor.name == "skills"),
        None,
    )
    return skills_parent.parent if skills_parent is not None else resolved


def resource_relative_path(raw: str, *, source: bool = False, canonical: bool = False) -> Path:
    """Validate portable package aliases or contained relative source paths."""
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("resource path must be a non-empty relative path")
    syntax = PathSyntax.inspect(raw)
    if syntax.posix.is_absolute() or syntax.windows.root or syntax.windows.drive:
        raise ValueError(f"resource path must be relative, got absolute path: {raw!r}")
    if syntax.has_backslash:
        raise ValueError(f"resource path must use forward slashes: {raw!r}")
    if not source and ".." in syntax.posix.parts:
        raise ValueError(f"resource path escapes skill package: {raw!r}")
    if (
        not syntax.posix.parts
        or syntax.has_windows_reserved_name
        or (canonical and syntax.posix.as_posix() != raw)
    ):
        raise ValueError(f"invalid resource path: {raw!r}")
    return syntax.native


def resolve_resource_source(root: Path, raw: str) -> Path:
    """Resolve a resource source within the enclosing skill package."""
    resolved = (root / resource_relative_path(raw, source=True)).resolve()
    if not resolved.is_relative_to(resource_containment_root(root)):
        raise ValueError(f"resource source escapes its package: {raw!r}")
    return resolved


def add_resource_sources(
    skill_path: Path, entries: list[ResourceEntry], sources: Mapping[str, str]
) -> list[ResourceEntry]:
    """Pin additional files; sources are relative to SKILL.md, aliases to its snapshot."""
    files = {("skill.md",)}
    directories: set[tuple[str, ...]] = set()
    for alias in [*(entry.path for entry in entries), *sources]:
        parts = tuple(
            part.casefold() for part in resource_relative_path(alias, canonical=True).parts
        )
        if (
            parts in files
            or parts in directories
            or any(parts[:i] in files for i in range(1, len(parts)))
        ):
            raise ValueError(f"explicit resource alias collision: {alias!r}")
        files.add(parts)
        directories.update(parts[:i] for i in range(1, len(parts)))
    root = skill_path.parent.resolve()
    result = list(entries)
    for alias, raw in sources.items():
        path = (root / resource_relative_path(raw, source=True)).resolve()
        if (
            not path.is_relative_to(resource_containment_root(root))
            or not path.is_file()
            or path.stat().st_size > RESOURCE_FILE_SIZE_LIMIT
        ):
            raise ValueError(f"missing, oversized or escaped explicit resource: {raw!r}")
        source = Path(os.path.relpath(path, root)).as_posix()
        result.append(_resource_entry(path, Path(alias), source=source))
    return sorted(result, key=lambda entry: entry.path)


def _referenced_local_resources(
    skill_path: Path, existing: list[ResourceEntry]
) -> list[ResourceEntry]:
    root = skill_path.parent
    root_resolved = root.resolve()
    known_paths = {entry.path for entry in existing}
    entries: list[ResourceEntry] = []
    for raw_path in _referenced_resource_paths(skill_path.read_text(encoding="utf-8")):
        try:
            relative = resource_relative_path(Path(raw_path).as_posix())
        except ValueError:
            continue
        if (
            ".calcifier" in relative.parts
            or relative.as_posix().casefold() == skill_path.name.casefold()
        ):
            continue
        source = root / relative
        if not source.resolve().is_relative_to(root_resolved):
            continue
        if source.is_dir():
            candidates = _entries_under(source, root, root_resolved)
        elif source.is_file() and source.stat().st_size <= RESOURCE_FILE_SIZE_LIMIT:
            candidates = [_resource_entry(source, relative)]
        else:
            continue
        entries.extend(entry for entry in candidates if entry.path not in known_paths)
        known_paths.update(entry.path for entry in candidates)
    return entries


def _referenced_external_resources(
    skill_path: Path, existing: list[ResourceEntry]
) -> list[ResourceEntry]:
    root = skill_path.parent.resolve()
    containment_root = resource_containment_root(root)
    known_paths = {entry.path for entry in existing}
    entries: list[ResourceEntry] = []
    for raw_path in _referenced_resource_paths(skill_path.read_text(encoding="utf-8")):
        source = (root / raw_path).resolve()
        if (
            source.is_relative_to(root)
            or not source.is_relative_to(containment_root)
            or not source.is_file()
        ):
            continue
        alias = _external_resource_alias(raw_path)
        if alias is None or alias in known_paths:
            continue
        size = source.stat().st_size
        if size > RESOURCE_FILE_SIZE_LIMIT:
            continue
        source_path = Path(os.path.relpath(source, root)).as_posix()
        entries.append(_resource_entry(source, Path(alias), source=source_path))
        known_paths.add(alias)
    return entries


def _referenced_resource_paths(skill_text: str) -> list[str]:
    paths = {
        match.group("path")
        for pattern in (_BASE_DIR_RESOURCE_RE, _MARKDOWN_RESOURCE_RE)
        for match in pattern.finditer(skill_text)
    }
    return sorted(paths)


def _external_resource_alias(raw_path: str) -> str | None:
    parts = Path(raw_path).parts
    for index, part in enumerate(parts):
        if part in RESOURCE_DIRS:
            parent_parts = tuple(item for item in parts[:index] if item not in {".", ".."})
            if parent_parts:
                return Path(part, *parent_parts, *parts[index + 1 :]).as_posix()
            return Path(*parts[index:]).as_posix()
    return None


def _entries_under(base: Path, root: Path, root_resolved: Path) -> list[ResourceEntry]:
    entries: list[ResourceEntry] = []
    for path in base.rglob("*"):
        relative = path.relative_to(root)
        if (
            {".calcifier", "__pycache__"}.intersection(relative.parts)
            or path.suffix in {".pyc", ".pyo"}
            or not path.is_file()
        ):
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root_resolved):
            continue
        size = path.stat().st_size
        if size > RESOURCE_FILE_SIZE_LIMIT:
            continue
        entries.append(_resource_entry(path, relative))
    return entries


def _resource_entry(path: Path, relative: Path, *, source: str | None = None) -> ResourceEntry:
    return ResourceEntry(
        path=relative.as_posix(),
        sha256=sha256_hex(path.read_bytes()),
        size=path.stat().st_size,
        source=source,
    )


def manifest_summary(entries: list[ResourceEntry]) -> str:
    """Render a deterministic, prompt-friendly resource manifest."""
    ordered = sorted(entries, key=lambda entry: entry.path)
    return "\n".join(f"{entry.path} (sha256:{entry.sha256})" for entry in ordered)


def generation_manifest_summary(
    skill_path: Path,
    entries: list[ResourceEntry],
    *,
    explicit_source_paths: Mapping[str, str] | None = None,
) -> str:
    """Include exact contents for explicitly named, bounded text parsing resources."""
    summary = manifest_summary(entries)
    skill_text = skill_path.read_text(encoding="utf-8")
    contents: dict[str, str] = {}
    for entry in sorted(entries, key=lambda item: item.path):
        explicit_source = (explicit_source_paths or {}).get(entry.path, entry.source)
        if not _resource_is_explicitly_named(entry.path, explicit_source, skill_text):
            continue
        source = skill_path.parent / (entry.source or entry.path)
        if (
            Path(entry.path).suffix.lower() not in GENERATION_CONTEXT_SUFFIXES
            or entry.size > GENERATION_CONTEXT_FILE_SIZE_LIMIT
            or not source.is_file()
        ):
            continue
        try:
            contents[entry.path] = source.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
    if not contents:
        return summary
    exact = json.dumps(contents, indent=2)
    return (
        summary + "\n\nExact contents of explicitly named text resources "
        "(authoritative compiler data; do not copy into recipe constants):\n" + exact
    )


def _resource_is_explicitly_named(path: str, source: str | None, skill_text: str) -> bool:
    return path in skill_text or (source is not None and source in skill_text)
