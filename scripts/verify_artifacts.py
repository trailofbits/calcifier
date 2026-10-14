"""Fail closed when release artifacts contain undeclared state or version drift."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import tomllib
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath

from packaging.version import InvalidVersion, Version

FORBIDDEN_PARTS = frozenset({".claude", ".hypothesis", ".vivisect", "__pycache__"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path, nargs="?")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--release-tag")
    args = parser.parse_args(argv)
    expected = _source_version(args.root)
    _verify_version_surfaces(args.root, expected)
    if args.release_tag is not None:
        _verify_release_tag(args.release_tag, expected)
        _verify_release_changelog(args.root, expected)
    if args.source_only:
        print(f"verified calcifier {expected} source version surfaces")
        return 0
    if args.dist is None:
        parser.error("dist is required unless --source-only is used")
    sdist = _single(args.dist, "*.tar.gz")
    wheel = _single(args.dist, "*.whl")
    _verify_sdist(sdist, expected)
    _verify_wheel(wheel, expected)
    print(f"verified calcifier {expected}: {sdist.name}, {wheel.name}")
    return 0


def _single(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {pattern} in {root}, found {len(matches)}")
    return matches[0]


def _source_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _distribution_version(version: str) -> str:
    try:
        return str(Version(version))
    except InvalidVersion as exc:
        raise SystemExit(f"invalid project version {version!r}") from exc


def _verify_release_tag(tag: str, source_version: str) -> None:
    if not tag.startswith("v"):
        raise SystemExit(f"release tag must start with v: {tag}")
    tag_version = _distribution_version(tag.removeprefix("v"))
    project_version = _distribution_version(source_version)
    if tag_version != project_version:
        raise SystemExit(
            f"pyproject.toml version {source_version} != release tag {tag.removeprefix('v')}"
        )


def _verify_release_changelog(root: Path, version: str) -> None:
    changelog = (root / "CHANGELOG.md").read_text()
    match = re.search(rf"^## v{re.escape(version)} — (.+)$", changelog, re.MULTILINE)
    if match is None:
        raise SystemExit(f"CHANGELOG.md has no v{version} release heading")
    release_date = match.group(1)
    try:
        parsed = date.fromisoformat(release_date)
    except ValueError as exc:
        raise SystemExit(
            f"CHANGELOG.md v{version} release heading must use a YYYY-MM-DD date"
        ) from exc
    if parsed.isoformat() != release_date:
        raise SystemExit(f"CHANGELOG.md v{version} release date is not canonical: {release_date}")


def _verify_sdist(path: Path, version: str) -> None:
    version = _distribution_version(version)
    expected_root = f"calcifier-{version}"
    with tarfile.open(path) as archive:
        names = archive.getnames()
    if path.name != f"{expected_root}.tar.gz":
        raise SystemExit(f"sdist filename does not match version {version}: {path.name}")
    for name in names:
        parts = PurePosixPath(name).parts
        if FORBIDDEN_PARTS.intersection(parts) or name.endswith((".pyc", ".pyo")):
            raise SystemExit(f"forbidden path in sdist: {name}")
        if not parts or parts[0] != expected_root:
            raise SystemExit(f"unexpected sdist root: {name}")
    required = (
        "/.claude-plugin/marketplace.json",
        "/CODE_OF_CONDUCT.md",
        "/CONTRIBUTING.md",
        "/LICENSE",
        "/README.md",
        "/SECURITY.md",
        "/SUPPORT.md",
        "/plugins/calcifier/",
        "/tests/",
        "/src/calcifier/",
    )
    for suffix in required:
        if not any(suffix in name for name in names):
            raise SystemExit(f"sdist is missing {suffix}")


def _verify_wheel(path: Path, version: str) -> None:
    version = _distribution_version(version)
    if not path.name.startswith(f"calcifier-{version}-"):
        raise SystemExit(f"wheel filename does not match version {version}: {path.name}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode()
    required = {
        "calcifier/py.typed",
        "calcifier/trace-schema.json",
    }
    if missing := required.difference(names):
        raise SystemExit(f"wheel is missing: {sorted(missing)}")
    if not any(name.endswith(".dist-info/licenses/LICENSE") for name in names):
        raise SystemExit("wheel is missing its license file")
    if f"Version: {version}\n" not in metadata:
        raise SystemExit("wheel metadata version does not match")


def _verify_version_surfaces(root: Path, version: str) -> None:
    init_text = (root / "src" / "calcifier" / "__init__.py").read_text()
    match = re.search(r'^__version__ = "([^"]+)"$', init_text, re.MULTILINE)
    plugin = json.loads((root / "plugins/calcifier/.claude-plugin/plugin.json").read_text())
    marketplace = json.loads((root / ".claude-plugin/marketplace.json").read_text())
    changelog = (root / "CHANGELOG.md").read_text()
    with (root / "uv.lock").open("rb") as handle:
        lock = tomllib.load(handle)
    project = next(item for item in lock["package"] if item["name"] == "calcifier")
    surfaces = {
        "src/calcifier/__init__.py": match.group(1) if match else None,
        "plugin.json": plugin.get("version"),
        "uv.lock": project.get("version"),
    }
    if mismatches := {name: value for name, value in surfaces.items() if value != version}:
        raise SystemExit(f"version surface mismatch: {mismatches}")
    if f"## v{version} — " not in changelog:
        raise SystemExit(f"CHANGELOG.md has no v{version} release heading")
    plugins = marketplace.get("plugins")
    if not isinstance(plugins, list):
        raise SystemExit("marketplace.json has no plugins list")
    calcifier_plugin = next(
        (item for item in plugins if isinstance(item, dict) and item.get("name") == "calcifier"),
        None,
    )
    if calcifier_plugin is None or calcifier_plugin.get("source") != "./plugins/calcifier":
        raise SystemExit("marketplace.json does not expose ./plugins/calcifier")
    author = plugin.get("author")
    if not isinstance(author, dict) or not isinstance(author.get("name"), str):
        raise SystemExit("plugin.json author must be an object with a name")


if __name__ == "__main__":
    raise SystemExit(main())
