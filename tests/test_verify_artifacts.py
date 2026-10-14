"""Tests for release artifact version normalization."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.verify_artifacts import (
    _distribution_version,
    _verify_release_changelog,
    _verify_release_tag,
    _verify_sdist,
    _verify_wheel,
)


def test_distribution_version_normalizes_prerelease_spelling():
    assert _distribution_version("0.1.0-rc.1") == "0.1.0rc1"


def test_release_tag_compares_normalized_versions():
    _verify_release_tag("v0.1.0-rc.1", "0.1.0rc1")


def test_release_tag_rejects_version_drift():
    with pytest.raises(SystemExit, match="version 0.1.0 != release tag 0.2.0"):
        _verify_release_tag("v0.2.0", "0.1.0")


def test_release_changelog_accepts_canonical_date(tmp_path: Path):
    (tmp_path / "CHANGELOG.md").write_text("## v0.1.0 — 2026-09-03\n")

    _verify_release_changelog(tmp_path, "0.1.0")


def test_release_changelog_rejects_pending_heading(tmp_path: Path):
    (tmp_path / "CHANGELOG.md").write_text("## v0.1.0 — Pending\n")

    with pytest.raises(SystemExit, match="must use a YYYY-MM-DD date"):
        _verify_release_changelog(tmp_path, "0.1.0")


def test_sdist_uses_normalized_prerelease_filename(tmp_path: Path):
    root = "calcifier-0.1.0rc1"
    path = tmp_path / f"{root}.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for suffix in (
            ".claude-plugin/marketplace.json",
            "CODE_OF_CONDUCT.md",
            "CONTRIBUTING.md",
            "LICENSE",
            "README.md",
            "SECURITY.md",
            "SUPPORT.md",
            "plugins/calcifier/plugin.json",
            "tests/test_package.py",
            "src/calcifier/__init__.py",
        ):
            archive.addfile(tarfile.TarInfo(f"{root}/{suffix}"))

    _verify_sdist(path, "0.1.0-rc.1")


def test_wheel_uses_normalized_prerelease_filename_and_metadata(tmp_path: Path):
    path = tmp_path / "calcifier-0.1.0rc1-py3-none-any.whl"
    dist_info = "calcifier-0.1.0rc1.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("calcifier/py.typed", "")
        archive.writestr("calcifier/trace-schema.json", "{}")
        archive.writestr(f"{dist_info}/licenses/LICENSE", "MIT")
        archive.writestr(f"{dist_info}/METADATA", "Version: 0.1.0rc1\n")

    _verify_wheel(path, "0.1.0-rc.1")
