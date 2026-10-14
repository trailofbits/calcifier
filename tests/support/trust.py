"""Test helpers for granting and checking trust by file path."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from calcifier.trust import TrustResult, check_trust_bytes, record_trust_bytes


def check_trust(recipe_path: Path) -> TrustResult:
    return check_trust_bytes(recipe_path, recipe_path.read_bytes())


def record_trust(
    recipe_path: Path,
    *,
    recorded_by: str,
    dependencies: Iterable[tuple[str | Path, str]] = (),
) -> TrustResult:
    return record_trust_bytes(
        recipe_path,
        recipe_path.read_bytes(),
        recorded_by=recorded_by,
        dependencies=dependencies,
    )
