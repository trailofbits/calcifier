"""Packaged-resource known answers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from calcifier.skill_package import RESOURCE_FILE_SIZE_LIMIT, ResourceEntry, discover_resources


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- skill package discovery ----------------------------------------------------


def test_discover_resources_applies_alias_duplicate_and_size_rules(
    tmp_path: Path,
) -> None:
    skills = tmp_path / "plugin" / "skills"
    skill_root = skills / "demo"
    (skill_root / "references").mkdir(parents=True)
    (skill_root / "references" / "local.md").write_bytes(b"local\n")
    for directory in ("references", "notes", "scripts"):
        (skills / directory).mkdir()
    exact = b"e" * RESOURCE_FILE_SIZE_LIMIT
    (skills / "references" / "local.md").write_bytes(b"external\n")
    (skills / "references" / "exact.txt").write_bytes(exact)
    (skills / "references" / "over.txt").write_bytes(b"o" * (RESOURCE_FILE_SIZE_LIMIT + 1))
    (skills / "notes" / "readme.txt").write_bytes(b"notes\n")
    script = b"print('ok')\n"
    (skills / "scripts" / "tool.py").write_bytes(script)
    referenced = ["local.md", "exact.txt", "over.txt"]
    lines = [f"- {{baseDir}}/../references/{name}" for name in referenced]
    lines += ["- {baseDir}/../notes/readme.txt", "- {baseDir}/../scripts/tool.py"]
    (skill_root / "SKILL.md").write_text("\n".join(lines) + "\n")

    entries = discover_resources(skill_root / "SKILL.md")

    assert entries == [
        ResourceEntry("references/exact.txt", _sha(exact), len(exact), "../references/exact.txt"),
        ResourceEntry("references/local.md", _sha(b"local\n"), 6, None),
        ResourceEntry("scripts/tool.py", _sha(script), len(script), "../scripts/tool.py"),
    ]
