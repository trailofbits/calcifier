"""Scratch relocation preserves host pins and original program bytes."""

from __future__ import annotations

import ast

import pytest

from calcifier.dry_run import _staged_source
from tests.support.recipes import _GOOD_RECIPE


def program_digest(module: ast.Module) -> str:
    """Dump the program without its provenance assignments."""
    return ast.dump(
        ast.Module(
            [
                node
                for node in module.body
                if not (
                    isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id.startswith("_CALCIFIER_")
                )
            ],
            [],
        )
    )


def test_staging_relocates_only_skill_path(tmp_path):
    source = _GOOD_RECIPE + '\n_CALCIFIER_SKILL = {"path": "../SKILL.md", "sha256": "pinned"}\n'
    target = tmp_path / "skills" / "Unicode café" / "SKILL.md"
    staged = _staged_source(source, target)
    assert program_digest(ast.parse(source)) == program_digest(ast.parse(staged))
    node = next(
        n
        for n in ast.parse(staged).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "_CALCIFIER_SKILL"
    )
    assert ast.literal_eval(node.value)["path"] == str(target)
    assert source.split("_CALCIFIER_SKILL")[0] == staged.split("_CALCIFIER_SKILL")[0]


@pytest.mark.parametrize(
    "source",
    [
        "not Python (",
        "x = 1\n",
        '_CALCIFIER_SKILL = dict(path="missing")\n',
        "_CALCIFIER_SKILL = 1\n",
        '_CALCIFIER_SKILL = {"sha256": "hash"}\n',
        '_CALCIFIER_SKILL = alias = {"path": "missing"}\n',
    ],
)
def test_staging_does_not_repair_or_introduce_provenance(source, tmp_path):
    assert _staged_source(source, tmp_path / "SKILL.md") == source


def test_staging_uses_python_line_coordinates_and_preserves_other_metadata(tmp_path):
    prefix = "# Unicode separator\u2028inside one Python line\r\n"
    source = prefix + '_CALCIFIER_SKILL = {"path": "old", "sha256": "retained"}\r\nOTHER = 1\r\n'
    staged = _staged_source(source, tmp_path / "SKILL.md")
    assert staged.startswith(prefix) and staged.endswith("OTHER = 1\r\n")
    node = next(
        n
        for n in ast.parse(staged).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "_CALCIFIER_SKILL"
    )
    assert ast.literal_eval(node.value) == {
        "path": str(tmp_path / "SKILL.md"),
        "sha256": "retained",
    }
    assert program_digest(ast.parse(source)) == program_digest(ast.parse(staged))


@pytest.mark.parametrize("ending", ["\r", "\r\n", "\n"])
def test_staging_matches_python_newlines_without_splitting_unicode_separators(tmp_path, ending):
    prefix = 'TEXT = """first' + ending + 'second\u2028same-line"""' + ending
    source = prefix + '_CALCIFIER_SKILL = {"path": "old", "sha256": "retained"}' + ending
    source += 'OTHER = "unchanged"' + ending
    staged = _staged_source(source, tmp_path / "SKILL.md")
    assert staged.startswith(prefix) and staged.endswith('OTHER = "unchanged"' + ending)
    assert program_digest(ast.parse(source)) == program_digest(ast.parse(staged))
    node = next(
        n
        for n in ast.parse(staged).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "_CALCIFIER_SKILL"
    )
    assert ast.literal_eval(node.value) == {
        "path": str(tmp_path / "SKILL.md"),
        "sha256": "retained",
    }
