"""Offline CLI dogfood for the four plugin recipes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import pytest

from calcifier import cli, mutations
from calcifier.metadata import parse_metadata_namespace, split_metadata
from calcifier.recipe import load_recipe
from calcifier.subprocesses import ProcessResult

_REPO_ROOT = Path(__file__).parents[1]
_SKILLS_ROOT = _REPO_ROOT / "plugins" / "calcifier" / "skills"


def _install_fake_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mutations,
        "run_process",
        lambda argv, **kwargs: ProcessResult(
            tuple(argv),
            0,
            b"no changes: the prepared recipe matches /fixture/.calcifier/recipe.py\n",
            b"",
        ),
    )


@pytest.mark.parametrize("skill", ["calcifier", "exec", "debug", "update"])
def test_plugin_recipe_metadata_matches_current_skill(skill: str) -> None:
    skill_path = _SKILLS_ROOT / skill / "SKILL.md"
    recipe_path = skill_path.parent / ".calcifier" / "recipe.py"
    _, metadata = split_metadata(recipe_path.read_text())
    assert metadata is not None
    recorded = cast(
        "dict[str, object]",
        parse_metadata_namespace(metadata)["_CALCIFIER_SKILL"],
    )
    assert isinstance(recorded, dict)
    assert recorded["sha256"] == hashlib.sha256(skill_path.read_bytes()).hexdigest()
    assert load_recipe(recipe_path).skill_root == skill_path.parent


@pytest.mark.parametrize(
    ("skill", "inputs", "expected_rc", "expected_text"),
    [
        ("calcifier", ["skill=fixture"], 1, "side effect blocked by hard dry-run"),
        ("exec", ["recipe_path=/fixture/recipe.py"], 1, "side effect blocked by hard dry-run"),
        ("debug", ["recipe_path=/fixture/recipe.py"], 1, "side effect blocked by hard dry-run"),
        ("update", ["skill_path=/fixture/SKILL.md"], 0, "OK"),
    ],
)
def test_plugin_recipe_offline_cli_dogfood(
    skill: str,
    inputs: list[str],
    expected_rc: int,
    expected_text: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_command(monkeypatch)
    recipe_path = _SKILLS_ROOT / skill / ".calcifier" / "recipe.py"
    assert not recipe_path.with_name("recipe.py.trust.json").is_file()
    argv = ["exec", str(recipe_path), "--trust", "--dry-run", "hard"]
    for value in inputs:
        argv.extend(("--input", value))

    rc = cli.main(argv)
    captured = capsys.readouterr()

    assert rc == expected_rc
    assert expected_text in captured.out


def test_exec_dogfood_passes_hostile_inputs_as_literal_argv(monkeypatch):
    recipe_path = _SKILLS_ROOT / "exec" / ".calcifier" / "recipe.py"
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):  # noqa: ANN001
        del kwargs
        seen.append(list(argv))
        return ProcessResult(tuple(argv), 0, b"ok", b"")

    monkeypatch.setattr(mutations, "run_process", fake_run)
    recipe_obj = load_recipe(recipe_path)
    hostile_recipe = "victim.py; touch /tmp/calcifier-pwned"  # noqa: S108
    hostile_input = "target=$(touch /tmp/calcifier-pwned)"  # noqa: S108

    run = recipe_obj.run(
        {
            "recipe_path": hostile_recipe,
            "inputs": hostile_input,
            "trust": False,
        }
    )

    assert run.ok
    assert seen == [
        [
            "calcifier",
            "exec",
            hostile_recipe,
            "--input",
            hostile_input,
        ]
    ]
