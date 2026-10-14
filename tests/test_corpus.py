"""A diverse-skill corpus: each scripted draft goes through the whole compile gate.

Every directory under ``tests/corpus`` holds a SKILL.md (and any packaged
resources), the final draft a compiler would return, and ``case.json``:

- ``kind``: ``baseline`` cases must compile on the first source call, dry run
  included; ``rejected`` cases record the current gate's refusal.
- ``rejection``: for rejected cases, text the refusal contains.
- ``flips_at``: for rejected cases, the plan step after which they must compile.
- ``run``: optional ``inputs``, per-role ``responses`` and ``artifacts`` (paths
  relative to the run directory) for a normal-mode run after compiling.
- ``children``: names of sibling cases under ``children/`` compiled first, so
  the parent's catalog pins them.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from calcifier.agent_events import AgentActivity, AgentResult
from calcifier.calcify import CalcifyError, calcify_file, render_dry_run
from calcifier.portable import lint_portable
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from tests.support.agents import StubProvider
from tests.support.compiler import ScriptedCompiler
from tests.support.trust import record_trust

pytestmark = pytest.mark.semantic_offline

_CORPUS = Path(__file__).parent / "corpus"
_SCRIPT_FILES = ("draft.py", "case.json", "children")


def _case(path: Path) -> dict[str, Any]:
    return json.loads((path / "case.json").read_text(encoding="utf-8"))


def _cases(kind: str) -> list[str]:
    return sorted(
        path.name
        for path in _CORPUS.iterdir()
        if (path / "case.json").is_file() and _case(path)["kind"] == kind
    )


class _Worker(StubProvider):
    """A semantic role that returns one scripted JSON response."""

    def __init__(self, name: str, response: Any = None) -> None:
        super().__init__()
        self.name = name
        self.response = response

    def invoke(self, prompt, **kwargs):  # noqa: ANN001, ANN003
        del prompt, kwargs
        text = json.dumps(self.response)
        return AgentResult(text, text, "", 0, self.name)


def _registry(roles: list[str], responses: dict[str, Any] | None = None) -> Registry:
    registry = Registry()
    for role in roles:
        registry.register(_Worker(role, (responses or {}).get(role)))
    return registry


def _compile(
    case: Path, skills: Path, activity: AgentActivity | None = None
) -> tuple[ScriptedCompiler, Path]:
    """Stage ``case`` as a skill under ``skills`` and compile it with its scripted drafts."""
    for child in _case(case).get("children", []):
        _compile(case / "children" / child, skills)
    skill_dir = skills / case.name
    shutil.copytree(case, skill_dir, ignore=shutil.ignore_patterns(*_SCRIPT_FILES))
    compiler = ScriptedCompiler(
        [(case / "draft.py").read_text(encoding="utf-8")],
        wrap_sources=True,
        repeat_last=True,
        default_timeout=300,
    )
    registry = _registry(compiler.roles)
    registry.register(compiler)
    target = calcify_file(skill_dir / "SKILL.md", registry, activity=activity)
    return compiler, target


@pytest.mark.parametrize("name", _cases("baseline"))
def test_a_baseline_case_compiles_on_its_first_draft(name: str, tmp_path: Path) -> None:
    activity = AgentActivity()
    compiler, recipe_path = _compile(_CORPUS / name, tmp_path / "skills", activity)

    assert compiler.source_calls == 1
    assert lint_portable(recipe_path) == []
    # The dry run finished, so its write list is complete, not merely unverified.
    summary = render_dry_run(activity)
    assert summary.startswith("dry run: passed\ndry-run writes: ")
    if "run" in _case(_CORPUS / name):
        assert not summary.endswith("writes: none")


@pytest.mark.parametrize("name", _cases("rejected"))
def test_a_rejected_case_records_the_current_refusal(name: str, tmp_path: Path) -> None:
    case = _CORPUS / name

    with pytest.raises(CalcifyError) as caught:
        _compile(case, tmp_path / "skills")

    assert _case(case)["rejection"] in str(caught.value)


@pytest.mark.parametrize(
    "name", [name for name in _cases("baseline") if "run" in _case(_CORPUS / name)]
)
def test_a_file_writing_case_writes_its_artifacts_in_a_normal_run(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _CORPUS / name
    compiler, recipe_path = _compile(case, tmp_path / "skills")
    spec = _case(case)["run"]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("CALCIFIER_HOME", str(tmp_path / "trust"))
    record_trust(recipe_path, recorded_by="test", dependencies=())
    monkeypatch.chdir(workspace)

    run = load_recipe(recipe_path).run(
        spec.get("inputs", {}), registry=_registry(compiler.roles, spec["responses"])
    )

    assert run.ok, run.error
    for artifact in spec["artifacts"]:
        assert (workspace / artifact).read_text(encoding="utf-8").strip()


def test_the_corpus_stays_small_enough_to_run_on_every_change() -> None:
    names = _cases("baseline") + _cases("rejected")

    assert 6 <= len(names) <= 10
