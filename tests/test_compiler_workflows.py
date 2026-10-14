"""Offline regressions for compiler workflows."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from calcifier.agent_events import AgentActivity
from calcifier.artifact import (
    _child_recipe_pins,
)
from calcifier.calcify import (
    calcify_file,
    render_dry_run,
)
from calcifier.catalog import (
    discover_sibling_skill_catalog,
)
from calcifier.registry import Registry
from tests.support.agents import StubProvider
from tests.support.compiler import (
    source_sequence,
)
from tests.support.compiler_cases import _TWO_ROLE_RECIPE, _compile_file_drafts
from tests.support.recipes import (
    _REPAIR_RECIPE,
    _REPAIR_SKILL,
)


def test_every_choice_of_a_dynamic_catalog_selector_is_pinned(tmp_path: Path) -> None:
    for name in ("brief", "detailed"):
        child = tmp_path / name / ".calcifier" / "recipe.py"
        child.parent.mkdir(parents=True)
        child.write_text("# child\n")
    target = tmp_path / "parent" / ".calcifier" / "recipe.py"
    source = textwrap.dedent(
        """
        from calcifier import call_catalog_recipe

        CHILDREN = {
            "brief": "../../brief/.calcifier/recipe.py",
            "detailed": "../../detailed/.calcifier/recipe.py",
        }

        def pick(style):
            return call_catalog_recipe(style, CHILDREN, {}).output
        """
    )
    catalog = {"brief": str(tmp_path / "brief" / ".calcifier" / "recipe.py")}

    pins = _child_recipe_pins(source, target, catalog)

    assert pins == {
        "brief": str((tmp_path / "brief" / ".calcifier" / "recipe.py").resolve()),
        "../../detailed/.calcifier/recipe.py": str(
            (tmp_path / "detailed" / ".calcifier" / "recipe.py").resolve()
        ),
    }


def test_calcify_file_repairs_syntax_the_configured_python_rejects(tmp_path: Path):
    (tmp_path / "ruff.toml").write_text(
        'target-version = "py311"\n\n[lint]\nselect = ["E", "F", "I"]\n'
    )
    skill = tmp_path / "SKILL.md"
    skill.write_text(_REPAIR_SKILL)
    incompatible = textwrap.dedent(
        r"""
        import re

        from calcifier import Budget, agent, recipe, step

        @step
        def inspect(target: str) -> bool:
            normalized = f"{re.sub(r'\t', '    ', target)}"
            return agent(
                "worker",
                "review {target}",
                working_directory=".",
                readable_paths=(".",),
                target=normalized,
            )

        @recipe(agents=("worker",), budget=Budget(2, 30))
        def writing_style(target: str = ".") -> bool:
            return inspect(target=target)

        if __name__ == "__main__":
            raise SystemExit(writing_style.cli())
        """
    ).strip()
    repaired = _REPAIR_RECIPE.replace("Budget(2, 180)", "Budget(2, 30)")
    compiler = source_sequence(incompatible, repaired)
    registry = Registry()
    registry.register(compiler)
    worker = StubProvider()
    worker.name = "worker"
    registry.register(worker)

    target = calcify_file(skill, registry)

    # Nothing rewrites the draft: the 3.11 grammar rejects it and one repair fixes it.
    assert len(compiler.source_prompts) == 2
    assert "emit syntax accepted by Python 3.11" in compiler.source_prompts[0]
    assert (
        "forbid every backslash inside an f-string replacement field"
        in (compiler.source_prompts[0])
    )
    assert "Python 3.11" in compiler.source_prompts[1]
    assert "re.sub" not in target.read_text()


def test_a_dry_run_that_cannot_start_is_reported_as_unverified(tmp_path: Path) -> None:
    requires_fixture = _TWO_ROLE_RECIPE.replace(
        'def writing_style(target: str = ".") -> bool:',
        'def writing_style(labels: list[object], target: str = ".") -> bool:',
    )
    assert requires_fixture != _TWO_ROLE_RECIPE
    activity = AgentActivity()

    _compile_file_drafts(tmp_path, requires_fixture, activity=activity)

    assert render_dry_run(activity) == (
        "dry run: unverified (a required recipe input cannot be synthesized)\n"
        "dry-run writes before it stopped (partial): none"
    )
    assert render_dry_run(AgentActivity()) == "dry run: not run"


def test_stray_agent_option_gets_the_dry_run_diagnostic_not_a_rewrite(tmp_path: Path) -> None:
    stray = _TWO_ROLE_RECIPE.replace(
        '"review {target}",\n        working_directory=".",',
        '"review {target}",\n        max_calls=2,\n        working_directory=".",',
    )
    assert "max_calls=2" in stray

    compiler, generated = _compile_file_drafts(tmp_path, stray, _TWO_ROLE_RECIPE)

    assert len(compiler.source_prompts) == 2
    assert "no {name} placeholder" in compiler.source_prompts[1]
    assert "max_calls" in compiler.source_prompts[1]
    assert "max_calls=" not in generated


def test_calcify_discovers_real_sibling_child_recipes(tmp_path: Path):
    parent = tmp_path / "skills" / "parent" / "SKILL.md"
    child = tmp_path / "skills" / "analysis-real" / ".calcifier" / "recipe.py"
    parent.parent.mkdir(parents=True)
    child.parent.mkdir(parents=True)
    parent.write_text("Run /analysis-*.")
    child.write_text("# child")

    assert discover_sibling_skill_catalog(parent) == {"analysis-real": str(child.resolve())}


def test_absent_default_input_does_not_request_a_paid_source_repair(tmp_path: Path) -> None:
    source = textwrap.dedent(
        """
        from pathlib import Path

        from calcifier import recipe, step

        @step
        def read_scope(target: str) -> str:
            return Path(target, "scope.json").read_text(encoding="utf-8")

        @recipe
        def review(target: str = "target") -> str:
            return read_scope(target)

        if __name__ == "__main__":
            raise SystemExit(review.cli())
        """
    )
    activity = AgentActivity()

    compiler, accepted = _compile_file_drafts(tmp_path, source, activity=activity)

    assert compiler.source_calls == 1
    assert 'Path(target, "scope.json").read_text' in accepted
    rendered = render_dry_run(activity)
    assert rendered.startswith("dry run: unverified (file beneath an absent default path")
    assert "scope.json" in rendered
    assert rendered.endswith("dry-run writes before it stopped (partial): none")
    event = next(item for item in activity.events if item["phase"] == "dry_run")
    assert event["status"] == "unverified"
    assert json.loads(str(event["detail"]))["complete"] is False


@pytest.mark.parametrize(
    "case,limit,calls,status",
    [
        ("corrected", 3, 2, "passed"),
        ("fixture", 3, 2, "unverified"),
        ("fixture_budget", 1, 1, "unverified"),
        ("invalid_advice", 3, 3, "passed"),
        ("downstream", 3, 2, "passed"),
    ],
)
def test_command_gaps_receive_one_bounded_repair(tmp_path, case, limit, calls, status):
    bad = repr([sys.executable, "--calcifier-invalid-flag"])
    good = repr([sys.executable, "-c", "print('\"done\"')"])
    command = (
        repr(
            [
                sys.executable,
                "-c",
                "import json, sys; from pathlib import Path; "
                "print(json.dumps(Path(sys.argv[1]).read_text()))",
            ]
        )
        + " + [str(data)]"
        if case in {"fixture", "fixture_budget", "downstream"}
        else bad
    )
    source = f"""from pathlib import Path
from calcifier import recipe, run_command, step, write_text

@step
def inspect(data: Path) -> str:
    return run_command({command}, timeout_seconds=10, mutating=False, output=str)

@step
def finish(value: str) -> str:
    write_text("after.txt", value)
    return {"str(1 // 0)" if case == "downstream" else "value"}

@recipe(capabilities=("Bash",), output_root="artifacts")
def workflow(data: Path) -> str:
    return finish(inspect(data))

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    corrected = source.replace(command, good).replace("str(1 // 0)", "value")
    if case == "fixture":
        corrected = source
    drafts = [source]
    if case == "invalid_advice":
        drafts.append("def broken(:")
    if limit > 1:
        drafts.append(corrected)
    compiler = source_sequence(*drafts)
    registry = Registry()
    registry.register(compiler)
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Inspect\nRead data and save the result.\n")
    activity = AgentActivity()
    calcify_file(skill, registry, max_compiler_calls=limit, activity=activity)
    assert compiler.source_calls == calls
    assert render_dry_run(activity).startswith(f"dry run: {status}")
    assert "artifacts/after.txt" in render_dry_run(activity)
    if limit > 1:
        prompt = compiler.source_prompts[1]
        repair = prompt.removeprefix(compiler.source_prompts[0])
        diagnostics = json.loads(
            repair.split("Structured diagnostics (JSON array):\n", 1)[1].split("\n", 1)[0]
        )
        (diagnostic,) = diagnostics
        assert diagnostic["phase"] == "source.validation"
        if case == "downstream":
            assert diagnostic["code"] == "dry_run"
            assert diagnostic["node"] == "finish"
            # Python versions use different wording for integer division by zero.
            assert "ZeroDivisionError:" in diagnostic["actual"]
        elif case == "fixture":
            assert diagnostic["code"] == "command_gap"
            assert diagnostic["node"] == "inspect"
        else:
            assert diagnostic["code"] == "dry_run"
            assert diagnostic["node"] == "inspect"
        if case != "fixture" and case != "downstream":
            assert "--calcifier-invalid-flag" in prompt
