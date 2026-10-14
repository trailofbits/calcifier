"""Artifact paths, metadata, writes, and summary rendering."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any, cast

import pytest

from calcifier.agent_events import AgentProvider
from calcifier.artifact import _atomic_write_text, _output_path
from calcifier.calcify import calcify_file, prepare_artifact, render_summary
from calcifier.metadata import attach_metadata, parse_metadata_namespace, split_metadata
from calcifier.portable import lint_portable
from calcifier.recipe import load_recipe
from calcifier.recipe_draft import draft_response as _draft
from calcifier.registry import Registry
from tests.support.compiler import ScriptedCompiler as _ScriptedCompiler
from tests.support.compiler import registry_emitting as _registry_emitting
from tests.support.recipes import _GOOD, _GOOD_RECIPE, _SKILL
from tests.support.recipes import write_skill as _write_skill
from tests.support.skill_source import SKILL
from tests.support.trust import record_trust

# -- _output_path ------------------------------------------------------------


def test_output_path_default_is_dot_calcifier_subdir(tmp_path: Path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("# skill")
    p = _output_path(skill, None)
    assert p == tmp_path / ".calcifier" / "recipe.py"


def test_output_path_override_wins(tmp_path: Path):
    skill = tmp_path / "SKILL.md"
    explicit = tmp_path / "elsewhere" / "r.py"
    assert _output_path(skill, explicit) == explicit


# -- metadata attachment -----------------------------------------------------


def test_attach_metadata_records_path_and_sha(tmp_path: Path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("# skill body\n")
    out = attach_metadata("# python source\n", skill, recipe_path=tmp_path / "recipe.py")
    assert "_CALCIFIER_SKILL" in out
    assert "'sha256'" in out
    assert "'path'" in out
    assert "'SKILL.md'" in out


def test_calcify_file_records_portable_relative_skill_path(tmp_path: Path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("# skill body\n")
    recipe_path = tmp_path / ".calcifier" / "recipe.py"

    out = attach_metadata("# python source\n", skill, recipe_path=recipe_path)
    _, metadata = split_metadata(out)
    namespace = parse_metadata_namespace(metadata or "")

    skill_metadata = cast("dict[str, object]", namespace["_CALCIFIER_SKILL"])
    assert skill_metadata["path"] == "../SKILL.md"


def test_render_summary_never_executes_class_body(tmp_path: Path):
    marker = tmp_path / "executed"
    recipe_path = tmp_path / "recipe.py"
    recipe_path.write_text(
        textwrap.dedent(
            f"""
            from calcifier import recipe, step

            class AgentOutput:
                __import__("pathlib").Path({str(marker)!r}).write_text("executed")

            @step
            def work(value: str) -> str:
                return value

            @recipe(version="0.1")
            def safe(value: str = "x") -> str:
                return work(value)

            if __name__ == "__main__":
                raise SystemExit(safe.cli())
            """
        )
    )

    summary = render_summary(recipe_path)

    assert "name: safe" in summary
    assert not marker.exists()


# -- calcify_file (atomic write + metadata) ----------------------------------


def test_atomic_recipe_writer_uses_utf8_with_windows_default_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_fdopen = os.fdopen

    def windows_fdopen(fd: int, mode: str = "r", **kwargs: Any) -> Any:
        if "b" not in mode:
            kwargs["encoding"] = kwargs.get("encoding") or "cp1252"
        return original_fdopen(fd, mode, **kwargs)

    monkeypatch.setattr(os, "fdopen", windows_fdopen)
    source = "# Neutral recipe — café\n"
    target = tmp_path / "recipe.py"

    _atomic_write_text(target, source)

    assert target.read_bytes() == source.encode("utf-8")


@pytest.mark.parametrize("publish", [False, True])
def test_file_apis_compile_a_recipe_in_one_call_by_default(tmp_path: Path, publish: bool):
    skill = tmp_path / "SKILL.md"
    skill.write_text(SKILL, encoding="utf-8")
    compiler = _ScriptedCompiler([_GOOD_RECIPE], wrap_sources=True)
    registry = Registry()
    registry.register(compiler)
    if publish:
        target = calcify_file(skill, registry)
        assert target.exists()
        body = target.read_text(encoding="utf-8")
    else:
        prepared = prepare_artifact(skill, registry)
        assert not prepared.path.exists()
        body = prepared.source
    assert "@recipe" in body
    assert "_CALCIFIER_SKILL" in body
    assert len(compiler.calls) == 1
    assert "Read every target." in compiler.all_prompts[0]
    assert (
        "Start with one agent session, then move concrete, repeatable operations into Python"
        in compiler.all_prompts[0]
    )


def test_calcify_file_formats_and_lints_body_and_appended_metadata(tmp_path: Path):
    (tmp_path / "ruff.toml").write_text(
        'target-version = "py311"\nline-length = 100\n\n[lint]\nselect = ["E", "F", "I"]\n'
    )
    skill = tmp_path / "SKILL.md"
    skill.write_text("# " + "long source evidence " * 12 + "\n")
    compact = _GOOD_RECIPE.replace("return bool(target)", "value = target; return bool(value)")
    target = calcify_file(skill, _registry_emitting(compact), compiler="compiler")
    source = target.read_text()

    assert "value = target\n    return bool(value)" in source
    assert "_CALCIFIER_SUBRECIPES = []" in source
    assert "_CALCIFIER_PLAN" not in source
    assert lint_portable(target) == []


def test_final_portability_failure_is_repaired_before_acceptance(tmp_path: Path):
    child_skill = tmp_path / "child.md"
    child_skill.write_text("# Count the input\n", encoding="utf-8")
    child = calcify_file(
        child_skill, _registry_emitting(_GOOD_RECIPE), output=tmp_path / "child.py"
    )
    skill = tmp_path / "SKILL.md"
    skill.write_text("# Delegate to the child\n", encoding="utf-8")
    source = (
        "from calcifier import call_catalog_recipe, recipe\n\n"
        'CHILDREN: dict[str, str] = {"child": "child.py"}\n\n'
        "@recipe\ndef run(target: str = '.') -> bool:\n"
        '    result = call_catalog_recipe("child", CHILDREN, {"target": target})\n'
        "    return bool(result.output)\n\n"
        'if __name__ == "__main__":\n    raise SystemExit(run.cli())\n'
    )
    compiler = _ScriptedCompiler(
        [source.replace('"child.py"}', '"child.py".strip()}'), source], wrap_sources=True
    )
    registry = _registry_for(compiler)
    artifact = prepare_artifact(
        skill, registry, output=tmp_path / "parent.py", skill_catalog={"child": str(child)}
    )
    assert len(compiler.calls) == 2
    repair = compiler.all_prompts[1].split("## Rejected candidate:", 1)[1]
    assert "catalog subrecipe mapping must be literal" in repair
    assert "_CALCIFIER_SKILL" not in repair
    artifact.path.write_text(artifact.source, encoding="utf-8")
    assert lint_portable(artifact.path) == []
    record_trust(child, recorded_by="test")
    from calcifier.hashing import sha256_hex

    record_trust(
        artifact.path, recorded_by="test", dependencies=[(child, sha256_hex(child.read_bytes()))]
    )
    run = load_recipe(artifact.path).run({"target": "."}, registry=registry)
    assert run.ok, run.error
    assert run.output is True


@pytest.mark.parametrize("absolute_path", [False, True])
def test_absolute_output_root_survives_compilation_and_runtime_override(tmp_path, absolute_path):
    root = tmp_path / "declared-output"
    destination = str(root / "result.txt") if absolute_path else "result.txt"
    source = (
        "from calcifier import recipe, write_text\n\n"
        f"@recipe(output_root={str(root)!r})\n"
        "def run() -> str:\n"
        f"    return str(write_text({destination!r}, 'complete'))\n\n"
        'if __name__ == "__main__":\n    raise SystemExit(run.cli())\n'
    )
    skill = tmp_path / "SKILL.md"
    skill.write_text("# Write the result\n", encoding="utf-8")
    target = calcify_file(skill, _registry_emitting(source))
    assert not root.exists(), "compiler dry run must not write the declared absolute root"
    loaded = load_recipe(target)
    assert loaded.output_root == root.resolve()
    result = loaded.run({})
    assert result.ok, result.error
    assert (root / "result.txt").read_text(encoding="utf-8") == "complete"
    if not absolute_path:
        redirected = tmp_path / "redirected"
        result = loaded.run({}, output_directory=redirected)
        assert result.ok, result.error
        assert (redirected / "result.txt").read_text(encoding="utf-8") == "complete"


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_final_validation_shares_the_call_budget_and_preserves_the_baseline(
    tmp_path, fallback, newline
):
    from calcifier.baseline import is_baseline
    from calcifier.diagnostics import CalcifyError
    from tests.support.agents import StubProvider

    skill = tmp_path / "SKILL.md"
    skill_bytes = b"# Complete original instructions" + newline
    skill.write_bytes(skill_bytes)
    source = _GOOD_RECIPE.replace(
        "from calcifier import recipe, step",
        "from calcifier import call_catalog_recipe, recipe, step",
    ) + (
        "\n\ndef dispatch(path: str) -> object:\n"
        '    return call_catalog_recipe("child", {"child": path}, {})\n'
    )
    compiler = _ScriptedCompiler([source, source], wrap_sources=True)
    registry = _registry_for(compiler)
    destination = tmp_path / "recipe.py"
    destination.write_bytes(b"original artifact\n")
    if fallback:
        registry.register(StubProvider())
        artifact = prepare_artifact(skill, registry, output=destination, max_compiler_calls=2)
        assert is_baseline(artifact.source)
    else:
        with pytest.raises(CalcifyError, match="global compiler-call budget"):
            prepare_artifact(skill, registry, output=destination, max_compiler_calls=2)
    assert len(compiler.calls) == 2
    assert destination.read_bytes() == b"original artifact\n"
    assert skill.read_bytes() == skill_bytes


def test_calcify_file_respects_output_override(tmp_path: Path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("# my skill\n")
    target_override = tmp_path / "custom" / "r.py"
    reg = _registry_emitting(_GOOD_RECIPE)
    written = calcify_file(skill, reg, compiler="compiler", output=target_override)
    assert written == target_override
    assert target_override.exists()


def test_calcify_attaches_resource_manifest(tmp_path: Path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("# my skill\n")
    (tmp_path / "references").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "references" / "notes.md").write_text("notes\n")
    (tmp_path / "scripts" / "run.sh").write_text("run\n")
    reg = _registry_emitting(_GOOD_RECIPE)

    target = calcify_file(skill, reg, compiler="compiler")
    _, metadata = split_metadata(target.read_text())
    namespace = parse_metadata_namespace(metadata or "")

    resources = namespace["_CALCIFIER_RESOURCES"]
    assert isinstance(resources, list)
    resource_data = cast("list[dict[str, object]]", resources)
    assert [entry["path"] for entry in resource_data] == ["references/notes.md", "scripts/run.sh"]
    assert all(entry["sha256"] for entry in resource_data)


# -- static recipe summary ----------------------------------------------------

_SUMMARY_TEMPLATE = """from calcifier import recipe, step


@step
def count_entries(target: str) -> bool:
    return bool(target)


{decorators}
def tally(target: str = ".") -> bool:
    return count_entries(target=target)
"""


@pytest.mark.parametrize(
    ("decorators", "version"),
    [
        ('@recipe(version="2.5")', "2.5"),
        ("@recipe", "0.1"),
        ('@recipe(agents=("worker",))', "0.1"),
        ("@recipe(version=3)", "0.1"),
        ('@recipe(output_root="out", version="2.5")', "2.5"),
        ('@traced(version="9.9")\n@recipe(version="2.5")', "2.5"),
    ],
)
def test_render_summary_reports_exact_declared_version(
    tmp_path: Path, decorators: str, version: str
) -> None:
    recipe_path = tmp_path / "recipe.py"
    recipe_path.write_text(_SUMMARY_TEMPLATE.format(decorators=decorators), encoding="utf-8")

    assert render_summary(recipe_path) == (
        f"name: tally\nversion: {version}\ninputs: target='.'\nsteps: ['count_entries']\n"
        "budget: none declared"
    )


# -- calcified file: formatted source plus provenance metadata ------------------

# Failure modes whose repr is exactly the 48-character inline limit, one past
# it, and one long enough to need three 60-character metadata chunks.


def _neutral_skill(root: Path) -> Path:
    return _write_skill(root / "neutral", _SKILL)


def _registry_for(compiler: AgentProvider) -> Registry:
    registry = Registry()
    registry.register(compiler)
    return registry


@pytest.mark.parametrize(
    ("relative", "recorded_skill_path"),
    [
        ("recipe.py", "neutral/SKILL.md"),
        ("nested/deeper/recipe.py", "../../neutral/SKILL.md"),
    ],
)
def test_calcify_writes_explicit_output_in_existing_or_missing_directories(
    tmp_path: Path, relative: str, recorded_skill_path: str
) -> None:
    skill = _neutral_skill(tmp_path)
    output = tmp_path / relative

    target = calcify_file(skill, _registry_for(_ScriptedCompiler([_draft(_GOOD)])), output=output)

    assert target == output
    lines = output.read_text(encoding="utf-8").splitlines()
    assert f'    "path": "{recorded_skill_path}",' in lines
    assert not list(output.parent.glob(".calcifier.tmp.*"))
