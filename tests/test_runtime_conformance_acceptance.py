"""Runtime-contract defects reject improvements or loading before any worker call.

The fixtures are neutral reductions of accepted-but-unrunnable recipes: one
named a packaged directory with ``resource(".")``, and others consumed a
Calcifier helper's result as a type it is not declared to return, failing only
after a paid worker call.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from calcifier.agent_events import AgentActivity, AgentResult
from calcifier.calcify import calcify_file, render_dry_run
from calcifier.exceptions import RecipeError
from calcifier.portable import lint_portable
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from calcifier.type_check import type_check_source
from tests.support.agents import StubProvider
from tests.support.compiler import ScriptedCompiler, source_sequence

pytestmark = pytest.mark.semantic_offline

_SKILL = """\
Review the supplied workspace against the packaged guide and record grounded findings.
Write the completed report to `<destination>/report.md`.
"""
_REVIEW, _PERSIST = _SKILL.splitlines()
_VALID = (
    textwrap.dedent(
        """
        from pathlib import Path

        from calcifier import (
            Budget,
            agent,
            emit,
            output_type,
            recipe,
            resource,
            step,
            verify_artifact,
            write_text,
        )

        @output_type
        class Finding:
            source: str
            start_line: int
            end_line: int
            category: str
            violated_invariant: str
            evidence: str

        @output_type
        class Review:
            findings: list[Finding]

        @step
        def review_workspace(working_directory: str) -> Review:
            guide = resource("references/guide.md").read_text()
            return agent(
                "worker",
                "Follow this guide, then return grounded findings for the workspace: {guide}",
                output=Review,
                working_directory=working_directory,
                readable_paths=(working_directory,),
                guide=guide,
            )

        @step
        def persist_report(review: Review, working_directory: str, destination: str) -> Path:
            text = "\\n".join(
                f"{finding.source}:{finding.start_line}-{finding.end_line} {finding.category}"
                for finding in review.findings
            )
            artifact = write_text(f"{destination}/report.md", text)
            verify_artifact(artifact)
            return emit(artifact, name="report")

        @recipe(agents=("worker",), budget=Budget(2, 30), output_root=".")
        def review_report(destination: str = "artifacts", working_directory: str = ".") -> Path:
            review = review_workspace(working_directory)
            return persist_report(review, working_directory, destination)

        if __name__ == "__main__":
            raise SystemExit(review_report.cli())
        """
    ).strip()
    + "\n"
)
_GUIDE = 'resource("references/guide.md")'
_RESOURCE_DIRECTORY = _VALID.replace(_GUIDE, 'resource(".")')
_VERIFY = "    verify_artifact(artifact)\n"
_HELPER_MISUSES = {
    "spread": _VALID.replace(_VERIFY, "    verify_artifact(Path(*artifact))\n"),
    "extend": _VALID.replace(
        _VERIFY,
        "    parts: list[str] = []\n"
        "    parts.extend(artifact)\n"
        "    verify_artifact(Path(*parts))\n",
    ),
    "loop": _VALID.replace(
        _VERIFY, "    for part in artifact:\n        verify_artifact(Path(part))\n"
    ),
}
_EXTRA_ARGUMENT = _VALID.replace(
    "    return persist_report(review, working_directory, destination)\n",
    "    return persist_report(review, working_directory, destination, destination)\n",
)
_FINDINGS = {
    "findings": [
        {
            "source": "alpha.py",
            "start_line": 1,
            "end_line": 1,
            "category": "unchecked-copy",
            "violated_invariant": "copies stay within the declared bound",
            "evidence": "value = copy(data)",
        }
    ]
}


class _CountingWorker(StubProvider):
    name = "worker"

    def __init__(self, response: str = "") -> None:
        super().__init__(default=response)
        self.invocations = 0

    def invoke(self, prompt: str, **kwargs: Any) -> AgentResult:
        self.invocations += 1
        return super().invoke(prompt, **kwargs)


class _Package:
    """A skill package with one packaged guide, plus a target workspace."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.skill = root / "plugin" / "skills" / "review" / "SKILL.md"
        (self.skill.parent / "references").mkdir(parents=True)
        self.skill.write_text(_SKILL)
        (self.skill.parent / "references" / "guide.md").write_text("# Guide\nFlag copies.\n")
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "alpha.py").write_text("value = copy(data)\n")
        self.recipe = root / "recipe.py"

    def compile(
        self, *drafts: str, activity: AgentActivity | None = None
    ) -> tuple[ScriptedCompiler, _CountingWorker, Path]:
        compiler = source_sequence(*drafts, repeat_last=True)
        worker = _CountingWorker()
        registry = Registry()
        registry.register(compiler)
        registry.register(worker)
        recipe_path = calcify_file(self.skill, registry, output=self.recipe, activity=activity)
        return compiler, worker, recipe_path

    def run(self, recipe_path: Path, monkeypatch: pytest.MonkeyPatch) -> _CountingWorker:
        inputs = {path: path.read_bytes() for path in self.workspace.rglob("*")}
        worker = _CountingWorker(json.dumps(_FINDINGS))
        registry = Registry()
        registry.register(worker)
        monkeypatch.chdir(self.root)

        run = load_recipe(recipe_path).run(
            {"destination": "out", "working_directory": str(self.workspace)}, registry=registry
        )

        assert run.ok, run.error
        assert (self.root / "out" / "report.md").read_text() == "alpha.py:1-1 unchecked-copy"
        assert {path: path.read_bytes() for path in self.workspace.rglob("*")} == inputs
        return worker


def test_packaged_guide_compiles_runs_and_persists_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _Package(tmp_path)

    compiler, compile_worker, recipe_path = package.compile(_VALID)

    assert len(compiler.source_prompts) == 1
    assert compile_worker.invocations == 0
    assert lint_portable(recipe_path) == []
    assert package.run(recipe_path, monkeypatch).invocations == 1


def _assert_fallback_without_a_worker_call(package: _Package, draft: str, *fragments: str) -> None:
    activity = AgentActivity()
    compiler, worker, recipe_path = package.compile(draft, activity=activity)

    assert len(compiler.source_prompts) == 3
    assert activity.max_model_calls == 5
    assert worker.invocations == 0
    retained = activity.events[-1]
    assert retained["phase"] == "baseline" and retained["status"] == "retained"
    for fragment in fragments:
        assert fragment in str(retained["detail"])
    assert "Calcifier baseline:" in recipe_path.read_text(encoding="utf-8")
    assert lint_portable(recipe_path) == []


def test_packaged_resource_directory_is_rejected_before_any_runtime_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rejected = _Package(tmp_path / "rejected")
    _assert_fallback_without_a_worker_call(
        rejected, _RESOURCE_DIRECTORY, "packaged resource path '.' names a directory"
    )

    repaired = _Package(tmp_path / "repaired")
    compiler, _, recipe_path = repaired.compile(_RESOURCE_DIRECTORY, _VALID)

    assert len(compiler.source_prompts) == 2
    assert '"code": "packaged_resources"' in compiler.source_prompts[1]
    assert repaired.run(recipe_path, monkeypatch).invocations == 1


def test_packaged_file_resolved_under_a_user_directory_is_repaired_to_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _Package(tmp_path)
    draft = _VALID.replace(_GUIDE, '(Path(working_directory) / "references/guide.md")')

    compiler, compile_worker, recipe_path = package.compile(draft, _VALID)

    assert len(compiler.source_prompts) == 2
    repair_prompt = compiler.source_prompts[1]
    assert '"code": "packaged_resources"' in repair_prompt
    assert "resource('references/guide.md')" in repair_prompt
    assert "remove any recipe input for it" in repair_prompt
    assert "ships with the recipe" in repair_prompt
    assert compile_worker.invocations == 0
    assert package.run(recipe_path, monkeypatch).invocations == 1


@pytest.mark.parametrize("misuse", ["spread", "extend", "loop"])
def test_helper_result_misuse_is_rejected_before_any_runtime_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, misuse: str
) -> None:
    draft = _HELPER_MISUSES[misuse]
    assert draft != _VALID
    assert type_check_source(draft) == ()
    rejected = _Package(tmp_path / "rejected")
    _assert_fallback_without_a_worker_call(
        rejected, draft, "dry run failed: TypeError:", "iterable"
    )

    repaired = _Package(tmp_path / "repaired")
    compiler, _, recipe_path = repaired.compile(draft, _VALID)

    assert len(compiler.source_prompts) == 2
    assert "dry run failed: TypeError:" in compiler.source_prompts[1]
    assert repaired.run(recipe_path, monkeypatch).invocations == 1


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        (
            "return persist_report(review, working_directory, destination)",
            "return persist_report(review, working_directory, destination, destination)",
            "ty[too-many-positional-arguments]",
        ),
        (
            _GUIDE,
            'resource("references")',
            "packaged resource path 'references' is not in the skill package manifest",
        ),
        (
            "verify_artifact(artifact)",
            "verify_artifact(artifact())",
            "ty[call-non-callable]",
        ),
    ],
)
def test_portable_lint_refuses_a_recipe_altered_after_compilation(
    tmp_path: Path, old: str, new: str, fragment: str
) -> None:
    package = _Package(tmp_path)
    _, _, recipe_path = package.compile(_VALID)
    source = recipe_path.read_text()
    assert old in source
    recipe_path.write_text(source.replace(old, new))

    assert any(fragment in item.actual for item in lint_portable(recipe_path))


def test_loading_refuses_a_packaged_resource_altered_after_compilation(tmp_path: Path) -> None:
    package = _Package(tmp_path)
    _, _, recipe_path = package.compile(_VALID)
    source = recipe_path.read_text()
    recipe_path.write_text(source.replace(_GUIDE, 'resource("references")'))
    fragment = "packaged resource path 'references' is not in the skill package manifest"

    with pytest.raises(RecipeError, match="violates Calcifier runtime contracts") as caught:
        load_recipe(recipe_path)
    assert fragment in str(caught.value)


def test_a_keyword_resource_directory_is_refused_before_any_worker_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _Package(tmp_path)
    _, _, recipe_path = package.compile(_VALID)
    source = recipe_path.read_text()
    altered = source.replace(_GUIDE, 'resource(rel_path=".")', 1)
    assert altered.count('resource(rel_path=".")') == 1
    recipe_path.write_text(altered)
    worker = _CountingWorker(json.dumps(_FINDINGS))
    registry = Registry()
    registry.register(worker)
    monkeypatch.chdir(package.root)

    with pytest.raises(RecipeError, match="packaged resource path '.' names a directory"):
        load_recipe(recipe_path).run(
            {"destination": "out", "working_directory": str(package.workspace)},
            registry=registry,
        )

    assert worker.invocations == 0
    assert not (package.root / "out").exists()


def test_a_step_called_with_the_wrong_arguments_is_rejected_before_any_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _EXTRA_ARGUMENT != _VALID
    rejected = _Package(tmp_path / "rejected")
    _assert_fallback_without_a_worker_call(
        rejected,
        _EXTRA_ARGUMENT,
        "ty[too-many-positional-arguments]",
        "Too many positional arguments to bound method `StepCallable.__call__`",
    )

    repaired = _Package(tmp_path / "repaired")
    compiler, _, recipe_path = repaired.compile(_EXTRA_ARGUMENT, _VALID)

    assert len(compiler.source_prompts) == 2
    assert "type-correct generated source" in compiler.source_prompts[1]
    assert repaired.run(recipe_path, monkeypatch).invocations == 1


_CRASHES_ON_OUTPUT = _VALID.replace(
    "        for finding in review.findings\n",
    "        for finding in review.findings[1:2] or [review.findings[3]]\n",
)


def test_stub_dry_run_repairs_behavior_after_an_agent_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _CRASHES_ON_OUTPUT != _VALID
    package = _Package(tmp_path)
    activity = AgentActivity()
    compiler, compile_worker, recipe_path = package.compile(
        _CRASHES_ON_OUTPUT, _VALID, activity=activity
    )
    assert len(compiler.source_prompts) == 2
    assert "dry run failed: IndexError: list index out of range" in compiler.source_prompts[1]
    assert compile_worker.invocations == 0
    rendered = render_dry_run(activity)
    assert rendered.startswith("dry run: passed\ndry-run writes: ")
    assert rendered.endswith("/artifacts/report.md")
    assert package.run(recipe_path, monkeypatch).invocations == 1
    assert not list(recipe_path.parent.glob(".calcifier-dry-run-*"))
