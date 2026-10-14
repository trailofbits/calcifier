"""Compiler failures retain a runnable, source-preserving agent workflow."""

from __future__ import annotations

import ast
import json

import pytest

from calcifier.agent_events import AgentActivity, AgentProvider, AgentResult
from calcifier.artifact import calcify_file, prepare_artifact, render_summary
from calcifier.baseline import is_baseline
from calcifier.compiler_activity import render_compiler_activity
from calcifier.portable import lint_portable
from calcifier.recipe import load_recipe
from calcifier.recipe_ast import (
    collect_structure,
    decorator_keyword_call,
    import_bindings,
    recipe_decorator,
    require_recipe,
)
from calcifier.recipe_draft import draft_response
from calcifier.registry import Registry
from tests.support.compiler import ScriptedCompiler
from tests.support.recipes import _GOOD_RECIPE

SKILL = (
    "---\r\nname: class\r\nallowed-tools: Read\r\n---\r\n"
    "# Report — café\r\n"
    "Read references/style.md before reviewing the target.\r\n"
    "## When applicable\r\nOnly include the appendix when requested.\r\n"
    "## Example (not a task)\r\n```python\r\nprint('example')\r\n```\r\n"
)


class Worker(AgentProvider):
    default_timeout = 30

    def __init__(self, name="worker", *, files=None, blocked_reason=""):
        self.name = name
        self.files = [] if files is None else files
        self.answer = "Résumé complete."
        self.blocked_reason = blocked_reason
        self.prompts = []
        self.references = []
        self.readable_paths = ()

    def invoke(self, prompt, *, timeout=None, filesystem=None, schema=None):
        self.prompts.append(prompt)
        assert filesystem is not None
        self.readable_paths = filesystem.readable_paths
        self.references = [
            p.read_text(encoding="utf-8")
            for root in filesystem.readable_paths
            for p in root.glob("references/style.md")
        ]
        text = json.dumps(
            {
                "answer": self.answer,
                "files": self.files,
                "blocked_reason": self.blocked_reason,
            }
        )
        return AgentResult(text, text, "", 0, self.name)


def setup_skill(tmp_path, worker):
    package = tmp_path / "package"
    package.mkdir()
    skill = package / "SKILL.md"
    skill.write_bytes(SKILL.encode())
    reference = package / "references/style.md"
    reference.parent.mkdir()
    reference.write_text("Preserve each distinct observation.\n", encoding="utf-8")
    registry = Registry()
    compiler = ScriptedCompiler(["invalid compiler response"], repeat_last=True)
    registry.register(compiler)
    registry.register(worker)
    return skill, registry, compiler


def test_failed_compilation_keeps_complete_instructions_and_delivers_real_files(
    tmp_path, monkeypatch
):
    worker = Worker(files=[{"path": "reports/example.md", "content": "Résumé complete.\n"}])
    skill, registry, compiler = setup_skill(tmp_path, worker)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "input.txt"
    target.write_text("Unchanged target.\n", encoding="utf-8")
    monkeypatch.chdir(workspace)
    activity = AgentActivity()
    path = calcify_file(skill, registry, activity=activity)

    assert len(compiler.calls) == 3
    assert lint_portable(path) == []
    assert "compilation: baseline" in render_summary(path)
    assert "baseline retained" in render_compiler_activity(activity)
    assert activity.events[-1]["status"] == "retained"
    run = load_recipe(path).run(
        {"request": "Review the input without an appendix.", "target": str(target)},
        registry=registry,
    )

    assert run.ok, run.error
    assert len(worker.prompts) == 1
    assert SKILL in worker.prompts[0]
    assert "Review the input without an appendix." in worker.prompts[0]
    assert worker.references == ["Preserve each distinct observation.\n"]
    assert (workspace / ".calcifier-output/reports/example.md").read_text(
        encoding="utf-8"
    ) == "Résumé complete.\n"
    assert target.read_text(encoding="utf-8") == "Unchanged target.\n"
    assert skill.read_bytes() == SKILL.encode()


@pytest.mark.parametrize("command", ["exec", "debug"])
def test_baseline_cli_delivers_to_predeclared_destination(tmp_path, monkeypatch, capsys, command):
    from calcifier import cli

    worker = Worker(files=[{"path": "report.md", "content": "Résumé complete.\n"}])
    skill, registry, compiler = setup_skill(tmp_path, worker)
    artifact = prepare_artifact(skill, registry, optimize=False)
    artifact.path.parent.mkdir(parents=True)
    artifact.path.write_text(artifact.source, encoding="utf-8")
    monkeypatch.setattr(cli, "_registry_for_args", lambda *args: registry)
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "target café"
    target.mkdir()
    destination = target / "calcifier-report"
    trace = tmp_path / "trace.json"
    args = [
        command,
        str(artifact.path),
        "--trust",
        "--output-dir",
        str(destination),
        "--input",
        "request=Write report.md",
        "--input",
        f"target={target}",
        "--trace-out",
        str(trace),
    ]
    assert cli.main([*args, "--preflight-only"]) == 0
    assert json.loads(capsys.readouterr().out)["output_directory"] == str(destination.resolve())
    assert not worker.prompts
    invalid = tmp_path / "existing-file"
    invalid.write_text("Preserve me.", encoding="utf-8")
    assert cli.main([*args, "--output-dir", str(invalid / "nested")]) == 2
    assert "not a directory" in capsys.readouterr().err
    assert invalid.read_text(encoding="utf-8") == "Preserve me."
    assert not worker.prompts
    assert cli.main(args) == 0
    assert (destination / "report.md").read_text(encoding="utf-8") == "Résumé complete.\n"
    assert not (tmp_path / ".calcifier-output").exists()
    result = json.loads(trace.read_text(encoding="utf-8"))
    assert result["output_directory"] == str(destination.resolve())
    assert result["output"]["files"] == [str((destination / "report.md").resolve())]
    assert f"output directory: {destination.resolve()}" in capsys.readouterr().out
    assert artifact.path.read_text(encoding="utf-8") == artifact.source
    assert not compiler.calls


@pytest.mark.parametrize("returned_path", [".git/config", "app.py"])
def test_baseline_cannot_overwrite_the_target_or_git_metadata(tmp_path, monkeypatch, returned_path):
    worker = Worker(
        files=[
            {"path": "report.md", "content": "New report."},
            {"path": returned_path, "content": "Untrusted output."},
        ]
    )
    skill, registry, _ = setup_skill(tmp_path, worker)
    artifact = prepare_artifact(skill, registry, optimize=False)
    path = artifact.path
    path.parent.mkdir(parents=True)
    path.write_text(artifact.source, encoding="utf-8")
    target = tmp_path / "target"
    target.mkdir()
    (target / ".git").mkdir()
    config = target / ".git/config"
    config.write_bytes(b"original configuration")
    source = target / "app.py"
    source.write_bytes(b"original source")
    monkeypatch.chdir(tmp_path)

    result = load_recipe(path).run(
        {"request": "Review the project and write report.md", "target": str(target)},
        registry=registry,
        output_directory=target,
    )

    assert not result.ok
    expected_error = (
        ".git paths" if returned_path.startswith(".git") else "refusing to replace an existing file"
    )
    assert expected_error in (result.error or "")
    assert (target / "report.md").read_bytes() == b"New report."
    assert source.read_bytes() == b"original source"
    assert config.read_bytes() == b"original configuration"
    assert len(worker.prompts) == 1


@pytest.mark.parametrize("publish", [False, True])
def test_fallback_uses_a_configured_runtime_role_and_respects_output_override(tmp_path, publish):
    skill, registry, _ = setup_skill(tmp_path, Worker("analyst"))
    target = tmp_path / "generated/recipe.py"
    if publish:
        result = calcify_file(skill, registry, output=target)
        assert result == target
        source = target.read_text(encoding="utf-8")
    else:
        artifact = prepare_artifact(skill, registry, output=target)
        assert artifact.path == target and not target.exists()
        source = artifact.source
    assert 'agents=("analyst",)' in source
    assert "Budget(2, 60)" in source


def _budget(source: str) -> dict[str, int]:
    module = ast.parse(source)
    recipes, _, _ = collect_structure(module)
    decorator = recipe_decorator(require_recipe(module, recipes))
    assert decorator is not None
    budget = decorator_keyword_call(decorator, "budget", import_bindings(module))
    assert budget is not None
    return {"max_agent_calls": budget[0], "max_wall_seconds": budget[1]}


def test_numbered_headings_do_not_multiply_the_worker_budget(tmp_path):
    worker = Worker("analyst")
    worker.default_timeout = 120
    skill, registry, compiler = setup_skill(tmp_path, worker)
    compiler.default_timeout = 10
    budgets = []
    for stages in (1, 8):
        skill.write_bytes("\n".join(f"## Step {n}: Analyze" for n in range(1, stages + 1)).encode())
        artifact = prepare_artifact(skill, registry, timeout=5, optimize=False)
        budgets.append(_budget(artifact.source))
    assert budgets[0] == {"max_agent_calls": 2, "max_wall_seconds": 240}
    assert budgets[1] == budgets[0]
    capped = prepare_artifact(skill, registry, runtime_wall_seconds=600, optimize=False)
    assert _budget(capped.source)["max_wall_seconds"] == 600


@pytest.mark.parametrize("failure", ["blocked", "escaped_output"])
def test_fallback_execution_reports_missing_work_or_failed_delivery(tmp_path, monkeypatch, failure):
    worker = Worker(
        blocked_reason="Required input is unavailable" if failure == "blocked" else "",
        files=[{"path": "../outside.md", "content": "Must not be written"}],
    )
    skill, registry, _ = setup_skill(tmp_path, worker)
    monkeypatch.chdir(tmp_path)
    path = calcify_file(skill, registry)
    run = load_recipe(path).run(
        {"request": "Perform the skill", "target": str(tmp_path)}, registry=registry
    )
    assert not run.ok
    assert run.error is not None
    assert not (tmp_path / "outside.md").exists()
    assert (
        "Required input" if failure == "blocked" else "escapes declared output root"
    ) in run.error


@pytest.mark.parametrize("accepted", [False, True])
def test_baseline_caps_the_entire_optimizer_at_five_calls(tmp_path, accepted):
    from calcifier.recipe_draft import draft_response
    from tests.support.recipes import _GOOD_RECIPE

    skill, registry, _ = setup_skill(tmp_path, Worker())
    invalid = [json.dumps({"version": f"invalid-{i}", "source": _GOOD_RECIPE}) for i in range(5)]
    compiler = ScriptedCompiler(
        invalid[:4] + [draft_response(_GOOD_RECIPE)] if accepted else invalid
    )
    registry.register(compiler)
    activity = AgentActivity()
    artifact = prepare_artifact(skill, registry, activity=activity)
    assert len(compiler.calls) == activity.attempts == activity.max_model_calls == 5
    assert is_baseline(artifact.source) is not accepted
    assert activity.events[-1]["status"] == ("replaced" if accepted else "retained")


@pytest.mark.parametrize("limit,accepted", [(None, False), (6, True), (8, True)])
def test_explicit_budget_can_fund_additional_source_repairs(tmp_path, limit, accepted):
    from calcifier.recipe_draft import draft_response
    from tests.support.recipes import _GOOD_RECIPE

    skill, registry, _ = setup_skill(tmp_path, Worker())
    responses = [json.dumps({"version": f"invalid-{i}", "source": _GOOD_RECIPE}) for i in range(5)]
    compiler = ScriptedCompiler(responses + [draft_response(_GOOD_RECIPE)])
    registry.register(compiler)
    activity = AgentActivity()
    artifact = prepare_artifact(skill, registry, activity=activity, max_compiler_calls=limit)
    assert len(compiler.calls) == activity.attempts == (6 if accepted else 5)
    assert activity.max_model_calls == (5 if limit is None else limit)
    assert is_baseline(artifact.source) is not accepted


@pytest.mark.parametrize("limit", [0, 9, -1, True, 3.5])
def test_invalid_compiler_call_limit_fails_before_dispatch(tmp_path, limit):
    from calcifier.diagnostics import CalcifyError

    skill, registry, compiler = setup_skill(tmp_path, Worker())
    with pytest.raises(CalcifyError, match="max_compiler_calls must be an integer from 1 to 8"):
        prepare_artifact(skill, registry, max_compiler_calls=limit)
    assert compiler.calls == []
    assert not (skill.parent / ".calcifier").exists()


def test_compiler_only_registry_cannot_return_an_unrunnable_fallback(tmp_path):
    from calcifier.diagnostics import CalcifyError

    skill = tmp_path / "SKILL.md"
    skill.write_text("Explain the input.", encoding="utf-8")
    registry = Registry()
    registry.register(ScriptedCompiler(["invalid"], repeat_last=True))
    with pytest.raises(CalcifyError, match="invalid RecipeDraft"):
        prepare_artifact(skill, registry)


@pytest.mark.parametrize("changed", ["skill", "resource"])
def test_input_drift_cannot_publish_a_prepared_baseline(tmp_path, monkeypatch, changed):
    from calcifier.diagnostics import CalcifyError

    skill, registry, _ = setup_skill(tmp_path, Worker())

    def drift(*args, **kwargs):
        target = skill if changed == "skill" else skill.parent / "references/style.md"
        target.write_text("Changed input", encoding="utf-8")
        raise CalcifyError("optimizer failed")

    monkeypatch.setattr("calcifier.artifact.compile_source", drift)
    with pytest.raises(CalcifyError, match="changed"):
        calcify_file(skill, registry)
    assert not (skill.parent / ".calcifier/recipe.py").exists()


@pytest.mark.parametrize("optimized_child", [False, True])
def test_graph_and_catalog_fallbacks_preserve_child_sources_and_execute(
    tmp_path, monkeypatch, optimized_child
):
    from calcifier.catalog import load_skill_catalog
    from calcifier.graph_build import build_graph
    from tests.support.skill_source import SKILL as SOURCE

    worker = Worker(files=[{"path": "report.md", "content": "Report complete."}])
    skill, registry, _ = setup_skill(tmp_path, worker)
    child = tmp_path / "child.md"
    child.write_text("Child original instructions.", encoding="utf-8")
    grandchild = tmp_path / "grandchild.md"
    grandchild.write_bytes(SOURCE.encode("utf-8"))
    compiler = ScriptedCompiler(
        [*([draft_response(_GOOD_RECIPE)] if optimized_child else []), "invalid"],
        repeat_last=True,
    )
    registry.register(compiler)
    spec = tmp_path / "graph.json"
    spec.write_text(
        json.dumps(
            {
                "version": 1,
                "entry": "parent",
                "output_directory": "built",
                "nodes": {
                    "grandchild": {"skill": "grandchild.md", "output": "grandchild.py"},
                    "child": {
                        "skill": "child.md",
                        "output": "child.py",
                        "dependencies": ["grandchild"],
                    },
                    "parent": {
                        "skill": "package/SKILL.md",
                        "output": "parent.py",
                        "dependencies": ["child"],
                        "interface": "Preserve all distinct observations.",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    catalog_path = build_graph(spec, registry)
    catalog = load_skill_catalog(catalog_path)
    assert len(compiler.calls) == (4 if optimized_child else 3)
    for name in ("parent", "child", "grandchild"):
        path = catalog_path.parent / f"{name}.py"
        assert is_baseline(path.read_text(encoding="utf-8"))
        assert lint_portable(path) == []

    # Catalog compilation also retains a runnable baseline, with no graph-only exclusion.
    standalone = prepare_artifact(
        skill, registry, output=tmp_path / "standalone.py", skill_catalog=catalog
    )
    standalone.path.write_text(standalone.source, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for path in (catalog_path.parent / "parent.py", standalone.path):
        run = load_recipe(path).run(
            {"request": "Inspect the target", "target": str(tmp_path)}, registry=registry
        )
        assert run.ok, run.error
        assert SOURCE in worker.prompts[-1] and SKILL in worker.prompts[-1]
        assert "Child original instructions." in worker.prompts[-1]
        assert child in worker.readable_paths and grandchild in worker.readable_paths
    assert "Preserve all distinct observations." in worker.prompts[-2]
    assert (tmp_path / ".calcifier-output/report.md").read_text() == "Report complete."


@pytest.mark.parametrize(
    "procedure",
    [
        "## Step 1: Inspect\nRead inputs.\n## Step 2: Report\nWrite output.",
        "### Phase 0 — Prepare\n### Phase 1 — Analyze\n### Phase 2 — Deliver",
        "## Workflow\n1. Read the document.\n2. Explain it.",
    ],
)
@pytest.mark.parametrize("blocked", [False, True])
def test_numbered_procedures_remain_intact_in_one_worker_session(
    tmp_path, monkeypatch, procedure, blocked
):
    worker = Worker(
        files=[
            {"path": "notes.md", "content": "Intermediate observations."},
            {"path": "report.md", "content": "Final report."},
        ],
        blocked_reason="Required independent review unavailable" if blocked else "",
    )
    skill, registry, _ = setup_skill(tmp_path, worker)
    source = "# Review\nPreserve all observations.\n" + procedure
    skill.write_bytes(source.encode("utf-8"))
    artifact = prepare_artifact(skill, registry, optimize=False)
    artifact.path.parent.mkdir(parents=True)
    artifact.path.write_text(artifact.source, encoding="utf-8")
    assert lint_portable(artifact.path) == []
    monkeypatch.chdir(tmp_path)
    workflow = load_recipe(artifact.path)
    assert workflow.budget is not None and workflow.budget.max_agent_calls == 2
    target = tmp_path / "input.txt"
    target.write_text("Unchanged input.", encoding="utf-8")
    run = workflow.run({"request": "Review the target", "target": str(target)}, registry=registry)
    assert len(worker.prompts) == 1
    assert source in worker.prompts[0]
    assert "Review the target" in worker.prompts[0]
    assert run.ok is not blocked, run.error
    notes = (tmp_path / ".calcifier-output/notes.md").resolve()
    report = (tmp_path / ".calcifier-output/report.md").resolve()
    assert notes.exists() is not blocked
    assert report.exists() is not blocked
    assert target.read_text(encoding="utf-8") == "Unchanged input."
    if not blocked:
        assert run.output is not None
        assert run.output.answer == worker.answer
        assert run.output.files == [str(notes), str(report)]
        assert notes.read_text(encoding="utf-8") == "Intermediate observations."
        assert report.read_text(encoding="utf-8") == "Final report."
