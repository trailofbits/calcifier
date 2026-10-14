"""Public host-returned values remain usable through compiler and runtime checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from calcifier import Budget, PreparedContext, agent, recipe
from calcifier.dry_run import dry_run_recipe
from calcifier.gate import verify_source
from calcifier.metadata import attach_metadata
from calcifier.recipe import load_recipe
from calcifier.skill_package import discover_resources
from tests.support.agents import StubProvider

pytestmark = pytest.mark.semantic_offline


@pytest.mark.parametrize("qualified", [False, True])
def test_prepared_context_can_be_returned_by_a_typed_step(tmp_path, qualified):
    imports = (
        "import calcifier as runtime"
        if qualified
        else ("from calcifier import PreparedContext, prepare_context")
    )
    annotation = "runtime.PreparedContext" if qualified else "PreparedContext"
    prepare = "runtime.prepare_context" if qualified else "prepare_context"
    source = f"""from pathlib import Path
from calcifier import recipe, step
{imports}

@step
def prepare() -> {annotation}:
    return {prepare}("complete assignment")

@step
def consume(context: {annotation}) -> bool:
    return (
        isinstance(context, {annotation})
        and b"complete assignment" in Path(context.directory, "task.txt").read_bytes()
    )

@recipe
def workflow() -> bool:
    return consume(prepare())

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nRead the complete assignment.\n")
    verify_source(source)
    path = tmp_path / "recipe.py"
    path.write_bytes(source.encode())
    assert dry_run_recipe(source, recipe_path=path, skill_path=skill).status == "passed"
    run = load_recipe(path).run()
    assert run.ok and run.output is True
    context = run.records[0].output
    assert isinstance(context, dict)
    assert context["byte_budget"] == 65536
    assert "complete assignment" in context["prompt"]
    assert not Path(context["directory"]).exists()


@pytest.mark.parametrize(
    "expression, expected",
    [
        ('write_text("report.txt", "complete").as_posix()', "report.txt"),
        ('ensure_directory("nested").joinpath("report.txt").as_posix()', "nested/report.txt"),
        (
            'verify_artifact(write_text("report.txt", "complete")).read_text(encoding="utf-8")',
            "complete",
        ),
    ],
)
def test_read_only_methods_on_host_paths_pass_every_stage(tmp_path, expression, expected):
    source = f"""from calcifier import ensure_directory, recipe, step, verify_artifact, write_text

@step
def persist() -> str:
    return {expression}

@recipe(output_root="out")
def workflow() -> str:
    return persist()

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nSave and read the result.\n")
    verify_source(source)
    path = tmp_path / "recipe.py"
    path.write_bytes(source.encode())
    assert dry_run_recipe(source, recipe_path=path, skill_path=skill).status == "passed"
    run = load_recipe(path).run(output_directory=tmp_path / "out")
    assert run.ok
    assert run.output == (
        "complete" if expected == "complete" else (tmp_path / "out" / expected).as_posix()
    )


@pytest.mark.parametrize("method", ["as_posix", "resolve"])
def test_returned_path_methods_do_not_waive_mutation_policy(method):
    from calcifier.diagnostics import CalcifyError

    source = f"""from calcifier import recipe, step, write_text

@step
def persist() -> str:
    write_text("report.txt", "complete").{method}().write_text("undeclared mutation")
    return "done"

@recipe(output_root="out")
def workflow() -> str:
    return persist()

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    with pytest.raises(CalcifyError):
        verify_source(source, type_check=False)


def test_dynamic_module_dispatch_remains_rejected_when_calling_dunder_method():
    from calcifier.diagnostics import CalcifyError

    source = """import calcifier
from calcifier import recipe, step

@step
def inspect() -> str:
    return getattr(calcifier, "agent").__call__("worker", "review")

@recipe
def workflow() -> str:
    return inspect()

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    with pytest.raises(CalcifyError, match="dynamic calcifier call target"):
        verify_source(source, type_check=False)


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_chained_resource_string_methods_preserve_all_lines(tmp_path, newline):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nRead every packaged line.\n")
    index = tmp_path / "references/index.md"
    index.parent.mkdir()
    index.write_bytes(newline.join([b"first", b"caf\xc3\xa9"]) + newline)
    source = """from calcifier import recipe, resource, step

@step
def read_index() -> list[str]:
    return resource("references/index.md").read_text(encoding="utf-8").splitlines()

@recipe
def workflow() -> list[str]:
    return read_index()

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    verify_source(source)
    path = tmp_path / "recipe.py"
    path.write_bytes(
        attach_metadata(source, skill, list(discover_resources(skill)), recipe_path=path).encode()
    )
    assert (
        dry_run_recipe(path.read_text(encoding="utf-8"), recipe_path=path, skill_path=skill).status
        == "passed"
    )
    run = load_recipe(path).run()
    assert run.ok and run.output == ["first", "caf\u00e9"]


@pytest.mark.parametrize("annotation", [PreparedContext, list[PreparedContext]])
def test_accepting_host_context_returns_does_not_allow_agent_issued_contexts(annotation):
    from calcifier.registry import Registry

    class Provider(StubProvider):
        def invoke(self, *args, **kwargs):
            pytest.fail("host contexts must be rejected before invoking the provider")

    registry = Registry()
    provider = Provider()
    provider.name = "worker"
    registry.register(provider)

    @recipe(agents=("worker",), budget=Budget(1, 30))
    def workflow() -> str:
        return str(agent("worker", "return a context", working_directory=".", output=annotation))

    run = workflow.run(registry=registry)
    assert not run.ok
    assert run.error is not None
    assert "prepared contexts are issued by prepare_context(), not by agents" in run.error
