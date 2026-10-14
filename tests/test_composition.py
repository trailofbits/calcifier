"""Tests for recipe composition."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import cast

import pytest

from calcifier.composition import SubrecipeError, call_catalog_recipe
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from calcifier.tools import Capability, ToolRegistry
from calcifier.trust import TrustStatus, compute_sha256, trust_path_for
from tests.support.agents import StubProvider
from tests.support.trust import check_trust, record_trust

_MARKER = "# --- calcifier metadata (skill provenance + sha256) ---"


def _with_metadata(tmp_path: Path, body: str) -> str:
    skill = tmp_path / "SKILL.md"
    skill.write_text("# test skill\n")
    return (
        body.rstrip()
        + "\n\n\n"
        + _MARKER
        + f"\n_CALCIFIER_SKILL = {{'path': {str(skill)!r}, 'sha256': {compute_sha256(skill)!r}}}\n"
    )


def _write_recipe(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(_with_metadata(tmp_path, body))
    return path


def _child_recipe(tmp_path: Path) -> Path:
    return _write_recipe(
        tmp_path,
        "child.py",
        """
from calcifier import output_type, recipe, step


@output_type
class Greeting:
    message: str


@step
def build(name: str) -> Greeting:
    return Greeting(message=f"hello {name}")


@recipe(name="child")
def child_recipe(name: str) -> Greeting:
    return build(name=name)
""",
    )


def _parent_recipe(tmp_path: Path, body: str) -> Path:
    return _write_recipe(tmp_path, "parent.py", body)


def _trust_parent(parent: Path, child: Path) -> None:
    record_trust(parent, recorded_by="test", dependencies=((child, compute_sha256(child)),))


def test_subrecipes_share_the_parent_agent_call_budget(tmp_path: Path):
    child = _write_recipe(
        tmp_path,
        "child.py",
        f"""
from calcifier import Budget, agent, recipe, step

@step
def judge() -> bool:
    return agent(
        "worker",
        "judge",
        working_directory={str(tmp_path)!r},
        readable_paths=({str(tmp_path)!r},),
    )

@recipe(name="child", agents=("worker",), budget=Budget(2, 30))
def child_recipe() -> bool:
    return judge()
""",
    )
    parent = _parent_recipe(
        tmp_path,
        """
from pathlib import Path
from calcifier import Budget, call_recipe, recipe

@recipe(agents=("worker",), budget=Budget(1, 30))
def parent(child_path: str) -> bool:
    first = call_recipe(Path(child_path), {}).output
    second = call_recipe(Path(child_path), {}).output
    return bool(first and second)
""",
    )
    _trust_parent(parent, child)
    registry = Registry()
    provider = StubProvider(default="true")
    provider.name = "worker"
    registry.register(provider)

    run = load_recipe(parent).run({"child_path": str(child)}, registry=registry)

    assert not run.ok
    assert "agent-call budget exhausted" in (run.error or "")


def test_call_recipe_two_recipe_workflow(tmp_path: Path):
    child = _child_recipe(tmp_path)
    parent = _parent_recipe(
        tmp_path,
        """
from pathlib import Path

from calcifier import call_recipe, recipe


@recipe
def parent(child_path: str, name: str) -> str:
    result = call_recipe(Path(child_path), {"name": name})
    assert result.child_name == "child"
    return result.output.message
""",
    )
    _trust_parent(parent, child)

    run = load_recipe(parent).run({"child_path": str(child), "name": "Ada"})

    assert run.ok
    assert run.output == "hello Ada"


def test_nested_child_is_authorized_by_verified_parent_metadata(tmp_path: Path):
    grandchild = _write_recipe(
        tmp_path,
        "grandchild.py",
        """
from calcifier import recipe


@recipe
def grandchild_recipe() -> str:
    return "nested"
""",
    )
    child_body = _with_metadata(
        tmp_path,
        """
from calcifier import call_recipe, recipe


@recipe
def child_recipe() -> str:
    return call_recipe("grandchild.py", {}).output
""",
    )
    child_body += (
        "_CALCIFIER_SUBRECIPES = "
        + repr(
            [
                {
                    "name": "grandchild",
                    "path": "grandchild.py",
                    "sha256": compute_sha256(grandchild),
                }
            ]
        )
        + "\n"
    )
    child = tmp_path / "child.py"
    child.write_text(child_body)
    parent = _parent_recipe(
        tmp_path,
        """
from calcifier import call_recipe, recipe


@recipe
def parent() -> str:
    return call_recipe("child.py", {}).output
""",
    )
    _trust_parent(parent, child)

    run = load_recipe(parent).run()

    assert run.ok
    assert run.output == "nested"


def test_catalog_recipe_dispatch_accepts_only_closed_real_names(tmp_path: Path):
    child = _child_recipe(tmp_path)
    parent = _parent_recipe(
        tmp_path,
        """
from calcifier import call_catalog_recipe, recipe


@recipe
def parent(name: str) -> str:
    return call_catalog_recipe(
        name,
        {"analysis-real": "child.py"},
        {"name": "Ada"},
    ).output.message
""",
    )
    _trust_parent(parent, child)

    accepted = load_recipe(parent).run({"name": "analysis-real"})
    rejected = load_recipe(parent).run({"name": "analysis-invented"})

    assert accepted.ok
    assert accepted.output == "hello Ada"
    assert not rejected.ok
    assert "unresolved catalog subrecipe 'analysis-invented'" in (rejected.error or "")


def test_child_uses_its_own_declared_tool_capabilities(tmp_path: Path):
    child = _write_recipe(
        tmp_path,
        "child.py",
        """
from calcifier import recipe, step, tool


@step
def inspect() -> str:
    return str(tool("Read", path="README.md"))


@recipe(capabilities=("Read",))
def child_recipe() -> str:
    return inspect()
""",
    )
    parent = _parent_recipe(
        tmp_path,
        """
from calcifier import call_recipe, recipe


@recipe(capabilities=("Write",))
def parent() -> str:
    return call_recipe("child.py", {}).output
""",
    )
    _trust_parent(parent, child)

    class Provider:
        def available(self, name: str) -> bool:
            return True

        def invoke(self, name: str, /, **kwargs: object) -> str:
            return f"{name}:{kwargs['path']}"

    tools = ToolRegistry()
    tools.register(Capability("Read", "test"), Provider())
    tools.register(Capability("Write", "test"), Provider())

    run = load_recipe(parent).run(tool_registry=tools)

    assert run.ok
    assert run.output == "Read:README.md"


@pytest.mark.parametrize("redirect", [False, True])
def test_child_mutations_inherit_parent_output_root(tmp_path: Path, redirect: bool):
    child = _write_recipe(
        tmp_path,
        "child.py",
        """
from calcifier import recipe, step, write_text


@step
def persist() -> bool:
    write_text("report.txt", "done")
    return True


@recipe(output_root="child-own-root")
def child_recipe() -> bool:
    return persist()
""",
    )
    requested = tmp_path / "requested"
    parent = _parent_recipe(
        tmp_path,
        f"""
from calcifier import call_recipe, recipe


@recipe(output_root={str(requested)!r})
def parent() -> bool:
    call_recipe("child.py", {{}})
    return bool(call_recipe("child.py", {{}}).output)
""",
    )
    _trust_parent(parent, child)

    destination = tmp_path / "redirected" if redirect else requested
    run = load_recipe(parent).run(output_directory=destination if redirect else None)

    assert run.ok
    assert (destination / "report.txt").read_text() == "done"
    if redirect:
        assert not requested.exists()
        repeat = load_recipe(parent).run(output_directory=destination)
        assert not repeat.ok
        assert "refusing to replace an existing file" in (repeat.error or "")
        assert (destination / "report.txt").read_text() == "done"
    assert not (Path.cwd() / "child-own-root" / "report.txt").exists()


def test_parent_output_root_does_not_grant_mutation_to_rootless_child(tmp_path: Path):
    child = _write_recipe(
        tmp_path,
        "child.py",
        """
from calcifier import recipe, step, write_text


@step
def persist() -> bool:
    write_text("report.txt", "done")
    return True


@recipe
def child_recipe() -> bool:
    return persist()
""",
    )
    requested = tmp_path / "requested"
    parent = _parent_recipe(
        tmp_path,
        f"""
from calcifier import call_recipe, recipe


@recipe(output_root={str(requested)!r})
def parent() -> bool:
    return bool(call_recipe("child.py", {{}}).output)
""",
    )
    _trust_parent(parent, child)

    run = load_recipe(parent).run()

    assert not run.ok
    assert "filesystem mutation requires a recipe with output_root" in (run.error or "")
    assert not (requested / "report.txt").exists()


def test_independent_child_recipes_can_run_in_parallel(tmp_path: Path):
    children = []
    for index in range(2):
        child = _write_recipe(
            tmp_path,
            f"child-{index}.py",
            f"""
from calcifier import Budget, agent, recipe, step

@step
def judge_{index}() -> bool:
    return agent(
        "worker",
        "child-{index}",
        working_directory={str(tmp_path)!r},
        readable_paths=({str(tmp_path)!r},),
    )

@recipe(name="child_{index}", agents=("worker",), budget=Budget(2, 30))
def child_recipe_{index}() -> bool:
    return judge_{index}()
""",
        )
        children.append(child)
    parent = _parent_recipe(
        tmp_path,
        """
from calcifier import Budget, call_recipe, parallel, recipe, step

@step
def invoke(paths: tuple[str, ...]) -> bool:
    results = parallel(lambda path: call_recipe(path, {}), list(paths), max_workers=2)
    return all(result.output for result in results)

@recipe(agents=("worker",), budget=Budget(4, 30))
def parent(paths: tuple[str, ...]) -> bool:
    return invoke(paths)
""",
    )
    record_trust(
        parent,
        recorded_by="test",
        dependencies=tuple((child, compute_sha256(child)) for child in children),
    )
    barrier = threading.Barrier(2)

    class BarrierProvider(StubProvider):
        name = "worker"

        def invoke(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            barrier.wait(timeout=2)
            return super().invoke(*args, **kwargs)

    registry = Registry()
    registry.register(BarrierProvider(default="true"))

    run = load_recipe(parent).run(
        {"paths": tuple(str(child) for child in children)},
        registry=registry,
    )

    assert run.ok
    assert len(run.records[0].children) == 2


@pytest.mark.parametrize("redirect", [False, True])
def test_parallel_mutating_children_receive_isolated_output_paths(tmp_path: Path, redirect: bool):
    child = _write_recipe(
        tmp_path,
        "child.py",
        """
from calcifier import recipe, step, write_text


@step
def persist(value: str) -> str:
    write_text("fixed.txt", value)
    return value


@recipe(name="child", output_root=".")
def child_recipe(value: str) -> str:
    return persist(value)
""",
    )
    requested = tmp_path / "requested"
    parent = _parent_recipe(
        tmp_path,
        f"""
from calcifier import call_recipe, parallel, recipe, step


@step
def invoke() -> list[tuple[str, str]]:
    results = parallel(
        lambda value: (
            call_recipe("child.py", {{"value": value + "-one"}}).output,
            call_recipe("child.py", {{"value": value + "-two"}}).output,
        ),
        ["first", "second"],
        max_workers=2,
    )
    return results


@recipe(output_root={str(requested)!r})
def parent() -> list[tuple[str, str]]:
    return invoke()
""",
    )
    _trust_parent(parent, child)

    destination = tmp_path / "redirected" if redirect else requested
    run = load_recipe(parent).run(output_directory=destination if redirect else None)

    first = destination / ".calcifier-parallel/lane-0001/call-0001-child/fixed.txt"
    second = destination / ".calcifier-parallel/lane-0002/call-0001-child/fixed.txt"
    first_repeat = destination / ".calcifier-parallel/lane-0001/call-0002-child/fixed.txt"
    second_repeat = destination / ".calcifier-parallel/lane-0002/call-0002-child/fixed.txt"
    assert run.ok
    if redirect:
        assert not requested.exists()
    assert first.read_text() == "first-one"
    assert second.read_text() == "second-one"
    assert first_repeat.read_text() == "first-two"
    assert second_repeat.read_text() == "second-two"
    assert run.output == [
        ("first-one", "first-two"),
        ("second-one", "second-two"),
    ]
    if redirect:
        repeat = load_recipe(parent).run(output_directory=destination)
        assert not repeat.ok
        assert "refusing to replace an existing file" in str(repeat.error)
        assert first.read_text() == "first-one"
        assert second.read_text() == "second-one"


def test_async_child_recipe_runs_inside_async_parent(tmp_path: Path):
    child = _write_recipe(
        tmp_path,
        "child.py",
        """
import asyncio

from calcifier import recipe, step


@step
async def transform(value: str) -> str:
    await asyncio.sleep(0)
    return value.upper()


@recipe(name="child")
async def child_recipe(value: str) -> str:
    return await transform(value)
""",
    )
    parent = _parent_recipe(
        tmp_path,
        """
import asyncio

from calcifier import call_recipe, recipe


@recipe
async def parent(value: str) -> str:
    await asyncio.sleep(0)
    return call_recipe("child.py", {"value": value}).output
""",
    )
    _trust_parent(parent, child)

    run = load_recipe(parent).run({"value": "nested"})

    assert run.ok
    assert run.output == "NESTED"
    assert run.records[0].name == "transform"


def test_parallel_resolves_async_child_recipes(tmp_path: Path):
    child = _write_recipe(
        tmp_path,
        "child.py",
        """
import asyncio

from calcifier import recipe, step


@step
async def transform(value: str) -> str:
    await asyncio.sleep(0)
    return value.upper()


@recipe(name="child")
async def child_recipe(value: str) -> str:
    return await transform(value)
""",
    )
    parent = _parent_recipe(
        tmp_path,
        """
from calcifier import call_recipe, parallel, recipe, step


@step
def invoke(values: list[str]) -> list[str]:
    return parallel(
        lambda value: call_recipe("child.py", {"value": value}).output,
        values,
        max_workers=2,
    )


@recipe
def parent(values: list[str]) -> list[str]:
    return invoke(values)
""",
    )
    _trust_parent(parent, child)

    run = load_recipe(parent).run({"values": ["one", "two"]})

    assert run.ok
    assert run.output == ["ONE", "TWO"]
    assert [record.output for record in run.records[0].children] == ["ONE", "TWO"]


def test_call_recipe_resolves_relative_child_from_parent_recipe(tmp_path: Path, monkeypatch):
    child = _child_recipe(tmp_path)
    parent = _parent_recipe(
        tmp_path,
        """
from calcifier import call_recipe, recipe


@recipe
def parent(name: str) -> str:
    return call_recipe("child.py", {"name": name}).output.message
""",
    )
    _trust_parent(parent, child)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    run = load_recipe(parent).run({"name": "Ada"})

    assert run.ok
    assert run.output == "hello Ada"


def test_composed_pipeline_passes_prior_documentation_through_planning_to_dispatch(
    tmp_path: Path,
):
    planner = _write_recipe(
        tmp_path,
        "planner.py",
        """
from calcifier import recipe, step


@step
def create_plan(target_context: str, documentation: str) -> str:
    if "generated API reference" not in documentation:
        raise ValueError("prior documentation was not supplied")
    return f"plan for {target_context}: parser"


@recipe(name="planner")
def planner_recipe(target_context: str, documentation: str = "") -> str:
    return create_plan(target_context, documentation)
""",
    )
    inspector = _write_recipe(
        tmp_path,
        "inspector.py",
        """
from calcifier import recipe, step


@step
def inspect(plan: str) -> str:
    if "parser" not in plan:
        raise ValueError("planning did not reach dispatch")
    return "inspection-dispatched"


@recipe(name="inspector")
def inspector_recipe(plan: str) -> str:
    return inspect(plan)
""",
    )
    parent = _parent_recipe(
        tmp_path,
        """
from calcifier import call_recipe, recipe, step


@step
def build_documentation() -> str:
    return "generated API reference"


@step
def create_plan(target_context: str, documentation: str) -> str:
    return call_recipe(
        "planner.py",
        {"target_context": target_context, "documentation": documentation},
    ).output


@step
def dispatch_inspector(plan: str) -> str:
    return call_recipe("inspector.py", {"plan": plan}).output


@recipe(name="pipeline")
def pipeline_recipe(target_context: str) -> str:
    documentation = build_documentation()
    plan = create_plan(target_context, documentation)
    return dispatch_inspector(plan)
""",
    )
    record_trust(
        parent,
        recorded_by="test",
        dependencies=(
            (planner, compute_sha256(planner)),
            (inspector, compute_sha256(inspector)),
        ),
    )

    run = load_recipe(parent).run({"target_context": "repository at revision"})

    assert run.ok
    assert run.output == "inspection-dispatched"


def test_call_recipe_reads_child_bytes_exactly_once(tmp_path: Path, monkeypatch):
    child = _child_recipe(tmp_path)
    parent = _parent_recipe(
        tmp_path,
        """
from pathlib import Path

from calcifier import call_recipe, recipe


@recipe
def parent(child_path: str, name: str) -> str:
    result = call_recipe(Path(child_path), {"name": name})
    return result.output.message
""",
    )
    _trust_parent(parent, child)
    original_read_bytes = Path.read_bytes
    child_reads = 0

    def counting_read_bytes(path: Path) -> bytes:
        nonlocal child_reads
        if path == child:
            child_reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)

    run = load_recipe(parent).run({"child_path": str(child), "name": "Ada"})

    assert run.ok
    assert run.output == "hello Ada"
    assert child_reads == 1


def test_call_recipe_resists_swap_between_check_and_load(tmp_path: Path, monkeypatch):
    child = _write_recipe(
        tmp_path,
        "child.py",
        """
from calcifier import recipe


@recipe(name="child")
def child_recipe(name: str) -> str:
    return f"trusted {name}"
""",
    )
    parent = _parent_recipe(
        tmp_path,
        """
from pathlib import Path

from calcifier import call_recipe, recipe


@recipe
def parent(child_path: str, name: str) -> str:
    result = call_recipe(Path(child_path), {"name": name})
    return result.output
""",
    )
    _trust_parent(parent, child)
    trusted_bytes = child.read_bytes()
    malicious_bytes = _with_metadata(
        tmp_path,
        """
from calcifier import recipe


@recipe(name="child")
def child_recipe(name: str) -> str:
    return f"malicious {name}"
""",
    ).encode()
    original_read_bytes = Path.read_bytes
    child_reads = 0

    def swapping_read_bytes(path: Path) -> bytes:
        nonlocal child_reads
        if path != child:
            return original_read_bytes(path)
        child_reads += 1
        if child_reads == 1:
            child.write_bytes(malicious_bytes)
            return trusted_bytes
        return malicious_bytes

    monkeypatch.setattr(Path, "read_bytes", swapping_read_bytes)

    run = load_recipe(parent).run({"child_path": str(child), "name": "Ada"})

    assert run.ok
    assert run.output == "trusted Ada"
    assert child_reads == 1


def test_call_recipe_child_failure_propagates(tmp_path: Path):
    child = _write_recipe(
        tmp_path,
        "child.py",
        """
from calcifier import fail, recipe


@recipe(name="child")
def child_recipe() -> bool:
    fail("child boom")
""",
    )
    parent = _parent_recipe(
        tmp_path,
        """
from pathlib import Path

from calcifier import call_recipe, recipe


@recipe
def parent(child_path: str) -> bool:
    call_recipe(Path(child_path), {})
    return True
""",
    )
    _trust_parent(parent, child)

    run = load_recipe(parent).run({"child_path": str(child)})

    assert not run.ok
    assert run.error == "subrecipe child failed: child boom"
    assert run.error_type is None


def test_call_recipe_child_crash_stays_an_exception(tmp_path: Path):
    child = _write_recipe(
        tmp_path,
        "child.py",
        """
from calcifier import recipe


@recipe(name="child")
def child_recipe() -> bool:
    return {}["missing"]
""",
    )
    parent = _parent_recipe(
        tmp_path,
        """
from pathlib import Path

from calcifier import call_recipe, recipe


@recipe
def parent(child_path: str) -> bool:
    call_recipe(Path(child_path), {})
    return True
""",
    )
    _trust_parent(parent, child)

    run = load_recipe(parent).run({"child_path": str(child)})

    assert run.error == "SubrecipeCrashError: subrecipe child failed: KeyError: 'missing'"
    assert run.error_type == "SubrecipeCrashError"


@pytest.mark.parametrize(
    ("name", "recipes", "message"),
    [
        ("", {"child": "child.py"}, "non-empty string"),
        (cast("str", 7), {"child": "child.py"}, "non-empty string"),
        ("child", cast("dict[str, str]", {"child": 1}), "string names and paths"),
    ],
)
def test_call_catalog_recipe_rejects_malformed_arguments(
    name: str, recipes: dict[str, str], message: str
):
    with pytest.raises(SubrecipeError, match=message):
        call_catalog_recipe(name, recipes, {})


def test_call_recipe_circular_call_rejected(tmp_path: Path):
    path = _write_recipe(
        tmp_path,
        "cycle.py",
        """
from pathlib import Path

from calcifier import call_recipe, recipe, step


@step
def call_self(path: str) -> bool:
    call_recipe(Path(path), {})
    return True


@recipe(name="cycle")
def cycle_recipe(path: str) -> bool:
    return call_self(path=path)
""",
    )
    record_trust(path, recorded_by="test", dependencies=((path, compute_sha256(path)),))

    run = load_recipe(path).run({"path": str(path)})

    assert not run.ok
    assert "circular subrecipe call detected" in (run.error or "")


def test_call_recipe_drifted_dependency_rejected(tmp_path: Path):
    child = _child_recipe(tmp_path)
    parent = _parent_recipe(
        tmp_path,
        """
from pathlib import Path

from calcifier import call_recipe, recipe


@recipe
def parent(child_path: str) -> bool:
    call_recipe(Path(child_path), {"name": "Ada"})
    return True
""",
    )
    _trust_parent(parent, child)
    child.write_text(child.read_text().replace("hello {name}", "goodbye {name}"))

    run = load_recipe(parent).run({"child_path": str(child)})

    assert not run.ok
    assert "dependency drift" in (run.error or "")


def test_call_recipe_unrecorded_dependency_rejected(tmp_path: Path):
    child = _child_recipe(tmp_path)
    parent = _parent_recipe(
        tmp_path,
        """
from pathlib import Path

from calcifier import call_recipe, recipe


@recipe
def parent(child_path: str) -> bool:
    call_recipe(Path(child_path), {"name": "Ada"})
    return True
""",
    )
    record_trust(parent, recorded_by="test", dependencies=())

    run = load_recipe(parent).run({"child_path": str(child)})

    assert not run.ok
    assert "undeclared dependency" in (run.error or "")


def test_call_recipe_attaches_to_parent_step_record(tmp_path: Path):
    child = _child_recipe(tmp_path)
    parent = _parent_recipe(
        tmp_path,
        """
from pathlib import Path

from calcifier import call_recipe, recipe, step


@step
def invoke(child_path: str, name: str) -> str:
    result = call_recipe(Path(child_path), {"name": name})
    return result.output.message


@recipe
def parent(child_path: str, name: str) -> str:
    return invoke(child_path=child_path, name=name)
""",
    )
    _trust_parent(parent, child)

    run = load_recipe(parent).run({"child_path": str(child), "name": "Ada"})

    parent_step = run.records[0]
    assert run.ok
    assert parent_step.name == "invoke"
    assert parent_step.subrecipe == {"name": "child", "hash": compute_sha256(child)}
    assert [record.name for record in parent_step.children] == ["build"]


def test_trust_record_round_trips_dependencies(tmp_path: Path):
    child = _child_recipe(tmp_path)
    parent = _parent_recipe(
        tmp_path,
        """
from calcifier import recipe


@recipe
def parent() -> bool:
    return True
""",
    )
    child_hash = compute_sha256(child)

    record_trust(parent, recorded_by="test", dependencies=((child, child_hash),))
    result = check_trust(parent)
    payload = json.loads(trust_path_for().read_text())
    entry = payload["entries"][str(parent.resolve())][compute_sha256(parent)]

    assert result.status is TrustStatus.TRUSTED
    assert entry["dependencies"] == [{"path": str(child), "sha256": child_hash}]
    assert result.dependency_drift == ()


def test_explicit_parent_recipe_path_resolves_trusted_child(tmp_path: Path):
    child = _child_recipe(tmp_path)
    parent = _parent_recipe(
        tmp_path,
        """
from calcifier import call_recipe, recipe

@recipe
def parent(trusted_parent: str) -> str:
    return call_recipe(
        "child.py", {"name": "Ada"}, parent_trust=trusted_parent
    ).output.message
""",
    )
    _trust_parent(parent, child)

    run = load_recipe(parent).run({"trusted_parent": str(parent)})

    assert run.ok and run.output == "hello Ada"
