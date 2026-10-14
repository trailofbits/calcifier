"""Tests for the @recipe decorator + Recipe runtime DSL."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from calcifier import Budget, agent, emit, fail, output_type, recipe, step
from calcifier.recipe import (
    Recipe,
    RecipeError,
    RecipeRun,
    load_recipe,
)
from calcifier.registry import Registry
from tests.support.agents import StubProvider

# Module-level dataclasses for forward-reference resolution under
# `from __future__ import annotations`.


@output_type
class _Verdict:
    approved: bool
    reasoning: str


# -- decorator surface --------------------------------------------------------


def test_recipe_decorator_no_arguments():
    @recipe
    def trivial(target: str = "x") -> bool:
        return bool(target)

    assert isinstance(trivial, Recipe)
    assert trivial.name == "trivial"
    assert trivial.version == "0.1"


def test_async_recipe_and_step_are_resolved_with_complete_trace():
    @step
    async def phase(value: str) -> str:
        await asyncio.sleep(0)
        return value.upper()

    @recipe
    async def run(value: str) -> str:
        return await phase(value)

    result = run.run({"value": "ready"})

    assert result.ok
    assert result.output == "READY"
    assert result.records[0].name == "phase"
    assert result.records[0].output == "READY"


@pytest.mark.skipif(os.name == "nt", reason="POSIX SIGTERM semantics")
def test_sigterm_flushes_an_interrupted_trace(tmp_path: Path):
    script = tmp_path / "sigterm_recipe.py"
    trace_path = tmp_path / "trace.json"
    ready_path = tmp_path / "ready"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            import time
            from pathlib import Path
            from calcifier import recipe, step

            @step
            def wait(ready: str) -> bool:
                Path(ready).write_text("ready")
                time.sleep(30)
                return True

            @recipe
            def run(ready: str) -> bool:
                return wait(ready)

            run.run({"ready": sys.argv[2]}, trace_path=Path(sys.argv[1]))
            """
        )
    )
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, str(script), str(trace_path), str(ready_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 5
    while not ready_path.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready_path.exists()

    process.terminate()
    process.communicate(timeout=5)

    trace = json.loads(trace_path.read_text())
    assert trace["state"] == "interrupted"
    assert trace["error"] == "SIGTERM: interrupted"


def test_recipe_decorator_with_version_and_name():
    @recipe(version="1.4", name="renamed")
    def fn(target: str) -> bool:
        return bool(target)

    assert fn.name == "renamed"
    assert fn.version == "1.4"


def test_recipe_decorator_records_explicit_agent_roles():
    @recipe(
        agents=("worker", "verifier"),
        budget=Budget(max_agent_calls=2, max_wall_seconds=30),
    )
    def fn(target: str) -> bool:
        return bool(target)

    assert fn.agent_names == ("worker", "verifier")
    assert fn.agent_names == ("worker", "verifier")


def test_recipe_decorator_rejects_invalid_or_duplicate_agent_roles():
    with pytest.raises(RecipeError, match="invalid declared agent"):

        @recipe(agents=("not a role",), budget=Budget(1, 30))
        def invalid() -> bool:
            return True

    with pytest.raises(RecipeError, match="must be unique"):

        @recipe(agents=("worker", "worker"), budget=Budget(1, 30))
        def duplicate() -> bool:
            return True


def test_recipe_callable_invokes_underlying_function():
    @recipe
    def echo(value: str) -> str:
        return value.upper()

    assert echo(value="hi") == "HI"


# -- Recipe.run --------------------------------------------------------------


def test_run_returns_ok_recipe_run():
    @recipe
    def passthrough(target: str = "x") -> bool:
        return bool(target)

    run = passthrough.run({"target": "hello"})
    assert isinstance(run, RecipeRun)
    assert run.ok
    assert run.output is True
    assert run.error is None
    assert run.name == "passthrough"
    assert run.duration_seconds >= 0.0


def test_run_captures_recipefailure_as_error_not_exception():
    @recipe
    def gated(target: str = "x") -> bool:
        if not target:
            fail("target required")
        return True

    run = gated.run({"target": ""})
    assert not run.ok
    assert run.error == "target required"
    assert run.exception is None
    assert run.output is None


def test_run_captures_arbitrary_exception_with_typename():
    @recipe
    def boom(_: str = "x") -> bool:
        raise ValueError("nope")

    run = boom.run({})
    assert not run.ok
    assert run.error == "ValueError: nope"
    assert run.exception is not None
    assert run.exception.type_name == "ValueError"
    assert run.exception.location is not None
    source, start = inspect.getsourcelines(boom.fn)
    raise_line = next(
        start + offset for offset, text in enumerate(source) if 'raise ValueError("nope")' in text
    )
    assert run.exception.location.line == raise_line
    assert run.exception.location.source == 'raise ValueError("nope")'


def test_run_records_step_frames_in_order():
    @step
    def gather(query: str) -> _Verdict:
        return _Verdict(approved=True, reasoning=query)

    @step
    def judge(reason: str) -> _Verdict:
        return _Verdict(approved=False, reasoning=reason)

    @recipe
    def two_phase(target: str = "x") -> bool:
        a = gather(query=target)
        b = judge(reason=a.reasoning)
        return b.approved is False

    run = two_phase.run({"target": "hello"})
    assert run.ok
    names = [r.name for r in run.records]
    assert names == ["gather", "judge"]


def test_run_records_emitted_artifact_before_boolean_status():
    @step
    def draft_report(target: str) -> str:
        return f"# Report\n\nTarget: {target}"

    @recipe
    def report_skill(target: str = "demo") -> bool:
        report = draft_report(target=target)
        if not report:
            fail("draft_report produced no report")
        emit(report, name="report")
        return True

    run = report_skill.run({"target": "repo"})
    assert run.ok
    assert run.output is True
    assert len(run.artifacts) == 1
    assert run.artifacts[0].name == "report"
    assert run.artifacts[0].value == "# Report\n\nTarget: repo"
    rendered = run.render()
    assert "artifact report:" in rendered
    assert "# Report" in rendered
    assert "output: True" in rendered


def test_emit_outside_active_run_returns_value():
    value = {"report": "ready"}
    assert emit(value, name="report") is value


def test_run_preserves_emitted_artifact_on_failure():
    @recipe
    def questions(target: str = "demo") -> bool:
        emit(["Which repo?", f"Target was {target}"], name="questions")
        fail("need clarification")

    run = questions.run({"target": "repo"})
    assert not run.ok
    assert run.error == "need clarification"
    assert len(run.artifacts) == 1
    assert run.artifacts[0].name == "questions"
    assert run.artifacts[0].value == ["Which repo?", "Target was repo"]
    rendered = run.render()
    assert "artifact questions:" in rendered
    assert "need clarification" in rendered


def test_run_failure_in_step_propagates_recipefailure():
    @step
    def inner(must: bool) -> _Verdict:
        if not must:
            fail("inner failed")
        return _Verdict(approved=True, reasoning="ok")

    @recipe
    def outer(target: str = "x") -> bool:
        inner(must=False)  # raises RecipeFailure
        return True

    run = outer.run({"target": "x"})
    assert not run.ok
    assert run.error == "inner failed"
    # The step's record carries the error.
    assert run.records[0].name == "inner"
    assert run.records[0].error == "inner failed"


# -- Recipe.run with registry / agent calls ----------------------------------


def test_run_threads_registry_to_active_context():
    @step
    def ask(target: str) -> _Verdict:
        return agent(
            "worker",
            "rate {target}",
            working_directory=".",
            readable_paths=(".",),
            target=target,
        )

    @recipe(agents=("worker",), budget=Budget(2, 30))
    def review(target: str) -> bool:
        result = ask(target=target)
        return result.approved

    reg = Registry()
    provider = StubProvider(default=json.dumps({"approved": True, "reasoning": "ok"}))
    provider.name = "worker"
    reg.register(provider)

    run = review.run({"target": "thing"}, registry=reg)
    assert run.ok
    assert run.output is True
    assert run.records[0].name == "ask"


def test_generic_agent_role_is_typed_traced_and_allowlisted():
    @step
    def ask(target: str) -> _Verdict:
        return agent(
            "worker",
            "rate {target}",
            working_directory=".",
            readable_paths=(".",),
            target=target,
        )

    @recipe(agents=("worker",), budget=Budget(2, 30))
    def review(target: str) -> bool:
        return ask(target=target).approved

    reg = Registry(configuration_hash="abc123")
    provider = StubProvider(default=json.dumps({"approved": True, "reasoning": "ok"}))
    provider.name = "worker"
    reg.register(provider)

    run = review.run({"target": "thing"}, registry=reg)

    assert run.ok
    call = run.records[0].agent_calls[0]
    assert call.role == "worker"
    assert call.configuration_hash == "abc123"
    assert call.attempts == 1
    assert call.error is None


def test_generic_agent_role_fails_closed_without_declaration():
    @step
    def ask() -> str:
        return agent("worker", "review", working_directory=".", readable_paths=(".",))

    @recipe
    def review() -> str:
        return ask()

    reg = Registry()
    provider = StubProvider(default='"unexpected"')
    provider.name = "worker"
    reg.register(provider)

    run = review.run(registry=reg)

    assert not run.ok
    assert "agent role 'worker' is not declared" in (run.error or "")
    assert run.records[0].agent_calls == []


def test_agent_trace_error_does_not_include_response_content():
    @step
    def ask() -> str:
        return agent("worker", "check", working_directory=".", readable_paths=(".",))

    @recipe(agents=("worker",), budget=Budget(2, 30))
    def review() -> str:
        return ask()

    reg = Registry()
    provider = StubProvider(default="sensitive unparseable response")
    provider.name = "worker"
    reg.register(provider)

    run = review.run(registry=reg)

    assert not run.ok
    assert "sensitive unparseable response" not in (run.error or "")
    call = run.records[0].agent_calls[0]
    assert call.attempts == 2
    assert call.error == "AgentError"


def test_explicit_agent_allowlist_rejects_undeclared_role():
    @step
    def ask() -> str:
        return agent("verifier", "check", working_directory=".", readable_paths=(".",))

    @recipe(agents=("worker",), budget=Budget(2, 30))
    def review() -> str:
        return ask()

    reg = Registry()
    provider = StubProvider(default='"ok"')
    provider.name = "verifier"
    reg.register(provider)

    run = review.run(registry=reg)

    assert not run.ok
    assert "not declared by the active recipe" in (run.error or "")


# -- Recipe.cli (argparse from signature) ------------------------------------


def test_cli_parses_required_string_input():
    @recipe
    def needs_target(target: str) -> str:
        return target

    rc = needs_target.cli(["--target", "hello"])
    assert rc == 0


def test_cli_returns_nonzero_on_failure():
    @recipe
    def failing(target: str = "x") -> bool:
        fail("nope")

    rc = failing.cli(["--target", "x"])
    assert rc == 1


def test_cli_parses_int_and_bool_flags(capsys):
    @recipe
    def widgets(count: int = 3, dry_run: bool = False) -> dict[str, object]:
        return {"count": count, "dry_run": dry_run}

    rc = widgets.cli(["--count", "7", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "count" in out


def test_recipe_cli_current_input_coercion_known_answer() -> None:
    observed: list[tuple[str, int, bool, list[int]]] = []

    @recipe
    def capture(target: str, count: int, enabled: bool, items: list[int]) -> bool:
        observed.append((target, count, enabled, items))
        return True

    assert capture.cli(["--target", "42", "--count", "7", "--no-enabled", "--items", "[1, 2]"]) == 0
    assert observed == [("42", 7, False, [1, 2])]


@pytest.mark.parametrize("has_budget", [False, True])
def test_recipe_cli_invalid_wall_override_matches_runner(
    has_budget: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    declared = Budget(1, 60) if has_budget else None

    @recipe(budget=declared)
    def plain() -> bool:
        return True

    with pytest.raises(SystemExit) as caught:
        plain.cli(["--max-wall-seconds", "0"])

    assert caught.value.code == 2
    assert capsys.readouterr().err.splitlines()[-1] == (
        "plain: error: --max-wall-seconds must be positive"
    )


@pytest.mark.parametrize("has_budget", [False, True])
def test_recipe_cli_accepts_positive_wall_override(
    has_budget: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    @recipe(budget=Budget(1, 60) if has_budget else None)
    def plain() -> bool:
        return True

    assert plain.cli(["--max-wall-seconds", "1"]) == 0
    assert "output: True" in capsys.readouterr().out.splitlines()


def test_cli_required_param_without_default_errors_when_missing():
    @recipe
    def required(target: str) -> str:
        return target

    with pytest.raises(SystemExit):
        required.cli([])


def test_direct_recipe_cli_requires_v1_config_and_applies_role_timeout(tmp_path: Path, monkeypatch):
    @step
    def model() -> bool:
        return agent(
            "worker",
            "review",
            working_directory=tmp_path,
            readable_paths=(tmp_path,),
        )

    @recipe(agents=("worker",), budget=Budget(2, 30))
    def run() -> bool:
        return model()

    with pytest.raises(SystemExit):
        run.cli([])

    config_path = tmp_path / "agents.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "agents": {
                    "worker": {
                        "target": {"builtin": "claude"},
                        "runtime": {"kind": "host"},
                        "timeout_seconds": 20,
                    }
                },
            }
        )
    )
    registry = Registry()
    provider = StubProvider(default="true")
    provider.name = "worker"
    registry.register(provider)
    configured_timeout = []

    def configured(config, roles):  # noqa: ANN001, ANN202
        configured_timeout.append(config.routes[roles[0]].timeout_seconds)
        return registry

    monkeypatch.setattr(sys.modules["calcifier.recipe"], "configured_registry", configured)

    assert run.cli(["--agent-config", str(config_path), "--agent-timeout", "worker=7"]) == 0
    assert configured_timeout == [7]


@pytest.mark.parametrize("name", ["tool_config", "agent_config", "help"])
def test_a_recipe_input_named_like_a_runner_flag_gets_a_clear_error(
    name: str, tmp_path: Path, capsys
):
    path = tmp_path / "r.py"
    path.write_text(
        f"from calcifier import recipe\n\n\n@recipe\ndef run({name}: str) -> str:\n"
        f"    return {name}\n"
    )

    with pytest.raises(SystemExit) as caught:
        load_recipe(path).cli([])

    assert caught.value.code == 2
    flag = "--" + name.replace("_", "-")
    assert f"recipe input {name!r} collides with the runner flag {flag}" in capsys.readouterr().err


# -- load_recipe -------------------------------------------------------------


def test_load_recipe_finds_single_recipe(tmp_path: Path):
    src = textwrap.dedent(
        """
        from calcifier import recipe

        @recipe(version="0.1")
        def hello(target: str = "x") -> bool:
            return bool(target)
        """
    )
    p = tmp_path / "r.py"
    p.write_text(src)
    obj = load_recipe(p)
    assert isinstance(obj, Recipe)
    assert obj.name == "hello"


def test_load_recipe_infers_skill_root_from_calcifier_directory(tmp_path: Path):
    calcifier_dir = tmp_path / ".calcifier"
    calcifier_dir.mkdir()
    path = calcifier_dir / "recipe.py"
    path.write_text(
        textwrap.dedent(
            """
            from calcifier import recipe

            @recipe
            def hello() -> bool:
                return True
            """
        )
    )

    assert load_recipe(path).skill_root == tmp_path


def test_load_recipe_retains_namespace_for_nested_postponed_dataclass(tmp_path: Path):
    src = textwrap.dedent(
        """
        from __future__ import annotations
        from calcifier import Budget, agent, output_type, recipe, step

        @output_type
        class Item:
            name: str

        @output_type
        class Result:
            items: list[Item]

        @step
        def collect() -> Result:
            return agent(
                "worker", "collect", working_directory=".", readable_paths=(".",)
            )

        @recipe(agents=("worker",), budget=Budget(2, 30))
        def runner() -> bool:
            return collect().items[0].name == "fixture"
        """
    )
    path = tmp_path / "nested.py"
    path.write_text(src)
    reg = Registry()
    provider = StubProvider(default='{"items":[{"name":"fixture"}]}')
    provider.name = "worker"
    reg.register(provider)

    loaded = load_recipe(path)
    run = loaded.run(registry=reg)

    assert loaded.agent_names == ("worker",)
    assert run.ok
    assert run.output is True


def test_load_recipe_uses_only_declared_semantic_roles(tmp_path: Path):
    src = textwrap.dedent(
        """
        from calcifier import Budget, agent, recipe, step

        def _judge() -> str:
            return agent(
                "judge", "judge", working_directory=".", readable_paths=(".",), output=str
            )

        @step
        def vote() -> str:
            return agent("voter", "vote", working_directory=".", readable_paths=(".",))

        @step
        def judged() -> str:
            return _judge()

        @recipe(agents=("judge", "voter"), budget=Budget(4, 30))
        def runner() -> bool:
            return bool(vote() and judged())
        """
    )
    path = tmp_path / "indirect.py"
    path.write_text(src)

    loaded = load_recipe(path)

    assert loaded.agent_names == ("judge", "voter")


def test_load_recipe_does_not_infer_agent_roles(tmp_path: Path):
    src = textwrap.dedent(
        """
        from calcifier import agent, recipe, step

        @step
        def judge() -> str:
            return agent(
                "reviewer", "review", working_directory=".", readable_paths=(".",)
            )

        @recipe
        def runner() -> str:
            return judge()
        """
    )
    path = tmp_path / "aliased.py"
    path.write_text(src)

    assert load_recipe(path).agent_names == ()


def test_load_recipe_ignores_agent_names_in_imported_library_code(tmp_path: Path):
    src = textwrap.dedent(
        """
        from calcifier import emit, fail, recipe, step

        @step
        def plain() -> str:
            return "no agents here"

        @recipe
        def runner() -> bool:
            return plain() == "no agents here"
        """
    )
    path = tmp_path / "agentless.py"
    path.write_text(src)

    loaded = load_recipe(path)

    assert loaded.agent_names == ()


def test_load_recipe_rejects_zero_recipes(tmp_path: Path):
    p = tmp_path / "r.py"
    p.write_text("X = 1\n")
    with pytest.raises(RecipeError, match="no @recipe"):
        load_recipe(p)


def test_load_recipe_rejects_multiple_recipes(tmp_path: Path):
    src = textwrap.dedent(
        """
        from calcifier import recipe

        @recipe
        def first(target: str = "x") -> bool:
            return True

        @recipe
        def second(target: str = "x") -> bool:
            return True
        """
    )
    p = tmp_path / "r.py"
    p.write_text(src)
    with pytest.raises(RecipeError, match="multiple @recipe"):
        load_recipe(p)


def test_load_recipe_rejects_syntax_error(tmp_path: Path):
    p = tmp_path / "r.py"
    p.write_text("def broken(:\n    pass\n")
    with pytest.raises(RecipeError, match="not valid Python"):
        load_recipe(p)


def test_load_recipe_resets_step_registry_between_loads(tmp_path: Path):
    src = textwrap.dedent(
        """
        from calcifier import recipe, step

        @step
        def collect(target: str) -> dict[str, str]:
            return {"target": target}

        @recipe
        def runner(target: str = "x") -> bool:
            return bool(target)
        """
    )
    p = tmp_path / "a.py"
    p.write_text(src)
    load_recipe(p)
    # Loading the same file a second time must succeed (no @step name conflict).
    load_recipe(p)


# -- parallel() context propagation (Codex #4 + concurrency review) -----------


def test_parallel_propagates_run_context_to_threads():
    """``parallel(fn, items)`` must run ``fn`` in worker threads with access
    to the active recipe's agent registry (so ``@step`` and ``agent(...)``
    calls work)."""
    from calcifier import parallel
    from calcifier.trace import current as current_ctx

    @step
    def per_item(item: str) -> _Verdict:
        assert current_ctx() is not None
        return _Verdict(approved=True, reasoning=item)

    @recipe
    def fan_out(items: list[str]) -> bool:
        results = parallel(lambda x: per_item(item=x), items)
        return all(r.approved for r in results)

    run = fan_out.run({"items": ["a", "b", "c", "d"]})
    assert run.ok
    assert run.output is True
    # Each step invocation produces a record. Order is non-deterministic;
    # check by sorted name only.
    names = sorted(r.name for r in run.records)
    assert names == ["per_item"] * 4


def test_parallel_merges_worker_artifacts_into_parent_run():
    from calcifier import parallel

    @step
    def per_item(item: str) -> bool:
        emit(f"finding for {item}", name=f"finding-{item}")
        return True

    @recipe
    def fan_out(items: list[str]) -> bool:
        return all(parallel(lambda x: per_item(item=x), items))

    run = fan_out.run({"items": ["a", "b", "c"]})
    assert run.ok
    assert run.output is True
    assert [artifact.name for artifact in run.artifacts] == [
        "finding-a",
        "finding-b",
        "finding-c",
    ]
    assert [artifact.value for artifact in run.artifacts] == [
        "finding for a",
        "finding for b",
        "finding for c",
    ]


def test_parallel_isolates_worker_run_contexts_under_overlap():
    """Force overlapping work in worker threads. Workers share a deliberate
    barrier so all threads are inside their step bodies simultaneously. The
    parent's record tree must end up with one record per worker, none lost,
    none cross-contaminated.

    Without per-worker ``RunContext`` isolation, frame-list races would
    corrupt the trace (workers popping each other's frames) or trip a
    ``ContextVar`` re-entry assertion.
    """
    import threading

    from calcifier import parallel
    from calcifier.trace import current as current_ctx
    from calcifier.trace import current_frame

    n_workers = 6
    barrier = threading.Barrier(n_workers)

    @step
    def slow_step(item: int) -> _Verdict:
        # Make all workers occupy their @step bodies at the same time.
        ctx = current_ctx()
        frame = current_frame()
        assert ctx is not None
        assert frame is not None and frame.name == "slow_step"
        barrier.wait(timeout=5.0)
        return _Verdict(approved=True, reasoning=str(item))

    @recipe
    def fan_out(items: list[int]) -> int:
        results = parallel(lambda x: slow_step(item=x), items, max_workers=n_workers)
        return sum(1 for r in results if r.approved)

    run = fan_out.run({"items": list(range(n_workers))})
    assert run.ok
    assert run.output == n_workers
    # All worker step records appear exactly once at the top level (no nesting,
    # no duplicates, no losses).
    assert len(run.records) == n_workers
    assert all(r.name == "slow_step" for r in run.records)
    reasons = sorted(r.inputs["item"] for r in run.records)
    assert reasons == list(range(n_workers))


# -- coerce_input (Codex #9 / #10) -------------------------------------------


def test_coerce_input_str_annotation_keeps_string_form():
    from calcifier.recipe import coerce_input

    # The CLI's eager ``_coerce`` would turn "42" into int 42; coerce_input
    # respects the str annotation and re-stringifies.
    assert coerce_input(42, str) == "42"
    assert coerce_input("hello", str) == "hello"


def test_coerce_input_bool_from_string():
    from calcifier.recipe import coerce_input

    assert coerce_input("true", bool) is True
    assert coerce_input("false", bool) is False
    assert coerce_input("yes", bool) is True
    assert coerce_input("no", bool) is False


def test_coerce_input_path_annotation():
    from calcifier.recipe import coerce_input

    out = coerce_input("./relative", Path)
    assert isinstance(out, Path)
    assert str(out) == "relative"


def test_coerce_input_passthrough_when_no_annotation():
    import inspect

    from calcifier.recipe import coerce_input

    assert coerce_input([1, 2, 3], inspect.Parameter.empty) == [1, 2, 3]
    assert coerce_input({"a": 1}, None) == {"a": 1}


def test_coerce_input_excludes_bool_from_int_fast_path():
    """``bool`` is a subclass of ``int``; without an explicit guard,
    ``coerce_input(True, int)`` would return ``True`` unchanged."""
    from calcifier.recipe import coerce_input

    assert coerce_input(True, int) == 1
    assert coerce_input(False, int) == 0
    # A real int is preserved.
    assert coerce_input(42, int) == 42


def test_coerce_input_excludes_bool_from_float_fast_path():
    from calcifier.recipe import coerce_input

    assert coerce_input(True, float) == 1.0
    assert coerce_input(False, float) == 0.0
    assert coerce_input(3.14, float) == 3.14


def test_coerce_input_validates_container_annotation():
    """Container annotations like ``list[int]`` are validated via pydantic
    when possible. JSON-string input is parsed and validated."""
    from calcifier.recipe import coerce_input

    out = coerce_input([1, 2, 3], list[int])
    assert out == [1, 2, 3]


def test_coerce_input_optional_annotation():
    from calcifier.recipe import coerce_input

    assert coerce_input(None, int | None) is None
    assert coerce_input(7, int | None) == 7
