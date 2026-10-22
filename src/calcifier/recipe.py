"""Recipe execution, typed steps, tracing, and signature-derived CLI inputs."""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import contextvars
import inspect
import itertools
import json
import logging
import signal
import sys
import tempfile
import threading
import time
import traceback
import types
from collections.abc import Awaitable, Callable, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    Any,
    NoReturn,
    Protocol,
    cast,
    get_type_hints,
    overload,
)

from . import trace as _trace
from .agent_config import BUILTIN_AGENTS, ROLE_RE, load_agent_config, override_agent_timeouts
from .budget import Budget, BudgetState
from .cli_options import (
    add_agent_flags,
    add_output_flag,
    add_tool_flags,
    resolve_output_directory,
    shorten_budget,
)
from .exceptions import RecipeError, RecipeFailure
from .hashing import sha256_hex
from .metadata import skill_root_from_metadata
from .outputs import OutputTypeError, schema_for
from .registry import configured_registry
from .run_result import ErrorLocation, RecipeRun, RunException
from .runtime_conformance import runtime_conformance_errors
from .skill_package import resolve_resource_source, resource_relative_path
from .tools import (
    ToolConfigError,
    configured_tool_registry,
    render_tool_preflight,
    tool_preflight,
)
from .trace import DryRunMode
from .trace_export import write_trace

logger = logging.getLogger(__name__)


class _GracefulTermination(KeyboardInterrupt):
    def __init__(self, signal_name: str) -> None:
        super().__init__(signal_name)
        self.signal_name = signal_name


@contextlib.contextmanager
def _graceful_termination():
    """Turn catchable SIGTERM into an interruption that can flush trace state."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGTERM"):
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum: int, _frame: types.FrameType | None) -> NoReturn:
        raise _GracefulTermination(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


class _NamedCallable[**P, R](Protocol):
    __name__: str
    __qualname__: str
    __code__: types.CodeType

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


def _resolve_awaitable_sync(value: Any) -> Any:
    """Resolve an awaitable while preserving the active runtime context.

    ``Recipe.run`` is intentionally synchronous. Generated recipes may still
    use ``async def`` entry points and steps, including when a synchronous
    subrecipe call occurs inside an already-running event loop. In that case,
    run the awaitable in one short-lived helper thread because ``asyncio.run``
    cannot nest event loops in the same thread.
    """
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)

    worker_context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(worker_context.run, asyncio.run, value).result()


def parallel[T, R](  # noqa: C901 - worker isolation and serial completion lifecycle
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int | None = None,
    on_result: Callable[[T, R], object] | None = None,
) -> list[R]:
    """Dispatch isolated workers and return results in input order.

    ``on_result(item, result)`` runs serially in the parent recipe context;
    async callbacks are awaited. Failures cancel unstarted workers.
    """
    parent = _trace.current()
    lane = parent.parallel_lane if parent is not None else None
    workers = [
        (
            contextvars.copy_context(),
            _trace.fork_run_context(
                parent, lane=f"{lane}.{index + 1:04d}" if lane else f"lane-{index + 1:04d}"
            ),
            item,
        )
        for index, item in enumerate(items)
    ]

    def invoke(worker: tuple[contextvars.Context, _trace.RunContext, T]) -> R:
        context, runtime, item = worker

        def run() -> R:
            with _trace.activate_context(runtime):
                try:
                    return _resolve_awaitable_sync(fn(item))
                finally:
                    _trace.checkpoint("worker_complete")

        return context.run(run)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            if on_result is None:
                return list(pool.map(invoke, workers))
            remaining = iter(workers)
            futures = {
                pool.submit(invoke, worker): worker[2]
                for worker in itertools.islice(remaining, pool._max_workers)
            }
            pending = set(futures)
            failure: Exception | None = None
            try:
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in sorted(done, key=lambda item: item.exception() is None):
                        pending.remove(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            failure = failure or exc
                            pending = {queued for queued in pending if not queued.cancel()}
                            continue
                        _resolve_awaitable_sync(on_result(futures[future], result))
                    if failure is None:
                        for worker in itertools.islice(remaining, len(done)):
                            future = pool.submit(invoke, worker)
                            futures[future] = worker[2]
                            pending.add(future)
            except BaseException as exc:
                for future in pending:
                    future.cancel()
                if failure is not None:
                    raise exc from failure
                raise
            if failure is not None:
                raise failure
            return [future.result() for future in futures]
    finally:
        _trace.order_worker_records(parent, [runtime for _, runtime, _ in workers])
        _trace.checkpoint("parallel_complete")


@contextlib.contextmanager
def _verified_skill_root(
    root: Path | None,
    resources: tuple[tuple[str, str, str | None], ...],
):
    """Verify and snapshot packaged resources for one recipe run."""
    if root is None or not resources:
        yield root
        return
    with tempfile.TemporaryDirectory(prefix="calcifier-resources-") as temp_dir:
        snapshot_root = Path(temp_dir)
        for rel_path, expected_sha, source_path in resources:
            relative = _validate_resource_rel_path(rel_path)
            source = _resolve_resource_source(root, source_path or rel_path)
            if not source.is_file():
                raise RecipeError(f"packaged resource does not exist: {rel_path!r}")
            data = source.read_bytes()
            actual_sha = sha256_hex(data)
            if actual_sha != expected_sha:
                raise RecipeError(
                    f"packaged resource drift: {rel_path!r} "
                    f"expected {expected_sha}, got {actual_sha}"
                )
            destination = snapshot_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            source_mode = source.stat().st_mode
            destination.chmod(0o500 if source_mode & 0o111 else 0o400)
        yield snapshot_root


def _resolve_resource_source(root: Path, raw_path: str) -> Path:
    try:
        return resolve_resource_source(root, raw_path)
    except ValueError as exc:
        raise RecipeError(str(exc)) from exc


def resource(rel_path: str) -> Path:
    """Resolve a packaged resource path for the active recipe run."""
    context = _trace.current()
    root = None if context is None else context.resource_root
    if root is None:
        raise RecipeError(
            "calcifier.resource() called outside an active recipe run with a skill package"
        )
    relative = _validate_resource_rel_path(rel_path)
    root_resolved = root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise RecipeError(f"resource path escapes skill package: {rel_path!r}")
    if not candidate.is_file():
        raise RecipeError(f"packaged resource does not exist: {rel_path!r}")
    return candidate


def _validate_resource_rel_path(rel_path: str) -> Path:
    try:
        return resource_relative_path(rel_path)
    except ValueError as exc:
        raise RecipeError(str(exc)) from exc


def fail(reason: str) -> NoReturn:
    """Abort the running recipe with ``reason``.

    ``Recipe.run()`` captures this as a failed result with a user-facing message.
    Completion depends on independently observed required checks and complete input
    coverage, not an agent's completed flag. Preserve enumeration provenance: exhaust
    pages or report capped/truncated scope as incomplete. When claiming measured
    effectiveness, require observed execution; distinguish empty inputs, blocked work,
    and unmeasured results from a completed measurement. Apply the source's actual
    requirements without inventing thresholds or replacing qualitative judgments.
    """
    raise RecipeFailure(reason)


def emit[T](value: T, *, name: str = "artifact") -> T:
    """Record a named artifact for the CLI and return ``value``.

    Artifacts are independent of the recipe's return value. Outside a run,
    this only validates the name and returns the value.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("artifact name must be a non-empty string")
    ctx = _trace.current()
    if ctx is not None:
        ctx.artifacts.append(_trace.ArtifactRecord(name=name, value=_trace.redact(value)))
        _trace.checkpoint("artifact")
    return value


# -- @step ---------------------------------------------------------------------


_RECIPE_LOAD_LOCK = threading.Lock()
_OUTPUT_ROOT_UNSET = object()


def _record_step_error(record: _trace.StepRecord | None, error: str) -> None:
    if record is not None:
        record.error = error


def _record_step_output(
    record: _trace.StepRecord | None,
    output: Any,
    redacted_kwargs: tuple[str, ...],
) -> None:
    if record is not None:
        record.output = _trace.redact(output, redacted_kwargs)


@dataclass
class StepCallable[**P, R]:
    """Run a step within a trace frame carrying its output annotation.

    Agent calls use the frame's annotation for structured-output parsing.
    """

    fn: _NamedCallable[P, R]
    name: str
    annotation: Any
    redacted_kwargs: tuple[str, ...] = ()

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if inspect.iscoroutinefunction(self.fn):
            # An async step's declared return type is already its coroutine.
            return cast("R", self._call_async(*args, **kwargs))
        with self._frame(args, kwargs) as record:
            return self._finish(record, lambda: self.fn(*args, **kwargs))

    async def _call_async(self, *args: Any, **kwargs: Any) -> Any:
        with _trace.async_step_context(), self._frame(args, kwargs) as record:
            try:
                output = await cast("Awaitable[Any]", self.fn(*args, **kwargs))
            except BaseException as exc:
                _record_step_error(record, _step_error_text(exc))
                raise
            _record_step_output(record, output, self.redacted_kwargs)
            return output

    def _frame(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        bound = inspect.signature(self.fn).bind(*args, **kwargs)
        bound.apply_defaults()
        return _trace.step_frame(
            self.name,
            self.annotation,
            dict(bound.arguments),
            extra_redacted=self.redacted_kwargs,
        )

    def _finish(self, record: _trace.StepRecord | None, body: Callable[[], R]) -> R:
        try:
            output = body()
        except BaseException as exc:
            _record_step_error(record, _step_error_text(exc))
            raise
        _record_step_output(record, output, self.redacted_kwargs)
        return output


def _step_error_text(exc: BaseException) -> str:
    return str(exc) if isinstance(exc, RecipeFailure) else f"{type(exc).__name__}: {exc}"


def _exception_location(exc: Exception, source_path: Path) -> ErrorLocation | None:
    """Select the last frame in the recipe source that caused the failure."""
    frames = [
        frame
        for frame in traceback.extract_tb(exc.__traceback__)
        if Path(frame.filename).resolve() == source_path.resolve()
    ]
    if not frames:
        return None
    frame = frames[-1]
    if frame.lineno is None:
        return None
    return ErrorLocation(frame.lineno, frame.line)


@overload
def step[**P, R](fn: _NamedCallable[P, R], /) -> StepCallable[P, R]: ...


@overload
def step[**P, R](
    *, redacted_kwargs: tuple[str, ...] = ()
) -> Callable[[_NamedCallable[P, R]], StepCallable[P, R]]: ...


def step(
    fn: _NamedCallable[..., Any] | None = None,
    *,
    redacted_kwargs: tuple[str, ...] = (),
) -> Any:
    """Decorate a function as a calcifier step.

    The function MUST have a return-type annotation that ``outputs.schema_for``
    can derive a JSON schema for. The decoration enforces this at definition
    time so unsupported types raise ``OutputTypeError`` long before the recipe
    runs.
    """

    def decorate(target: _NamedCallable[..., Any]) -> StepCallable[..., Any]:
        annotation = _resolve_return_annotation(target)
        if annotation is None or annotation is inspect.Parameter.empty:
            raise OutputTypeError(f"@step {target.__qualname__} requires a return-type annotation")
        try:
            schema_for(annotation)
        except OutputTypeError as exc:
            raise OutputTypeError(
                f"@step {target.__qualname__}: return annotation {annotation!r} "
                f"not supported: {exc}"
            ) from exc
        return StepCallable(
            fn=target,
            name=target.__name__,
            annotation=annotation,
            redacted_kwargs=tuple(redacted_kwargs),
        )

    if fn is not None:
        return decorate(fn)
    return decorate


def _resolve_return_annotation(fn: _NamedCallable[..., Any]) -> Any:
    try:
        hints = get_type_hints(fn)
    except NameError as exc:
        raise OutputTypeError(
            f"@step {fn.__qualname__}: cannot resolve forward reference: {exc}"
        ) from exc
    return hints.get("return")


# -- @recipe -------------------------------------------------------------------


@dataclass
class Recipe:
    """Wrapper produced by ``@recipe`` decoration.

    Exposes ``run(**inputs)`` for programmatic invocation, ``cli(argv)`` for
    a sysv-friendly entry point, and ``__call__`` for direct calls (handy
    in unit tests).
    """

    fn: _NamedCallable[..., Any]
    name: str
    version: str
    skill_root: Path | None = None
    skill_path: Path | None = None
    skill_sha256: str = ""
    interface_text: str = ""
    recipe_hash: str = ""
    capabilities: tuple[str, ...] = ()
    agent_names: tuple[str, ...] = ()
    resources: tuple[tuple[str, str, str | None], ...] = ()
    budget: Budget | None = None
    output_root: Path | None = None
    composed: bool = False
    source_bytes: bytes | None = None
    _tool_capabilities: tuple[str, ...] | None = None

    @property
    def _provider_capabilities(self) -> tuple[str, ...]:
        return self.capabilities if self._tool_capabilities is None else self._tool_capabilities

    def __call__(self, **inputs: Any) -> Any:
        return self.fn(**inputs)

    def run(
        self,
        inputs: Mapping[str, Any] | None = None,
        *,
        registry: Any | None = None,
        tool_registry: Any | None = None,
        dry_run: DryRunMode = DryRunMode.OFF,
        budget: Budget | None = None,
        trace_path: Path | None = None,
        output_directory: Path | None = None,
    ) -> RecipeRun:
        """Execute the recipe with configured roles, tools, budgets and trace output."""
        return self._execute(
            inputs,
            registry=registry,
            tool_registry=tool_registry,
            dry_run=dry_run,
            budget=budget,
            trace_path=trace_path,
            output_directory=output_directory,
        )

    def _execute(  # noqa: C901 - one branch per budget guard and exception type
        self,
        inputs: Mapping[str, Any] | None = None,
        *,
        registry: Any | None = None,
        tool_registry: Any | None = None,
        dry_run: DryRunMode = DryRunMode.OFF,
        budget: Budget | None = None,
        trace_path: Path | None = None,
        output_directory: Path | None = None,
        _budget_state: BudgetState | None = None,
        _parallel_worker: bool = False,
        _parallel_lane: str | None = None,
        _output_root: Path | None | object = _OUTPUT_ROOT_UNSET,
        _output_writes: _trace.OutputWrites | None = None,
        _hooks: _trace.RunHooks | None = None,
        _child: bool = False,
    ) -> RecipeRun:
        """Execute the recipe under an active ``trace`` run.

        ``registry`` is the configured semantic-role registry used by
        ``agent(...)`` calls inside ``@step`` bodies. ``inputs`` are
        kwargs forwarded to the recipe function (signature-validated).
        ``output_directory`` redirects a declared output root for this run.
        """
        if self.source_bytes is None and getattr(self.fn, "__module__", None) == "__main__":
            self = load_recipe(Path(self.fn.__code__.co_filename))
        kwargs = dict(inputs or {})
        record = RecipeRun(name=self.name, inputs=dict(kwargs), recipe_hash=self.recipe_hash)
        started = time.time()
        try:
            effective_output_root = resolve_output_directory(
                self.output_root
                if _output_root is _OUTPUT_ROOT_UNSET
                else cast("Path | None", _output_root),
                output_directory,
            )
        except (ValueError, OSError) as exc:
            return _failed_run(record, str(exc), trace_path, started)
        if effective_output_root is not None:
            record.output_directory = str(effective_output_root)
        graph = None
        if self.composed and not _child:
            from .graph import inspect_graph

            path = Path(self.fn.__code__.co_filename)
            if _hooks is not None and _hooks.parent_recipe_path is not None:
                path = _hooks.parent_recipe_path(path)
            graph = inspect_graph(path, entry_bytes=self.source_bytes)
        if graph is not None:
            missing_roles = set(graph.agents) - set(registry.known() if registry else ())
            missing_tools = set(graph.provider_capabilities) - set(
                tool_registry.known() if tool_registry else ()
            )
            if missing_roles or missing_tools:
                return _failed_run(
                    record,
                    f"graph preflight missing roles {sorted(missing_roles)} "
                    f"and capabilities {sorted(missing_tools)}",
                    trace_path,
                    started,
                )
        effective_budget = budget or self.budget
        if effective_budget is not None:
            record.budget = {
                "max_agent_calls": effective_budget.max_agent_calls,
                "max_wall_seconds": effective_budget.max_wall_seconds,
                "attempted_agent_calls": 0,
            }
        if (
            registry is not None
            and self.agent_names
            and getattr(registry, "configuration_hash", "")
        ):
            record.agent_runtime = registry.runtime_summary(self.agent_names)
        if (self.agent_names or (graph is not None and graph.agents)) and effective_budget is None:
            return _failed_run(
                record, "agent-backed recipe has no declared Budget", trace_path, started
            )
        if dry_run is DryRunMode.HARD and effective_output_root is not None:
            return _failed_run(
                record,
                f"side effects blocked by hard dry-run: output root {effective_output_root}",
                trace_path,
                started,
            )
        budget_state = (
            _budget_state
            if _budget_state is not None
            else BudgetState(effective_budget)
            if effective_budget is not None
            else None
        )
        checkpoint = _checkpoint_writer(record, trace_path, started)
        with (
            _graceful_termination(),
            _verified_skill_root(self.skill_root, self.resources) as verified_root,
            _trace.active_run(
                registry=registry,
                tool_registry=tool_registry,
                dry_run=dry_run,
                hooks=_hooks,
                recipe_hash=self.recipe_hash,
                allowed_agents=frozenset(self.agent_names),
                allowed_capabilities=frozenset(self.capabilities),
                output_root=effective_output_root,
                output_writes=(
                    _output_writes
                    if _output_writes is not None
                    else _trace.OutputWrites(protect_existing=output_directory is not None)
                ),
                resource_root=verified_root,
                budget_state=budget_state,
                checkpoint=checkpoint,
                parallel_worker=_parallel_worker,
                parallel_lane=_parallel_lane,
            ) as ctx,
        ):
            try:
                from .instruction_context import instruction_context
                from .prepared_context import prepared_context_run

                with (
                    prepared_context_run(),
                    instruction_context(
                        skill_path=self.skill_path,
                        skill_sha256=self.skill_sha256,
                        skill_root=self.skill_root,
                        resources=self.resources,
                        resource_root=verified_root,
                        interface=self.interface_text,
                    ),
                ):
                    body = StepCallable(self.fn, self.name, None)
                    with body._frame((), kwargs) as frame:
                        ctx._recipe_body = frame
                        record.output = body._finish(
                            frame, lambda: _resolve_awaitable_sync(self.fn(**kwargs))
                        )
            except (KeyboardInterrupt, SystemExit) as exc:
                interruption = (
                    exc.signal_name if isinstance(exc, _GracefulTermination) else type(exc).__name__
                )
                record.error = f"{interruption}: interrupted"
                record.state = "interrupted"
                _copy_context_to_run(record, ctx, budget_state)
                _trace.checkpoint("interrupted")
                raise
            except RecipeFailure as exc:
                record.error = str(exc)
            except Exception as exc:  # noqa: BLE001 — recipes get faithful errors
                record.error = f"{type(exc).__name__}: {exc}"
                record.exception = RunException.capture(
                    exc,
                    location=_exception_location(exc, Path(self.fn.__code__.co_filename)),
                )
                # gremlin: pardon[equivalent] exc_info flag affects logger side effect only
                logger.debug("recipe %s raised", self.name, exc_info=True)
            _copy_context_to_run(record, ctx, budget_state)
            if budget_state is not None and record.error is None:
                try:
                    budget_state.check_wall()
                except RecipeFailure as exc:
                    record.error = str(exc)
            record.state = "completed" if record.error is None else "failed"
            _trace.checkpoint("run")
        record.duration_seconds = time.time() - started
        record.agent_preflight = _agent_preflight_report(registry, self.agent_names)
        return record

    def cli(self, argv: list[str] | None = None) -> int:  # noqa: C901 - CLI setup and exit policies
        """Argparse-driven entry point with the same role configuration contract."""
        if self.source_bytes is None and getattr(self.fn, "__module__", None) == "__main__":
            from .exceptions import CalcifierError

            try:
                return load_recipe(Path(self.fn.__code__.co_filename)).cli(argv)
            except CalcifierError as exc:
                print(str(exc), file=sys.stderr)
                return 1
        parser = self._argparser()
        ns = parser.parse_args(argv)
        raw_inputs = {k: v for k, v in vars(ns).items() if not k.startswith("_")}
        try:
            kwargs = coerce_cli_inputs(self, raw_inputs)
        except (ValueError, TypeError) as exc:
            parser.error(f"input coercion failed: {exc}")
        registry = None
        preflight_recipe = self
        if self.composed:
            from .graph import inspect_graph

            graph = inspect_graph(Path(self.fn.__code__.co_filename), entry_bytes=self.source_bytes)
            preflight_recipe = replace(
                self,
                agent_names=graph.agents,
                capabilities=graph.capabilities,
                _tool_capabilities=graph.provider_capabilities,
            )
        if preflight_recipe.agent_names:
            if ns._agent_config is None:
                parser.error("agent-backed recipes require --agent-config")
            configuration = override_agent_timeouts(
                load_agent_config(ns._agent_config), ns._agent_timeout
            )
            registry = configured_registry(configuration, preflight_recipe.agent_names)
        tool_registry = preflight_recipe._cli_tool_registry(parser, ns)
        if tool_registry is None:
            return 2
        try:
            effective_budget = shorten_budget(
                self.budget, ns._max_wall_seconds, ns._max_agent_calls
            )
            output_directory = resolve_output_directory(self.output_root, ns._output_directory)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        run = self.run(
            kwargs,
            registry=registry,
            tool_registry=tool_registry,
            budget=effective_budget,
            output_directory=output_directory,
        )
        print(run.render())
        return 0 if run.ok else 1

    def _cli_tool_registry(
        self, parser: argparse.ArgumentParser, ns: argparse.Namespace
    ) -> Any | None:
        """Build the tool registry ``calcifier exec`` would; None after a failed preflight."""
        try:
            tools = configured_tool_registry(self._provider_capabilities, ns._tool_config)
        except ToolConfigError as exc:
            parser.error(f"tool config failed: {exc}")
        report = tool_preflight(tools, self._provider_capabilities)
        if not report.ok and not ns._skip_tool_preflight:
            print(render_tool_preflight(report), file=sys.stderr)
            return None
        return tools

    def _argparser(self) -> argparse.ArgumentParser:
        sig = inspect.signature(self.fn)
        try:
            hints = get_type_hints(self.fn)
        except NameError:
            hints = {}
        parser = argparse.ArgumentParser(prog=self.name)
        add_agent_flags(parser, internal=True)
        add_tool_flags(parser, internal=True)
        add_output_flag(parser, internal=True)
        for param_name, param in sig.parameters.items():
            flag = f"--{param_name.replace('_', '-')}"
            if flag in parser._option_string_actions:
                parser.error(
                    f"recipe input {param_name!r} collides with the runner flag {flag}; "
                    "rename the input"
                )
            kw: dict[str, Any] = {"dest": param_name}
            ann = hints.get(param_name)
            if ann is bool:
                kw["action"] = argparse.BooleanOptionalAction
            else:
                kw["type"] = str
            if param.default is inspect.Parameter.empty:
                kw["required"] = True
            else:
                kw["default"] = param.default
            parser.add_argument(flag, **kw)
        return parser


def _coerce_for(annotation: Any) -> Callable[[str], Any]:
    """Coerce scalar CLI flags directly; parse other annotations as JSON."""
    if annotation in (str, inspect.Parameter.empty):
        return str
    if annotation is int:
        return int
    if annotation is float:
        return float
    if annotation is Path:
        return Path
    return _json_coerce


def _agent_preflight_report(
    registry: Any | None,
    names: tuple[str, ...],
) -> _trace.AgentPreflightReport | None:
    if registry is None:
        return None
    entries = []
    for name in names:
        if name not in registry.known():
            entries.append(
                _trace.AgentPreflightEntry(
                    role=name,
                    target_type="builtin" if name in BUILTIN_AGENTS else "command",
                    target_name=name,
                    runtime={"kind": "host"},
                    available=False,
                    remediation="configure this declared agent role",
                )
            )
            continue
        provider = registry.get(name)
        metadata = provider.trace_metadata()
        entries.append(
            _trace.AgentPreflightEntry(
                role=name,
                target_type=metadata["target_type"],
                target_name=metadata["target_name"],
                runtime=metadata["runtime"],
                available=provider.is_available(),
                remediation=provider.remediation,
            )
        )
    return _trace.AgentPreflightReport(agents=tuple(entries))


_CHECKPOINT_INTERVAL_SECONDS = 0.25
# Routine progress events coalesce into at most one snapshot per interval.
# "attempt" is deliberately absent: it fires just before a provider call blocks,
# so the snapshot on disk during every long wait already holds completed steps;
# only the in-flight call's usage/model telemetry can be lost to a crash.
_THROTTLED_CHECKPOINT_EVENTS = frozenset(
    {"step", "prompt", "worker_complete", "agent", "usage", "model"}
)


def _checkpoint_due(event: str, since_last: float) -> bool:
    """Throttle routine snapshots; always flush effects, artifacts, and failures.

    Each snapshot serializes the whole trace. The terminal trace is separate.
    """
    if event in _THROTTLED_CHECKPOINT_EVENTS:
        return since_last >= _CHECKPOINT_INTERVAL_SECONDS
    return True


def _checkpoint_writer(
    record: RecipeRun,
    trace_path: Path | None,
    started: float,
) -> Callable[[str, _trace.RunContext], None] | None:
    if trace_path is None:
        return None
    lock = threading.Lock()
    last_snapshot = float("-inf")

    def checkpoint(event: str, ctx: _trace.RunContext) -> None:
        nonlocal last_snapshot
        with lock:
            now = time.monotonic()
            if not _checkpoint_due(event, now - last_snapshot):
                return
            last_snapshot = now
            record.duration_seconds = time.time() - started
            _copy_context_to_run(record, ctx, ctx.budget_state)
            try:
                write_trace(record, trace_path, checkpoint=True)
            except OSError as exc:
                logger.warning("incremental trace checkpoint failed: %s", exc)

    return checkpoint


def _copy_context_to_run(
    record: RecipeRun,
    ctx: _trace.RunContext,
    budget_state: BudgetState | None,
) -> None:
    record.records = list(ctx.records)
    record.artifacts = list(ctx.artifacts)
    record.agent_transport = list(ctx.agent_transport)
    record.agent_calls = list(ctx.agent_calls)
    # Preserve flat traces, including checkpoints, for step-only recipes.
    body = ctx._recipe_body
    if body is not None and not (body.side_effects or body.tool_calls):
        record.records = list(body.children)
        record.agent_calls.extend(body.agent_calls)
    if budget_state is not None and record.budget is not None:
        record.budget["attempted_agent_calls"] = budget_state.attempted_calls


def _write_terminal_trace(record: RecipeRun, trace_path: Path | None, started: float) -> None:
    record.duration_seconds = time.time() - started
    if trace_path is None:
        return
    write_trace(record, trace_path)


def _failed_run(
    record: RecipeRun, error: str, trace_path: Path | None, started: float
) -> RecipeRun:
    record.error = error
    record.state = "failed"
    _write_terminal_trace(record, trace_path, started)
    return record


def _json_coerce(text: str) -> Any:
    """Parse ``text`` as JSON; fall back to ``str`` if it isn't JSON."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


_TRUTHY = frozenset({"true", "1", "yes", "on"})


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return bool(value)


def _to_str(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return value if isinstance(value, str) else str(value)


_PRIMITIVE_COERCERS: dict[Any, Callable[[Any], Any]] = {
    bool: lambda v: _to_bool(v),
    str: lambda v: _to_str(v),
    int: lambda v: v if isinstance(v, int) and not isinstance(v, bool) else int(v),
    float: lambda v: float(v),
    Path: lambda v: v if isinstance(v, Path) else Path(str(v)),
}


def coerce_input(value: Any, annotation: Any) -> Any:
    """Apply a recipe parameter's annotation to an input from either CLI.

    Numeric coercers exclude bool, which Python otherwise treats as an int.
    """
    if annotation is inspect.Parameter.empty or annotation is None:
        return value
    coercer = _PRIMITIVE_COERCERS.get(annotation)
    if coercer is not None:
        return coercer(value)
    return _coerce_via_pydantic(value, annotation)


def coerce_cli_inputs(recipe_obj: Recipe, raw: dict[str, Any]) -> dict[str, Any]:
    """Apply recipe annotations to inputs from either command-line syntax."""
    try:
        hints = get_type_hints(recipe_obj.fn)
    except NameError:
        hints = {}
    parameters = inspect.signature(recipe_obj.fn).parameters
    return {
        name: coerce_input(
            _coerce_for(hints.get(name, inspect.Parameter.empty))(value)
            if isinstance(value, str)
            else value,
            hints.get(name, inspect.Parameter.empty),
        )
        if name in parameters
        else value
        for name, value in raw.items()
    }


def _coerce_via_pydantic(value: Any, annotation: Any) -> Any:
    """Validate containers, unions, and literals through Pydantic.

    Leave values unchanged when Pydantic cannot build the input schema.
    """
    from pydantic import TypeAdapter
    from pydantic.errors import PydanticSchemaGenerationError, PydanticUserError

    try:
        adapter = TypeAdapter(annotation)
    except (PydanticSchemaGenerationError, PydanticUserError, TypeError):
        return value
    return adapter.validate_python(value)


@overload
def recipe(fn: _NamedCallable[..., Any], /) -> Recipe: ...


@overload
def recipe(
    *,
    version: str = "0.1",
    name: str | None = None,
    capabilities: tuple[str, ...] = (),
    agents: tuple[str, ...] | None = None,
    budget: Budget | None = None,
    output_root: str | Path | None = None,
) -> Callable[[_NamedCallable[..., Any]], Recipe]: ...


def recipe(  # noqa: C901 - validates each declared-agent-role constraint independently
    fn: _NamedCallable[..., Any] | None = None,
    *,
    version: str = "0.1",
    name: str | None = None,
    capabilities: tuple[str, ...] = (),
    agents: tuple[str, ...] | None = None,
    budget: Budget | None = None,
    output_root: str | Path | None = None,
) -> Any:
    """Declare a recipe's roles, budget, capabilities, and output root.

    The function's parameters become typed CLI inputs. ``name`` defaults to
    its function name; ``version`` records the recipe's declared version.
    """

    def decorate(target: _NamedCallable[..., Any]) -> Recipe:
        declared_agents = None if agents is None else tuple(agents)
        if declared_agents is not None:
            invalid = [
                role
                for role in declared_agents
                if not isinstance(role, str) or ROLE_RE.fullmatch(role) is None
            ]
            if invalid:
                raise RecipeError(f"invalid declared agent roles: {invalid!r}")
            if len(set(declared_agents)) != len(declared_agents):
                raise RecipeError("declared agent roles must be unique")
            provider_roles = sorted(set(declared_agents) & BUILTIN_AGENTS)
            if provider_roles:
                raise RecipeError(f"provider names cannot be semantic roles: {provider_roles!r}")
            if declared_agents and budget is None:
                raise RecipeError("agent-backed @recipe requires budget=Budget(...)")
        resolved_output_root = None
        if output_root is not None:
            root_path = Path(output_root)
            if not root_path.is_absolute():
                root_path = Path.cwd() / root_path
            resolved_output_root = root_path.resolve()
        return Recipe(
            fn=target,
            name=name or target.__name__,
            version=version,
            capabilities=tuple(capabilities),
            agent_names=declared_agents or (),
            budget=budget,
            output_root=resolved_output_root,
        )

    if fn is not None:
        return decorate(fn)
    return decorate


# -- recipe loading ------------------------------------------------------------


@dataclass(frozen=True)
class RecipeSnapshot:
    path: Path
    data: bytes
    sha256: str


def read_recipe_snapshot(path: str | Path) -> RecipeSnapshot:
    recipe_path = Path(path).resolve()
    data = recipe_path.read_bytes()
    return RecipeSnapshot(recipe_path, data, sha256_hex(data))


def load_recipe(path: str | Path) -> Recipe:
    """Import a recipe file, find its ``@recipe``-decorated entry, return it.

    A recipe file must
    define exactly one module-level ``Recipe`` instance (i.e., one
    ``@recipe``-decorated function). Multiple recipes in one file are not
    supported; split separate entry points into separate files.
    """
    return load_recipe_snapshot(read_recipe_snapshot(path))


def load_recipe_snapshot(snapshot: RecipeSnapshot) -> Recipe:
    """Execute exactly the bytes held by ``snapshot`` and return its recipe."""
    p = snapshot.path
    # Loading temporarily installs the module in sys.modules; serialize that
    # short phase so independent call_recipe() workers cannot race.
    with _RECIPE_LOAD_LOCK:
        module_name = _loaded_recipe_module_name(p, snapshot.sha256)
        namespace = _execute_snapshot(snapshot, module_name)

    candidates = [v for v in namespace.values() if isinstance(v, Recipe)]
    if not candidates:
        raise RecipeError(
            f"Recipe {p} defines no @recipe-decorated function "
            "(import calcifier.recipe and decorate one with @recipe)."
        )
    if len(candidates) > 1:
        names = sorted(c.name for c in candidates)
        raise RecipeError(
            f"Recipe {p} defines multiple @recipe-decorated functions {names}; "
            "recipes support exactly one entry point per file."
        )
    found = candidates[0]
    found.skill_root = _skill_root_from_namespace(namespace, p)
    found.recipe_hash = snapshot.sha256
    found.source_bytes = snapshot.data
    from .recipe_ast import import_bindings, subrecipe_calls, tool_provider_capabilities

    module = ast.parse(snapshot.data)
    found._tool_capabilities = tool_provider_capabilities(module, found.capabilities)
    found.composed = bool(subrecipe_calls(module, import_bindings(module)))
    found.resources = _resources_from_namespace(namespace, found.skill_root)
    _require_runtime_conformance(snapshot, namespace, found.resources)
    from .metadata import interface_text, module_metadata

    try:
        metadata = module_metadata(module, unique=True)
        found.interface_text = interface_text(metadata)
    except ValueError as exc:
        raise RecipeError(f"Recipe {p} has invalid provenance metadata: {exc}") from exc
    skill = metadata.get("_CALCIFIER_SKILL", {})
    if not isinstance(skill, dict):
        raise RecipeError("_CALCIFIER_SKILL metadata must be an object")
    if isinstance(skill.get("path"), str):
        digest = skill.get("sha256")
        if not isinstance(digest, str) or not digest:
            raise RecipeError("_CALCIFIER_SKILL provenance requires a nonempty sha256")
        found.skill_path = p.parent / skill["path"]
        found.skill_sha256 = digest
    return found


def _require_runtime_conformance(
    snapshot: RecipeSnapshot,
    namespace: dict[str, Any],
    resources: tuple[tuple[str, str, str | None], ...],
) -> None:
    """Refuse known calls that violate the native recipe admission requirements.

    Runs before any step, so a recipe that bypassed the compiler's acceptance
    gate still fails before its first model call.
    """
    module = ast.parse(snapshot.data.decode("utf-8"), filename=str(snapshot.path))
    packaged = (
        frozenset(path for path, _, _ in resources) if "_CALCIFIER_RESOURCES" in namespace else None
    )
    errors = runtime_conformance_errors(module, packaged)
    if errors:
        rendered = "; ".join(f"line {item.line}: {item.actual}" for item in errors)
        raise RecipeError(
            f"Recipe {snapshot.path} violates Calcifier runtime contracts: {rendered}"
        )


def _execute_snapshot(snapshot: RecipeSnapshot, module_name: str) -> dict[str, Any]:
    p = snapshot.path
    previous_module = sys.modules.get(module_name)
    try:
        source = snapshot.data.decode("utf-8")
        code = compile(source, str(p), "exec")
        module = types.ModuleType(module_name)
        namespace = module.__dict__
        namespace.update(
            {
                "__file__": str(p),
                "__cached__": None,
                "__package__": "",
            }
        )
        sys.modules[module_name] = module
        exec(code, namespace)  # noqa: S102 - exact bytes passed through the trust boundary.
    except UnicodeDecodeError as exc:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        raise RecipeError(f"Recipe {p} is not valid UTF-8: {exc}") from exc
    except SyntaxError as exc:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        raise RecipeError(f"Recipe {p} is not valid Python: {exc}") from exc
    except Exception as exc:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        raise RecipeError(f"Recipe {p} failed while loading: {type(exc).__name__}: {exc}") from exc

    return namespace


def _loaded_recipe_module_name(path: Path, recipe_hash: str) -> str:
    path_hash = sha256_hex(str(path.resolve()).encode())
    return f"_calcifier_recipe_{path_hash[:16]}_{recipe_hash[:16]}"


def _skill_root_from_namespace(namespace: dict[str, Any], recipe_path: Path) -> Path | None:
    return skill_root_from_metadata(recipe_path, namespace.get("_CALCIFIER_SKILL"))


def _resources_from_namespace(
    namespace: dict[str, Any],
    skill_root: Path | None,
) -> tuple[tuple[str, str, str | None], ...]:
    raw = namespace.get("_CALCIFIER_RESOURCES", [])
    if not isinstance(raw, list):
        raise RecipeError("_CALCIFIER_RESOURCES metadata must be a list")
    resources: list[tuple[str, str, str | None]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RecipeError("_CALCIFIER_RESOURCES entries must be objects")
        path = item.get("path")
        sha256 = item.get("sha256")
        source = item.get("source")
        if not isinstance(path, str) or not isinstance(sha256, str):
            raise RecipeError("_CALCIFIER_RESOURCES entries require path and sha256")
        if source is not None and not isinstance(source, str):
            raise RecipeError("_CALCIFIER_RESOURCES source must be a string or null")
        _validate_resource_rel_path(path)
        if source is not None:
            if skill_root is None:
                raise RecipeError(
                    "_CALCIFIER_RESOURCES source requires resolvable _CALCIFIER_SKILL metadata"
                )
            _resolve_resource_source(skill_root, source)
        resources.append((path, sha256, source))
    return tuple(resources)
