"""Deterministic, declared filesystem mutations for generated recipes."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from . import trace as _trace
from .atomic_io import atomic_write as _atomic_write
from .exceptions import RecipeError
from .hashing import sha256_hex
from .outputs import _adapter, output_type
from .path_syntax import PathSyntax
from .side_effects import record_side_effect
from .subprocesses import ProcessError, ProcessStartError, ProcessTimeoutError, run_process
from .trace import DryRunMode
from .validation import bounded_timeout

if TYPE_CHECKING:
    from pydantic import TypeAdapter
    from typing_extensions import TypeForm

    from .budget import BudgetState


@output_type(frozen=True)
class CommandResult:
    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str


def _output_path(  # noqa: C901 - write boundaries plus compiler scratch redirection
    path: str | Path, *, writing: bool = True, redirect_scratch: bool = True
) -> Path:
    ctx = _trace.current()
    if ctx is None or ctx.output_root is None:
        raise RecipeError("filesystem mutation requires a recipe with output_root=...")
    if not ctx.frames:
        raise RecipeError("filesystem mutation requires an active @step")
    if ctx.parallel_worker:
        raise RecipeError("parallel workers cannot mutate shared state")
    root = ctx.output_root.resolve()
    syntax = PathSyntax.inspect(path)
    candidate = syntax.native
    candidate = candidate if candidate.is_absolute() else root / candidate
    resolved = candidate.resolve(strict=False)
    if writing:
        parts = (*syntax.windows.parts, *candidate.parts, *resolved.parts)
        if any(part.casefold() == ".git" for part in parts):
            raise RecipeError(f"artifact writes cannot modify .git paths: {path!s}")
        if syntax.has_windows_reserved_name:
            raise RecipeError(f"artifact path contains a Windows reserved name: {path!s}")
    if not resolved.is_relative_to(root):
        raise RecipeError(f"artifact path escapes declared output root: {path!s}")
    scratch = ctx.hooks.scratch_output_root
    if (
        redirect_scratch
        and ctx.dry_run is DryRunMode.SOFT
        and scratch is not None
        and not root.is_relative_to(scratch)
    ):
        resolved = scratch.joinpath("absolute-output", *root.parts[1:], resolved.relative_to(root))
    return resolved


def ensure_directory(
    path: str | Path = ".",
    *,
    output_files: Iterable[str | Path] = (),
    exclude_roots: Iterable[str | Path] = (),
    overwrite: bool = False,
) -> Path:
    """Create a directory inside the recipe output root and return its path.

    Requires an active recipe step (the recipe body is also traced as a step).
    Parallel workers cannot mutate shared state; dry-run policy applies.
    Before any mkdir or subprocess, pass the complete planned ``output_files`` set,
    relative to this directory, including intermediates. Preflight confines all paths,
    rejects symlinked files and collisions, and excludes resolved ``exclude_roots``
    such as an audited project. ``overwrite=True`` permits existing regular files only
    when the run's output policy permits replacement; --output-dir protection still applies.
    This validates destinations before external writes; it does not sandbox the subprocess.
    """
    target = _output_path(path)
    _preflight_outputs(path, tuple(output_files), tuple(exclude_roots), overwrite=overwrite)
    ctx = cast("_trace.RunContext", _trace.current())
    effect = record_side_effect(
        "file_write",
        str(target),
        payload={"operation": "mkdir"},
        pending=True,
    )
    if _suppress_output(ctx, target):
        return target
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        effect.completed = False
        effect.error = str(exc)
        raise RecipeError(f"failed to create artifact directory {target}: {exc}") from exc
    effect.completed = ctx.dry_run is DryRunMode.OFF
    _trace.checkpoint("side-effect-complete")
    return target


def _preflight_outputs(
    path: str | Path,
    output_files: tuple[str | Path, ...],
    exclude_roots: tuple[str | Path, ...],
    *,
    overwrite: bool,
) -> None:
    if not output_files and not exclude_roots:
        return
    directory = _output_path(path, redirect_scratch=False)
    excluded = tuple(Path(root).resolve() for root in exclude_roots)
    candidates = tuple(directory / PathSyntax.inspect(file).native for file in output_files)
    resolved = tuple(_output_path(file, redirect_scratch=False) for file in candidates)
    if any(target.is_relative_to(root) for target in (directory, *resolved) for root in excluded):
        raise RecipeError("planned output is inside an excluded root")
    if directory.exists() and not directory.is_dir():
        raise RecipeError(f"artifact directory is not a directory: {directory}")
    for candidate, target in zip(candidates, resolved, strict=True):
        if target == directory or not target.is_relative_to(directory):
            raise RecipeError(f"planned output escapes the selected directory: {candidate}")
        _preflight_output_file(candidate, overwrite=overwrite)
    if any(a != b and b.is_relative_to(a) for a in resolved for b in resolved):
        raise RecipeError("a planned output file is also another output's parent directory")


def _preflight_output_file(path: Path, *, overwrite: bool) -> None:
    ctx = cast("_trace.RunContext", _trace.current())
    if path.is_symlink():
        raise RecipeError(f"planned output is a symlink: {path}")
    # Compiler scratch redirection must not hide a collision at the actual destination.
    for target in {_output_path(path, redirect_scratch=False), _output_path(path)}:
        if target.exists():
            if not target.is_file():
                raise RecipeError(f"planned output is not a regular file: {path}")
            protected = (
                ctx.output_writes.protect_existing and target not in ctx.output_writes.created
            )
            if not overwrite or protected:
                raise RecipeError(f"refusing to replace an existing planned output: {path}")
        if any(parent.exists() and not parent.is_dir() for parent in target.parents):
            raise RecipeError(f"planned output has a non-directory parent: {path}")


def _suppress_output(ctx: _trace.RunContext, target: Path) -> bool:
    if ctx.dry_run is not DryRunMode.SOFT:
        return False
    scratch = ctx.hooks.scratch_output_root
    if scratch is None:
        return True
    if not target.is_relative_to(scratch):
        raise RecipeError(f"compiler dry-run output escapes scratch storage: {target}")
    return False


def write_text(path: str | Path, text: str, *, reuse_existing: bool = False) -> Path:
    """Atomically write UTF-8 text inside the declared output root.

    Requires an active recipe step, including the traced recipe body. Parallel
    workers cannot write shared files. With --output-dir, existing files are
    protected; reuse_existing=True accepts only byte-identical contents.
    """
    if not isinstance(text, str):
        raise RecipeError("write_text() requires a string")
    target = _output_path(path)
    ctx = cast("_trace.RunContext", _trace.current())  # _output_path requires an active run.
    with ctx.output_writes.lock:
        return _write_text(target, text, ctx, reuse_existing=reuse_existing)


def _write_text(
    target: Path, text: str, ctx: _trace.RunContext, *, reuse_existing: bool = False
) -> Path:
    data = text.encode()
    overwrite = not ctx.output_writes.protect_existing or target in ctx.output_writes.created
    reused = False
    if not overwrite and target.exists():
        reused = (
            reuse_existing
            and target.is_file()
            and target.stat().st_size == len(data)
            and target.read_bytes() == data
        )
        if not reused:
            raise RecipeError(f"refusing to replace an existing file in --output-dir: {target}")
    digest = sha256_hex(data)
    effect = record_side_effect(
        "file_write",
        str(target),
        payload={"operation": "write_text", "sha256": digest},
        pending=True,
    )
    if _suppress_output(ctx, target) and not reused:
        return target
    try:
        if not reused:
            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(target, data, overwrite=overwrite)
    except FileExistsError as exc:
        effect.completed = False
        effect.error = str(exc)
        raise RecipeError(
            f"refusing to replace an existing file in --output-dir: {target}"
        ) from exc
    except OSError as exc:
        effect.completed = False
        effect.error = str(exc)
        raise RecipeError(f"failed to write artifact {target}: {exc}") from exc
    effect.completed = ctx.dry_run is DryRunMode.OFF
    ctx.output_writes.created.add(target)
    _trace.checkpoint("side-effect-complete")
    return target


def write_json(path: str | Path, value: Any, *, reuse_existing: bool = False) -> Path:
    """Write sorted, indented JSON through write_text(), with its output and dry-run policy."""
    try:
        text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    except (TypeError, ValueError) as exc:
        raise RecipeError(f"write_json() value is not JSON serializable: {exc}") from exc
    return write_text(path, text, reuse_existing=reuse_existing)


def verify_artifact(
    path: str | Path,
    *,
    sha256: str | None = None,
    nonempty: bool = True,
) -> Path:
    """Require one output-root file, optionally checking its SHA-256 and nonempty content.

    Requires an active recipe step, including the recipe body. Records the
    verified path, hash and size; ordinary soft dry runs suppress filesystem checks.
    Before announcing saved delivery, verify the trusted declared destination, never a
    worker-provided path. Existence and a worker's saved/completed flag do not establish
    complete content: parse the saved artifact and compare its records with the full
    independently gathered inventory, or check a hash of independently expected bytes.
    Preserve a usable alternate answer or fail delivery when these checks fail.
    """
    target = _output_path(path, writing=False)
    ctx = cast("_trace.RunContext", _trace.current())
    if _suppress_output(ctx, target):
        return target
    if not target.is_file():
        raise RecipeError(f"required artifact does not exist: {target}")
    data = target.read_bytes()
    if nonempty and not data:
        raise RecipeError(f"required artifact is empty: {target}")
    actual = sha256_hex(data)
    if sha256 is not None and actual != sha256:
        raise RecipeError(f"artifact hash mismatch for {target}: expected {sha256}, got {actual}")
    if ctx is not None:
        ctx.artifacts.append(
            _trace.ArtifactRecord(
                name="verified-artifact",
                value={"path": str(target), "sha256": actual, "size": len(data)},
            )
        )
        _trace.checkpoint("artifact")
    return target


def _check_wall(budget_state: BudgetState | None) -> None:
    """Fail when the run's shared wall-clock deadline has passed."""
    if budget_state is not None:
        budget_state.check_wall()


def _wall_timeout(budget_state: BudgetState | None, timeout_seconds: float) -> float:
    """Clamp a command timeout to the time left before the shared wall deadline."""
    _check_wall(budget_state)
    if budget_state is None:
        return timeout_seconds
    return bounded_timeout(timeout_seconds, budget_state.remaining_wall_seconds())


def run_command[T = CommandResult](  # noqa: C901 - effect lifecycle, simulation, and typed result validation
    argv: list[str],
    *,
    timeout_seconds: float,
    working_directory: str | Path = ".",
    mutating: bool = True,
    output: TypeForm[T] | None = None,
) -> T:
    """Run argv without a shell, declaring mutations before execution.

    Returns a CommandResult when output is omitted, otherwise the validated output value.
    ``output=Type`` requires a zero exit code and strictly validates JSON stdout.
    Soft dry runs synthesize mutating command results; read-only commands execute.
    Launch failures and local timeouts raise RecipeError with ``command_failure``
    set to "launch" or "timeout". Optional-command fallbacks may handle these;
    re-raise other RecipeError values rather than hiding validation or output-limit failures.
    Shared deadline exhaustion raises BudgetExceeded and must stop the workflow.
    Retain the complete environment-specific argv prefix for fallback commands;
    an import probe in one environment does not establish bare executable availability.
    Classify effective behavior: analysis wrappers that compile targets or load executable
    project configuration are mutating. Require explicit recipe input opt-in before target
    builds, and run them in an isolated environment; mutating=True does not isolate a process.
    Preserve a non-building/manual route when the source workflow permits one.
    """
    ctx = _command_context(mutating)
    if not argv or any(not isinstance(part, str) or not part for part in argv):
        raise RecipeError("run_command() argv must contain non-empty strings")
    cwd = Path(working_directory).resolve()
    if not cwd.is_dir():
        raise RecipeError(f"run_command() working directory does not exist: {cwd}")
    budget_state = ctx.budget_state if ctx is not None else None
    timeout_seconds = _wall_timeout(budget_state, timeout_seconds)
    adapter = None if output is None else _adapter(output)
    effect = None
    if mutating:
        effect = record_side_effect(
            "external_object",
            argv[0],
            payload={"operation": "run_command", "argv": list(argv), "cwd": str(cwd)},
            pending=True,
        )
    if mutating and ctx is not None and ctx.dry_run is DryRunMode.SOFT:
        if adapter is not None:
            return _command_stub(adapter)
        return cast("T", CommandResult(list(argv), 0, "", ""))
    try:
        completed = run_process(
            argv,
            timeout_seconds=timeout_seconds,
            max_output_bytes=8 * 1024 * 1024,
            cwd=cwd,
        )
    except ProcessError as exc:
        if effect is not None:
            effect.completed = False
            effect.error = type(exc).__name__
            _trace.checkpoint("side-effect-failed")
        _check_wall(budget_state)
        error = RecipeError(f"command failed to execute: {type(exc).__name__}: {exc}")
        if isinstance(exc, ProcessStartError):
            error.command_failure = "launch"
        elif isinstance(exc, ProcessTimeoutError):
            error.command_failure = "timeout"
        raise error from exc
    if effect is not None:
        effect.completed = True
        _trace.checkpoint("side-effect-complete")
    if adapter is not None:
        if completed.returncode != 0:
            detail = completed.stderr_tail[-4096:].decode("utf-8", errors="replace")
            if detail:
                print(f"command stderr: {detail}", file=sys.stderr)
            error = f"command exited {completed.returncode}"
            if (
                ctx is not None
                and ctx.hooks.command_gap is not None
                and ctx.hooks.command_gap(
                    argv, working_directory, f"{error}: {detail}" if detail else error
                )
            ):
                return _command_stub(adapter)
            raise RecipeError(error)
        try:
            return adapter.validate_json(completed.stdout, strict=True)
        except ValidationError as exc:
            raise RecipeError("command stdout does not match its declared output type") from exc
    return cast(
        "T",
        CommandResult(
            list(argv),
            completed.returncode,
            completed.stdout.decode("utf-8", errors="replace"),
            completed.stderr_tail.decode("utf-8", errors="replace"),
        ),
    )


def _command_stub[T](adapter: TypeAdapter[T]) -> T:
    """Validate synthesized command results with the same contract as real stdout."""
    from .outputs import STUB_REJECTED, StubJsonSchema, synthesize

    data = synthesize(adapter.json_schema(schema_generator=StubJsonSchema))
    try:
        return adapter.validate_json(json.dumps(data), strict=True)
    except ValidationError as exc:
        raise RecipeError(f"command stub {STUB_REJECTED}") from exc


def _command_context(mutating: bool) -> _trace.RunContext | None:
    ctx = _trace.current()
    if ctx is not None and ctx.parallel_worker and mutating:
        raise RecipeError("parallel workers cannot mutate shared state")
    return ctx
