"""Pinned recipe composition with shared runtime budgets."""

from __future__ import annotations

import contextvars
import inspect
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import trace as _trace
from .exceptions import CalcifierError, RecipeFailure
from .metadata import parse_metadata_namespace, split_metadata
from .trust import compute_sha256, compute_sha256_bytes, trusted_entry

if TYPE_CHECKING:
    from .recipe import Recipe, RecipeRun
    from .trace import ArtifactRecord, StepRecord


class SubrecipeError(RecipeFailure):
    """Raised when a child recipe cannot be safely invoked."""


class SubrecipeCrashError(CalcifierError):
    """Raised when a child recipe crashed, so the parent's run reports a crash, not ``fail()``."""


@dataclass(frozen=True)
class SubrecipeResult:
    """Result returned from a successful child recipe call."""

    child_name: str
    child_hash: str
    output: Any
    artifacts: list[ArtifactRecord]
    records: list[StepRecord]


@dataclass(frozen=True)
class _StackEntry:
    name: str
    recipe_hash: str
    path: Path


_CALL_STACK: contextvars.ContextVar[tuple[_StackEntry, ...]] = contextvars.ContextVar(
    "calcifier_recipe_call_stack",
    default=(),
)
_PARALLEL_CHILD_CALL_INDEX: contextvars.ContextVar[int] = contextvars.ContextVar(
    "calcifier_parallel_child_call_index",
    default=0,
)


def call_recipe(
    child_path: str | Path,
    inputs: Mapping[str, Any],
    *,
    parent_trust: str | Path | None = None,
) -> SubrecipeResult:
    """Run a declared child recipe under the active recipe context."""
    parent_ctx = _require_context()
    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    parent_path = _parent_recipe_path(parent_trust, caller)
    child_file = Path(child_path)
    if not child_file.is_absolute():
        child_file = (parent_path.parent / child_file).resolve()
    child_bytes = child_file.read_bytes()
    child_hash = compute_sha256_bytes(child_bytes)
    _require_skill_metadata_bytes(child_bytes)
    verify_dependency = parent_ctx.hooks.verify_dependency or _verify_dependency
    verify_dependency(parent_path, child_file, child_hash)
    chain = _active_chain(parent_ctx, parent_path)
    _reject_circular_call(chain, child_file.stem, child_hash)
    child = _load_recipe_from_bytes(child_bytes, child_file)
    token = _CALL_STACK.set((*chain, _StackEntry(child.name, child_hash, child_file.resolve())))
    try:
        run = _run_child(child, inputs, parent_ctx)
    finally:
        _CALL_STACK.reset(token)
    _attach_child_records(parent_ctx, child, child_hash, run)
    if not run.ok:
        message = f"subrecipe {child.name} failed: {run.error}"
        if run.error_type is not None:
            raise SubrecipeCrashError(message)
        raise SubrecipeError(message)
    return SubrecipeResult(child.name, child_hash, run.output, run.artifacts, run.records)


def call_catalog_recipe(
    name: str,
    recipes: Mapping[str, str],
    inputs: Mapping[str, Any],
) -> SubrecipeResult:
    """Resolve one child from a generated, closed name-to-recipe catalog."""
    if not isinstance(name, str) or not name:
        raise SubrecipeError("catalog subrecipe name must be a non-empty string")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in recipes.items()):
        raise SubrecipeError("catalog subrecipe mapping must contain string names and paths")
    normalized_name = name.removeprefix("/")
    child_path = recipes.get(normalized_name)
    if child_path is None:
        raise SubrecipeError(
            f"unresolved catalog subrecipe {name!r}; expected one of {sorted(recipes)!r}"
        )
    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    parent_path = _parent_recipe_path(None, caller)
    return call_recipe(child_path, inputs, parent_trust=parent_path)


def _require_context() -> _trace.RunContext:
    ctx = _trace.current()
    if ctx is None:
        raise SubrecipeError("call_recipe requires an active recipe run")
    return ctx


def _parent_recipe_path(parent_trust: str | Path | None, caller: Any) -> Path:
    if parent_trust is not None:
        path = _path_from_trust_or_recipe(Path(parent_trust))
    else:
        filename = caller.f_globals.get("__file__") if caller is not None else None
        if not isinstance(filename, str):
            raise SubrecipeError("parent trust record could not be inferred")
        path = Path(filename)
    ctx = _trace.current()
    return ctx.hooks.parent_recipe_path(path) if ctx and ctx.hooks.parent_recipe_path else path


def _path_from_trust_or_recipe(path: Path) -> Path:
    suffix = ".trust.json"
    if path.name.endswith(suffix):
        return path.with_name(path.name[: -len(suffix)])
    return path


def _require_skill_metadata_bytes(child_bytes: bytes) -> None:
    _, metadata = split_metadata(child_bytes.decode())
    parsed = parse_metadata_namespace(metadata or "")
    if not isinstance(parsed.get("_CALCIFIER_SKILL"), Mapping):
        raise SubrecipeError("child recipe is missing _CALCIFIER_SKILL metadata")


def _load_recipe_from_bytes(child_bytes: bytes, child_path: Path) -> Recipe:
    """Load the trusted child snapshot.

    Callers hash and validate ``child_bytes`` first; the snapshot loader then
    compiles and executes those exact bytes without reopening the path.
    """
    from .recipe import RecipeSnapshot, load_recipe_snapshot

    snapshot = RecipeSnapshot(child_path.resolve(), child_bytes, compute_sha256_bytes(child_bytes))
    return load_recipe_snapshot(snapshot)


def _verify_dependency(parent_path: Path, child_path: Path, child_hash: str) -> None:
    entry = trusted_entry(parent_path, verify_dependencies=False)
    if entry is None:
        _verify_nested_dependency(parent_path, child_path, child_hash)
        return
    dependencies = _dependencies(entry)
    if not _dependency_declared(parent_path, dependencies, child_path):
        raise SubrecipeError(f"undeclared dependency: {child_path}")
    for dependency in dependencies:
        current = _current_dependency_sha(parent_path, dependency, child_path, child_hash)
        if current != dependency["sha256"]:
            raise SubrecipeError(f"dependency drift: {dependency['path']}")
    if not _dependency_hash_declared(parent_path, dependencies, child_path, child_hash):
        raise SubrecipeError(f"dependency drift: {child_path}")


def _verify_nested_dependency(parent_path: Path, child_path: Path, child_hash: str) -> None:
    """Authorize a grandchild through the already-verified active parent snapshot."""
    stack = _CALL_STACK.get()
    parent = stack[-1] if stack else None
    if parent is None or parent.path.resolve() != parent_path.resolve():
        raise SubrecipeError(f"parent recipe is not trusted: {parent_path}")
    try:
        parent_bytes = parent_path.read_bytes()
    except OSError as exc:
        raise SubrecipeError(f"cannot read active parent recipe: {parent_path}") from exc
    if compute_sha256_bytes(parent_bytes) != parent.recipe_hash:
        raise SubrecipeError(f"dependency drift: {parent_path}")
    try:
        _, metadata = split_metadata(parent_bytes.decode())
        raw = parse_metadata_namespace(metadata or "").get("_CALCIFIER_SUBRECIPES", [])
    except (UnicodeDecodeError, ValueError, SyntaxError) as exc:
        raise SubrecipeError(f"invalid subrecipe metadata: {parent_path}") from exc
    if not isinstance(raw, list):
        raise SubrecipeError(f"invalid subrecipe metadata: {parent_path}")
    dependencies = [
        dependency
        for item in raw
        if isinstance(item, Mapping) and (_dependency := _dependency_from_item(item)) is not None
        for dependency in (_dependency,)
    ]
    if not _dependency_declared(parent_path, dependencies, child_path):
        raise SubrecipeError(f"undeclared dependency: {child_path}")
    if not _dependency_hash_declared(parent_path, dependencies, child_path, child_hash):
        raise SubrecipeError(f"dependency drift: {child_path}")


def _dependencies(entry: Mapping[str, Any]) -> list[dict[str, str]]:
    raw = entry.get("dependencies")
    if not isinstance(raw, list):
        return []
    return [_dependency for item in raw if (_dependency := _dependency_from_item(item))]


def _dependency_from_item(item: Any) -> dict[str, str] | None:
    if not isinstance(item, Mapping):
        return None
    path = item.get("path")
    sha = item.get("sha256")
    if isinstance(path, str) and isinstance(sha, str):
        return {"path": path, "sha256": sha}
    return None


def _dependency_declared(
    parent_path: Path,
    dependencies: list[dict[str, str]],
    child_path: Path,
) -> bool:
    return any(
        _same_path(parent_path, dependency["path"], child_path) for dependency in dependencies
    )


def _dependency_hash_declared(
    parent_path: Path,
    dependencies: list[dict[str, str]],
    child_path: Path,
    child_hash: str,
) -> bool:
    return any(
        dependency["sha256"] == child_hash
        for dependency in dependencies
        if _same_path(parent_path, dependency["path"], child_path)
    )


def _dependency_current_sha(parent_path: Path, dependency_path: str) -> str | None:
    path = _dependency_path(parent_path, dependency_path)
    return compute_sha256(path) if path.exists() else None


def _current_dependency_sha(
    parent_path: Path,
    dependency: dict[str, str],
    child_path: Path,
    child_hash: str,
) -> str | None:
    if _same_path(parent_path, dependency["path"], child_path):
        return child_hash
    return _dependency_current_sha(parent_path, dependency["path"])


def _same_path(parent_path: Path, dependency_path: str, child_path: Path) -> bool:
    return _dependency_path(parent_path, dependency_path).resolve() == child_path.resolve()


def _dependency_path(parent_path: Path, dependency_path: str) -> Path:
    path = Path(dependency_path)
    return path if path.is_absolute() else parent_path.parent / path


def _active_chain(parent_ctx: _trace.RunContext, parent_path: Path) -> tuple[_StackEntry, ...]:
    stack = _CALL_STACK.get()
    if stack or not parent_ctx.recipe_hash:
        return stack
    return (_StackEntry(parent_path.stem, parent_ctx.recipe_hash, parent_path.resolve()),)


def _reject_circular_call(
    chain: tuple[_StackEntry, ...],
    child_name: str,
    child_hash: str,
) -> None:
    if child_hash not in {entry.recipe_hash for entry in chain}:
        return
    names = [entry.name for entry in chain] + [child_name]
    raise SubrecipeError(f"circular subrecipe call detected: {' -> '.join(names)}")


def _run_child(
    child: Recipe,
    inputs: Mapping[str, Any],
    parent_ctx: _trace.RunContext,
) -> RecipeRun:
    isolated_root = _parallel_child_root(child, parent_ctx)
    isolated_parallel_child = isolated_root is not None
    child_output_root = parent_ctx.output_root if child.output_root is not None else None
    if isolated_root is not None:
        child_output_root = isolated_root
    return child._execute(
        inputs,
        registry=parent_ctx.registry,
        tool_registry=parent_ctx.tool_registry,
        dry_run=parent_ctx.dry_run,
        _hooks=parent_ctx.hooks,
        _budget_state=parent_ctx.budget_state,
        _parallel_worker=parent_ctx.parallel_worker and not isolated_parallel_child,
        _parallel_lane=parent_ctx.parallel_lane,
        _output_root=child_output_root,
        _output_writes=parent_ctx.output_writes,
        _child=True,
    )


def _parallel_child_root(child: Recipe, parent_ctx: _trace.RunContext) -> Path | None:
    """Contain every mutating parallel child call beneath a private root."""
    if (
        not parent_ctx.parallel_worker
        or parent_ctx.parallel_lane is None
        or child.output_root is None
        or parent_ctx.output_root is None
    ):
        return None
    call_index = _PARALLEL_CHILD_CALL_INDEX.get() + 1
    _PARALLEL_CHILD_CALL_INDEX.set(call_index)
    child_component = re.sub(r"[^A-Za-z0-9_.-]+", "-", child.name).strip(".-") or "child"
    call_component = f"call-{call_index:04d}-{child_component}"
    return (
        parent_ctx.output_root / ".calcifier-parallel" / parent_ctx.parallel_lane / call_component
    )


def _attach_child_records(
    parent_ctx: _trace.RunContext,
    child: Recipe,
    child_hash: str,
    run: RecipeRun,
) -> None:
    marker = {"name": child.name, "hash": child_hash}
    if parent_ctx.record_stack:
        parent_record = parent_ctx.record_stack[-1]
        parent_record.children.extend(run.records)
        parent_record.subrecipe = marker
    else:
        parent_ctx.records.extend(run.records)
