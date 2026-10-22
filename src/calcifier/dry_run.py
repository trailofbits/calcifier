"""Run a candidate recipe once against schema-valid stub agents.

The compiler calls ``dry_run_recipe`` on a candidate that already passed every
static check. The candidate runs in a separate interpreter under a soft dry
run. Declared files are materialized only in scratch storage; mutating commands remain
suppressed. Each declared agent role
answers with output synthesized from the call's JSON schema, and each declared
tool returns an empty result. A Python exception during the run is a recipe
defect the model can repair; a stub answer that fails a validator the schema
cannot express, such as a word count, is inconclusive. A missing file beneath
an absent default input path is also unverified: this workspace cannot tell a
missing fixture from a misspelled path inside that fixture. Nonzero command
exits involving synthetic or absent inputs are gaps; validated output stubs
let remaining steps run. Other command failures reject the candidate.
Required Path inputs use a separate temporary tree from the working/output workspace.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import json
import os
import re
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal, NamedTuple, get_args, get_origin, get_type_hints

from .agent_events import AgentProvider, AgentResult
from .filesystem_contract import FilesystemContract
from .hashing import sha256_hex
from .outputs import (
    STUB_REJECTED,
    STUB_TEXT,
    StubJsonSchema,
    _adapter,
    object_root_schema,
    synthesize,
    validated_structured,
)
from .recipe import Recipe, RecipeError, RecipeRun, load_recipe
from .registry import Registry
from .skill_package import ResourceEntry, discover_resources
from .subprocesses import ProcessError, run_process
from .tools import Capability, ToolRegistry
from .trace import DryRunMode, OutputWrites, RunHooks, current, current_frame

_TIMEOUT_SECONDS = 120
_OUTPUT_LIMIT = 1024 * 1024
# Reported write paths use "/" on every platform, so a failure reads the same everywhere.
_NATIVE_SEPARATOR = os.sep


class DryRunResult(NamedTuple):
    """Outcome of one dry run: passed, failed with a step and error, or unverified.

    ``writes`` lists the files the candidate declared it would write, in order;
    scratch writes are removed after execution. It is complete only when the
    run passed.
    """

    status: Literal["passed", "failed", "unverified"]
    step: str | None = None
    error: str | None = None
    writes: tuple[str, ...] = ()
    command_gap: bool = False


def dry_run_recipe(
    source: str,
    *,
    recipe_path: Path,
    skill_path: Path,
    resource_entries: tuple[ResourceEntry, ...] | None = None,
    agent_capabilities: dict[str, tuple[str, ...] | None] | None = None,
) -> DryRunResult:
    """Run ``source`` once in a separate interpreter and report how it ended.

    The candidate is staged in a temporary directory, never in the destination
    tree, and its relative child-recipe paths resolve against ``recipe_path``.

    Args:
        source: Complete candidate recipe module.
        recipe_path: Where the accepted recipe will be written.
        skill_path: The SKILL.md whose package supplies resources.
        resource_entries: Packaged resources; discovered from ``skill_path`` when omitted.

    Returns:
        The dry run's outcome.
    """
    recipe_path = recipe_path.resolve()
    skill_path = skill_path.resolve()
    with tempfile.TemporaryDirectory(prefix="calcifier-dry-run-") as directory:
        candidate = Path(directory) / "candidate.py"
        candidate.write_text(_staged_source(source, skill_path), encoding="utf-8", newline="\n")
        argv = _candidate_argv(
            candidate,
            skill_path,
            recipe_path,
            resource_entries,
            agent_capabilities=agent_capabilities,
        )
        try:
            result = run_process(
                argv,
                timeout_seconds=_TIMEOUT_SECONDS,
                max_output_bytes=_OUTPUT_LIMIT,
                cwd=directory,
            )
        except ProcessError as exc:
            return DryRunResult("failed", None, f"dry run did not finish: {exc}")
    lines = result.stdout.decode(errors="replace").strip().splitlines()
    try:
        payload = json.loads(lines[-1])
        return DryRunResult(
            payload["status"],
            payload.get("step"),
            payload.get("error"),
            tuple(payload.get("writes", ())),
            payload.get("command_gap", False),
        )
    except (IndexError, json.JSONDecodeError, KeyError, TypeError):
        detail = result.stderr_tail.decode(errors="replace").strip()
        return DryRunResult("failed", None, f"dry run crashed: {detail or 'no result'}")


def _candidate_argv(
    candidate: Path,
    skill_path: Path,
    recipe_path: Path,
    resource_entries: tuple[ResourceEntry, ...] | None,
    *,
    agent_capabilities: dict[str, tuple[str, ...] | None] | None,
) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "calcifier.dry_run",
        str(candidate),
        str(skill_path),
        str(recipe_path),
    ]
    if resource_entries is not None or agent_capabilities is not None:
        entries = discover_resources(skill_path) if resource_entries is None else resource_entries
        manifest = candidate.parent / "resources.json"
        manifest.write_text(json.dumps([asdict(entry) for entry in entries]), encoding="utf-8")
        argv.append(str(manifest))
    if agent_capabilities is not None:
        path = candidate.parent / "agent-capabilities.json"
        path.write_text(json.dumps(agent_capabilities), encoding="utf-8", newline="\n")
        argv.append(str(path))
    return argv


def _staged_source(source: str, skill_path: Path) -> str:
    """Relocate only host skill provenance when copying a candidate to scratch."""
    from .metadata import metadata_assignment

    try:
        module = ast.parse(source)
    except SyntaxError:
        return source  # The native loader reports malformed Python in the usual result.
    lines = re.split(r"(?<=\n)|(?<=\r)(?!\n)", source)
    for node in reversed(module.body):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_CALCIFIER_SKILL"
        ):
            try:
                metadata = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue  # Do not repair malformed provenance.
            if isinstance(metadata, dict) and "path" in metadata:
                metadata["path"] = str(skill_path)
                lines[node.lineno - 1 : node.end_lineno] = [
                    metadata_assignment("_CALCIFIER_SKILL", metadata)
                ]
    return "".join(lines)


def main(argv: list[str]) -> int:
    """Subprocess entry point: run one candidate and print a JSON outcome line."""
    entries = (
        tuple(
            ResourceEntry(**entry)
            for entry in json.loads(Path(argv[4]).read_text(encoding="utf-8"))
        )
        if len(argv) > 4
        else None
    )
    capabilities = (
        {
            role: tuple(value) if value is not None else None
            for role, value in json.loads(Path(argv[5]).read_text(encoding="utf-8")).items()
        }
        if len(argv) > 5
        else None
    )
    result = _run(
        Path(argv[1]), Path(argv[2]), Path(argv[3]), entries, agent_capabilities=capabilities
    )
    print(json.dumps(result._asdict()))
    return 0


def _skip_dependency_pin(parent_path: Path, child_path: Path, child_hash: str) -> None:
    """Let child recipes run without the trust store; installed in the dry-run process only."""
    del parent_path, child_path, child_hash


def _parent_at_destination(candidate: Path, recipe_path: Path) -> Callable[[Path], Path]:
    """Resolve the candidate's children from where the recipe will live, not the staging copy."""
    staged = candidate.resolve()

    def resolve(path: Path) -> Path:
        return recipe_path if path.resolve() == staged else path

    return resolve


def _run(
    candidate: Path,
    skill_path: Path,
    recipe_path: Path,
    resource_entries: tuple[ResourceEntry, ...] | None = None,
    *,
    agent_capabilities: dict[str, tuple[str, ...] | None] | None = None,
) -> DryRunResult:
    hooks = RunHooks(
        parent_recipe_path=_parent_at_destination(candidate, recipe_path),
        verify_dependency=_skip_dependency_pin,
    )
    # A relative output_root resolves at load time; load and run in the same workspace.
    # The previous directory comes back before cleanup, which Windows requires.
    with (
        tempfile.TemporaryDirectory(prefix="calcifier-dry-run-workspace-") as directory,
        contextlib.chdir(directory),
    ):
        workspace = Path(directory)
        result = _run_in_workspace(
            candidate,
            skill_path,
            workspace,
            hooks,
            resource_entries,
            agent_capabilities=agent_capabilities,
        )
        placeholders = {str(workspace.resolve()): "<workspace>", str(workspace): "<workspace>"}
    placeholders |= {str(candidate.resolve()): "<recipe>", str(candidate): "<recipe>"}
    return _with_placeholders(result, placeholders)


def _run_in_workspace(
    candidate: Path,
    skill_path: Path,
    workspace: Path,
    hooks: RunHooks,
    resource_entries: tuple[ResourceEntry, ...] | None,
    *,
    agent_capabilities: dict[str, tuple[str, ...] | None] | None = None,
) -> DryRunResult:
    try:
        recipe = load_recipe(candidate)
    except (RecipeError, SyntaxError) as exc:
        return DryRunResult("failed", None, f"{type(exc).__name__}: {exc}")
    with tempfile.TemporaryDirectory(prefix="calcifier-dry-run-input-") as directory:
        input_root = Path(directory).resolve() / "input"
        _prepare_workspace(recipe, skill_path, workspace, resource_entries, input_root=input_root)
        result = _run_prepared_recipe(
            recipe, workspace, hooks, input_root=input_root, agent_capabilities=agent_capabilities
        )
        return _with_placeholders(result, {str(input_root): "<input>"})


def _run_prepared_recipe(
    recipe: Recipe,
    workspace: Path,
    hooks: RunHooks,
    *,
    input_root: Path,
    agent_capabilities: dict[str, tuple[str, ...] | None] | None,
) -> DryRunResult:
    text_root = workspace / STUB_TEXT
    inputs = _inputs(recipe, input_root)
    if inputs is None:
        return DryRunResult("unverified", None, "a required recipe input cannot be synthesized")
    missing_roots = _missing_default_roots(recipe, workspace)
    gaps: list[tuple[str | None, str]] = []
    artifact_gaps: list[tuple[str | None, str]] = []

    def command_gap(argv: list[str], cwd: str | Path, error: str) -> bool:
        if not _command_input_gap(
            argv, cwd, workspace, (*missing_roots, text_root), synthetic_roots=(input_root,)
        ):
            return False
        frame = current_frame()
        gaps.append((frame.name if frame else None, error))
        return True

    def artifact_gap(error: str) -> None:
        frame = current_frame()
        artifact_gaps.append((frame.name if frame else None, error))

    hooks = replace(
        hooks,
        scratch_output_root=workspace.resolve(),
        command_gap=command_gap,
        artifact_gap=artifact_gap,
    )
    output_writes = OutputWrites()
    outcome = _outcome(
        _execute(recipe, inputs, hooks, agent_capabilities, output_writes=output_writes),
        missing_input_roots=missing_roots,
        unmaterialized_outputs=output_writes.unmaterialized,
    )
    return _with_gaps(outcome, gaps, artifact_gaps)


def _prepare_workspace(
    recipe: Recipe,
    skill_path: Path,
    workspace: Path,
    resource_entries: tuple[ResourceEntry, ...] | None,
    *,
    input_root: Path,
) -> None:
    entries = discover_resources(skill_path) if resource_entries is None else resource_entries
    recipe.skill_root = skill_path.parent
    recipe.skill_path = skill_path
    if not recipe.skill_sha256:
        recipe.skill_sha256 = sha256_hex(skill_path.read_bytes())
    recipe.resources = tuple((entry.path, entry.sha256, entry.source) for entry in entries)
    for root in (workspace, workspace / STUB_TEXT, input_root):
        root.mkdir(exist_ok=True)
        (root / "example.txt").write_bytes(b"example\n")


def _with_gaps(
    outcome: DryRunResult,
    command_gaps: list[tuple[str | None, str]],
    artifact_gaps: list[tuple[str | None, str]],
) -> DryRunResult:
    """Keep observed gaps even when caught, without hiding deterministic failures."""
    combined = [*command_gaps, *artifact_gaps]
    if not combined or outcome.status == "failed":
        return outcome
    errors = [error for _, error in combined] + ([outcome.error] if outcome.error else [])
    return DryRunResult(
        "unverified",
        combined[0][0],
        "; ".join(dict.fromkeys(errors)),
        outcome.writes,
        bool(command_gaps),
    )


def _command_input_gap(
    argv: list[str],
    cwd: str | Path,
    workspace: Path,
    missing_roots: tuple[Path, ...],
    *,
    synthetic_roots: tuple[Path, ...] = (),
) -> bool:
    """Only invocations naming synthetic or unavailable inputs can be inconclusive."""
    workspace = workspace.resolve()
    roots = tuple(root.resolve() for root in missing_roots)
    for value, base in [*((part, Path(cwd)) for part in argv[1:]), (str(cwd), workspace)]:
        if value.startswith("-") and "=" in value:
            value = value.split("=", 1)[1]
        try:
            path = Path(value)
            resolved = (base / path).resolve()
            if any(
                resolved in {root, root / "example.txt"}
                for root in (workspace / STUB_TEXT, *synthetic_roots)
            ):
                return True
            if not resolved.exists() and any(resolved.is_relative_to(root) for root in roots):
                return True
            if (
                path.is_absolute()
                and resolved.is_relative_to(workspace)
                and (
                    resolved == workspace
                    or resolved == workspace / "example.txt"
                    or not resolved.exists()
                )
            ):
                return True
        except (OSError, ValueError):
            continue  # Arbitrary command arguments need not be filesystem paths.
    return False


def _missing_default_roots(recipe: Recipe, workspace: Path) -> tuple[Path, ...]:
    """Record absent input defaults before execution, only within this workspace."""
    roots: list[Path] = []
    workspace = workspace.resolve()
    for parameter in inspect.signature(recipe.fn).parameters.values():
        value = parameter.default
        if not isinstance(value, (str, Path)) or not str(value):
            continue
        try:
            if Path(value).is_absolute():
                continue
            path = (workspace / value).resolve()
            if path != workspace and path.is_relative_to(workspace) and not path.exists():
                roots.append(path)
        except (OSError, ValueError):
            # A text default need not be a filesystem path.
            continue
    return tuple(roots)


def _with_placeholders(result: DryRunResult, placeholders: dict[str, str]) -> DryRunResult:
    """Name temporary paths by placeholders, so the same failure always reads the same."""

    def replaced(text: str) -> str:
        # Exceptions repr-escape Windows paths; writes carry their unescaped spelling.
        spellings = {
            spelling: name
            for path, name in placeholders.items()
            for spelling in (path, path.replace("\\", "\\\\"), path.replace("\\", "/"))
        }
        # Longest first: a resolved macOS path (/private/var/...) contains the unresolved one.
        for path in sorted(spellings, key=len, reverse=True):
            text = text.replace(path, spellings[path])
        for name in set(placeholders.values()):
            text = text.replace(f"{name}\\\\", f"{name}/")
            text = text.replace(f"{name}\\", f"{name}/")
        return text

    return result._replace(
        error=None if result.error is None else replaced(result.error),
        writes=tuple(replaced(write).replace(_NATIVE_SEPARATOR, "/") for write in result.writes),
    )


def _inputs(recipe: Recipe, input_root: Path) -> dict[str, Any] | None:
    hints = get_type_hints(recipe.fn)
    inputs: dict[str, Any] = {}
    for name, parameter in inspect.signature(recipe.fn).parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            continue
        value = _input_value(hints.get(name), input_root)
        if value is None:
            return None
        inputs[name] = value
    return inputs


def _input_value(annotation: Any, input_root: Path) -> Any:
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin in {list, tuple} and arguments:
        elements = arguments[:-1] if arguments[-1] is Ellipsis else arguments
        values = [_input_value(element, input_root) for element in elements]
        return None if any(value is None for value in values) else origin(values)
    if annotation is Path:
        return input_root
    if annotation is bool:
        return False
    if annotation in {int, float}:
        return annotation(1)
    # Neutral text also names a seeded directory relative to the active workspace.
    return STUB_TEXT if annotation in {str, None} else None


def _execute(
    recipe: Recipe,
    inputs: dict[str, Any],
    hooks: RunHooks,
    agent_capabilities: dict[str, tuple[str, ...] | None] | None = None,
    *,
    output_writes: OutputWrites | None = None,
) -> RecipeRun:
    roles, capabilities = recipe.agent_names, recipe.capabilities
    if recipe.composed:
        from .graph import inspect_graph

        path = Path(recipe.fn.__code__.co_filename)
        if hooks.parent_recipe_path is not None:
            path = hooks.parent_recipe_path(path)
        graph = inspect_graph(path, entry_bytes=recipe.source_bytes)
        roles, capabilities = graph.agents, graph.capabilities
    registry = _StubRegistry(agent_capabilities)
    for role in roles:
        registry.get(role)
    tools = ToolRegistry()
    for name in capabilities:
        tools.register(Capability(name, "dry-run"), _EmptyTool())
    return recipe._execute(
        inputs,
        registry=registry,
        tool_registry=tools,
        dry_run=DryRunMode.SOFT,
        _hooks=hooks,
        _output_writes=output_writes,
    )


def _outcome(
    run: RecipeRun,
    *,
    missing_input_roots: tuple[Path, ...] = (),
    unmaterialized_outputs: set[Path] | None = None,
) -> DryRunResult:
    writes = _file_writes(run.records)
    if run.ok:
        return DryRunResult("passed", writes=writes)
    error = run.error or ""
    step, located = _failing_step(run.records)
    if (
        STUB_REJECTED in error
        or run.exception is None
        or any(
            exc.type_name == "UnmaterializedArtifactError"
            for exc in (run.exception, *run.exception.causes)
        )
    ):
        return DryRunResult("unverified", step, error, writes)
    detail = located or error
    location = run.exception.location
    if location is not None:
        source = f": {location.source}" if location.source else ""
        detail = f"{detail} (line {location.line}{source})"
    if _missing_file(run, paths=unmaterialized_outputs):
        return DryRunResult(
            "unverified",
            step,
            f"declared output is unavailable after a suppressed effect: {detail}; "
            "saved content and downstream checks remain unverified",
            writes,
        )
    if _missing_file(run, roots=missing_input_roots):
        return DryRunResult(
            "unverified",
            step,
            "file beneath an absent default path is unavailable in the synthetic workspace: "
            f"{detail}",
            writes,
        )
    return DryRunResult("failed", step, detail, writes)


def _missing_file(
    run: RecipeRun,
    *,
    roots: tuple[Path, ...] = (),
    paths: set[Path] | None = None,
) -> bool:
    if run.exception is None:
        return False
    for exc in (run.exception, *run.exception.causes):
        if exc.type_name != "FileNotFoundError" or exc.filename is None:
            continue
        try:
            path = Path(exc.filename).resolve()
            if path in (paths or ()) or any(path.is_relative_to(root) for root in roots):
                return True
        except (OSError, ValueError):
            continue
    return False


def _file_writes(records: list[Any]) -> tuple[str, ...]:
    """Return the file-write targets the run recorded, in execution order."""
    effects: list[Any] = []
    pending = list(reversed(records))
    while pending:
        record = pending.pop()
        effects.extend(effect for effect in record.side_effects if effect.kind == "file_write")
        pending.extend(reversed(record.children))
    ordered = sorted(effects, key=lambda effect: effect._recorded_at_ns)
    return tuple(dict.fromkeys(effect.target for effect in ordered))


def _failing_step(records: list[Any]) -> tuple[str | None, str | None]:
    """Return the innermost step that recorded an error, and that error."""
    failing: tuple[str | None, str | None] = (None, None)
    pending = list(records)
    while pending:
        record = pending.pop()
        if record.error is not None:
            failing = (record.name, record.error)
            pending = list(record.children)
    return failing


class _StubRegistry(Registry):
    """Answer any role with a schema stub, including roles of children not yet pinned."""

    def __init__(self, capabilities: dict[str, tuple[str, ...] | None] | None = None) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._capabilities = dict(capabilities or {})

    def get(self, name: str) -> AgentProvider:
        with self._lock:
            if name not in self.known():
                self.register(_SchemaStub(name, self._capabilities.get(name)))
        return super().get(name)


class _EmptyTool:
    def available(self, name: str) -> bool:
        del name
        return True

    def invoke(self, name: str, /, **kwargs: Any) -> Any:
        del name, kwargs
        return ""


class _SchemaStub(AgentProvider):
    """Answer every call with output synthesized from its JSON schema."""

    def __init__(self, role: str, capabilities: tuple[str, ...] | None = None) -> None:
        self.name = role
        self.default_timeout = 30
        self.capabilities = capabilities

    def _structured_schema(self, annotation: Any) -> dict[str, Any]:
        # Native normalization loses tuple positions; exercise the exact host shape.
        return object_root_schema(_adapter(annotation).json_schema(schema_generator=StubJsonSchema))

    def _parse_structured(self, text: str, annotation: Any) -> Any:
        return validated_structured(text, annotation)

    def invoke(
        self,
        prompt: str,
        *,
        timeout: float | None = None,
        filesystem: FilesystemContract | None = None,
        schema: dict[str, Any] | None = None,
    ) -> AgentResult:
        del prompt, timeout
        ctx = current()
        if ctx is not None and filesystem is not None:
            with ctx.output_writes.lock:
                ctx.output_writes.unmaterialized.update(
                    path
                    for path in ctx.output_writes.planned
                    if any(path.is_relative_to(root) for root in filesystem.writable_paths)
                )
        text = STUB_TEXT if schema is None else json.dumps(synthesize(schema))
        return AgentResult(text, text, "", 0, self.name)


# gremlin: pardon[untestable] subprocess entry guard; main() carries the behavior and is tested.
if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
