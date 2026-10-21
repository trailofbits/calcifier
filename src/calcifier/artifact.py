"""Prepare, validate, summarize, and write recipe artifacts."""

from __future__ import annotations

import ast
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .agent_events import AgentActivity, AgentProvider
from .atomic_io import atomic_write
from .catalog import _merge_skill_catalogs, discover_sibling_skill_catalog
from .compiler_activity import DEFAULT_COMPILATION_TIMEOUT, CompilerDeadline
from .diagnostics import CalcifyError, render_diagnostics
from .dry_run import DryRunResult
from .hashing import sha256_hex
from .metadata import attach_metadata, attach_script_metadata
from .recipe_ast import (
    collect_structure,
    decorator_keyword_call,
    import_bindings,
    parse_module,
    recipe_decorator,
    recipe_version,
    require_recipe,
    signature_for_node,
)
from .registry import Registry
from .repair import _MAX_COMPILER_CALLS, SourceCompileContext, compile_source
from .skill_package import add_resource_sources, discover_resources, manifest_summary
from .source_quality import SourceQualityError, format_source, resolve_python_version
from .validation import is_positive_number

logger = logging.getLogger(__name__)


def _child_recipe_pins(source: str, target: Path, catalog: dict[str, str]) -> dict[str, str]:
    """Name every child recipe the source calls, so trust pins each one.

    Children come from the recipe itself, including every choice of a dynamic
    catalog selector; a child the catalog does not name is keyed by its path.
    """
    from .portable import called_child_recipes

    names = {Path(path).resolve(): name for name, path in catalog.items()}
    return {
        names.get(child, Path(os.path.relpath(child, target.parent.resolve())).as_posix()): str(
            child
        )
        for child in called_child_recipes(target, ast.parse(source))
    }


def _atomic_write_text(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(target, text.encode("utf-8"))


def _output_path(skill_path: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    return skill_path.parent / ".calcifier" / "recipe.py"


def _configured_runtime_wall_seconds(
    provider: AgentProvider,
    timeout: int | None,
) -> int | None:
    """Resolve the host-owned runtime ceiling independently of model output."""
    configured = timeout if timeout is not None else provider.default_timeout
    if not is_positive_number(configured, finite=True):
        return None
    return math.ceil(cast("int | float", configured))


@dataclass(frozen=True)
class PreparedArtifact:
    """A validated recipe artifact with its provenance, not yet written."""

    path: Path
    source: str
    outcome: str = "compiled"
    dry_run: DryRunResult | None = None
    suitability_note: str | None = None


def prepare_artifact(  # noqa: C901 - baseline preparation and compiler improvement share inputs
    skill_path: Path,
    registry: Registry,
    *,
    compiler: str = "compiler",
    output: Path | None = None,
    timeout: int | None = None,
    compilation_timeout: int | None = None,
    max_compiler_calls: int | None = None,
    skill_catalog: dict[str, str] | None = None,
    activity: AgentActivity | None = None,
    python_version: str | None = None,
    interface_requirements: str = "",
    runtime_wall_seconds: int | None = None,
    resource_sources: dict[str, str] | None = None,
    optimize: bool = True,
) -> PreparedArtifact:
    """Prepare a runnable baseline, then retain a validated compiler improvement.

    Return the destination and source without writing there; dry runs use scratch
    storage. ``timeout`` limits each call, ``compilation_timeout`` the whole
    compilation, and ``max_compiler_calls`` includes repairs (1–8, default five
    with a baseline, eight without). ``optimize=False`` prepares only the baseline.
    Invalid inputs or failure without a usable baseline raise CalcifyError.
    """
    if max_compiler_calls is not None and (
        type(max_compiler_calls) is not int or not 1 <= max_compiler_calls <= _MAX_COMPILER_CALLS
    ):
        raise CalcifyError(f"max_compiler_calls must be an integer from 1 to {_MAX_COMPILER_CALLS}")
    deadline = CompilerDeadline(
        DEFAULT_COMPILATION_TIMEOUT if compilation_timeout is None else compilation_timeout
    )
    provider = registry.get(compiler)
    skill_bytes = skill_path.read_bytes()
    resources = discover_resources(skill_path)
    if resource_sources is not None:
        try:
            resources = add_resource_sources(skill_path, resources, resource_sources)
        except (OSError, ValueError) as exc:
            raise CalcifyError(f"invalid explicit compiler resources: {exc}") from exc
    target = _output_path(skill_path, output)
    python_version = resolve_python_version(target, python_version)
    skill_text = skill_bytes.decode("utf-8")
    agent_roles = tuple(role for role in registry.known() if role != compiler)
    wall_seconds = runtime_wall_seconds or _configured_runtime_wall_seconds(provider, timeout)
    effective_catalog = _merge_skill_catalogs(
        discover_sibling_skill_catalog(skill_path),
        skill_catalog or {},
    )
    baseline = None
    if agent_roles:
        from .baseline import baseline_source, catalog_context

        instructions, references = catalog_context(effective_catalog, target)
        worker_role = "worker" if "worker" in agent_roles else agent_roles[0]
        source = baseline_source(
            skill_text,
            role=worker_role,
            resources=tuple(entry.path for entry in resources),
            worker_timeout_seconds=_configured_runtime_wall_seconds(registry.get(worker_role), None)
            or 900,
            wall_seconds=runtime_wall_seconds,
            instructions=interface_requirements + "\n" + instructions,
            reference_paths=references,
        )
        source = attach_metadata(
            source,
            skill_path,
            resources,
            recipe_path=target,
            skill_sha256=sha256_hex(skill_bytes),
            outcome="baseline_retained",
        )
        baseline = PreparedArtifact(
            target, _checked_artifact(source, target, python_version), "baseline_retained"
        )
    activity = activity or AgentActivity()
    if baseline is not None:
        activity.record_event("baseline", "prepared")
    if not optimize:
        if baseline is None:
            raise CalcifyError("a baseline requires a configured runtime agent role")
        activity.outcome = baseline.outcome
        activity.retained_reason = "optimization disabled"
        activity.dry_run = baseline.dry_run
        activity.record_event("baseline", "retained", detail="optimization disabled")
        return baseline

    def finalize_source(source: str) -> str:
        return attach_metadata(
            source,
            skill_path,
            resources,
            recipe_path=target,
            subrecipes=_child_recipe_pins(source, target, effective_catalog),
            skill_sha256=sha256_hex(skill_bytes),
            interface_requirements=interface_requirements,
        )

    try:
        context = SourceCompileContext(
            skill_text=skill_text,
            provider=provider,
            resource_manifest=manifest_summary(resources) or "(none)",
            agent_roles=agent_roles,
            agent_capabilities={role: registry.get(role).capabilities for role in agent_roles},
            timeout=timeout,
            runtime_wall_seconds=wall_seconds,
            catalog={
                name: Path(
                    os.path.relpath(Path(path).resolve(), target.parent.resolve())
                ).as_posix()
                for name, path in effective_catalog.items()
            },
            activity=activity,
            source_path=target,
            python_version=python_version,
            resource_paths=frozenset(entry.path for entry in resources),
            skill_path=skill_path,
            resource_entries=tuple(resources),
            interface_requirements=interface_requirements,
            max_calls=max_compiler_calls
            if max_compiler_calls is not None
            else (5 if baseline is not None else _MAX_COMPILER_CALLS),
            deadline=deadline,
            finalize_source=finalize_source,
        )
        compiled = compile_source(context)
        result = PreparedArtifact(
            target, compiled.source, "compiled", compiled.dry_run, compiled.suitability_note
        )
        deadline.remaining()
    except CalcifyError as exc:
        if baseline is None:
            raise
        # A rejected improvement cannot make a stale baseline publishable.
        if skill_path.read_bytes() != skill_bytes:
            raise CalcifyError("skill changed during compilation") from exc
        from .compiler_resources import stage_compiler_resources

        with tempfile.TemporaryDirectory(prefix="calcifier-baseline-") as directory:
            stage_compiler_resources(Path(directory), skill_path, skill_text, tuple(resources))
        activity.outcome = baseline.outcome
        activity.retained_reason = str(exc)
        activity.suitability_note = None
        activity.dry_run = baseline.dry_run
        activity.record_event("baseline", "retained", detail=str(exc))
        logger.warning("Compiler improvement rejected; retaining baseline: %s", exc)
        return baseline
    activity.outcome = result.outcome
    activity.dry_run = result.dry_run
    activity.record_event("baseline", "replaced")
    return result


def _checked_artifact(source: str, target: Path, python_version: str | None) -> str:
    """Format the recipe with its metadata, then check it as `lint --portable` will."""
    try:
        source = attach_script_metadata(source, python_version)
        source = format_source(source, target, python_version=python_version)
    except SourceQualityError as exc:
        raise CalcifyError(f"final generated artifact failed validation: {exc}") from exc
    from .portable import lint_portable_source

    diagnostics = lint_portable_source(source, target, python_version=python_version)
    if diagnostics:
        raise CalcifyError(
            f"final generated artifact is not portable: {render_diagnostics(diagnostics)}"
        )
    return source


def calcify_file(
    skill_path: Path,
    registry: Registry,
    *,
    compiler: str = "compiler",
    output: Path | None = None,
    timeout: int | None = None,
    compilation_timeout: int | None = None,
    max_compiler_calls: int | None = None,
    skill_catalog: dict[str, str] | None = None,
    activity: AgentActivity | None = None,
    python_version: str | None = None,
) -> Path:
    """Compile a recipe and atomically write it."""
    artifact = prepare_artifact(
        skill_path,
        registry,
        compiler=compiler,
        output=output,
        timeout=timeout,
        compilation_timeout=compilation_timeout,
        max_compiler_calls=max_compiler_calls,
        skill_catalog=skill_catalog,
        activity=activity,
        python_version=python_version,
    )
    _atomic_write_text(artifact.path, artifact.source)
    logger.info("Calcified %s -> %s", skill_path, artifact.path)
    return artifact.path


def render_summary(recipe_path: Path) -> str:
    """Pretty-print a calcified recipe without executing it."""
    try:
        module = parse_module(recipe_path.read_text(encoding="utf-8"))
        recipes, steps, _ = collect_structure(module)
        recipe_node = require_recipe(module, recipes)
    except (OSError, CalcifyError) as exc:
        return f"<could not statically summarise recipe: {exc}>"
    bindings = import_bindings(module)
    lines = [
        f"name: {recipe_node.name}",
        f"version: {recipe_version(recipe_node, bindings)}",
        f"inputs: {signature_for_node(recipe_node)}",
        f"steps: {[step.name for step in steps]}",
    ]
    from .metadata import module_metadata

    if module_metadata(module).get("_CALCIFIER_COMPILATION") == "baseline_retained":
        lines.append("compilation: baseline (workflow delegated to an agent; not optimized)")
    decorator = recipe_decorator(recipe_node, bindings)
    budget = None if decorator is None else decorator_keyword_call(decorator, "budget", bindings)
    if budget is not None:
        lines.append(f"budget: {budget[0]} agent calls, {budget[1]} seconds")
    else:
        lines.append("budget: none declared")
    return "\n".join(lines)
