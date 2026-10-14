"""Deterministic acceptance checks for generated recipe source."""

from __future__ import annotations

import ast
import functools
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from .agent_config import BUILTIN_AGENTS
from .agent_events import captured_agent_activities
from .diagnostics import (
    CalcifyError,
    Diagnostic,
    diagnostic_from_error,
    lint_diagnostics,
    render_diagnostics,
    require_no_findings,
    source_node_name,
)
from .effect_methods import forbidden_method
from .output_annotations import output_annotation_error
from .recipe_ast import (
    _OUTPUT_PLACEHOLDER_RE,
    MUTATION_TARGETS,
    TOOL_TARGETS,
    _TopFunction,
    collect_structure,
    decorator_string_keyword,
    defined_names,
    import_bindings,
    import_resolver,
    parse_module,
    primitive_targets,
    recipe_agents,
    recipe_capabilities,
    recipe_decorator,
    references_calcifier,
    reject_bare_cli_footer,
    require_footer,
    require_recipe,
    resolve_call_target,
)
from .runtime_conformance import packaged_resource_errors
from .skill_package import resource_relative_path
from .source_quality import SourceQualityError, lint_source_quality, quality_error
from .taint import model_derived_sink_findings
from .type_check import type_check_source


def _check_non_agent_call(
    call: ast.Call,
    target: str | None,
    bindings: dict[str, str],
) -> None:
    module_lookup = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and node.args
        and (resolve_call_target(node.args[0], bindings) or "").startswith("calcifier")
        for node in ast.walk(call.func)
    )
    if (
        target is None
        and references_calcifier(call.func, bindings)
        and (not isinstance(call.func, ast.Attribute) or module_lookup)
    ):
        raise CalcifyError(
            f"dynamic calcifier call target at line {call.lineno}: {ast.unparse(call.func)}. "
            "Call the public Calcifier entry point directly.",
            line=call.lineno,
        )


def _check_agent_declarations(
    module: ast.Module,
    recipe_node: _TopFunction,
) -> None:
    bindings = import_bindings(module)
    declared = recipe_agents(recipe_node, bindings)
    if declared is not None and len(set(declared)) != len(declared):
        raise CalcifyError(
            f"recipe agents must be unique at line {recipe_node.lineno}", line=recipe_node.lineno
        )
    provider_roles = sorted(set(declared or ()) & BUILTIN_AGENTS)
    if provider_roles:
        raise CalcifyError(
            f"provider names cannot be semantic roles: {provider_roles!r}",
            line=recipe_node.lineno,
        )
    _check_each(
        _tool_calls([module]),
        lambda call: _check_agent_call(call, declared, bindings),
        code="agent_declarations",
    )
    if declared:
        entry_decorator = recipe_decorator(recipe_node, bindings)
        if entry_decorator is None or not any(
            keyword.arg == "budget" for keyword in entry_decorator.keywords
        ):
            raise CalcifyError(
                f"agent-backed recipe requires Budget at line {recipe_node.lineno}",
                line=recipe_node.lineno,
            )


_DECLARED_MUTATIONS = MUTATION_TARGETS
_OUTPUT_ROOT_PRIMITIVES = _DECLARED_MUTATIONS | {"calcifier.verify_artifact"}
_FORBIDDEN_EFFECT_CALLS = {
    "os.popen",
    "os.system",
    "shutil.copy",
    "shutil.copy2",
    "shutil.copyfile",
    "shutil.move",
    "shutil.rmtree",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.run",
}


def _check_effect_contracts(
    module: ast.Module,
    recipe_node: _TopFunction,
) -> None:
    bindings = import_bindings(module)
    calls = _tool_calls([module])
    output_root_primitives = {
        target
        for call in calls
        if (target := resolve_call_target(call.func, bindings)) in _OUTPUT_ROOT_PRIMITIVES
    }

    def check_call(call: ast.Call) -> None:
        target = resolve_call_target(call.func, bindings)
        if target in _OUTPUT_ROOT_PRIMITIVES:
            return
        if target == "calcifier.record_side_effect":
            raise CalcifyError(
                "generated recipes must use deterministic mutation primitives "
                f"at line {call.lineno}",
                line=call.lineno,
            )
        if target in _FORBIDDEN_EFFECT_CALLS or (
            isinstance(call.func, ast.Name) and call.func.id == "open"
        ):
            raise CalcifyError(
                f"arbitrary generated effect at line {call.lineno}", line=call.lineno
            )
        forbidden = forbidden_method(call)
        if forbidden is not None:
            qualifier = "path receiver" if forbidden == "replace" else "write"
            raise CalcifyError(
                f"arbitrary generated {qualifier} method {forbidden!r} "
                f"at line {call.lineno}; "
                "use Calcifier's declared "
                "primitives for filesystem mutation",
                line=call.lineno,
            )

    checks: list[Callable[[], None]] = [
        lambda: _check_each(calls, check_call, code="effect_contracts")
    ]
    if output_root_primitives:
        checks.extend(
            [
                lambda: _require_generated_output_root(
                    recipe_node, output_root_primitives, bindings
                ),
                lambda: _reject_double_rooted_output_paths(module, recipe_node, bindings),
            ]
        )
    _check_each(checks, lambda check: check(), code="effect_contracts")


def _require_generated_output_root(
    recipe_node: _TopFunction,
    output_root_primitives: set[str],
    bindings: dict[str, str],
) -> None:
    decorator = recipe_decorator(recipe_node, bindings)
    output_keyword = (
        None
        if decorator is None
        else next((item for item in decorator.keywords if item.arg == "output_root"), None)
    )
    if (
        output_keyword is None
        or not isinstance(output_keyword.value, ast.Constant)
        or not isinstance(output_keyword.value.value, str)
        or not output_keyword.value.value
        or _OUTPUT_PLACEHOLDER_RE.search(output_keyword.value.value) is not None
    ):
        if output_root_primitives == {"calcifier.verify_artifact"}:
            raise CalcifyError(
                "verify_artifact() requires a literal output_root; "
                f"declare one for the recipe or remove the check (line {recipe_node.lineno})",
                line=recipe_node.lineno,
            )
        raise CalcifyError(
            "mutating recipe requires a concrete literal output_root without "
            f"placeholders at line {recipe_node.lineno}",
            line=recipe_node.lineno,
        )


def _reject_double_rooted_output_paths(
    module: ast.Module,
    recipe_node: _TopFunction,
    bindings: dict[str, str],
) -> None:
    _reject_target_rooted_output_paths(module)
    decorator = recipe_decorator(recipe_node, bindings)
    if decorator is None:
        return
    root = decorator_string_keyword(decorator, "output_root")
    if root in {None, "."} or Path(root).is_absolute():
        return
    normalized_root = root.rstrip("/")
    parameter_defaults = _recipe_string_defaults(recipe_node)
    double_rooted_names = {
        name
        for name, default in parameter_defaults.items()
        if default == normalized_root or default.startswith(normalized_root + "/")
    }
    for call in _tool_calls([*module.body]):
        if resolve_call_target(call.func, bindings) not in _OUTPUT_ROOT_PRIMITIVES or not call.args:
            continue
        argument = call.args[0]
        literal = argument.value if isinstance(argument, ast.Constant) else None
        repeats_root = isinstance(literal, str) and (
            literal == normalized_root or literal.startswith(normalized_root + "/")
        )
        repeats_parameter = any(
            isinstance(node, ast.Name) and node.id in double_rooted_names
            for node in ast.walk(argument)
        )
        if repeats_root or repeats_parameter:
            raise CalcifyError(
                f"output path is rooted twice beneath output_root={root!r} at line "
                f"{call.lineno}; pass a path relative to output_root or declare output_root='.' "
                "when a recipe input supplies the complete relative destination",
                line=call.lineno,
            )


def _reject_target_rooted_output_paths(module: ast.Module) -> None:
    for function in (
        item for item in module.body if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
    ):
        aliases = _path_role_aliases(function)
        for expression in ast.walk(function):
            if not isinstance(expression, ast.BinOp) or not isinstance(expression.op, ast.Div):
                continue
            left = _path_role_flags(expression.left, aliases)
            right = _path_role_flags(expression.right, aliases)
            if left & 1 and right & 2 or left & 2 and right & 1:
                raise CalcifyError(
                    f"output path is incorrectly rooted beneath working_directory at line "
                    f"{expression.lineno}; keep the read target separate and resolve "
                    "output_path relative to the recipe invocation directory/output_root",
                    line=expression.lineno,
                )


def _path_role_aliases(function: _TopFunction) -> dict[str, int]:
    aliases: dict[str, int] = {"working_directory": 1, "output_path": 2}
    assignments = [
        item
        for item in ast.walk(function)
        if isinstance(item, ast.Assign | ast.AnnAssign | ast.NamedExpr) and item.value is not None
    ]
    changed = True
    while changed:
        changed = False
        for assignment in assignments:
            flags = _path_role_flags(cast("ast.AST", assignment.value), aliases)
            targets = (
                list(assignment.targets)
                if isinstance(assignment, ast.Assign)
                else [assignment.target]
            )
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                combined = aliases.get(target.id, 0) | flags
                if combined != aliases.get(target.id, 0):
                    aliases[target.id] = combined
                    changed = True
    return aliases


def _path_role_flags(node: ast.AST, aliases: dict[str, int]) -> int:
    flags = 0
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            flags |= aliases.get(item.id, 0)
        elif isinstance(item, ast.Attribute):
            if item.attr == "working_directory":
                flags |= 1
            elif item.attr == "output_path":
                flags |= 2
    return flags


def _recipe_string_defaults(recipe_node: _TopFunction) -> dict[str, str]:
    positional = [*recipe_node.args.posonlyargs, *recipe_node.args.args]
    values: dict[str, str] = {}
    first_default = len(positional) - len(recipe_node.args.defaults)
    for argument, default in zip(
        positional[first_default:], recipe_node.args.defaults, strict=True
    ):
        if isinstance(default, ast.Constant) and isinstance(default.value, str):
            values[argument.arg] = default.value
    for argument, default in zip(
        recipe_node.args.kwonlyargs, recipe_node.args.kw_defaults, strict=True
    ):
        if isinstance(default, ast.Constant) and isinstance(default.value, str):
            values[argument.arg] = default.value
    return values


def _check_agent_call(
    call: ast.Call,
    declared: tuple[str, ...] | None,
    bindings: dict[str, str],
) -> None:
    target = resolve_call_target(call.func, bindings)
    if target not in primitive_targets("agent"):
        _check_non_agent_call(call, target, bindings)
        return
    role = _called_name(call)
    if role is None:
        raise CalcifyError(f"dynamic agent role at line {call.lineno}", line=call.lineno)
    if declared is None or role not in declared:
        raise CalcifyError(f"undeclared agent role {role} at line {call.lineno}", line=call.lineno)
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    directory = keywords.get("working_directory")
    context = keywords.get("context")
    if (directory is None or isinstance(directory, ast.Constant) and directory.value is None) and (
        context is None or isinstance(context, ast.Constant) and context.value is None
    ):
        raise CalcifyError(
            f"agent call requires working_directory or PreparedContext at line {call.lineno}",
            line=call.lineno,
        )


def _called_name(call: ast.Call) -> str | None:
    if not call.args:
        return None
    value = call.args[0]
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _is_tool_call(node: ast.Call, bindings: dict[str, str]) -> bool:
    return resolve_call_target(node.func, bindings) in TOOL_TARGETS


def _tool_calls(containers: list[ast.AST]) -> list[ast.Call]:
    return [
        node
        for container in containers
        for node in ast.walk(container)
        if isinstance(node, ast.Call)
    ]


def _check_tool_capabilities(
    module: ast.Module,
    recipe_node: _TopFunction,
) -> None:
    bindings = import_bindings(module)
    declared = recipe_capabilities(recipe_node, bindings)
    _check_each(
        _tool_calls([module]),
        lambda call: _check_tool_call(call, declared, bindings),
        code="tool_capabilities",
    )


def _check_tool_call(call: ast.Call, declared: tuple[str, ...], bindings: dict[str, str]) -> None:
    if not _is_tool_call(call, bindings):
        return
    name = _called_name(call)
    if name is None:
        raise CalcifyError(f"dynamic tool capability at line {call.lineno}", line=call.lineno)
    if len(call.args) != 1:
        raise CalcifyError(
            f"tool capability {name} at line {call.lineno} must pass the capability "
            "name as its only positional argument; pass tool inputs as keywords",
            line=call.lineno,
        )
    if name not in declared:
        raise CalcifyError(
            f"undeclared tool capability {name} at line {call.lineno}", line=call.lineno
        )


def verify_source(
    source: str,
    *,
    python_version: str | None = None,
    resource_paths: frozenset[str] | None = None,
    type_check: bool = True,
    lint_path: Path | None = None,
    agent_roles: tuple[str, ...] | None = None,
) -> None:
    """Acceptance gate: statically validate generated Python before writing.

    ``resource_paths`` is the skill package manifest; when known, every literal
    packaged-resource path must name one of its files. ``type_check`` can be
    disabled for isolated checks that do not have the installed runtime API.
    ``lint_path`` is where the recipe will live; when given, its Ruff configuration
    lints ``source``, which must already carry that configuration's formatting.
    ``agent_roles`` are the configured roles; when given, the recipe may declare only them.
    """
    module = parse_module(source, python_version=python_version)
    # Ruff, shape and ty findings need only the text, so one repair sees them together.
    # The semantic tiers below assume a well-shaped, type-correct module.
    _require_static_checks(
        source,
        module,
        lint_path=lint_path,
        python_version=python_version,
        type_check=type_check,
    )
    recipes, steps, _ = collect_structure(module)
    recipe_node = recipes[0]
    checks = _source_checks(module, recipe_node, steps, defined_names(module))
    checks.extend(_bound_checks(module, recipe_node, agent_roles))
    checks.append(
        ("runtime_conformance", lambda: _check_runtime_conformance(module, resource_paths))
    )
    diagnostics, validation_tiers = _run_source_checks(module, recipe_node, steps, checks)
    unique = tuple(dict.fromkeys(diagnostics))
    _record_validation_metrics(module, validation_tiers, "failed" if unique else "accepted")
    if unique:
        raise CalcifyError(
            render_diagnostics(unique),
            diagnostics=unique,
        )


def _bound_checks(
    module: ast.Module,
    recipe_node: _TopFunction,
    agent_roles: tuple[str, ...] | None,
) -> list[tuple[str, Callable[[], None]]]:
    """Check configured roles; source tool pre-approvals are not an exclusive allowlist."""
    checks: list[tuple[str, Callable[[], None]]] = []
    if agent_roles is not None:
        checks.append(
            (
                "configured_roles",
                functools.partial(
                    _check_configured_roles, recipe_node, agent_roles, import_bindings(module)
                ),
            )
        )
    return checks


def _check_configured_roles(
    recipe_node: _TopFunction, agent_roles: tuple[str, ...], bindings: dict[str, str]
) -> None:
    unknown = [
        role for role in recipe_agents(recipe_node, bindings) or () if role not in agent_roles
    ]
    if unknown:
        raise CalcifyError(
            f"@recipe at line {recipe_node.lineno} declares agent roles {unknown} that are "
            f"not configured; configured roles: {sorted(agent_roles)}",
            line=recipe_node.lineno,
        )


_Failure = tuple[str, tuple[Diagnostic, ...]]


def _require_static_checks(
    source: str,
    module: ast.Module,
    *,
    lint_path: Path | None,
    python_version: str | None,
    type_check: bool,
) -> None:
    """Collect Ruff, shape and ty findings plus structural effect checks on one snapshot."""
    failures: list[_Failure] = []
    if lint_path is not None:
        failures.extend(
            _quality_failures(
                "Ruff validation failed",
                lambda: lint_source_quality(source, lint_path, python_version=python_version),
                source,
            )
        )
    shape_failures = _shape_failures(source, module)
    failures.extend(shape_failures)
    if type_check:
        failures.extend(
            _quality_failures(
                "ty validation failed",
                lambda: type_check_source(source, python_version=python_version),
                source,
            )
        )
    if failures:
        if not shape_failures:
            # These checks use syntax and structure, not valid annotations, so the
            # first repair can see their findings too.
            recipes, steps, _ = collect_structure(module)
            checks = [
                (code, check)
                for code, check in _source_checks(module, recipes[0], steps, defined_names(module))
                if code in {"model_derived_values", "effect_contracts"}
            ]
            diagnostics, tiers = _run_source_checks(module, recipes[0], steps, checks)
            failures.extend((item.actual, (item,)) for item in diagnostics)
            _record_validation_metrics(module, tiers, "failed")
        raise CalcifyError(
            "; ".join(message for message, _ in failures),
            diagnostics=tuple(dict.fromkeys(item for _, found in failures for item in found)),
        )


def _quality_failures(
    title: str, run: Callable[[], Sequence[Diagnostic]], source: str
) -> list[_Failure]:
    """Run one tool's check; a tool that cannot run is itself a failure."""
    try:
        findings = run()
    except SourceQualityError as exc:
        return [(str(exc), lint_diagnostics(exc))]
    if not findings:
        return []
    error = quality_error(title, findings, source)
    return [(str(error), lint_diagnostics(error))]


def _shape_failures(source: str, module: ast.Module) -> list[_Failure]:
    """Run each independent shape check, keeping every failure."""
    checks: tuple[Callable[[], None], ...] = (
        lambda: reject_bare_cli_footer(source),
        lambda: _check_portable_imports(module),
        lambda: _check_output_type_declarations(module),
        lambda: _check_recipe_structure(module),
    )
    failures: list[_Failure] = []
    for check in checks:
        try:
            check()
        except CalcifyError as exc:
            found = exc.diagnostics or (diagnostic_from_error("source_shape", exc, {}),)
            failures.append((str(exc), found))
    return list(dict.fromkeys(failures))


def _check_recipe_structure(module: ast.Module) -> None:
    recipes, steps, footers = collect_structure(module)
    recipe_node = require_recipe(module, recipes)
    require_footer(footers, recipe_node.name)


def _source_checks(
    module: ast.Module,
    recipe_node: _TopFunction,
    steps: list[_TopFunction],
    defined_names: set[str],
) -> list[tuple[str, Callable[[], None]]]:
    """Return the ordered, independently diagnosed source validation tiers."""
    return [
        ("step_annotations", lambda: _check_step_annotations(steps, defined_names)),
        ("output_annotations", lambda: _check_output_annotations(module, steps)),
        (
            "agent_declarations",
            lambda: _check_agent_declarations(module, recipe_node),
        ),
        (
            "tool_capabilities",
            lambda: _check_tool_capabilities(module, recipe_node),
        ),
        ("resource_usage", lambda: _check_resource_usage(module)),
        ("model_derived_values", lambda: _check_model_derived_values(module)),
        ("subrecipe_calls", lambda: _check_subrecipe_call_contracts(module)),
        ("sync_primitives", lambda: _check_sync_primitive_awaits(module)),
        ("async_steps", lambda: _check_async_step_calls(module, steps)),
        (
            "effect_contracts",
            lambda: _check_effect_contracts(module, recipe_node),
        ),
    ]


def _check_step_annotations(steps: list[_TopFunction], defined_names: set[str]) -> None:
    annotation_diagnostics: list[Diagnostic] = []
    for step_node in steps:
        try:
            _check_step_annotation(step_node, defined_names)
        except CalcifyError as exc:
            annotation_diagnostics.append(
                Diagnostic(
                    code="step_annotations",
                    phase="source.validation",
                    node=step_node.name,
                    line=step_node.lineno,
                    actual=str(exc),
                    expected="declare a loadable explicit return annotation",
                    safe_fix="replace only the named step function annotation",
                )
            )
    if annotation_diagnostics:
        raise CalcifyError(
            render_diagnostics(annotation_diagnostics),
            diagnostics=tuple(annotation_diagnostics),
        )


def _check_output_annotations(module: ast.Module, steps: list[_TopFunction]) -> None:
    annotation_diagnostics: list[Diagnostic] = []
    bindings = import_bindings(module)
    for step_node in steps:
        if step_node.returns is None:
            continue
        schema_error = output_annotation_error(module, [step_node.returns], bindings)
        if schema_error is None:
            continue
        annotation_diagnostics.append(
            Diagnostic(
                code="output_annotations",
                phase="source.validation",
                node=step_node.name,
                line=step_node.lineno,
                actual=schema_error,
                expected="declare a supported concrete wire output type",
                safe_fix="replace only the named step return annotation",
            )
        )
    if annotation_diagnostics:
        raise CalcifyError(
            render_diagnostics(annotation_diagnostics),
            diagnostics=tuple(annotation_diagnostics),
        )


def _run_source_checks(
    module: ast.Module,
    recipe_node: _TopFunction,
    steps: list[_TopFunction],
    checks: list[tuple[str, Callable[[], None]]],
) -> tuple[list[Diagnostic], list[dict[str, object]]]:
    """Run every tier, collecting diagnostics and timing."""
    diagnostics: list[Diagnostic] = []
    validation_tiers: list[dict[str, object]] = []
    functions_by_name = {node.name: node for node in [*steps, recipe_node]}
    for code, check in checks:
        tier_started = time.perf_counter()
        try:
            check()
        except CalcifyError as exc:
            validation_tiers.append(
                {
                    "name": code,
                    "status": "failed",
                    "wall_seconds": round(time.perf_counter() - tier_started, 6),
                    "diagnostic_count": max(1, len(exc.diagnostics)),
                }
            )
            if exc.diagnostics:
                diagnostics.extend(
                    replace(item, node=item.node or source_node_name(item.line, functions_by_name))
                    for item in exc.diagnostics
                )
                continue
            diagnostics.append(diagnostic_from_error(code, exc, functions_by_name))
        else:
            validation_tiers.append(
                {
                    "name": code,
                    "status": "accepted",
                    "wall_seconds": round(time.perf_counter() - tier_started, 6),
                    "diagnostic_count": 0,
                }
            )
    return diagnostics, validation_tiers


def _record_validation_metrics(
    module: ast.Module,
    tiers: Sequence[dict[str, object]],
    status: str,
) -> None:
    detail = json.dumps(
        {
            "ast_nodes": sum(1 for _ in ast.walk(module)),
            "check_count": len(tiers),
            "outcome": status,
            "tiers": tiers,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    for activity in captured_agent_activities():
        activity.record_event("source.validation", "metrics", detail=detail)


def _check_output_type_declarations(module: ast.Module) -> None:
    bindings = import_bindings(module)
    for statement in module.body:
        if not isinstance(statement, ast.ClassDef):
            continue
        for decorator in statement.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if resolve_call_target(target, bindings) == "dataclasses.dataclass":
                raise CalcifyError(
                    f"open dataclass output; use @output_type at line {statement.lineno}",
                    line=statement.lineno,
                )


_SYNC_PRIMITIVE_TARGETS = primitive_targets(
    "agent", "call_recipe", "call_catalog_recipe", "parallel", "tool"
)


def _check_sync_primitive_awaits(module: ast.Module) -> None:
    bindings = import_bindings(module)

    def check(node: ast.Await) -> None:
        target = _awaited_sync_target(node.value, bindings)
        if target in _SYNC_PRIMITIVE_TARGETS:
            name = target.rsplit(".", 1)[-1]
            raise CalcifyError(
                f"synchronous Calcifier primitive {name}() is awaited at line "
                f"{node.lineno}; call {name}(...) directly",
                line=node.lineno,
            )

    _check_each(
        [node for node in ast.walk(module) if isinstance(node, ast.Await)],
        check,
        code="sync_primitives",
    )


def _check_async_step_calls(module: ast.Module, steps: list[_TopFunction]) -> None:
    async_steps = {step.name for step in steps if isinstance(step, ast.AsyncFunctionDef)}
    if not async_steps:
        return
    parents = {
        child: parent for parent in ast.walk(module) for child in ast.iter_child_nodes(parent)
    }

    def check(call: ast.Call) -> None:
        if not isinstance(call.func, ast.Name) or call.func.id not in async_steps:
            return
        if isinstance(parents.get(call), ast.Await):
            return
        raise CalcifyError(
            f"async @step {call.func.id!r} is called without await at line {call.lineno}; "
            "await it from an async caller",
            line=call.lineno,
        )

    _check_each(
        [node for node in ast.walk(module) if isinstance(node, ast.Call)],
        check,
        code="async_steps",
    )


def _awaited_sync_target(value: ast.expr, bindings: dict[str, str]) -> str:
    call = value
    if isinstance(value, ast.Attribute) and value.attr == "output":
        call = value.value
    if not isinstance(call, ast.Call):
        return ""
    return resolve_call_target(call.func, bindings) or ""


def _check_subrecipe_call_contracts(module: ast.Module) -> None:
    """Keep child calls positional, so the pin collector can read every child path."""
    bindings = import_bindings(module)

    def check(call: ast.Call) -> None:
        target = resolve_call_target(call.func, bindings)
        if target == "calcifier.call_recipe":
            expected = 2
            signature = "call_recipe(path, {input_name: value, ...})"
        elif target == "calcifier.call_catalog_recipe":
            expected = 3
            signature = "call_catalog_recipe(name, catalog, {input_name: value, ...})"
        else:
            return
        if len(call.args) != expected or call.keywords:
            raise CalcifyError(
                f"invalid child recipe call at line {call.lineno}; use exactly "
                f"{signature} without expanded keyword arguments",
                line=call.lineno,
            )

    _check_each(
        [node for node in ast.walk(module) if isinstance(node, ast.Call)],
        check,
        code="subrecipe_calls",
    )


def _check_resource_usage(module: ast.Module) -> None:
    bindings = import_bindings(module)
    parents = {
        child: parent for parent in ast.walk(module) for child in ast.iter_child_nodes(parent)
    }

    def check(call: ast.Call) -> None:
        if resolve_call_target(call.func, bindings) != "calcifier.resource":
            return
        if len(call.args) != 1 or call.keywords:
            raise CalcifyError(
                f"resource() at line {call.lineno} requires one path argument", line=call.lineno
            )
        if isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
            raw_path = call.args[0].value
            try:
                resource_relative_path(raw_path)
            except ValueError as exc:
                raise CalcifyError(
                    f"invalid packaged resource path: {raw_path!r}: {exc}", line=call.lineno
                ) from exc
        parent = parents.get(call)
        while parent is not None and not isinstance(parent, ast.stmt):
            if (isinstance(parent, ast.Attribute) and parent.attr in {"parent", "parents"}) or (
                isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div)
            ):
                raise CalcifyError(
                    f"packaged resource path escapes its manifest at line {call.lineno}; "
                    "use the exact resource() path directly",
                    line=call.lineno,
                )
            parent = parents.get(parent)

    _check_each(
        [node for node in ast.walk(module) if isinstance(node, ast.Call)],
        check,
        code="resource_usage",
    )


def _check_runtime_conformance(module: ast.Module, resource_paths: frozenset[str] | None) -> None:
    """Require packaged-resource paths to use resource() with a manifest file."""
    require_no_findings(
        packaged_resource_errors(module, import_resolver(module), resource_paths),
        code="packaged_resources",
        expected="every packaged file uses resource() with one skill package manifest path",
        safe_fix="call resource() with one manifest file path for each packaged file",
    )


def _check_model_derived_values(module: ast.Module) -> None:
    """Reject agent or child-recipe output reaching a path, tool request, or command."""
    bindings = import_bindings(module)
    functions = [node for node in module.body if isinstance(node, _TopFunction)]
    findings = model_derived_sink_findings(
        functions, lambda target: resolve_call_target(target, bindings)
    )
    require_no_findings(
        findings,
        code="model_derived_values",
        expected="model output never selects a path, tool request, or command",
        safe_fix="derive sink arguments from trusted inputs or deterministic tools",
    )


def _check_step_annotation(statement: _TopFunction, defined_names: set[str]) -> None:
    annotation = statement.returns
    if annotation is None:
        raise CalcifyError(
            f"@step {statement.name!r} requires an explicit return annotation "
            f"at line {statement.lineno}",
            line=statement.lineno,
        )
    unknown = sorted(
        {
            node.id
            for node in ast.walk(annotation)
            if isinstance(node, ast.Name) and node.id not in defined_names
        }
    )
    if unknown:
        label = (
            f"unsupported step annotation `{unknown[0]}`"
            if len(unknown) == 1
            else f"unsupported step annotation names {unknown!r}"
        )
        raise CalcifyError(
            f"{label} at line {annotation.lineno} (no matching definition)", line=annotation.lineno
        )


def _check_portable_imports(module: ast.Module) -> None:
    import calcifier

    allowed = {*sys.stdlib_module_names, "__future__", "calcifier"}

    def check(statement: ast.AST) -> None:
        if isinstance(statement, ast.Import):
            modules = [alias.name for alias in statement.names]
        elif isinstance(statement, ast.ImportFrom):
            if statement.level:
                raise CalcifyError(
                    f"portable recipe cannot use a relative import at line {statement.lineno}",
                    line=statement.lineno,
                )
            modules = [statement.module or ""]
        else:
            return
        private_names = (
            [alias.name for alias in statement.names if alias.name != "*"]
            if isinstance(statement, ast.ImportFrom) and statement.module == "calcifier"
            else []
        )
        if any(name.startswith("calcifier.") for name in modules) or set(private_names) - set(
            calcifier.__all__
        ):
            raise CalcifyError(
                "generated recipes may import only the public Calcifier API "
                f"at line {statement.lineno}",
                line=statement.lineno,
            )
        unavailable = sorted(
            {module for module in modules if module.split(".", 1)[0] not in allowed}
        )
        if unavailable:
            raise CalcifyError(
                f"portable recipe imports unavailable top-level dependencies {unavailable!r} "
                f"at line {statement.lineno}; use the standard library, calcifier primitives, "
                "tools, scripts, or packaged resources",
                line=statement.lineno,
            )

    _check_each(
        [
            lambda: _check_each(list(ast.walk(module)), check, code="source_shape"),
            lambda: _check_public_attributes(module),
        ],
        lambda check: check(),
        code="source_shape",
    )


def _check_public_attributes(module: ast.Module) -> None:
    import calcifier

    bindings = import_bindings(module)

    def check(node: ast.Attribute) -> None:
        target = resolve_call_target(node, bindings)
        if (
            target
            and target.startswith("calcifier.")
            and target.split(".")[1] not in calcifier.__all__
        ):
            raise CalcifyError(
                f"generated recipes may access only the public Calcifier API at line {node.lineno}",
                line=node.lineno,
            )

    _check_each(
        [node for node in ast.walk(module) if isinstance(node, ast.Attribute)],
        check,
        code="source_shape",
    )


def _check_each[T](items: Sequence[T], check: Callable[[T], None], *, code: str) -> None:
    """Collect independent call violations before asking for another repair."""
    findings: list[Diagnostic] = []
    for item in items:
        try:
            check(item)
        except CalcifyError as error:
            findings.extend(error.diagnostics or (diagnostic_from_error(code, error, {}),))
    if findings:
        raise CalcifyError(render_diagnostics(findings), diagnostics=tuple(findings))
