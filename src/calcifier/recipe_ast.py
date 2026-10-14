"""Shared AST helpers for names bound by recipe imports."""

from __future__ import annotations

import ast
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Never

from .diagnostics import CalcifyError, Diagnostic
from .outputs import BUILTIN_OUTPUT_TYPES
from .source_quality import python_feature_version

type CallResolver = Callable[[ast.expr], str | None]


def primitive_targets(*names: str) -> frozenset[str]:
    """Qualified public and implementation names for the requested runtime primitives."""
    modules = {
        "agent": "agents",
        "tool": "tools",
        "parallel": "recipe",
        "call_recipe": "composition",
        "call_catalog_recipe": "composition",
        "write_text": "mutations",
        "write_json": "mutations",
        "ensure_directory": "mutations",
        "verify_artifact": "mutations",
        "run_command": "mutations",
        "record_side_effect": "side_effects",
    }
    return frozenset(
        target
        for name in names
        for target in (f"calcifier.{name}", f"calcifier.{modules[name]}.{name}")
    )


TOOL_TARGETS = primitive_targets("tool")
CHILD_TARGETS = primitive_targets("call_recipe", "call_catalog_recipe")
WRITE_TARGETS = primitive_targets("write_text", "write_json")
MUTATION_TARGETS = WRITE_TARGETS | primitive_targets("ensure_directory")


def import_bindings(module: ast.Module) -> dict[str, str]:
    """Map module-level imported names to their qualified targets."""
    bindings: dict[str, str] = {}
    for statement in module.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                local_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                bindings[local_name] = alias.name if alias.asname else local_name
        elif isinstance(statement, ast.ImportFrom):
            prefix = statement.module or ""
            if statement.module == "calcifier" and any(
                alias.name == "*" for alias in statement.names
            ):
                raise CalcifyError(
                    f"wildcard calcifier import at line {statement.lineno} cannot be verified",
                    line=statement.lineno,
                )
            for alias in statement.names:
                bindings[alias.asname or alias.name] = ".".join(
                    part for part in (prefix, alias.name) if part
                )
    return bindings


def resolve_call_target(target: ast.expr, bindings: dict[str, str]) -> str | None:
    """Resolve a name or attribute expression through module imports."""
    if isinstance(target, ast.Name):
        return bindings.get(target.id)
    if isinstance(target, ast.Attribute):
        parent = resolve_call_target(target.value, bindings)
        return None if parent is None else f"{parent}.{target.attr}"
    return None


def subrecipe_calls(module: ast.Module, imports: dict[str, str]) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(module)
        if isinstance(call, ast.Call) and resolve_call_target(call.func, imports) in CHILD_TARGETS
    ]


def called_subrecipes(
    path: Path,
    module: ast.Module,
    calls: list[ast.Call],
    imports: dict[str, str],
) -> tuple[list[Diagnostic], set[Path]]:
    """Return source findings and every child the calls statically name."""
    diagnostics: list[Diagnostic] = []
    called_children: set[Path] = set()
    for call in calls:
        if resolve_call_target(call.func, imports) == "calcifier.call_catalog_recipe":
            children, literal = _catalog_children(path, module, call)
            message = "catalog subrecipe mapping must be literal"
        else:
            children, literal = _literal_child(path, call)
            message = "subrecipe path must be literal"
        # Children named before a non-literal catalog value still count toward topology.
        called_children |= children
        if not literal:
            diagnostics.append(Diagnostic(line=call.lineno, actual=message))
    return diagnostics, called_children


def _catalog_children(
    path: Path,
    module: ast.Module,
    call: ast.Call,
) -> tuple[set[Path], bool]:
    mapping = call.args[1] if len(call.args) > 1 else None
    if isinstance(mapping, ast.Name):
        assignments = [
            item.value
            for item in ast.walk(module)
            if (
                isinstance(item, ast.Assign)
                and len(item.targets) == 1
                and isinstance(item.targets[0], ast.Name)
                and item.targets[0].id == mapping.id
            )
            or (
                isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and item.target.id == mapping.id
            )
        ]
        mapping = assignments[0] if len(assignments) == 1 else None
    if not isinstance(mapping, ast.Dict):
        return set(), False
    children: set[Path] = set()
    for value in mapping.values:
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            return children, False
        children.add(_resolve_child(path, value.value))
    return children, True


def _literal_child(path: Path, call: ast.Call) -> tuple[set[Path], bool]:
    if (
        not call.args
        or not isinstance(call.args[0], ast.Constant)
        or not isinstance(call.args[0].value, str)
    ):
        return set(), False
    return {_resolve_child(path, call.args[0].value)}, True


def _resolve_child(path: Path, value: str) -> Path:
    raw_path = Path(value)
    return raw_path.resolve() if raw_path.is_absolute() else (path.parent / raw_path).resolve()


def import_resolver(module: ast.Module) -> CallResolver:
    """Return a resolver for calls in one module."""
    bindings = import_bindings(module)
    return lambda target: resolve_call_target(target, bindings)


def tool_provider_capabilities(module: ast.Module, declared: tuple[str, ...]) -> tuple[str, ...]:
    """Require providers for possible tool calls, keeping indirect dispatch conservative."""
    resolve = import_resolver(module)
    calls = [node for node in ast.walk(module) if isinstance(node, ast.Call)]
    attribute_roots = {node.value for node in ast.walk(module) if isinstance(node, ast.Attribute)}
    modules = {"calcifier", "calcifier.tools"} | {
        alias.name
        for statement in module.body
        if isinstance(statement, ast.Import)
        for alias in statement.names
        if alias.name.startswith("calcifier.")
    }
    if any(
        isinstance(node, ast.Name | ast.Attribute)
        and resolve(node) in modules
        and node not in attribute_roots
        for node in ast.walk(module)
    ):
        return declared
    direct_tools = {call.func for call in calls if resolve(call.func) in TOOL_TARGETS}
    if any(
        isinstance(node, ast.Name | ast.Attribute)
        and resolve(node) in TOOL_TARGETS
        and node not in direct_tools
        for node in ast.walk(module)
    ):
        return declared
    used: set[str] = set()
    for call in calls:
        if (
            isinstance(call.func, ast.Name)
            and call.func.id == "getattr"
            and call.args
            and (resolve(call.args[0]) or "").startswith("calcifier")
        ):
            return declared
        if resolve(call.func) in TOOL_TARGETS:
            if not call.args or not isinstance(call.args[0], ast.Constant):
                return declared
            if isinstance(call.args[0].value, str):
                used.add(call.args[0].value)
    return tuple(name for name in declared if name in used)


_BARE_CLI_RE = re.compile(
    r"^[ \t]*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:\s*"
    r"[A-Za-z_][A-Za-z0-9_]*\.cli\(\s*\)\s*$",
    re.MULTILINE,
)

_OUTPUT_PLACEHOLDER_RE = re.compile(r"(?:<[^<>]+>|\{[^{}]+\}|\$\{[^{}]+\})")

_TopFunction = ast.FunctionDef | ast.AsyncFunctionDef


def parse_module(source: str, *, python_version: str | None = None) -> ast.Module:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SyntaxWarning)
            module = ast.parse(source, feature_version=python_feature_version(python_version))
            compile(module, "<generated-recipe>", "exec")
    except SyntaxError as exc:
        # gremlin: pardon[untestable] SyntaxError with lineno=None is not portably synthesizable
        line = exc.lineno or 1
        # gremlin: pardon[untestable] same for offset=None — not portably synthesizable
        column = exc.offset or 1
        target = "" if python_version is None else f" {python_version}"
        raise CalcifyError(
            f"agent output was not valid Python{target} at line {line}, column {column}: {exc.msg}",
            line=line,
        ) from exc
    syntax_warnings = [item for item in caught if issubclass(item.category, SyntaxWarning)]
    if syntax_warnings:
        warning = syntax_warnings[0]
        line = warning.lineno or 1
        raise CalcifyError(
            f"agent output emitted SyntaxWarning at line {line}: {warning.message}", line=line
        )
    return module


def reject_bare_cli_footer(source: str) -> None:
    if match := _BARE_CLI_RE.search(source):
        line = source.count("\n", 0, match.start()) + 1
        raise CalcifyError(
            "agent output ends with `<name>.cli()` instead of "
            "`raise SystemExit(<name>.cli())`; the bare form discards the "
            f"non-zero return code on failure. (line {line})",
            line=line,
        )


def decorator_name(decorator: ast.expr, bindings: dict[str, str] | None = None) -> str | None:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if bindings is not None:
        resolved = resolve_call_target(target, bindings)
        return (
            resolved.removeprefix("calcifier.")
            if resolved and resolved.startswith("calcifier.")
            else None
        )
    return target.id if isinstance(target, ast.Name) else None


def function_role(statement: _TopFunction, bindings: dict[str, str]) -> str:
    decorator_names = {
        decorator_name(decorator, bindings) for decorator in statement.decorator_list
    }
    if "recipe" in decorator_names:
        return "recipe"
    if "step" in decorator_names:
        return "step"
    return "allowed"


def _is_module_docstring(statement: ast.stmt, index: int) -> bool:
    return (
        index == 0
        and isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def assignment_name(statement: ast.stmt) -> str | None:
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
    elif isinstance(statement, ast.AnnAssign):
        target = statement.target
    else:
        return None
    return target.id if isinstance(target, ast.Name) else None


def _is_main_check(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _system_exit_cli_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Raise) or not isinstance(statement.exc, ast.Call):
        return None
    call = statement.exc
    if not isinstance(call.func, ast.Name) or call.func.id != "SystemExit":
        return None
    if len(call.args) != 1 or call.keywords:
        return None
    inner = call.args[0]
    if not isinstance(inner, ast.Call) or inner.args or inner.keywords:
        return None
    func = inner.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id if func.attr == "cli" else None
    return None


def _footer_target(statement: ast.If) -> str | None:
    if not _is_main_check(statement.test):
        return None
    if len(statement.body) != 1 or statement.orelse:
        return None
    return _system_exit_cli_target(statement.body[0])


def _raise_forbidden_statement(statement: ast.stmt) -> Never:
    raise CalcifyError(
        f"forbidden top-level statement at line {statement.lineno}: {type(statement).__name__}",
        line=statement.lineno,
    )


def _imported_names(statement: ast.Import | ast.ImportFrom) -> set[str]:
    names = set()
    for alias in statement.names:
        if alias.asname:
            names.add(alias.asname)
        elif isinstance(statement, ast.Import):
            names.add(alias.name.split(".", maxsplit=1)[0])
        else:
            names.add(alias.name)
    return names


def defined_names(module: ast.Module) -> set[str]:
    names = set(BUILTIN_OUTPUT_TYPES)
    for statement in module.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            names.update(_imported_names(statement))
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(statement.name)
        if name := assignment_name(statement):
            names.add(name)
    return names


def recipe_capabilities(
    statement: _TopFunction, bindings: dict[str, str] | None = None
) -> tuple[str, ...]:
    for decorator in statement.decorator_list:
        if decorator_name(decorator, bindings) == "recipe" and isinstance(decorator, ast.Call):
            return _capabilities_kwarg(decorator)
    return ()


def _capabilities_kwarg(decorator: ast.Call) -> tuple[str, ...]:
    for keyword in decorator.keywords:
        if keyword.arg == "capabilities":
            return literal_string_sequence(keyword.value, keyword.value.lineno, "capabilities")
    return ()


def literal_string_sequence(
    value: ast.expr,
    lineno: int,
    label: str,
) -> tuple[str, ...]:
    if not isinstance(value, ast.Tuple | ast.List):
        raise CalcifyError(
            f"recipe {label} must be a literal tuple/list at line {lineno}", line=lineno
        )
    out: list[str] = []
    for element in value.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            raise CalcifyError(
                f"recipe {label} must contain string literals at line {lineno}", line=lineno
            )
        out.append(element.value)
    return tuple(out)


def recipe_agents(
    statement: _TopFunction, bindings: dict[str, str] | None = None
) -> tuple[str, ...] | None:
    for decorator in statement.decorator_list:
        if decorator_name(decorator, bindings) != "recipe" or not isinstance(decorator, ast.Call):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "agents":
                return literal_string_sequence(keyword.value, keyword.value.lineno, "agents")
    return None


def recipe_decorator(
    statement: _TopFunction, bindings: dict[str, str] | None = None
) -> ast.Call | None:
    return next(
        (
            decorator
            for decorator in statement.decorator_list
            if decorator_name(decorator, bindings) == "recipe" and isinstance(decorator, ast.Call)
        ),
        None,
    )


def references_calcifier(target: ast.expr, bindings: dict[str, str]) -> bool:
    return any(
        isinstance(node, ast.Name)
        and bindings.get(node.id, "").split(".", maxsplit=1)[0] == "calcifier"
        for node in ast.walk(target)
    )


def _is_verified_resource_read(target: ast.expr, bindings: dict[str, str]) -> bool:
    return (
        isinstance(target, ast.Attribute)
        and target.attr in {"read_bytes", "read_text"}
        and isinstance(target.value, ast.Call)
        and resolve_call_target(target.value.func, bindings) == "calcifier.resource"
    )


def _classify_statement(
    statement: ast.stmt, index: int, bindings: dict[str, str]
) -> tuple[str, str | None]:
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return ("allowed", None)
    if isinstance(statement, ast.ClassDef):
        return ("allowed", None)
    if _is_module_docstring(statement, index):
        return ("allowed", None)
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return (function_role(statement, bindings), statement.name)
    if assignment_name(statement):
        return ("allowed", None)
    if isinstance(statement, ast.If) and (target := _footer_target(statement)):
        return ("footer", target)
    _raise_forbidden_statement(statement)


def collect_structure(
    module: ast.Module,
) -> tuple[list[_TopFunction], list[_TopFunction], list[tuple[int, str]]]:
    recipes: list[_TopFunction] = []
    steps: list[_TopFunction] = []
    footers: list[tuple[int, str]] = []
    bindings = import_bindings(module)
    for index, statement in enumerate(module.body):
        kind, name = _classify_statement(statement, index, bindings)
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if kind == "recipe":
                recipes.append(statement)
            if kind == "step":
                steps.append(statement)
        if kind == "footer" and name is not None:
            footers.append((statement.lineno, name))
    return recipes, steps, footers


def _module_line(module: ast.Module) -> int:
    return module.body[0].lineno if module.body else 1


def require_recipe(module: ast.Module, recipes: list[_TopFunction]) -> _TopFunction:
    if not recipes:
        raise CalcifyError(f"no @recipe at line {_module_line(module)}", line=_module_line(module))
    if len(recipes) > 1:
        raise CalcifyError(f"multiple @recipe at line {recipes[1].lineno}", line=recipes[1].lineno)
    return recipes[0]


def require_footer(footers: list[tuple[int, str]], recipe_name: str) -> None:
    if len(footers) != 1:
        # gremlin: pardon[equivalent] outer guard excludes len==1; >/>= diverge on unreachable input
        line = footers[1][0] if len(footers) > 1 else 1
        raise CalcifyError(f"no CLI footer at line {line}", line=line)
    footer_line, footer_name = footers[0]
    if footer_name != recipe_name:
        raise CalcifyError(
            f"no CLI footer at line {footer_line}: footer calls `{footer_name}.cli()` "
            f"but the @recipe function is `{recipe_name}`",
            line=footer_line,
        )


def decorator_keyword_call(
    decorator: ast.Call,
    keyword_name: str,
    bindings: dict[str, str],
) -> tuple[int, int] | None:
    keyword = next((item for item in decorator.keywords if item.arg == keyword_name), None)
    if keyword is None or not isinstance(keyword.value, ast.Call):
        return None
    if resolve_call_target(keyword.value.func, bindings) != "calcifier.Budget":
        return None
    values: list[object] = [
        argument.value if isinstance(argument, ast.Constant) else None
        for argument in keyword.value.args
    ]
    named = {
        item.arg: item.value.value
        for item in keyword.value.keywords
        if item.arg is not None and isinstance(item.value, ast.Constant)
    }
    max_calls = values[0] if values else named.get("max_agent_calls")
    max_wall = values[1] if len(values) > 1 else named.get("max_wall_seconds")
    if isinstance(max_calls, bool) or not isinstance(max_calls, int):
        return None
    if isinstance(max_wall, bool) or not isinstance(max_wall, int):
        return None
    return max_calls, max_wall


def decorator_string_keyword(decorator: ast.Call, keyword_name: str) -> str | None:
    keyword = next((item for item in decorator.keywords if item.arg == keyword_name), None)
    if keyword is None:
        return None
    if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
        return keyword.value.value
    return None


def recipe_version(recipe_node: _TopFunction, bindings: dict[str, str] | None = None) -> str:
    for decorator in recipe_node.decorator_list:
        if decorator_name(decorator, bindings) != "recipe" or not isinstance(decorator, ast.Call):
            continue
        for keyword in decorator.keywords:
            if (
                keyword.arg == "version"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                return keyword.value.value
    return "0.1"


def signature_for_node(recipe_node: _TopFunction) -> str:
    arguments = recipe_node.args
    positional = [*arguments.posonlyargs, *arguments.args]
    required_count = len(positional) - len(arguments.defaults)
    parts = [f"{arg.arg} (required)" for arg in positional[:required_count]]
    parts.extend(
        f"{arg.arg}={ast.unparse(default)}"
        for arg, default in zip(positional[required_count:], arguments.defaults, strict=True)
    )
    if arguments.vararg is not None:
        parts.append(f"*{arguments.vararg.arg}")
    for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
        parts.append(
            f"{arg.arg} (required)" if default is None else f"{arg.arg}={ast.unparse(default)}"
        )
    if arguments.kwarg is not None:
        parts.append(f"**{arguments.kwarg.arg}")
    return ", ".join(parts) or "(none)"
