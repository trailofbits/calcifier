"""Keep model-produced values out of filesystem locations, tool requests, and commands.

Agent output and child-recipe output can carry injected instructions, so their
values cannot supply effect arguments. Python may select between trusted values
using a model judgment, as with ordinary if/else control flow. The check reads
only the recipe: ``agent()`` and child-recipe
results are the sources, taint flows through assignments, loops, step
arguments and step results to a fixed point, and a tainted value reaching a
sink is an error.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from .diagnostics import Diagnostic
from .recipe_ast import TOOL_TARGETS, WRITE_TARGETS, primitive_targets

Resolve = Callable[[ast.expr], str | None]
Function = ast.FunctionDef | ast.AsyncFunctionDef
Comprehension = ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp

_SOURCES = primitive_targets("agent", "call_recipe", "call_catalog_recipe")
_TOOLS = TOOL_TARGETS
_COMMANDS = primitive_targets("run_command")
_WRITERS = WRITE_TARGETS
_FILESYSTEM_KEYWORDS = frozenset(
    {
        "cwd",
        "file_path",
        "path",
        "paths",
        "readable_paths",
        "root",
        "working_directory",
        "writable_paths",
    }
)
_ADVICE = "derive it from a trusted recipe input or a deterministic tool result"


def model_derived_sink_findings(
    functions: Sequence[Function], resolve: Resolve
) -> list[Diagnostic]:
    """Return one located finding per sink reached by model output.

    Args:
        functions: Every top-level function of the recipe module.
        resolve: Maps a call's function expression to its canonical import name.

    Returns:
        Findings in source order; empty when no sink receives model output.
    """
    scopes = _scoped_names(functions)
    functions = [
        node for root in functions for node in ast.walk(root) if isinstance(node, Function)
    ]
    flow = _Flow(resolve, {scopes[function]: function for function in functions}, scopes)
    flow.bindings = _callable_bindings(functions, scopes)
    flow.host_paths = {
        node: path
        for function in functions
        for node, path in _final_host_paths(function, flow).items()
    }
    names: set[str] = set()
    changed = True
    while changed:
        changed = False
        before = names.copy()
        for function in functions:
            names = flow.names(function, names)
            changed |= flow.mark_return(function, names)
        changed |= names != before
    return [finding for function in functions for finding in _sink_errors(function, flow, names)]


def _callable_bindings(
    functions: Sequence[Function], scopes: dict[ast.AST, str]
) -> dict[str, list[ast.expr]]:
    bindings: dict[str, list[ast.expr]] = {}
    for function in functions:
        for node in _scope_nodes(function, lambdas=True):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign | ast.NamedExpr) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings.setdefault(scopes[target], []).append(value)
    return bindings


def _scope_nodes(function: Function | ast.Lambda, *, lambdas: bool = False) -> Iterator[ast.AST]:
    """Walk one lexical scope, including declarations but excluding nested bodies."""
    pending: list[ast.AST] = (
        list(function.body) if isinstance(function, Function) else [function.body]
    )
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, Function):
            pending.extend([node.args, *node.decorator_list])
        elif isinstance(node, ast.Lambda) and not lambdas:
            pending.append(node.args)
        elif not isinstance(node, ast.ClassDef):
            pending.extend(ast.iter_child_nodes(node))


def _scoped_names(functions: Sequence[Function]) -> dict[ast.AST, str]:  # noqa: C901 - lexical bindings
    """Resolve local, captured, and lambda names without changing the AST."""
    result: dict[ast.AST, str] = {}
    module = {function.name: f"{function.name}@{id(function)}" for function in functions}
    pending: list[tuple[ast.AST, dict[str, str]]] = [(function, module) for function in functions]
    while pending:
        node, names = pending.pop()
        if isinstance(node, ast.Name):
            result[node] = names.get(node.id, node.id)
        if isinstance(node, Comprehension):
            inner = names.copy()
            for generator in node.generators:
                inner.update(
                    (item.id, f"{item.id}@{id(node)}")
                    for item in ast.walk(generator.target)
                    if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
                )
                pending.extend((child, inner) for child in [generator.target, *generator.ifs])
                pending.append(
                    (generator.iter, names if generator is node.generators[0] else inner)
                )
            values = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
            pending.extend((value, inner) for value in values)
        elif isinstance(node, Function | ast.Lambda):
            inner = names.copy()
            if isinstance(node, Function):
                result[node] = names.get(node.name, f"{node.name}@{id(node)}")
            local = list(_scope_nodes(node))
            targets = {
                target
                for item in local
                if isinstance(item, ast.comprehension)
                for target in ast.walk(item.target)
            }
            external = {
                name
                for item in local
                if isinstance(item, ast.Global | ast.Nonlocal)
                for name in item.names
            }
            bound = {
                item.id
                for item in local
                if isinstance(item, ast.Name)
                and isinstance(item.ctx, ast.Store)
                and item not in targets
            }
            bound |= {item.name for item in local if isinstance(item, Function | ast.ClassDef)}
            inner.update((name, f"{name}@{id(node)}") for name in bound - external)
            inner.update(
                (name, module.get(name, name))
                for item in local
                if isinstance(item, ast.Global)
                for name in item.names
            )
            for parameter in ast.iter_child_nodes(node.args):
                if isinstance(parameter, ast.arg):
                    name = f"{parameter.arg}@{id(node)}"
                    inner[parameter.arg] = result[parameter] = name
            body = node.body if isinstance(node, Function) else [node.body]
            pending.extend(
                (child, names) for child in ast.iter_child_nodes(node) if child not in body
            )
            pending.extend((child, inner) for child in body)
        else:
            pending.extend((child, names) for child in ast.iter_child_nodes(node))
    return result


@dataclass
class _Flow:
    """What is known to carry model output: step parameters, step results, local names."""

    imported: Resolve
    by_name: dict[str, Function]
    scoped_names: dict[ast.AST, str]
    returns: set[str] = field(default_factory=set)
    host_paths: dict[ast.expr, ast.expr] = field(default_factory=dict)
    bindings: dict[str, list[ast.expr]] = field(default_factory=dict)

    def resolve(self, node: ast.expr) -> str | None:
        root = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and self.scoped_names[root] != root.id:
            return None
        return self.imported(node)

    def invocations(self, call: ast.Call) -> Iterator[ast.Call]:
        """Expand callable bindings and partials, retaining every possible target."""
        pending = [(call, frozenset())]
        while pending:
            call, seen = pending.pop()
            target = call.func
            if target in seen:
                continue
            seen = seen | {target}
            if isinstance(target, ast.Name) and self.scoped_names[target] in self.bindings:
                pending.extend(
                    (ast.Call(func=value, args=call.args, keywords=call.keywords), seen)
                    for value in self.bindings[self.scoped_names[target]]
                )
            elif isinstance(target, ast.NamedExpr):
                pending.append(
                    (ast.Call(func=target.value, args=call.args, keywords=call.keywords), seen)
                )
            elif isinstance(target, ast.IfExp):
                pending.extend(
                    (ast.Call(func=value, args=call.args, keywords=call.keywords), seen)
                    for value in (target.body, target.orelse)
                )
            elif (
                isinstance(target, ast.Call)
                and self.resolve(target.func) == "functools.partial"
                and target.args
            ):
                pending.append(
                    (
                        ast.Call(
                            func=target.args[0],
                            args=[*target.args[1:], *call.args],
                            keywords=[*target.keywords, *call.keywords],
                        ),
                        seen,
                    )
                )
            else:
                yield call

    def tainted(  # noqa: C901 - case per expression/source kind
        self, node: ast.AST, names: set[str]
    ) -> bool:
        """Whether ``node`` can evaluate to model output, given the tainted local ``names``."""
        stack = [node]
        while stack:
            item = stack.pop()
            if isinstance(item, ast.expr) and item in self.host_paths:
                stack.append(self.host_paths[item])
                continue
            if isinstance(item, ast.IfExp):
                # The condition selects a branch; it is not part of the returned value.
                # Calls and assignments inside it are checked separately as usual.
                stack.extend((item.body, item.orelse))
                continue
            if isinstance(item, ast.Name) and self.scoped_names[item] in names:
                return True
            if (
                isinstance(item, ast.Attribute)
                and item.attr == "directory"
                and isinstance(item.value, ast.Call)
                and self.resolve(item.value.func) == "calcifier.prepare_context"
            ):
                continue  # This immutable field is host-issued, unlike the prompt.
            if isinstance(item, ast.Call):
                invocations = list(self.invocations(item))
                if len(invocations) != 1 or invocations[0] is not item:
                    stack.extend(invocations)
                    continue
                if isinstance(item.func, ast.Lambda):
                    stack.append(item.func.body)
                    continue
                target = self.resolve(item.func)
                if target in _SOURCES or (
                    isinstance(item.func, ast.Name) and self.scoped_names[item.func] in self.returns
                ):
                    return True
                if target == "calcifier.parallel":
                    worker = next(_parallel_calls(item, self.resolve), None)
                    if worker is not None:
                        stack.append(
                            worker.func.body if isinstance(worker.func, ast.Lambda) else worker
                        )
                        # Callbacks can mutate the returned result.
                        stack.extend(kw.value for kw in item.keywords if kw.arg == "on_result")
                        continue
                path = _written_path(item, target)
                if path is not None:
                    # These host functions return their destination, never their content.
                    stack.append(path)
                    continue
                if isinstance(item.func, ast.Name) and self.scoped_names[item.func] in self.by_name:
                    # A recipe function returning clean data may ignore tainted arguments.
                    continue
            stack.extend(ast.iter_child_nodes(item))
        return False  # gremlin: pardon[equivalent] callers only test truthiness; None == False

    def names(self, function: Function, known: set[str]) -> set[str]:
        """Grow the function's tainted names to a fixed point over its bindings."""
        names = known.copy()
        bindings = list(_bindings(function, self))
        changed = True
        while changed:
            changed = False
            for value, target in bindings:
                if not self.tainted(value, names):
                    continue
                nodes = (target,) if isinstance(target, ast.arg) else ast.walk(target)
                new = {
                    self.scoped_names[item]
                    for item in nodes
                    if isinstance(item, ast.Name | ast.arg)
                } - names
                if new:
                    names |= new
                    changed = True
        return names

    def mark_return(self, function: Function, names: set[str]) -> bool:
        name = self.scoped_names[function]
        if name in self.returns:
            return False
        if not any(
            isinstance(node, ast.Return)
            and node.value is not None
            and self.tainted(node.value, names)
            for node in _scope_nodes(function)
        ):
            return False
        self.returns.add(name)
        return True


def _written_path(call: ast.Call, target: str | None) -> ast.expr | None:
    """Known write results depend on the path argument; ambiguous calls stay conservative."""
    if (
        target not in _WRITERS
        or any(isinstance(argument, ast.Starred) for argument in call.args)
        or any(keyword.arg is None for keyword in call.keywords)
    ):
        return None
    return (
        call.args[0]
        if call.args
        else next((keyword.value for keyword in call.keywords if keyword.arg == "path"), None)
    )


def _final_host_paths(function: Function, flow: _Flow) -> dict[ast.expr, ast.expr]:
    """Resolve final host Path assignments and PreparedContext directory projections.

    Paths are immutable, unlike arbitrary containers. A top-level assignment must
    finish before subsequent local reads, and no later binding (even in a branch
    or loop) may replace it. Keep earlier reads and implicit bindings conservative.
    """
    if any(
        node is not function
        and isinstance(
            node,
            ast.FunctionDef
            | ast.AsyncFunctionDef
            | ast.ClassDef
            | ast.Import
            | ast.ImportFrom
            | ast.Match
            | ast.ExceptHandler
            | ast.Global
            | ast.Nonlocal,
        )
        for node in ast.walk(function)
    ):
        return {}
    last = _last_bindings(function, flow)
    resets: dict[str, tuple[int, ast.expr]] = {}
    directories: dict[str, int] = {}
    for statement in function.body:
        if not isinstance(statement, ast.Assign | ast.AnnAssign):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if not (
            len(targets) == 1
            and isinstance(target := targets[0], ast.Name)
            and isinstance(statement.value, ast.Call)
            and last[flow.scoped_names[target]] == (target.lineno, target.col_offset)
        ):
            continue
        path = _written_path(statement.value, flow.resolve(statement.value.func))
        if path is not None:
            resets[target.id] = (statement.end_lineno or statement.lineno, path)
        if flow.resolve(statement.value.func) == "calcifier.prepare_context":
            directories[target.id] = statement.end_lineno or statement.lineno
    reads = set(_local_reads(function))
    result: dict[ast.expr, ast.expr] = {
        node: resets[node.id][1]
        for node in reads
        if node.id in resets and node.lineno > resets[node.id][0]
    }
    result |= {
        node: ast.Constant(None)
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and node.attr == "directory"
        and isinstance(node.value, ast.Name)
        and node.value in reads
        and node.value.id in directories
        and node.lineno > directories[node.value.id]
    }
    return result


def _local_reads(function: Function) -> Iterator[ast.Name]:
    """Name reads in this scope, excluding deferred and shadowed nested scopes."""
    pending: list[ast.AST] = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            yield node
        pending.extend(ast.iter_child_nodes(node))


def _last_bindings(function: Function, flow: _Flow) -> dict[str, tuple[int, int]]:
    last: dict[str, tuple[int, int]] = {}
    for _, target in _bindings(function, flow):
        for node in ast.walk(target):
            if isinstance(node, ast.Name):
                position = (node.lineno, node.col_offset)
                name = flow.scoped_names[node]
                last[name] = max(last.get(name, (0, 0)), position)
    return last


def _bindings(  # noqa: C901 - assignment and mutation binding forms
    function: Function, flow: _Flow
) -> Iterator[tuple[ast.AST, ast.AST]]:
    """Yield ``(value, target)`` for every binding whose target takes on the value's taint."""
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                yield node.value, target
        elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr) and node.value:
            yield node.value, node.target
        elif isinstance(node, ast.For | ast.AsyncFor | ast.comprehension):
            yield node.iter, node.target
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            yield node.context_expr, node.optional_vars
        elif isinstance(node, ast.Call):
            yield from _call_bindings(node, flow)
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "append",
                "extend",
                "add",
                "update",
                "insert",
                "setdefault",
                "__setitem__",
            }:
                target = node.func.value
                while True:
                    if isinstance(target, ast.Attribute | ast.Subscript):
                        target = target.value
                    elif isinstance(target, ast.Call) and isinstance(target.func, ast.Attribute):
                        target = target.func.value
                    else:
                        break
                values = node.args[1:] if node.func.attr == "insert" else node.args
                if isinstance(target, ast.Name):
                    for value in [*values, *(kw.value for kw in node.keywords)]:
                        yield value, target


def _parallel_calls(call: ast.Call, resolve: Resolve) -> Iterator[ast.Call]:
    """Model parallel dispatch as worker(item), then on_result(item, result)."""
    if resolve(call.func) != "calcifier.parallel":
        return
    arguments = dict(zip(("fn", "items"), call.args, strict=False))
    arguments.update((kw.arg, kw.value) for kw in call.keywords if kw.arg is not None)
    if "fn" not in arguments or "items" not in arguments:
        return
    worker = ast.Call(func=arguments["fn"], args=[arguments["items"]], keywords=[])
    yield worker
    if callback := arguments.get("on_result"):
        result = worker.func.body if isinstance(worker.func, ast.Lambda) else worker
        yield ast.Call(func=callback, args=[arguments["items"], result], keywords=[])


def _call_bindings(call: ast.Call, flow: _Flow) -> Iterator[tuple[ast.AST, ast.AST]]:
    for invocation in flow.invocations(call):
        yield from _invocation_bindings(invocation, flow)


def _invocation_bindings(  # noqa: C901 - positional, keyword, default, and variadic arguments
    call: ast.Call, flow: _Flow
) -> Iterator[tuple[ast.AST, ast.AST]]:
    """Bind calls to recipe functions and callbacks, including their captured defaults."""
    if flow.resolve(call.func) == "functools.partial":
        return
    if flow.resolve(call.func) == "calcifier.parallel":
        invocations = _parallel_calls(call, flow.resolve)
    elif isinstance(call.func, ast.Lambda) or (
        isinstance(call.func, ast.Name) and flow.scoped_names[call.func] in flow.by_name
    ):
        invocations = iter((call,))
    else:
        arguments = [*call.args, *(kw.value for kw in call.keywords)]
        invocations = (
            ast.Call(
                func=argument,
                args=[],
                keywords=[
                    ast.keyword(arg=None, value=other)
                    for other in arguments
                    if other is not argument
                ],
            )
            for argument in arguments
            if isinstance(argument, ast.Lambda | ast.Name)
        )
    for invocation in (bound for value in invocations for bound in flow.invocations(value)):
        callee = (
            invocation.func
            if isinstance(invocation.func, ast.Lambda)
            else flow.by_name.get(flow.scoped_names[invocation.func])
            if isinstance(invocation.func, ast.Name)
            else None
        )
        if callee is None:
            continue
        args = callee.args
        parameters = [*args.posonlyargs, *args.args]
        values = dict(zip(args.kwonlyargs, args.kw_defaults, strict=True))
        values.update(zip(reversed(parameters), reversed(args.defaults), strict=False))
        values.update(zip(parameters, invocation.args, strict=False))
        unpacked = [arg.value for arg in invocation.args if isinstance(arg, ast.Starred)]
        unpacked.extend(kw.value for kw in invocation.keywords if kw.arg is None)
        for parameter in ast.iter_child_nodes(args):
            if isinstance(parameter, ast.arg):
                for value in unpacked:
                    yield value, parameter
        keywords = {kw.arg: kw.value for kw in invocation.keywords}
        for parameter in [*args.args, *args.kwonlyargs]:
            if (value := keywords.pop(parameter.arg, None)) is not None:
                values[parameter] = value
        yield from ((value, parameter) for parameter, value in values.items() if value is not None)
        if args.vararg is not None:
            for value in invocation.args[len(parameters) :]:
                yield value, args.vararg
        if args.kwarg is not None:
            for value in keywords.values():
                yield value, args.kwarg


def _sink_errors(function: Function, flow: _Flow, names: set[str]) -> Iterator[Diagnostic]:
    for call in sorted(
        (node for node in _scope_nodes(function, lambdas=True) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    ):
        consumer = None
        invocations = (
            bound
            for expanded in flow.invocations(call)
            for invocation in (expanded, *_parallel_calls(expanded, flow.resolve))
            for bound in flow.invocations(invocation)
        )
        for invocation in invocations:
            consumer = _tainted_sink(invocation, flow, names)
            if consumer is not None:
                break
        if consumer is not None:
            yield Diagnostic(
                line=call.lineno,
                node=function.name,
                actual=f"model-derived value reaches {consumer} at line {call.lineno} in "
                f"{function.name!r}: agent and child-recipe values cannot supply a path, "
                f"tool request, or command; {_ADVICE}",
            )


def _tainted_sink(call: ast.Call, flow: _Flow, names: set[str]) -> str | None:
    target = flow.resolve(call.func)
    keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
    if target == "calcifier.PreparedContext":
        directories = [
            *call.args[1:2],
            *(arg.value for arg in call.args if isinstance(arg, ast.Starred)),
            *(kw.value for kw in call.keywords if kw.arg in {"directory", None}),
        ]
        if any(flow.tainted(value, names) for value in directories):
            return "an agent filesystem contract"
    if target in _COMMANDS:
        argv = call.args[0] if call.args else keywords.get("argv")
        cwd = keywords.get("working_directory")
        if any(node is not None and flow.tainted(node, names) for node in (argv, cwd)):
            return "a command"
        return None
    if target == "calcifier.prepare_context" and _tainted_keywords(
        call.keywords, {"files", "readable_roots"}, flow, names
    ):
        return "a prepared-context file read"
    if target == "calcifier.ensure_directory" and _tainted_keywords(
        call.keywords, {"output_files", "exclude_roots"}, flow, names
    ):
        return "an output preflight path"
    if target in _TOOLS | {"calcifier.agent"} and any(
        name in _FILESYSTEM_KEYWORDS and flow.tainted(value, names)
        for name, value in keywords.items()
    ):
        return "an agent filesystem contract" if target == "calcifier.agent" else "a tool path"
    return None


def _tainted_keywords(
    keywords: list[ast.keyword],
    sinks: set[str],
    flow: _Flow,
    names: set[str],
) -> bool:
    for keyword in keywords:
        if keyword.arg is not None:
            if keyword.arg in sinks and flow.tainted(keyword.value, names):
                return True
        elif not isinstance(keyword.value, ast.Dict):
            return True  # An opaque mapping can supply any filesystem keyword.
        else:
            for key, value in zip(keyword.value.keys, keyword.value.values, strict=True):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    return True
                if key.value in sinks and flow.tainted(value, names):
                    return True
    return False
