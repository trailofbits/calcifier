"""Model-produced values never reach a path, tool request, or command."""

from __future__ import annotations

import ast
import textwrap

import pytest

from calcifier.diagnostics import Diagnostic
from calcifier.taint import model_derived_sink_findings

_NAMES = {
    "agent": "calcifier.agent",
    "call_recipe": "calcifier.call_recipe",
    "call_catalog_recipe": "calcifier.call_catalog_recipe",
    "tool": "calcifier.tool",
    "run_command": "calcifier.run_command",
    "parallel": "calcifier.parallel",
    "write_text": "calcifier.write_text",
    "write_json": "calcifier.write_json",
    "ensure_directory": "calcifier.ensure_directory",
    "prepare_context": "calcifier.prepare_context",
    "PreparedContext": "calcifier.PreparedContext",
    "partial": "functools.partial",
    "functools.partial": "functools.partial",
}


def _resolve(function: ast.expr) -> str | None:
    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
        return _NAMES.get(f"{function.value.id}.{function.attr}")
    return _NAMES.get(function.id) if isinstance(function, ast.Name) else None


def _errors(source: str) -> list[Diagnostic]:
    module = ast.parse(textwrap.dedent(source))
    functions = [
        node for node in module.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    return model_derived_sink_findings(functions, _resolve)


def _line(source: str, text: str) -> int:
    lines = textwrap.dedent(source).splitlines()
    return next(number for number, line in enumerate(lines, 1) if text in line)


@pytest.mark.parametrize("annotation", ["str", "list[str]", "dict[str, str]", "str | None"])
@pytest.mark.parametrize("model_path", [False, True])
def test_parameter_annotations_do_not_taint_type_names(annotation, model_path):
    directory = "message" if model_path else "root"
    errors = _errors(f"""
def forward(root: str, message: {annotation}):
    return agent("reader", "Read: {{message}}", message=message, working_directory=root)
def run(root):
    message = agent("writer", "Describe", working_directory=root)
    forward({directory}, message)
    return run_command(["cat", str(root)], timeout_seconds=1)
""")
    assert bool(errors) == model_path


@pytest.mark.parametrize("keyword", ["files", "readable_roots"])
def test_prepared_context_file_reads_require_trusted_paths(keyword):
    source = f"""
def run(root):
    answer = agent("worker", "answer", working_directory=root)
    return prepare_context("review", {keyword}=(answer,))
"""
    findings = _errors(source)
    assert len(findings) == 1
    assert "a prepared-context file read" in findings[0].actual


@pytest.mark.parametrize("keyword", ["output_files", "exclude_roots"])
def test_output_preflight_requires_trusted_paths(keyword):
    findings = _errors(f"""
def run(root):
    answer = agent("worker", "answer", working_directory=root)
    return ensure_directory(".", {keyword}=(answer,))
""")
    assert len(findings) == 1
    assert "an output preflight path" in findings[0].actual


@pytest.mark.parametrize(
    "call,sink",
    [
        (
            'prepare_context("review", **{"files": (answer,), "readable_roots": (root,)})',
            "a prepared-context file read",
        ),
        (
            'prepare_context("review", **{"files": (root,), "readable_roots": (answer,)})',
            "a prepared-context file read",
        ),
        ('ensure_directory(".", **{"output_files": (answer,)})', "an output preflight path"),
        ('ensure_directory(".", **{"exclude_roots": (answer,)})', "an output preflight path"),
        ('prepare_context("review", **options)', "a prepared-context file read"),
        ('ensure_directory(".", **options)', "an output preflight path"),
    ],
)
def test_unpacking_cannot_hide_context_and_preflight_path_sinks(call, sink):
    findings = _errors(f"""
def run(root, options):
    answer = agent("worker", "answer", working_directory=root)
    return {call}
""")
    assert len(findings) == 1
    assert sink in findings[0].actual


def test_unpacked_context_data_can_remain_tainted_with_trusted_paths():
    assert (
        _errors("""
def run(root):
    answer = agent("worker", "answer", working_directory=root)
    return prepare_context("review", **{
        "files": (root,), "readable_roots": (root,), "prior_results": answer,
    })
""")
        == []
    )


def test_model_text_remains_usable_as_context_data_with_trusted_file_paths():
    assert (
        _errors("""
def run(root, paths):
    answer = agent("worker", "answer", working_directory=root)
    return prepare_context("review", prior_results=answer, files=paths, readable_roots=(root,))
""")
        == []
    )


@pytest.mark.parametrize("field", ["directory", "prompt"])
@pytest.mark.parametrize("declaration", ["context", "context: PreparedContext"])
def test_prepared_context_directory_is_host_issued_but_prompt_stays_tainted(field, declaration):
    source = f"""
def run(root):
    answer = agent("worker", "answer", working_directory=root)
    {declaration} = prepare_context("review", prior_results=answer)
    return agent("reviewer", "review", working_directory=context.{field})
"""
    assert bool(_errors(source)) == (field == "prompt")


@pytest.mark.parametrize(
    "replacement",
    [
        "context = answer",
        "if root:\n        context = answer",
        "context.directory = answer",
    ],
)
def test_reassigned_context_cannot_supply_a_trusted_directory(replacement):
    source = f"""
def run(root):
    answer = agent("worker", "answer", working_directory=root)
    context = prepare_context("review", prior_results=answer)
    {replacement}
    return agent("reviewer", "review", working_directory=context.directory)
"""
    assert _errors(source)


def test_direct_prepared_context_directory_is_trusted():
    assert not _errors("""
def run(root):
    answer = agent("worker", "answer", working_directory=root)
    return agent("reviewer", "review", working_directory=prepare_context(answer).directory)
""")


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("directory,tainted", [("answer", True), ('"."', False)])
def test_constructed_context_directory_is_a_filesystem_sink(keyword, directory, tainted):
    argument = f"directory={directory}" if keyword else directory
    assert (
        bool(
            _errors(f"""
def run():
    answer = agent("worker", "choose", working_directory=".")
    context = PreparedContext(answer, {argument})
    return agent("reviewer", "review", context=context)
""")
        )
        == tainted
    )


@pytest.mark.parametrize(
    "fields,annotation", [("*fields", "tuple[str, str]"), ("**fields", "dict[str, str]")]
)
def test_constructed_context_checks_unpacked_model_values(fields, annotation):
    assert _errors(f"""
def run():
    fields = agent("worker", "choose", working_directory=".", output={annotation})
    context = PreparedContext({fields})
    return agent("reviewer", "review", context=context)
""")


@pytest.mark.parametrize("worker", ["reason", "lambda item: reason(item)"])
@pytest.mark.parametrize(
    "observer", [None, "observe", "lambda item, result: observe(item, result)"]
)
@pytest.mark.parametrize(
    "arguments", ["{worker}, items", "{worker}, items=items", "fn={worker}, items=items"]
)
@pytest.mark.parametrize(
    "result,items,tainted",
    [
        ('agent("worker", "choose", working_directory=".")', "[1]", True),
        ("item", 'agent("worker", "choose", working_directory=".")', True),
        ('"fixed"', 'agent("worker", "choose", working_directory=".")', False),
    ],
)
def test_parallel_results_preserve_worker_taint(
    worker, observer, arguments, result, items, tainted
):
    arguments = arguments.format(worker=worker)
    call = (
        f"observe(0, parallel({arguments})[0])"
        if observer is None
        else f"parallel({arguments}, on_result={observer})"
    )
    errors = _errors(f"""
def reason(item):
    return {result}
def observe(item, result):
    return run_command(["cat", result], timeout_seconds=1)
def run():
    items = {items}
    return {call}
""")
    assert bool(errors) == tainted


def test_parallel_callback_can_taint_mutable_results():
    assert _errors("""
def run():
    results = parallel(
        lambda item: [],
        [1],
        on_result=lambda item, result: result.append(
            agent("worker", "choose", working_directory=".")
        ),
    )
    return run_command(["cat", results[0][0]], timeout_seconds=1)
""")


@pytest.mark.parametrize(
    "binding",
    [
        'callback = lambda item, answer: run_command(["cat", answer], timeout_seconds=1)',
        'callback: object = lambda item, answer: run_command(["cat", answer], timeout_seconds=1)',
        'first = lambda item, answer: run_command(["cat", answer], timeout_seconds=1)\n'
        "    callback = first",
        "callback = observer",
        "callback = partial(observer)",
        "callback = observer if choice else lambda item, answer: None",
        "callback = lambda item, answer: None\n    if choice:\n        callback = observer",
        "callback = observer\n    callback = callback",
    ],
)
@pytest.mark.parametrize(
    "result,tainted",
    [('agent("worker", "choose", working_directory=".")', True), ('"trusted.txt"', False)],
)
def test_assigned_callbacks_preserve_worker_result_taint(binding, result, tainted):
    assert (
        bool(
            _errors(f"""
def reason(item):
    return {result}
def observer(item, answer):
    return run_command(["cat", answer], timeout_seconds=1)
def run(choice):
    {binding}
    return parallel(reason, ["trusted.txt"], on_result=callback)
""")
        )
        == tainted
    )


def test_assigned_callback_parameters_do_not_taint_an_outer_name():
    assert not _errors("""
def reason(item):
    return agent("worker", "choose", working_directory=".")
def run():
    answer = "trusted.txt"
    callback = lambda item, answer: len(answer)
    parallel(reason, [1], on_result=callback)
    return run_command(["cat", answer], timeout_seconds=1)
""")


@pytest.mark.parametrize("binding", ["identity", "lambda value: value"])
@pytest.mark.parametrize("tainted", [False, True])
def test_assigned_workers_propagate_return_taint(binding, tainted):
    value = 'agent("worker", "choose", working_directory=".")' if tainted else '"trusted"'
    assert (
        bool(
            _errors(f"""
def identity(value):
    return value
def run():
    worker = {binding}
    alias = worker
    values = parallel(alias, [{value}])
    return run_command(["cat", values[0]], timeout_seconds=1)
""")
        )
        == tainted
    )


@pytest.mark.parametrize("placement", ["inline", "assigned", "walrus"])
@pytest.mark.parametrize("tainted", [False, True])
def test_source_gate_checks_inline_and_assigned_callbacks_equally(placement, tainted):
    from calcifier.diagnostics import CalcifyError
    from calcifier.gate import verify_source

    value = (
        'agent("worker", "Review {item}", item=item, working_directory=".")' if tainted else "item"
    )
    callback = (
        'lambda item, answer: run_command(["cat", answer], timeout_seconds=10, mutating=False)'
    )
    passed = (
        "callback"
        if placement == "assigned"
        else f"(callback := {callback})"
        if placement == "walrus"
        else callback
    )
    source = f"""from calcifier import Budget, agent, parallel, recipe, run_command, step

@step
def reason(item: str) -> str:
    return {value}

@recipe(agents=("worker",), capabilities=("Bash",), budget=Budget(2, 30))
def workflow() -> list[str]:
    {f"callback = {callback}" if placement == "assigned" else "pass"}
    return parallel(reason, ["trusted.txt"], on_result={passed})

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    if tainted:
        with pytest.raises(CalcifyError, match="model-derived value reaches a command"):
            verify_source(source)
    else:
        verify_source(source)


def test_an_assigned_worker_can_ignore_model_derived_arguments():
    assert not _errors("""
def run():
    answer = agent("worker", "choose", working_directory=".")
    worker = lambda value: "trusted.txt"
    alias = worker
    values = parallel(alias, [answer])
    return run_command(["cat", values[0]], timeout_seconds=1)
""")


def test_assigned_callback_arguments_reach_sinks_through_an_unknown_caller():
    assert _errors("""
def run():
    answer = agent("worker", "choose", working_directory=".")
    callback = lambda result: run_command(["cat", result], timeout_seconds=1)
    invoke(callback=callback, result=answer)
""")


@pytest.mark.parametrize("async_def,await_call", [("def", ""), ("async def", "await ")])
@pytest.mark.parametrize("result,tainted", [("item", True), ("answer", True), ('"trusted"', False)])
def test_nested_workers_and_callbacks_track_parameters_and_captures(
    async_def, await_call, result, tainted
):
    assert (
        bool(
            _errors(f"""
async def run():
    answer = agent("worker", "choose", working_directory=".")
    {async_def} worker(item):
        return {result}
    def checkpoint(item, value):
        run_command(["cat", value], timeout_seconds=1)
    parallel(worker, [answer], on_result=checkpoint)
    return run_command(["cat", {await_call}worker(answer)], timeout_seconds=1)
""")
        )
        == tainted
    )


def test_nested_callback_mutation_taints_the_captured_container():
    assert _errors("""
def run():
    results = []
    def worker(item):
        return agent("worker", "choose", working_directory=".")
    def checkpoint(item, result):
        results.append(result)
    parallel(worker, [1], on_result=checkpoint)
    return run_command(["cat", results[0]], timeout_seconds=1)
""")


@pytest.mark.parametrize(
    "expression",
    [
        "[agent for agent in [1]]",
        "{agent for agent in [1]}",
        "{agent: agent for agent in [1]}",
        "(agent for agent in [1])",
    ],
)
def test_comprehension_binding_does_not_shadow_an_outer_source(expression):
    assert _errors(f"""
def run():
    answer = agent("worker", "choose", working_directory=".")
    ignored = {expression}
    return run_command(["cat", answer], timeout_seconds=1)
""")


@pytest.mark.parametrize("default,tainted", [("answer", True), ('"."', False)])
def test_nested_callback_defaults_follow_their_enclosing_scope(default, tainted):
    assert (
        bool(
            _errors(f"""
def run():
    answer = agent("worker", "choose", working_directory=".")
    def checkpoint(item, result, *, root={default}):
        agent("reader", "read", working_directory=root)
    parallel(str, [1], on_result=checkpoint)
""")
        )
        == tainted
    )


def test_nested_function_names_and_returns_stay_in_their_own_scope():
    assert not _errors("""
def dirty():
    def worker(item):
        return agent("worker", "choose", working_directory=".")
    return parallel(worker, [1])
def clean():
    def worker(item):
        def unused():
            return agent("worker", "choose", working_directory=".")
        return "trusted"
    return parallel(worker, dirty())
def run():
    return run_command(["cat", clean()[0]], timeout_seconds=1)
""")


def test_nested_default_assignments_bind_in_the_enclosing_scope():
    assert _errors("""
def run():
    answer = agent("worker", "choose", working_directory=".")
    def use(value=(saved := answer)):
        return run_command(["cat", saved], timeout_seconds=1)
    return use()
""")


@pytest.mark.parametrize(
    "expression", ["(lambda value=(saved := answer): value)()", "(lambda: (saved := answer))()"]
)
def test_lambda_assignments_do_not_leak_between_enclosing_functions(expression):
    assert not _errors(f"""
def first():
    answer = agent("worker", "choose", working_directory=".")
    return {expression}
def second():
    answer = "clean"
    result = {expression}
    return run_command(["cat", result], timeout_seconds=1)
""")


@pytest.mark.parametrize("factory", ["partial", "functools.partial"])
@pytest.mark.parametrize(
    "result,tainted",
    [
        ("prefix", True),
        ('"trusted"', False),
        ('agent("worker", "choose", working_directory=".")', True),
    ],
)
def test_partial_workers_and_callbacks_preserve_bound_arguments(factory, result, tainted):
    assert (
        bool(
            _errors(f"""
def run():
    answer = agent("worker", "choose", working_directory=".")
    def worker(prefix, item):
        return {result}
    def checkpoint(root, item, result):
        return run_command(["cat", result], working_directory=root, timeout_seconds=1)
    parallel({factory}(worker, answer), [1], on_result={factory}(checkpoint, "."))
""")
        )
        == tainted
    )


@pytest.mark.parametrize(
    "call,tainted",
    [
        ('partial(run_command, ["cat", answer], timeout_seconds=1)()', True),
        ('parallel(partial(agent, "worker", working_directory=answer), ["inspect"])', True),
        (
            'partial(parallel, partial(agent, "worker", working_directory=answer), ["inspect"])()',
            True,
        ),
        ('partial(run_command, ["cat", answer], timeout_seconds=1)', False),
    ],
)
def test_partial_sink_calls_are_checked_when_invoked(call, tainted):
    assert (
        bool(
            _errors(f"""
def run():
    answer = agent("worker", "choose", working_directory=".")
    return {call}
""")
        )
        == tainted
    )


@pytest.mark.parametrize(
    "parameters,result,tainted",
    [
        ("item, fallback=answer", "fallback", True),
        ("item, *, fallback=answer", "fallback", True),
        ("item=answer", "item", False),
        ("item, fallback=answer", '"fixed"', False),
    ],
)
def test_parallel_lambda_defaults_follow_parameter_use(parameters, result, tainted):
    errors = _errors(f"""
def run():
    answer = agent("worker", "choose", working_directory=".")
    results = parallel(lambda {parameters}: {result}, [1])
    return run_command(["cat", results[0]], timeout_seconds=1)
""")
    assert bool(errors) == tainted


def test_parallel_lambda_default_does_not_taint_outer_variable():
    assert not _errors("""
def run():
    answer = agent("worker", "choose", working_directory=".")
    fallback = "trusted.txt"
    parallel(lambda item, fallback=answer: fallback, [1])
    return run_command(["cat", fallback], timeout_seconds=1)
""")


@pytest.mark.parametrize(
    "mutation,tainted",
    [
        ("out.append(item)", True),
        ("out.extend([item])", True),
        ("out.insert(0, item)", True),
        ("out[0] = item", True),
        ("out.count(item)", False),
        ('out.insert(item, "trusted")', False),
    ],
)
def test_parallel_worker_container_contents_carry_taint(mutation, tainted):
    assert (
        bool(
            _errors(f"""
def worker(item):
    out = ["trusted"]
    {mutation}
    return out
def run():
    items = agent("worker", "choose", working_directory=".")
    results = parallel(worker, items)
    return run_command(["cat", results[0][0]], timeout_seconds=1)
""")
        )
        == tainted
    )


@pytest.mark.parametrize(
    "receiver",
    [
        "groups[path]",
        "groups.setdefault(path, [])",
        "groups.get(path)",
        "groups.__getitem__(path)",
        "groups[path].entries",
        "groups.setdefault(str(path), {})[key]",
    ],
)
@pytest.mark.parametrize(
    "operand,tainted", [("path", False), ("key", False), ("str(groups)", True)]
)
def test_container_mutation_does_not_taint_lookup_keys(receiver, operand, tainted):
    errors = _errors(f"""
def run(paths):
    groups = {{}}
    for path in paths:
        key = "details"
        verdict = agent("worker", "choose", working_directory=".")
        {receiver}.append(verdict)
        run_command(["inspect", {operand}], timeout_seconds=1)
""")
    assert bool(errors) == tainted


@pytest.mark.parametrize("keyword_only", ["", "*, "])
@pytest.mark.parametrize("default,tainted", [("answer", True), ('"."', False)])
@pytest.mark.parametrize("callback", [False, True])
def test_parallel_lambda_defaults_reach_sinks(keyword_only, default, tainted, callback):
    parameters = "item, result" if callback else "item"
    worker = (
        f"lambda {parameters}, {keyword_only}root={default}: "
        'agent("reader", "read", working_directory=root)'
    )
    arguments = f"str, [1], on_result={worker}" if callback else f"{worker}, [1]"
    assert (
        bool(
            _errors(f"""
def run():
    answer = agent("worker", "choose", working_directory=".")
    parallel({arguments})
""")
        )
        == tainted
    )


@pytest.mark.parametrize("default,tainted", [("answer", True), ('"."', False)])
def test_nested_lambda_defaults_use_the_enclosing_scope(default, tainted):
    assert (
        bool(
            _errors(f"""
def run():
    answer = agent("worker", "choose", working_directory=".")
    root = answer
    parallel(
        lambda item, root={default}: parallel(
            lambda item, selected=root: agent("reader", "read", working_directory=selected),
            [1],
        ),
        [1],
    )
""")
        )
        == tainted
    )


def test_lambda_analysis_preserves_the_callers_ast():
    module = ast.parse("""
def run():
    return parallel(lambda item, root=".": root, [1])
""")
    before = ast.dump(module, include_attributes=True)
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    assert not model_derived_sink_findings(functions, _resolve)
    assert ast.dump(module, include_attributes=True) == before


@pytest.mark.parametrize("tainted", [False, True])
def test_deep_elif_chain_preserves_taint_without_recursing(tainted):
    value = 'agent("worker", "choose", working_directory=".")' if tainted else '"trusted"'
    branches = "\n".join(
        f"    {'if' if index == 0 else 'elif'} choice == {index}:\n        value = {value}"
        for index in range(400)
    )
    source = f"""def run(choice):
{branches}
    return run_command(["cat", value], timeout_seconds=1)
"""
    assert bool(_errors(source)) == tainted


@pytest.mark.parametrize(
    "expression,tainted",
    [
        ('(lambda run_command: run_command(answer))(lambda value: "clean")', False),
        (
            'run_command(["cat", (lambda identity: identity(answer))(lambda value: value)], '
            "timeout_seconds=1)",
            True,
        ),
        (
            'run_command(["cat", (lambda prepare_context: prepare_context(answer).directory)'
            "(lambda value: SimpleNamespace(directory=value))], timeout_seconds=1)",
            True,
        ),
    ],
)
def test_lambda_shadowing_preserves_call_identity(expression, tainted):
    assert (
        bool(
            _errors(f"""
from types import SimpleNamespace
def identity(value):
    return "clean"
def run():
    answer = agent("worker", "choose", working_directory=".")
    return {expression}
""")
        )
        == tainted
    )


@pytest.mark.parametrize("mutation", ['root.append("clean")', '(root := "clean")'])
@pytest.mark.parametrize("context", [False, True])
def test_lambda_mutation_does_not_invalidate_outer_host_path(mutation, context):
    value = "prepare_context(root)" if context else 'write_text("report.txt", root)'
    path = "root.directory" if context else "root"
    assert not _errors(f"""
def run():
    root = agent("worker", "choose", working_directory=".")
    root = {value}
    parallel(lambda root: {mutation}, [[]])
    return run_command(["cat", {path}], timeout_seconds=1)
""")


@pytest.mark.parametrize(
    "expression,tainted",
    [
        ("parallel(lambda *items: items, answer)[0][0]", True),
        ("parallel(lambda item, *tail: tail, answer)[0]", False),
        ("(lambda root: root)(answer)", True),
        ("(lambda root: root)(root=answer)", True),
        ("(lambda *, root: root)(root=answer)", True),
        ('(lambda root: root)(**{"root": answer})', True),
        ('(lambda first, root: root)(*[".", answer])', True),
        ('(lambda root, **extras: extras)(".", extra=answer)', True),
        ("(lambda root, **extras: extras)(root=answer)", False),
    ],
)
def test_lambda_arguments_preserve_taint(expression, tainted):
    assert (
        bool(
            _errors(f"""
def run():
    answer = agent("worker", "choose", working_directory=".")
    result = {expression}
    return run_command(["cat", result], timeout_seconds=1)
""")
        )
        == tainted
    )


def test_variadic_callback_receives_worker_result_taint():
    assert _errors("""
def run():
    parallel(
        lambda item: agent("worker", "choose", working_directory="."),
        [1],
        on_result=lambda item, *results: run_command(["cat", results[0]], timeout_seconds=1),
    )
""")


def test_unpacked_lambda_arguments_reach_inner_sinks():
    assert _errors("""
def run():
    answer = agent("worker", "choose", working_directory=".")
    (lambda root: run_command(["cat", root], timeout_seconds=1))(**{"root": answer})
""")


@pytest.mark.parametrize(
    "call,tainted",
    [
        (
            'list(map(lambda item, root=root: run_command(["cat", root], timeout_seconds=1), [1]))',
            True,
        ),
        (
            'invoke(callback=lambda *, root=root: run_command(["cat", root], timeout_seconds=1))',
            True,
        ),
        (
            'invoke(callback=lambda root=".": agent("reader", "read", working_directory=root))',
            False,
        ),
    ],
)
def test_passed_lambda_arguments_and_defaults_reach_sinks(call, tainted):
    source = f"""
def run():
    root = agent("worker", "choose", working_directory=".")
    {call}
"""
    assert bool(_errors(source)) == tainted


def test_indirect_lambda_keeps_existing_sink_detection():
    assert _errors("""
def run():
    root = agent("worker", "choose", working_directory=".")
    work = lambda root=root: run_command(["cat", root], timeout_seconds=1)
    work()
""")


_CHILD_PATH = """
def pick(root):
    return call_recipe("child.py", {"root": root}).output

def read(path):
    return tool("Read", path=path)

def run(root):
    chosen = pick(root)
    return read(chosen)
"""

_TUPLE_WRAPPED = """
def propose(root) -> tuple[str, str]:
    answer = agent("worker", "pick two files", working_directory=root, readable_paths=(root,))
    return (answer, root)

def run(root):
    items = propose(root)
    return tool("Read", path=items[0])
"""

_COMMAND_ARGV = """
def run(root):
    sha = agent("worker", "name a commit", working_directory=root, readable_paths=(root,))
    return run_command(["git", "show", sha], timeout_seconds=30, mutating=False)
"""


@pytest.mark.parametrize(
    ("source", "sink", "call"),
    [
        (_CHILD_PATH, "a tool path", 'tool("Read"'),
        (_TUPLE_WRAPPED, "a tool path", 'tool("Read"'),
        (_COMMAND_ARGV, "a command", "run_command("),
    ],
    ids=["child-recipe-path", "tuple-wrapped-agent-output", "agent-output-argv"],
)
def test_model_output_reaching_a_sink_is_reported_at_its_call(
    source: str, sink: str, call: str
) -> None:
    (error,) = _errors(source)

    assert error.actual.startswith(f"model-derived value reaches {sink} ")
    assert error.line == _line(source, call)


@pytest.mark.parametrize(
    ("body", "sink"),
    [
        ('run_command(["ls"], working_directory=picked)', "a command"),
        ('run_command(argv=["cat", picked])', "a command"),
        (
            'agent("worker", "look", working_directory=picked, readable_paths=(root,))',
            "an agent filesystem contract",
        ),
        (
            'agent("worker", "look", working_directory=root, readable_paths=(picked,))',
            "an agent filesystem contract",
        ),
        (
            'agent("worker", "fix", working_directory=root, writable_paths=(picked,))',
            "an agent filesystem contract",
        ),
        ('tool("Glob", root=picked)', "a tool path"),
    ],
)
def test_every_sink_argument_is_checked(body: str, sink: str) -> None:
    source = f"""
def run(root):
    picked = agent("worker", "pick", working_directory=root, readable_paths=(root,))
    return {body}
"""
    (error,) = _errors(source)

    assert f"reaches {sink}" in error.actual
    assert error.line == 4 and error.node == "run"


@pytest.mark.parametrize(
    "binding",
    [
        "for path in picked:\n        target = path",
        "target = [path for path in picked][0]",
        "with open_list(picked) as target:\n        pass",
        "target = root\n    target += picked",
        "if (target := picked):\n        pass",
    ],
    ids=["for", "comprehension", "with", "augmented", "walrus"],
)
def test_taint_flows_through_every_binding_form(binding: str) -> None:
    source = f"""
def run(root):
    picked = agent("worker", "pick", working_directory=root, readable_paths=(root,))
    {binding}
    return tool("Read", path=target)
"""
    assert len(_errors(source)) == 1


def test_a_lambda_given_model_output_by_parallel_receives_it() -> None:
    source = """
def run(root):
    paths = agent("worker", "pick", working_directory=root, readable_paths=(root,))
    return parallel(lambda path: tool("Read", path=path), paths, max_workers=2)
"""
    assert len(_errors(source)) == 1


def test_taint_crosses_steps_in_both_directions_until_stable() -> None:
    source = """
def relay(value):
    return value

def read(path):
    return tool("Read", path=path)

def run(root):
    picked = relay(call_catalog_recipe("child", {"child": "child.py"}, {}).output)
    return read(relay(picked))
"""
    (error,) = _errors(source)

    assert error.node == "read"


def test_a_loop_can_carry_taint_to_a_binding_written_earlier() -> None:
    source = """
def run(root, items):
    previous = root
    for item in items:
        path = previous
        previous = agent("worker", "pick", working_directory=root, readable_paths=(root,))
    return tool("Read", path=path)
"""
    assert len(_errors(source)) == 1


def test_a_caller_defined_before_its_tainting_callee_is_rechecked() -> None:
    source = """
def run(root):
    chosen = pick(root)
    return tool("Read", path=chosen)

def pick(root):
    return agent("worker", "pick", working_directory=root, readable_paths=(root,))
"""
    (error,) = _errors(source)

    assert error.node == "run"


@pytest.mark.parametrize(
    "source",
    [
        # Trusted input chooses the path.
        """
def run(root):
    agent("worker", "summarize", working_directory=root, readable_paths=(root,))
    return tool("Read", path=root)
""",
        # Model output as report data is fine; only authority over locations is not.
        """
def run(root):
    summary = agent("worker", "summarize", working_directory=root, readable_paths=(root,))
    return run_command(["echo", "done"], timeout_seconds=5, mutating=False), summary
""",
        # A step that ignores its tainted argument returns clean data.
        """
def constant(value):
    return "README.md"

def run(root):
    picked = agent("worker", "pick", working_directory=root, readable_paths=(root,))
    return tool("Read", path=constant(picked))
""",
    ],
    ids=["trusted-input", "report-data", "clean-step-result"],
)
def test_values_that_model_output_does_not_reach_are_accepted(source: str) -> None:
    assert _errors(source) == []


@pytest.mark.parametrize(
    "selection",
    [
        'path = "first.txt" if decision else "second.txt"',
        'path = root if decision else "second.txt"',
        'path = "first.txt" if decision else (root if decision else "third.txt")',
        'if decision:\n        path = "first.txt"\n    else:\n        path = "second.txt"',
    ],
)
def test_judgment_selects_trusted_paths_without_supplying_them(selection):
    assert (
        _errors(f"""
def choose(root, decision):
    {selection}
    return path

def run(root):
    decision = agent("worker", "assess", working_directory=root, readable_paths=(root,))
    path = choose(root, decision)
    return run_command(["inspect", path], working_directory=root)
""")
        == []
    )


@pytest.mark.parametrize(
    "selection",
    [
        'picked if flag else "second.txt"',
        '"first.txt" if flag else picked',
        '"first.txt" if flag else (root if flag else picked)',
        '"first.txt" if run_command(["inspect", picked]) else "second.txt"',
    ],
)
def test_conditional_paths_cannot_hide_model_values_or_nested_effects(selection):
    assert (
        len(
            _errors(f"""
def run(root, flag):
    picked = agent("worker", "assess", working_directory=root, readable_paths=(root,))
    path = {selection}
    return run_command(["inspect", path], working_directory=root)
""")
        )
        == 1
    )


@pytest.mark.parametrize(
    "write",
    [
        'write_text("report.txt", answer)',
        "write_text(path=root, text=answer)",
        'write_json("report.json", {"summary": answer})',
        'write_json(value={"summary": answer}, path=root)',
        'write_text("report.txt", answer).resolve().as_posix()',
    ],
)
def test_host_write_returns_its_destination_not_the_model_content(write):
    assert (
        _errors(f"""
def run(root):
    answer = agent("worker", "summarize", working_directory=root, readable_paths=(root,))
    written = {write}
    return run_command(["inspect", str(written)], working_directory=root)
""")
        == []
    )


@pytest.mark.parametrize(
    "write",
    [
        'write_text(answer, "content")',
        'write_text(text="content", path=answer)',
        'write_json(answer, {"summary": "content"})',
        "write_json(value={}, path=answer)",
        'write_text(*[answer, "content"])',
        'write_json(**{"path": answer, "value": {}})',
        'unknown_writer("report.txt", answer)',
        'write_text(answer, "content").resolve().as_posix()',
    ],
)
def test_writing_does_not_launder_model_destinations_or_unknown_calls(write):
    errors = _errors(f"""
def run(root):
    answer = agent("worker", "choose", working_directory=root, readable_paths=(root,))
    written = {write}
    return run_command(["inspect", str(written)], working_directory=root)
""")
    assert len(errors) == 1
    assert "reaches a command" in errors[0].actual


def test_write_content_still_checks_nested_command_sinks():
    errors = _errors("""
def run(root):
    answer = agent("worker", "summarize", working_directory=root, readable_paths=(root,))
    written = write_text("report.txt", run_command(["inspect", answer]).stdout)
    return run_command(["inspect", str(written)])
""")
    assert len(errors) == 1
    assert errors[0].line == 4


@pytest.mark.parametrize(
    "write", ['write_text("report.txt", answer)', 'write_json("report.json", answer)']
)
@pytest.mark.parametrize("handoff", ["path", 'verify_artifact_file(path, role="report").path'])
def test_final_unconditional_host_path_overwrites_an_earlier_model_value(write, handoff):
    assert (
        _errors(f"""
def run(root):
    answer = agent("worker", "reason", working_directory=root, readable_paths=(root,))
    path = answer
    path = {write}
    delivered = {handoff}
    return run_command(["reader", delivered], working_directory=root)
""")
        == []
    )


@pytest.mark.parametrize(
    "body",
    [
        'if flag:\n        path = write_text("report.txt", answer)',
        'path = write_text("report.txt", answer)\n    if flag:\n        path = answer',
        'path = write_text("report.txt", answer)\n    for item in [answer]:\n'
        '        run_command(["reader", path])\n        path = item',
        "path = write_text(path, answer)",
        'path = write_text(*[answer, "content"])',
        'path = unknown_writer("report.txt", answer)',
        'path = write_text("report.txt", run_command(["reader", path]).stdout)',
        'path = write_text("report.txt", answer)\n    def nested(path):\n'
        '        return run_command(["reader", path])\n    nested(answer)',
        'path = write_text("report.txt", answer)\n'
        '    parallel(lambda path: run_command(["reader", path]), [answer])',
        'path = write_text("report.txt", answer)\n    path /= answer',
        'path = write_text("report.txt", answer)\n'
        "    match answer:\n        case path:\n            pass",
        'path = write_text("report.txt", answer)\n    def path():\n        return answer',
        'path = write_text("report.txt", answer)\n    try:\n        raise ValueError(answer)\n'
        '    except ValueError as path:\n        run_command(["reader", str(path)])',
    ],
)
def test_overwrites_do_not_launder_conditional_later_or_nested_model_values(body):
    assert _errors(f"""
def run(root, flag):
    answer = agent("worker", "reason", working_directory=root, readable_paths=(root,))
    path = answer
    {body}
    return run_command(["reader", path], working_directory=root)
""")


def test_earlier_sinks_stay_tainted_after_a_later_host_path_assignment():
    errors = _errors("""
def run(root):
    path = agent("worker", "reason", working_directory=root, readable_paths=(root,))
    run_command(["reader", path])
    path = write_text("report.txt", "text")
    return run_command(["reader", path])
""")
    assert len(errors) == 1
    assert errors[0].line == 4


def test_shared_paths_are_not_treated_as_final_local_overwrites():
    assert _errors("""
def run(root):
    global path
    path = agent("worker", "reason", working_directory=root, readable_paths=(root,))
    path = write_text("report.txt", "text")
    change_shared_state()
    return run_command(["reader", path])
""")
