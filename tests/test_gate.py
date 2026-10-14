"""Known-answer tests for the compiler's source-validation checks.

Each row is a small generated recipe fed through the source acceptance gate
(``verify_source``) and pins the exact outcome: accepted, or rejected with the
exact diagnostic codes and values.  Rows sit on the boundaries of the source
constructs the checks recognize (comparison shapes, literal values, slice
bounds, annotations, call arity, catalog label maps), so an off-by-one or
inverted condition changes a row.  A few rows observe the same contracts
through public entry points instead: ``calcify_source`` with a stub compiler
(source-generation prompt contracts and verbatim max_calls keywords) and
``render_summary`` (recipe input signatures).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from calcifier.calcify import (
    CalcifyError,
    calcify_source,
    render_summary,
)
from calcifier.gate import verify_source
from calcifier.prompts import build_prompt
from tests.support.compiler import ScriptedCompiler
from tests.support.recipes import _GOOD_RECIPE

Outcome = tuple[tuple[str, str], ...]
ACCEPTED: Outcome = ()


def test_wildcard_runtime_import_reports_its_source_location() -> None:
    source = _GOOD_RECIPE.replace(
        "from calcifier import recipe, step",
        "from calcifier import *\nfrom calcifier import recipe, step",
    )
    with pytest.raises(CalcifyError, match="wildcard calcifier import") as caught:
        verify_source(source)
    assert [(item.code, item.line, item.node) for item in caught.value.diagnostics] == [
        ("source_shape", 2, None)
    ]
    verify_source(_GOOD_RECIPE)


def test_discarded_async_step_cannot_skip_required_work() -> None:
    source = textwrap.dedent(
        """\
        from calcifier import recipe, step

        @step
        async def inspect_scope(target: str) -> bool:
            return bool(target)

        @recipe()
        async def run(target: str) -> bool:
            inspect_scope(target)
            return True

        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )
    # The return types agree even though the step never runs without await.
    with pytest.raises(CalcifyError, match="called without await") as caught:
        verify_source(source)
    assert [(item.code, item.line, item.node) for item in caught.value.diagnostics] == [
        ("async_steps", 9, "run")
    ]
    verify_source(source.replace("    inspect_scope(target)", "    await inspect_scope(target)"))


def _effect_root_recipe(output_path: str, default: str | None = None) -> str:
    parameter = "" if default is None else f'output_path: str = "{default}"'
    return textwrap.dedent(
        f"""\
        from calcifier import recipe, step, write_text

        @step
        def persist(output_path: str) -> bool:
            write_text({output_path}, "report")
            return True

        @recipe(output_root="artifacts")
        def run({parameter}) -> bool:
            return persist(output_path)

        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )


@pytest.mark.parametrize(
    "source",
    [
        _effect_root_recipe("output_path", "artifacts"),
        _effect_root_recipe('"artifacts"', "report.txt"),
    ],
)
def test_output_root_cannot_be_repeated_as_a_default_or_literal(source: str) -> None:
    with pytest.raises(CalcifyError) as caught:
        verify_source(source)
    assert "effect_contracts" in {item.code for item in caught.value.diagnostics}


def test_unrelated_zero_argument_call_does_not_look_like_an_output_path() -> None:
    source = textwrap.dedent(
        """\
        from calcifier import ensure_directory, recipe, step

        @step
        def persist() -> bool:
            value = str()
            ensure_directory("reports")
            return value == ""

        @recipe(output_root="artifacts")
        def run() -> bool:
            return persist()

        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )
    verify_source(source)


def test_output_destination_can_be_routed_beneath_invocation_root(tmp_path, monkeypatch) -> None:
    from calcifier.recipe import load_recipe

    source = textwrap.dedent(
        """\
        from pathlib import Path
        from calcifier import recipe, step, write_text
        @step
        def persist(destination: str) -> str:
            return str(write_text(Path(destination) / "report.txt", "report"))
        @recipe(output_root=".")
        def run(destination: str) -> str:
            return persist(destination)
        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )
    verify_source(source)
    path = tmp_path / "recipe.py"
    path.write_text(source, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    generated = load_recipe(path)
    run = generated.run({"destination": "requested"})
    assert run.ok
    assert isinstance(run.output, str)
    assert Path(run.output) == tmp_path / "requested" / "report.txt"
    assert Path(run.output).read_text(encoding="utf-8") == "report"
    escaped = generated.run({"destination": "../escaped"})
    assert not escaped.ok
    assert escaped.error is not None
    assert "escapes declared output root" in escaped.error


@pytest.mark.parametrize(
    ("primitive", "arguments"),
    [
        ("write_text", '"report.md", "report"'),
        ("write_json", '"report.json", {}'),
        ("ensure_directory", '"reports"'),
        ("verify_artifact", '"report.md"'),
    ],
)
def test_output_root_primitives_accept_recipe_body(primitive: str, arguments: str) -> None:
    source = textwrap.dedent(
        f"""\
        from calcifier import {primitive}, recipe, step

        @step
        def done() -> bool:
            return True

        @recipe(output_root="out")
        def run() -> bool:
            {primitive}({arguments})
            return done()

        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )

    verify_source(source, type_check=False)


def test_builtin_writes_allow_unused_tool_authorization() -> None:
    source = textwrap.dedent(
        """\
        from calcifier import recipe, step, write_text

        @step
        def save() -> bool:
            write_text("report.md", "report")
            return True

        @recipe(output_root="out", capabilities=("Write",))
        def run() -> bool:
            return save()

        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )

    verify_source(source, type_check=False)


@pytest.mark.parametrize("path", ["", "/outside/report.md", "../report.md", "a\\b.md"])
def test_resource_path_must_be_a_contained_relative_file(path: str) -> None:
    source = textwrap.dedent(
        f"""\
        from calcifier import recipe, resource, step

        @step
        def load() -> str:
            return resource({path!r}).read_text()

        @recipe()
        def run() -> str:
            return load()

        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )
    with pytest.raises(CalcifyError) as caught:
        verify_source(source)
    assert "resource_usage" in {item.code for item in caught.value.diagnostics}


def test_resource_call_needs_exactly_one_argument() -> None:
    source = textwrap.dedent(
        """\
        from calcifier import recipe, resource, step

        @step
        def load() -> str:
            return str(resource())

        @recipe()
        def run() -> str:
            return load()

        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )
    with pytest.raises(CalcifyError) as caught:
        verify_source(source, type_check=False)
    assert "resource_usage" in {item.code for item in caught.value.diagnostics}


def test_packaged_file_read_from_a_user_input_has_a_located_repairable_diagnostic() -> None:
    source = textwrap.dedent(
        """\
        from pathlib import Path
        from calcifier import recipe, resource

        @recipe()
        def run(guidelines_directory: str) -> str:
            return (Path(guidelines_directory) / "references/guide.md").read_text(encoding="utf-8")

        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )
    manifest = frozenset({"references/guide.md"})

    with pytest.raises(CalcifyError) as caught:
        verify_source(source, resource_paths=manifest)

    (diagnostic,) = caught.value.diagnostics
    assert (diagnostic.code, diagnostic.node, diagnostic.line) == ("packaged_resources", "run", 6)
    assert "resource('references/guide.md')" in diagnostic.actual
    assert "remove any recipe input for it" in diagnostic.actual
    assert "resource()" in diagnostic.safe_fix
    repaired = source.replace("guidelines_directory: str", "").replace(
        '(Path(guidelines_directory) / "references/guide.md")',
        'resource("references/guide.md")',
    )
    verify_source(repaired, resource_paths=manifest)


def _ty(*messages: str) -> Outcome:
    """ty rejects ill-typed code before any semantic check runs."""
    return tuple(("source_quality", f"ty[{message}") for message in messages)


_GUARD_RECIPE = """
__IMPORTS__
from calcifier import Budget, agent, output_type, recipe, step

@output_type
class Finding:
    __FIELDS__

@output_type
class FindingBatch:
    findings: list[Finding]

@step
def produce(working_directory: str) -> FindingBatch:
    return agent(
        "worker",
        __PROMPT__,
        output=FindingBatch,
        working_directory=working_directory,
        readable_paths=(working_directory,),
    )

@step
def validate(batch: FindingBatch) -> list[Finding]:
    __GUARD__
    return batch.findings

@recipe(agents=("worker",), budget=Budget(2, 60))
def run(working_directory: str = ".") -> list[Finding]:
    batch = produce(working_directory)
    return validate(batch)

if __name__ == "__main__":
    raise SystemExit(run.cli())
"""

_DEFAULT_PROMPT = "Return every supported finding."


def _guard_recipe(
    guard: str,
    *,
    fields: str = "title: str",
    prompt: str = _DEFAULT_PROMPT,
    imports: str = "",
) -> str:
    source = _GUARD_RECIPE.replace("__IMPORTS__", imports).replace("__PROMPT__", repr(prompt))
    source = source.replace("    __FIELDS__", textwrap.indent(fields, "    "))
    return source.replace("    __GUARD__", textwrap.indent(guard, "    "))


# --- word-count guards on model-produced text ------------------------------------------------


# --- collection cardinality guards -------------------------------------------------------------


# --- identity guards on model-produced labels --------------------------------------------------


# --- regex guards -------------------------------------------------------------------------------


# --- deterministic relay steps -----------------------------------------------------------------


# --- instructional templates and persisted outputs ---------------------------------------------

_TEMPLATE_SKILL = """---
name: summarize-records
---
Analyze the visible records and summarize their properties.

## Output Format

Write the summary to `<destination>/summary.md` following the template in
[the summary template](templates/summary.md).
"""


def _error(message: str) -> Outcome:
    return (("plan_cross_check", message),)


# --- optional Markdown prerequisites -----------------------------------------------------------


# --- recipe budgets ---------------------------------------------------------------


# --- catalog children: serialized wire keys ----------------------------------------------------

_JSON_CHILD = """
import json
from calcifier import recipe, step

@step
def render() -> str:
    __BODY__

@recipe()
def transform(__PARAMETERS__) -> str:
    __PREAMBLE__
    value = render()
    return value

if __name__ == "__main__":
    raise SystemExit(transform.cli())
"""

_JSON_CONSUMER = """
import json
from calcifier import call_recipe, recipe, step

@step
def transform() -> str:
    return call_recipe("transform.py", {}).output

@step
def consume(value: str) -> list[str]:
    decoded = json.loads(value)
    return decoded.get("__KEY__", [])

@recipe()
def run() -> list[str]:
    value = transform()
    return consume(value)

if __name__ == "__main__":
    raise SystemExit(run.cli())
"""

_WIRE_SKILL = "Run /transform-child and read its items."
_WIRE_MISMATCH = _error(
    "catalog wire contract mismatch in node 'consume': child 'transform-child' serializes "
    "top-level keys ['normalized_items'], but the consumer reads ['items']"
)
_INLINE_DUMP = 'return json.dumps({"normalized_items": []})'
_WIRE_ROWS = [
    pytest.param(
        'payload = {"normalized_items": []}\nreturn json.dumps(payload)',
        "",
        "pass",
        _WIRE_MISMATCH,
        id="assigned-dictionary",
    ),
    pytest.param(_INLINE_DUMP, "", "pass", _WIRE_MISMATCH, id="inline-dictionary"),
    pytest.param(
        'payload = alias = {"normalized_items": []}\nreturn json.dumps(payload)',
        "",
        "pass",
        ACCEPTED,
        id="chained-assignment-not-traced",
    ),
    pytest.param(
        'extra = {}\nreturn json.dumps({"normalized_items": [], **extra})',
        "",
        "pass",
        _WIRE_MISMATCH,
        id="spread-key-ignored",
    ),
    pytest.param(
        'return json.dumps({1: [], "normalized_items": []})',
        "",
        "pass",
        _WIRE_MISMATCH,
        id="integer-key-ignored",
    ),
    pytest.param(
        _INLINE_DUMP, 'label: str = ""', "label.strip()", _WIRE_MISMATCH, id="method-call-input"
    ),
    pytest.param(
        _INLINE_DUMP, 'label: str = ""', "str(label)", _WIRE_MISMATCH, id="builtin-call-input"
    ),
]


# --- catalog children: semantic label maps ------------------------------------------------------

_CHILD_METADATA = (
    "\n# --- calcifier metadata (skill provenance + sha256) ---\n"
    "_CALCIFIER_SKILL = {'path': 'SKILL.md'}\n"
)
_LEAF_RECIPE = """
from calcifier import recipe, step

@step
def noop(target: str) -> bool:
    return bool(target)

@recipe()
def leaf(target: str = "x") -> bool:
    return noop(target)

if __name__ == "__main__":
    raise SystemExit(leaf.cli())
"""
_COORDINATOR_RECIPE = """
from calcifier import recipe, step

@step
def render() -> str:
    return "- **Processor Class:** fast"

@recipe()
def coordinate() -> str:
    value = render()
    return value

if __name__ == "__main__":
    raise SystemExit(coordinate.cli())
"""
_DISPATCH_RECIPE = """
from calcifier import call_catalog_recipe, call_recipe, parallel, recipe, step
__MAPPING__

@step
def create() -> str:
    return call_recipe("coordinator.py", {}).output

@step
def dispatch(plan: str) -> list[bool]:
    label = plan.split("**Processor Class:**", 1)[1].strip()
    return parallel(
        lambda name: bool(
            call_catalog_recipe(
                name,
                {"analyze-static": "analyze_static.py", "analyze-dynamic": "analyze_dynamic.py"},
                {},
            ).output
        ),
        [__ITEM__],
        max_workers=2,
    )

@recipe()
def run() -> list[bool]:
    plan = create()
    return dispatch(plan)

if __name__ == "__main__":
    raise SystemExit(run.cli())
"""
_DISPATCH_SKILL = "Run /coordinate-real, then dispatch each processor class to /analyze-* recipes."
_LABEL_CATALOG = {
    "coordinate-real": "coordinator.py",
    "analyze-static": "analyze_static.py",
    "analyze-dynamic": "analyze_dynamic.py",
    "report-static": "report_static.py",
}


def _write_label_catalog(base: Path, label_lines: str) -> None:
    (base / "SKILL.md").write_text("## Processor Catalog\n" + label_lines)
    (base / "coordinator.py").write_text(_COORDINATOR_RECIPE + _CHILD_METADATA)
    for name in ("analyze_static.py", "analyze_dynamic.py", "report_static.py"):
        (base / name).write_text(_LEAF_RECIPE)


_FAST = "- `fast` -> `/analyze-static`\n"
_LABEL_MAP = 'WORKER_CATALOG_NAMES = {"fast": "analyze-static"}'
_LABEL_MISMATCH = _error(
    "catalog label contract mismatch in node 'dispatch': child 'coordinate-real' emits "
    "semantic labels requiring exact mapping {'fast': 'analyze-static'}; define and use that "
    "literal label-to-catalog dictionary before dispatch instead of guessing catalog names"
)
_LABEL_ROWS = [
    pytest.param(_FAST, "", "label", _LABEL_MISMATCH, id="unmapped-label"),
    pytest.param(
        _FAST, _LABEL_MAP, 'WORKER_CATALOG_NAMES.get(label, "")', ACCEPTED, id="mapped-label"
    ),
    pytest.param(_FAST + _FAST, "", "label", _LABEL_MISMATCH, id="repeated-consistent-mapping"),
    pytest.param(
        _FAST + "- `fast` -> `/analyze-dynamic`\n", "", "label", ACCEPTED, id="conflicting-mapping"
    ),
    pytest.param(
        "- `analyze-static` -> `/analyze-static`\n", "", "label", ACCEPTED, id="identity-mapping"
    ),
    pytest.param("- `fast` -> `/report-static`\n", "", "label", ACCEPTED, id="unselected-child"),
    pytest.param("- `fast` -> `/analyze-missing`\n", "", "label", ACCEPTED, id="unknown-child"),
]


# --- source-generation prompt contracts ---------------------------------------------------------


def _source_prompt(skill: str, *, python_version: str | None = None) -> str:
    return build_prompt(skill, agent_roles=("worker",), python_version=python_version)


_INSPECT_SKILL = "Inspect the supplied target and return the result."
_BACKSLASH_RULE = "Python 3.11 and earlier forbid every backslash inside an f-string"


@pytest.mark.parametrize(
    ("python_version", "has_backslash_rule"),
    [
        pytest.param("3.10", True),
        pytest.param("3.11", True),
        pytest.param("3.12", False),
        pytest.param("3.13", False),
        pytest.param("4.0", False),
        pytest.param("next", False),
    ],
)
def test_python_compatibility_prompt_known_answers(
    python_version: str, has_backslash_rule: bool
) -> None:
    prompt = _source_prompt(_INSPECT_SKILL, python_version=python_version)

    assert f"emit syntax accepted by Python {python_version}; newer syntax is invalid" in prompt
    assert (_BACKSLASH_RULE in prompt) is has_backslash_rule


# --- static recipe summaries -------------------------------------------------------------------

_SUMMARY_RECIPE = """
from calcifier import recipe, step

@step
def noop(target: str) -> bool:
    return bool(target)

@recipe(version="0.3")
def emitted(__PARAMETERS__) -> bool:
    return noop("x")

if __name__ == "__main__":
    raise SystemExit(emitted.cli())
"""


@pytest.mark.parametrize(
    ("parameters", "inputs"),
    [
        pytest.param("", "(none)", id="no-inputs"),
        pytest.param("target: str", "target (required)", id="required"),
        pytest.param('target: str = "x"', "target='x'", id="default"),
        pytest.param(
            'first: str, second: str = "y", *rest: str, flag: bool, depth: int = 2, **extra: str',
            "first (required), second='y', *rest, flag (required), depth=2, **extra",
            id="every-kind",
        ),
    ],
)
def test_recipe_summary_input_known_answers(tmp_path: Path, parameters: str, inputs: str) -> None:
    path = tmp_path / "recipe.py"
    path.write_text(_SUMMARY_RECIPE.replace("__PARAMETERS__", parameters))

    assert render_summary(path).splitlines()[:4] == [
        "name: emitted",
        "version: 0.3",
        f"inputs: {inputs}",
        "steps: ['noop']",
    ]


# --- dynamic persisted-output components --------------------------------------------------------


# --- child recipe call arity -------------------------------------------------------------------

_CHILD_CALL_RECIPE = """
from calcifier import call_recipe, recipe, step

@step
def noop(target: str) -> bool:
    return bool(__EXPRESSION__)

@recipe(version="0.1")
def emitted(target: str = "x") -> bool:
    return noop(target=target)

if __name__ == "__main__":
    raise SystemExit(emitted.cli())
"""


_CALL_RECIPE_SHAPE = (
    "subrecipe_calls",
    "invalid child recipe call; use exactly call_recipe(path, "
    "{input_name: value, ...}) without expanded keyword arguments",
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        pytest.param('call_recipe("child.py", {"value": target}).output', ACCEPTED, id="exact"),
        pytest.param('call_recipe("child.py", {"value": target})', ACCEPTED, id="result-kept"),
        pytest.param(
            'call_recipe("child.py").output',
            _ty(
                "missing-argument]: No argument provided for required parameter `inputs` "
                "of function `call_recipe`"
            ),
            id="missing-inputs",
        ),
        pytest.param(
            'call_recipe("child.py", {}, {}).output',
            _ty(
                "too-many-positional-arguments]: Too many positional arguments to function "
                "`call_recipe`: expected 2, got 3"
            ),
            id="extra-positional",
        ),
        pytest.param(
            'call_recipe("child.py", {}, parent_trust="trust.json").output',
            (_CALL_RECIPE_SHAPE,),
            id="runtime-keyword",
        ),
    ],
)
def test_child_recipe_call_shape_known_answers(expression: str, expected: Outcome) -> None:
    source = _CHILD_CALL_RECIPE.replace("__EXPRESSION__", expression)

    try:
        verify_source(source)
    except CalcifyError as exc:
        outcome = tuple((item.code, item.actual) for item in exc.diagnostics)
    else:
        outcome = ACCEPTED
    assert outcome == expected


# --- plan-owned max_calls normalization ---------------------------------------------------------


@pytest.mark.parametrize(
    "keyword",
    [
        pytest.param("max_calls=2", id="plan-value"),
        pytest.param("max_calls=3", id="other-value"),
        pytest.param("max_calls=2.0", id="float-value"),
        pytest.param("max_calls=True", id="bool-value"),
        pytest.param("max_calls=LIMIT", id="named-value"),
    ],
)
def test_stray_max_calls_keyword_is_kept_verbatim(keyword: str) -> None:
    # Only the dry run, which needs a destination, reports the unused template argument.
    source = (
        _guard_recipe("")
        .replace(
            "readable_paths=(working_directory,),\n    )",
            f"readable_paths=(working_directory,),\n        {keyword},\n    )",
        )
        .replace("@output_type\nclass Finding", "LIMIT = 2\n\n@output_type\nclass Finding")
    )
    assert keyword in source
    compiler = ScriptedCompiler([source], wrap_sources=True)

    generated = calcify_source(_DEFAULT_PROMPT, compiler, agent_roles=("worker",))

    assert keyword in generated


@pytest.mark.parametrize("validator", ["ruff", "ty"])
def test_validator_launch_failure_prevents_source_acceptance(tmp_path, monkeypatch, validator):
    import calcifier.gate as gate
    from calcifier.source_quality import SourceQualityError

    message = f"{validator} execution failed: validator could not start"

    def unavailable(*args, **kwargs):
        raise SourceQualityError(message)

    monkeypatch.setattr(
        gate,
        "lint_source_quality" if validator == "ruff" else "type_check_source",
        unavailable,
    )
    with pytest.raises(CalcifyError, match=message):
        verify_source(_GOOD_RECIPE, lint_path=tmp_path / "recipe.py")


def test_one_gate_pass_collects_all_independent_role_and_effect_findings():
    source = """from calcifier import Budget, agent, recipe, step
@step
def work() -> str:
    agent("missing-one", "request")
    agent("missing-two", "request")
    open("one.txt", "w")
    open("two.txt", "w")
    return "done"
@recipe(agents=(), budget=Budget(2, 30))
def workflow() -> str:
    return work()
if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""
    with pytest.raises(CalcifyError) as caught:
        verify_source(source)
    findings = caught.value.diagnostics
    roles = [item for item in findings if item.code == "agent_declarations"]
    effects = [item for item in findings if item.code == "effect_contracts"]
    assert [item.line for item in roles] == [4, 5]
    assert [item.line for item in effects] == [6, 7]
    assert all(item.node == "work" for item in [*roles, *effects])
    assert all(f"at line {item.line}" not in item.actual for item in findings)
