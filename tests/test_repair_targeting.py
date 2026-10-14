"""Source repair: diagnostic attribution and the one repair loop."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from calcifier.calcify import (
    CalcifyError,
    calcify_source,
)
from calcifier.compiler_activity import compiler_request_count
from calcifier.diagnostics import Diagnostic, diagnostic_from_error, lint_diagnostics
from calcifier.source_quality import SourceQualityError, format_and_lint_source
from tests.support.compiler import (
    ScriptedCompiler,
    one_draft,
    source_sequence,
)
from tests.support.recipes import _REPAIR_SKILL

pytestmark = pytest.mark.semantic_offline

_MODULE = ast.parse("def scan_sources():\n    pass\n\n\ndef main():\n    pass\n")
_FUNCTIONS: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
    node.name: node for node in _MODULE.body if isinstance(node, ast.FunctionDef)
}


@pytest.mark.parametrize(
    ("line", "expected"),
    [(1, "scan_sources"), (2, "scan_sources"), (3, None), (5, "main"), (6, "main"), (7, None)],
)
def test_structured_line_attributes_a_diagnostic_to_its_function(line, expected) -> None:
    error = CalcifyError("tool contract failed", line=line)
    diagnostic = diagnostic_from_error("source_contract", error, _FUNCTIONS)
    assert diagnostic.node == expected
    assert diagnostic.line == line
    assert diagnostic.render() == f"line {line}: tool contract failed"


def test_message_words_do_not_set_a_diagnostic_location() -> None:
    error = CalcifyError("tool contract failed in 'scan_sources'")
    diagnostic = diagnostic_from_error("source_contract", error, _FUNCTIONS)
    assert diagnostic.node is None
    assert diagnostic.line is None
    assert diagnostic.render() == "tool contract failed in 'scan_sources'"


_LINT_SOURCE = (
    "value = eval('1')\n"
    "\n"
    "\n"
    "@step\n"
    "def write_segment() -> str:\n"
    "    return eval('2')\n"
    "\n"
    "\n"
    "def helper() -> str:\n"
    "    return 'ok'\n"
    "\n"
    "\n"
    "def helper() -> str:\n"
    "    return eval('3')\n"
)


def test_ruff_failures_carry_line_diagnostics_and_the_linted_source(tmp_path: Path) -> None:
    with pytest.raises(SourceQualityError) as caught:
        format_and_lint_source(
            "def write_segment() -> str:\n    return missing()\n", tmp_path / "recipe.py"
        )

    assert [item.line for item in caught.value.diagnostics] == [2]
    assert caught.value.diagnostics[0].actual.startswith("F821:")
    assert "def write_segment" in caught.value.source


def testlint_diagnostics_name_only_uniquely_defined_enclosing_functions() -> None:
    error = SourceQualityError(
        "Ruff validation failed",
        diagnostics=tuple(Diagnostic(line=line, actual="S307: eval") for line in (1, 4, 6, 14)),
        source=_LINT_SOURCE,
    )

    diagnostics = lint_diagnostics(error)

    assert [(item.code, item.node, item.line) for item in diagnostics] == [
        ("source_quality", None, 1),
        ("source_quality", "write_segment", 4),
        ("source_quality", "write_segment", 6),
        ("source_quality", None, 14),
    ]
    assert all(item.actual == "S307: eval" for item in diagnostics)


def test_unparseable_lint_source_yields_unattributed_diagnostics() -> None:
    error = SourceQualityError(
        "Ruff validation failed",
        diagnostics=(Diagnostic(line=1, actual="E999: syntax"),),
        source="def broken(:\n",
    )

    assert [item.node for item in lint_diagnostics(error)] == [None]


_TYPED_RECIPE = """\
from __future__ import annotations
from calcifier import Budget, agent, output_type, recipe, step

@output_type
class Verdict:
    passed: bool

@step
def inspect(target: str) -> Verdict:
    reply = agent("worker", "review the target", working_directory=target, readable_paths=(target,))
    return Verdict(passed=bool(reply))

@recipe(agents=("worker",), budget=Budget(2, 180))
def writing_style(target: str = ".") -> Verdict:
    return inspect(target=target)

if __name__ == "__main__":
    raise SystemExit(writing_style.cli())
"""
# ty reports the unknown keyword inside `inspect`, but the fix belongs to the output type.
_MISNAMED_RECIPE = _TYPED_RECIPE.replace("    passed: bool\n", "    accepted: bool = False\n")


def test_a_repair_may_fix_code_outside_the_diagnosed_function() -> None:
    compiler = source_sequence(_MISNAMED_RECIPE, _TYPED_RECIPE)

    generated = calcify_source(_REPAIR_SKILL, compiler, agent_roles=("worker",))

    assert "passed: bool" in generated
    assert len(compiler.source_prompts) == 2


def test_a_repair_request_shows_the_source_its_diagnostics_refer_to() -> None:
    compiler = source_sequence(_MISNAMED_RECIPE, _TYPED_RECIPE)
    calcify_source(_REPAIR_SKILL, compiler, agent_roles=("worker",))
    initial, repair = compiler.source_prompts

    request = repair.removeprefix(initial)
    diagnostics = json.loads(
        request.split("Structured diagnostics (JSON array):\n")[1].split("\n")[0]
    )
    shown = json.loads(request.split("Rejected source (JSON string):\n")[1].split("\n")[0])

    (diagnostic,) = diagnostics
    assert diagnostic["node"] == "inspect"
    assert diagnostic["actual"].startswith("ty[unknown-argument]: ")
    assert "`passed`" in diagnostic["actual"]
    assert "Verdict(passed=bool(reply))" in shown.splitlines()[diagnostic["line"] - 1]


def _compile_calls(compiler: ScriptedCompiler) -> tuple[int, str]:
    with pytest.raises(CalcifyError) as caught:
        calcify_source("# synthetic skill\n", compiler, agent_roles=("worker",))
    assert caught.value.activity is not None
    return compiler_request_count(caught.value.activity), str(caught.value)


@pytest.mark.parametrize(
    "compiler_type",
    [
        lambda: one_draft("invalid — Python"),
        lambda: source_sequence(*("pass\n" * number + "invalid —" for number in (1, 2, 3))),
    ],
    ids=["repeated", "varying"],
)
def test_the_same_invalid_response_stops_after_three_source_attempts(
    compiler_type: Any,
) -> None:
    # A syntax error that only moves to another line is still the same failure.
    calls, message = _compile_calls(compiler_type())

    assert calls == 3  # three identical source failures
    assert "made no progress" in message
    assert "latest validation error: invalid RecipeDraft" in message
