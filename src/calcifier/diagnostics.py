"""Structured diagnostics and errors from the recipe compiler."""

from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from .exceptions import CalcifierError, ErrorFamily

if TYPE_CHECKING:
    from .agent_events import AgentActivity
    from .source_quality import SourceQualityError

_TopFunction = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class Diagnostic:
    """Stable repair-facing source diagnostic."""

    code: str = ""
    phase: str = ""
    node: str | None = None
    line: int | None = None
    actual: str = ""
    expected: str = ""
    safe_fix: str = ""

    def __post_init__(self) -> None:
        if self.line is not None:
            actual = re.sub(rf" at line {self.line}\b|\(line {self.line}\)", "", self.actual)
            object.__setattr__(self, "actual", actual.strip())

    def render(self) -> str:
        return f"line {self.line}: {self.actual}" if self.line is not None else self.actual


class _RenderableDiagnostic(Protocol):
    def render(self) -> str: ...


def render_diagnostics(diagnostics: Sequence[_RenderableDiagnostic]) -> str:
    """Render the first twenty findings and count the rest."""
    rendered = "; ".join(item.render() for item in diagnostics[:20])
    suffix = "" if len(diagnostics) <= 20 else f"; {len(diagnostics) - 20} more"
    return f"{rendered}{suffix}"


class CalcifyError(CalcifierError):
    family = ErrorFamily.COMPILATION

    def __init__(
        self,
        message: str,
        *,
        activity: AgentActivity | None = None,
        diagnostics: tuple[Diagnostic, ...] = (),
        source: str | None = None,
        draft: str | None = None,
        line: int | None = None,
        node: str | None = None,
    ) -> None:
        super().__init__(message)
        self.activity = activity
        self.diagnostics = diagnostics
        # The exact source the failed check read, so diagnostic lines match it.
        self.source = source
        # The model's raw response, when the failure is about what it keeps returning.
        self.draft = draft
        self.line = line
        self.node = node


class CalcifierDefectError(CalcifyError):
    """A deterministic compiler-stage failure that model retries cannot fix."""


class SkillDeclarationError(CalcifyError):
    """The SKILL.md must declare something a person decides; no repair can supply it."""


def source_node_name(line: int | None, functions_by_name: dict[str, _TopFunction]) -> str | None:
    """Locate a finding in its enclosing source function."""
    return next(
        (
            function.name
            for function in functions_by_name.values()
            if line is not None
            and function.lineno <= line <= (function.end_lineno or function.lineno)
        ),
        None,
    )


def diagnostic_from_error(
    code: str,
    error: CalcifyError,
    functions_by_name: dict[str, _TopFunction],
) -> Diagnostic:
    """Build a repair diagnostic from an error's structured location."""
    node_name = error.node or source_node_name(error.line, functions_by_name)
    return Diagnostic(
        code=code,
        phase="source.validation",
        node=node_name,
        line=error.line,
        actual=str(error),
        expected=f"satisfy the deterministic {code} contract",
        safe_fix="repair only this contract and preserve all validated structure",
    )


def lint_diagnostics(error: SourceQualityError) -> tuple[Diagnostic, ...]:
    """Attribute Ruff or ty findings to uniquely named top-level functions."""
    try:
        module = ast.parse(error.source)
    except SyntaxError:
        module = ast.Module(body=[], type_ignores=[])
    functions = [node for node in module.body if isinstance(node, _TopFunction)]
    names = Counter(node.name for node in functions)
    spans = [
        (
            min([node.lineno, *(item.lineno for item in node.decorator_list)]),
            node.end_lineno or node.lineno,
            node.name,
        )
        for node in functions
        if names[node.name] == 1
    ]
    return tuple(
        Diagnostic(
            "source_quality",
            "lint",
            next(
                (
                    name
                    for start, end, name in spans
                    if item.line is not None and start <= item.line <= end
                ),
                None,
            ),
            item.line,
            item.actual,
            (
                "type-correct generated source for the installed Calcifier API"
                if item.actual.startswith("ty[")
                else "Ruff-clean generated source under the project lint policy"
            ),
            "fix the finding where it originates, in any part of the module",
        )
        for item in error.diagnostics
    )


def require_no_findings(
    findings: Sequence[Diagnostic],
    *,
    code: str,
    expected: str,
    safe_fix: str,
) -> None:
    """Add repair context to located findings and report all of them together."""
    diagnostics = tuple(
        replace(item, code=code, phase="source.validation", expected=expected, safe_fix=safe_fix)
        for item in findings
    )
    if diagnostics:
        raise CalcifyError(render_diagnostics(diagnostics), diagnostics=diagnostics)
