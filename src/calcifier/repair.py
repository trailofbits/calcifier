"""Bounded source generation and repair of rejected recipe drafts."""

from __future__ import annotations

import ast
import json
import re
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import cast

from .agent_events import (
    AgentActivity,
    AgentProvider,
    capture_agent_activity,
    captured_agent_activities,
)
from .compiler_activity import (
    CompilerDeadline,
    check_compiler_deadline,
    compiler_timeout,
    record_compiler_start,
)
from .compiler_resources import stage_compiler_resources
from .diagnostics import (
    CalcifierDefectError,
    CalcifyError,
    Diagnostic,
    SkillDeclarationError,
    lint_diagnostics,
)
from .dry_run import DryRunResult, dry_run_recipe
from .filesystem_contract import AgentError, FilesystemContract
from .gate import verify_source
from .metadata import split_metadata
from .outputs import strict_schema_for
from .prompts import build_prompt
from .recipe_draft import RecipeDraft, RecipeDraftError, parse_recipe_draft
from .skill_frontmatter import AllowedToolsError, declared_allowed_tools
from .skill_package import ResourceEntry
from .source_quality import (
    SourceQualityError,
    format_source,
    python_feature_version,
    resolve_line_length,
    resolve_python_version,
)

# One host-enforced ceiling on compiler model calls: the first draft and every repair.
_MAX_COMPILER_CALLS = 8
# No smoke-run candidate was accepted after three identical failures in a row.
_STALLED_REPEATS = 3
_LINE_NUMBER = re.compile(r"\bline \d+")


SOURCE_REPAIR_REQUEST = """

## Rejected candidate: repair {repair}

Calcifier rejected the RecipeDraft returned for the instructions above. Return one complete,
corrected RecipeDraft for the whole module that fixes every diagnostic below. Any part of the
module may change, including output types, constants, and helpers. Line numbers refer to the
rejected source exactly as shown.

Validation error (JSON string):
{error}

Structured diagnostics (JSON array):
{diagnostics}

Rejected source (JSON string):
{source}
"""


def calcify_source(
    skill_text: str,
    provider: AgentProvider,
    *,
    resource_manifest: str = "(none)",
    agent_roles: tuple[str, ...] = ("worker",),
    timeout: int | None = None,
    runtime_wall_seconds: int | None = None,
    skill_catalog: dict[str, str] | None = None,
    activity: AgentActivity | None = None,
    source_path: Path | None = None,
    python_version: str | None = None,
) -> str:
    """Compile a SKILL.md into checked recipe source.

    ``source_path`` is where the recipe will live; when given, the destination's Ruff
    configuration formats and lints the candidate.
    """
    context = SourceCompileContext(
        skill_text=skill_text,
        provider=provider,
        resource_manifest=resource_manifest,
        agent_roles=agent_roles,
        timeout=timeout,
        runtime_wall_seconds=runtime_wall_seconds,
        catalog=dict(skill_catalog or {}),
        activity=activity or AgentActivity(),
        source_path=source_path,
        python_version=python_version,
    )
    return compile_source(context).source


@dataclass
class CompilerCallState:
    calls: int = 0


@dataclass(frozen=True)
class SourceCompileContext:
    """Resolved inputs for one compiler run."""

    skill_text: str
    provider: AgentProvider
    resource_manifest: str
    agent_roles: tuple[str, ...]
    timeout: int | None
    runtime_wall_seconds: int | None
    catalog: dict[str, str]
    source_path: Path | None
    python_version: str | None
    activity: AgentActivity
    max_calls: int = _MAX_COMPILER_CALLS
    call_state: CompilerCallState = field(default_factory=CompilerCallState)
    resource_paths: frozenset[str] | None = None
    skill_path: Path | None = None
    resource_entries: tuple[ResourceEntry, ...] = ()
    interface_requirements: str = ""
    deadline: CompilerDeadline | None = None
    finalize_source: Callable[[str], str] | None = None

    @property
    def calls(self) -> int:
        return self.call_state.calls

    @property
    def catalog_base(self) -> Path | None:
        return self.source_path.parent if self.source_path is not None else None


@dataclass(frozen=True)
class CompiledSource:
    source: str
    dry_run: DryRunResult | None = None


def compile_source(context: SourceCompileContext) -> CompiledSource:
    """Compile and validate within one context's call and wall budgets."""
    try:
        declared_allowed_tools(context.skill_text)
    except AllowedToolsError as exc:
        raise SkillDeclarationError(
            f"the SKILL.md cannot be compiled until its allowed-tools field is fixed: {exc}"
        ) from exc
    if context.source_path is not None:
        context = replace(
            context,
            python_version=resolve_python_version(context.source_path, context.python_version),
        )
    compiler_activity, deadline = context.activity, context.deadline
    compiler_activity.max_model_calls = context.max_calls
    compiler_activity.model = context.provider.model
    compiler_activity.reasoning_effort = context.provider.reasoning_effort
    record_compiler_start(compiler_activity, deadline)
    try:
        with capture_agent_activity(context.activity):
            try:
                with tempfile.TemporaryDirectory(prefix="calcifier-compiler-") as compiler_dir:
                    manifest = context.resource_manifest
                    if context.skill_path is not None:
                        manifest = (
                            stage_compiler_resources(
                                Path(compiler_dir),
                                context.skill_path,
                                context.skill_text,
                                context.resource_entries,
                            )
                            or "(none)"
                        )
                    filesystem = FilesystemContract.build(compiler_dir, (compiler_dir,))
                    prompt = build_prompt(
                        context.skill_text,
                        manifest,
                        context.agent_roles,
                        skill_catalog=context.catalog,
                        catalog_base=context.catalog_base,
                        python_version=context.python_version,
                        line_length=(
                            None
                            if context.source_path is None
                            else resolve_line_length(context.source_path)
                        ),
                        wall_seconds=context.runtime_wall_seconds,
                        instruction_delivery=context.skill_path is not None,
                    )
                    if context.interface_requirements:
                        prompt += (
                            "\n\nGraph interface requirements:\n" + context.interface_requirements
                        )
                    source = _generate_source(context, prompt=prompt, filesystem=filesystem)
                    check_compiler_deadline(deadline)
                    context.activity.outcome = "source_accepted"
                    context.activity.record_event("compiler", "accepted")
                    return source
            except AgentError as exc:
                raise CalcifyError(
                    f"agent invocation failed: {exc}", activity=context.activity
                ) from exc
    except CalcifyError as exc:
        # Keep the error's type, source and draft; only attach the telemetry.
        exc.activity = compiler_activity
        compiler_activity.outcome = "failed"
        compiler_activity.record_event("compiler", "failed", detail=str(exc))
        raise
    except KeyboardInterrupt:
        # The capture has unwound: the callback writes the terminal elapsed time
        # and the usage known so far, without inventing usage for this attempt.
        compiler_activity.outcome = "interrupted"
        compiler_activity.record_event("compiler", "interrupted")
        raise


def _generate_source(
    context: SourceCompileContext,
    *,
    prompt: str,
    filesystem: FilesystemContract,
) -> CompiledSource:
    """Request source, then repair the latest rejected candidate until the call budget is spent.

    Every repair request is the original prompt plus the source that failed and
    only that source's diagnostics, and it returns a complete module.
    """
    phase, request, repairs = "source.initial", prompt, 0
    review_command_gaps = True
    failures: list[tuple[str, ...]] = []
    while True:
        timeout = compiler_timeout(context.provider, context.timeout, context.deadline)
        if context.calls >= context.max_calls:
            raise CalcifyError("source generation exhausted the global compiler-call budget")
        context.call_state.calls += 1
        context.activity.model_calls = context.calls
        context.activity.record_event(phase, "request_started")
        draft = _request_recipe_draft(
            context.provider, request, timeout=timeout, filesystem=filesystem
        )
        context.activity.record_event(phase, "response_received")
        try:
            source = _prepare_recipe_candidate(
                draft,
                context,
                review_command_gaps=review_command_gaps and context.calls < context.max_calls,
            )
        except (CalcifierDefectError, SkillDeclarationError):
            raise
        except CalcifyError as error:
            if any(item.code == "command_gap" for item in error.diagnostics):
                review_command_gaps = False
            context.activity.record_event(
                phase,
                "validation_failed",
                detail=json.dumps(
                    {"diagnostic": str(error), "checked_source": error.source, "draft": draft},
                    ensure_ascii=False,
                ),
            )
            failures.append(_failure_key(error))
            if failures[-_STALLED_REPEATS:] == [failures[-1]] * _STALLED_REPEATS:
                raise _stalled_repair_error(context, draft, error) from error
            if context.calls >= context.max_calls:
                raise CalcifyError(
                    "source generation exhausted the global compiler-call budget; "
                    f"latest validation error: {error}",
                    activity=context.activity,
                    diagnostics=error.diagnostics,
                    source=error.source,
                ) from error
            repairs += 1
            phase = f"source.repair_{repairs}"
            request = prompt + _source_repair_request(draft, error, repairs)
            continue
        context.activity.record_event(phase, "accepted")
        return source


def _failure_key(error: CalcifyError) -> tuple[str, ...]:
    """Identify a candidate's failure as the multiset of its diagnostics, without line numbers.

    A failure with no structured diagnostics, such as a parse error, is its message.
    """
    if not error.diagnostics:
        return (_LINE_NUMBER.sub("line", str(error)),)
    return tuple(
        sorted(
            f"{item.code}\0{item.node or ''}\0{_LINE_NUMBER.sub('line', item.actual)}"
            for item in error.diagnostics
        )
    )


def _stalled_repair_error(
    context: SourceCompileContext, draft: str, error: CalcifyError
) -> CalcifyError:
    """Stop repairs that keep producing the same failure; more calls will not change it."""
    return CalcifyError(
        f"source repair made no progress: the same diagnostics came back "
        f"{_STALLED_REPEATS} times in a row, so a compiler transformation or a check the "
        f"recipe cannot satisfy is the likely cause; latest validation error: {error}",
        activity=context.activity,
        diagnostics=error.diagnostics,
        source=error.source,
        draft=draft,
    )


def _source_repair_request(draft: str, error: CalcifyError, repair: int) -> str:
    """Render the rejected source, exactly as checked, with its own diagnostics."""
    return SOURCE_REPAIR_REQUEST.format(
        repair=repair,
        error=json.dumps(str(error), ensure_ascii=False),
        diagnostics=json.dumps([asdict(item) for item in error.diagnostics], ensure_ascii=False),
        source=json.dumps(
            draft if error.source is None else split_metadata(error.source)[0], ensure_ascii=False
        ),
    )


def _prepare_source_candidate(
    source: str,
    context: SourceCompileContext,
    *,
    review_command_gaps: bool = False,
) -> CompiledSource:
    if context.finalize_source is not None:
        source = context.finalize_source(source)
    if context.source_path is not None:
        try:
            source = format_source(
                source, context.source_path, python_version=context.python_version
            )
        except SourceQualityError as exc:
            raise CalcifyError(str(exc), diagnostics=lint_diagnostics(exc), source=source) from exc
    outcome = None
    try:
        verify_source(
            source,
            python_version=context.python_version,
            resource_paths=context.resource_paths,
            lint_path=context.source_path,
            agent_roles=context.agent_roles,
        )
        if context.finalize_source is not None and context.source_path is not None:
            from .portable import artifact_metadata_diagnostics

            findings = artifact_metadata_diagnostics(
                ast.parse(source), context.source_path, context.python_version
            )
            if findings:
                raise CalcifyError("; ".join(item.render() for item in findings))
        if context.source_path is not None and context.skill_path is not None:
            outcome = _require_dry_run_success(
                source,
                recipe_path=context.source_path,
                skill_path=context.skill_path,
                resource_entries=context.resource_entries,
                review_command_gaps=review_command_gaps,
            )
            context.activity.dry_run = outcome
    except CalcifyError as exc:
        exc.source = source
        raise
    return CompiledSource(source, outcome)


def _require_dry_run_success(
    source: str,
    *,
    recipe_path: Path,
    skill_path: Path,
    resource_entries: tuple[ResourceEntry, ...] | None = None,
    review_command_gaps: bool = False,
) -> DryRunResult:
    """Run concrete candidate work until completion or an unverified frontier."""
    outcome = dry_run_recipe(
        source, recipe_path=recipe_path, skill_path=skill_path, resource_entries=resource_entries
    )
    review_gap = review_command_gaps and outcome.command_gap
    if outcome.status != "failed" and not review_gap:
        _record_dry_run(outcome)
        return outcome
    diagnostic = Diagnostic(
        code="command_gap" if review_gap else "dry_run",
        phase="source.validation",
        node=outcome.step,
        line=None,
        actual=f"dry run {'command gap' if review_gap else 'failed'}: {outcome.error}",
        expected="concrete Python work satisfies its contracts before external fixtures are needed",
        safe_fix=(
            "review the invocation and stderr; correct invocation mistakes, or return the same "
            "source if required inputs are unavailable; never fabricate inputs or agent answers"
            if review_gap
            else "fix the named deterministic failure; do not fabricate inputs or agent answers"
        ),
    )
    raise CalcifyError(diagnostic.actual, diagnostics=(diagnostic,))


def _record_dry_run(outcome: DryRunResult) -> None:
    """Record the dry-run outcome without promoting coverage gaps to passes."""
    detail = json.dumps(
        {
            "complete": outcome.status == "passed",
            "step": outcome.step,
            "reason": outcome.error,
            "writes": list(outcome.writes),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    for activity in captured_agent_activities():
        activity.record_event("dry_run", outcome.status, detail=detail)


def render_dry_run(activity: AgentActivity) -> str:
    """Summarize the accepted recipe's dry run, including the files it declared it writes."""
    outcome = activity.dry_run
    if outcome is None:
        return "dry run: not run"
    reason = "" if outcome.error is None else f" ({outcome.error})"
    label = "writes" if outcome.status == "passed" else "writes before it stopped (partial)"
    writes = ", ".join(outcome.writes) or "none"
    return f"dry run: {outcome.status}{reason}\ndry-run {label}: {writes}"


def _prepare_recipe_candidate(
    draft: str,
    context: SourceCompileContext,
    *,
    review_command_gaps: bool = False,
) -> CompiledSource:
    """Parse the complete candidate, then validate its final source."""
    try:
        source = parse_recipe_draft(
            _normalize_uniformly_indented_recipe_draft(
                draft, python_version=context.python_version
            ),
            python_version=context.python_version,
        )
    except RecipeDraftError as exc:
        raise CalcifyError(f"invalid RecipeDraft: {exc}") from exc
    return _prepare_source_candidate(source, context, review_command_gaps=review_command_gaps)


def _normalize_uniformly_indented_recipe_draft(
    value: str,
    *,
    python_version: str | None,
) -> str:
    """Recover only a whole module shifted by one uniform indentation prefix."""
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return value
    if not isinstance(payload, dict) or not isinstance(payload.get("source"), str):
        return value
    source = cast("str", payload["source"])
    feature_version = python_feature_version(python_version)
    try:
        ast.parse(source, feature_version=feature_version)
        return value
    except SyntaxError:
        pass
    try:
        candidate = _dedent_module_preserving_literals(source, feature_version=feature_version)
    except (SyntaxError, RecursionError):
        return value
    if candidate is None:
        return value
    payload["source"] = candidate
    return json.dumps(payload, ensure_ascii=False)


def _dedent_module_preserving_literals(
    source: str, *, feature_version: tuple[int, int] | None
) -> str | None:
    """Remove only the enclosing block's indent; keep multiline literal bytes."""
    wrapped = ast.parse("if True:\n" + source, feature_version=feature_version)
    if len(wrapped.body) != 1:
        return None
    body = cast("ast.If", wrapped.body[0]).body
    lines = source.splitlines(keepends=True)
    first = lines[body[0].lineno - 2]
    prefix = first[: len(first) - len(first.lstrip(" \t"))]
    # The wrapper adds one line. Protect every literal continuation line,
    # including its closing quote; f-strings include their original text too.
    protected = {
        line
        for node in ast.walk(wrapped)
        if isinstance(node, ast.Constant | ast.JoinedStr)
        for line in range(node.lineno - 1, (node.end_lineno or node.lineno) - 1)
    }
    candidate = "".join(
        line[len(prefix) :] if index not in protected and line.startswith(prefix) else line
        for index, line in enumerate(lines)
    )
    expected = ast.Module(body=body, type_ignores=[])
    recovered = ast.parse(candidate, feature_version=feature_version)
    return candidate if ast.dump(recovered) == ast.dump(expected) else None


def _request_recipe_draft(
    provider: AgentProvider,
    prompt: str,
    *,
    timeout: float | None,
    filesystem: FilesystemContract,
) -> str:
    schema = strict_schema_for(RecipeDraft)
    result = provider.complete(
        prompt,
        timeout=timeout,
        filesystem=filesystem,
        schema=schema,
    )
    # Parsing is candidate validation, so malformed responses consume the same
    # bounded repairs as invalid syntax/semantics, not an untracked provider retry.
    return result.text
