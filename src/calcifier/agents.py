"""Recipe-facing agent calls."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args, get_type_hints

from . import trace as trace_module
from .agent_config import BUILTIN_AGENTS, ROLE_RE
from .filesystem_contract import AgentError, FilesystemContract, _filesystem_prompt
from .validation import bounded_timeout

if TYPE_CHECKING:
    from typing_extensions import TypeForm

# -- user-facing Agent --------------------------------------------------------


class Agent:
    """Recipe-author handle for an agent provider.

    Calling the handle inside a ``@step`` body invokes the registered provider
    in structured-output mode using the active step's return annotation. The
    annotation can be overridden per call with ``output=<Type>``.

    The active ``Registry`` is read from the trace ``RunContext``; the runtime
    sets it on entry to ``execute()``. Calling outside of an active context
    raises ``AgentError``.
    """

    def __init__(self, provider_name: str) -> None:
        self.name = provider_name

    def __call__(
        self,
        prompt: str,
        /,
        *,
        working_directory: str | Path | None = None,
        readable_paths: tuple[str | Path, ...] = (),
        writable_paths: tuple[str | Path, ...] = (),
        output: Any | None = None,
        timeout: float | None = None,
        context: Any | None = None,
        **template_args: Any,
    ) -> Any:
        ctx = trace_module.current()
        if ctx is None or ctx.registry is None:
            raise AgentError(
                f"Agent {self.name!r} called outside an active recipe run (no registry in context)"
            )
        if ctx.allowed_agents is not None and self.name not in ctx.allowed_agents:
            raise AgentError(
                f"agent role {self.name!r} is not declared by the active recipe; "
                f"allowed: {sorted(ctx.allowed_agents)}"
            )
        active_frame = trace_module.current_frame()
        provider = ctx.registry.get(self.name)
        annotation = output
        if annotation is None:
            annotation = (
                ctx.parent_step_annotation if active_frame is None else active_frame.annotation
            )
            if annotation is None:
                raise AgentError(
                    f"Agent {self.name!r} called outside a @step body and no `output=` schema given"
                )
        rendered = _rendered_prompt(self.name, prompt, template_args)
        _require_agent_answer_type(annotation)
        rendered, readable_paths, host_delivery = _prepared_agent_context(
            rendered, readable_paths, context
        )
        filesystem = FilesystemContract.build(
            _agent_working_directory(working_directory, context),
            readable_paths,
            writable_paths,
            resource_root=ctx.resource_root,
        )
        rendered = _filesystem_prompt(filesystem, rendered, host_delivery=host_delivery)
        effective_timeout = provider.effective_timeout(timeout)
        if effective_timeout is None:
            raise AgentError(f"agent role {self.name!r} has no configured timeout")
        if ctx.budget_state is not None:
            ctx.budget_state.check_wall()
            effective_timeout = bounded_timeout(
                effective_timeout, ctx.budget_state.remaining_wall_seconds()
            )
        with trace_module.agent_call_frame(
            role=self.name,
            filesystem=filesystem.as_dict(),
            configuration_hash=ctx.registry.configuration_hash,
            **provider.trace_metadata(),
        ):
            if filesystem.writable_paths and ctx.dry_run is trace_module.DryRunMode.HARD:
                from .side_effects import SideEffectBlocked

                raise SideEffectBlocked("side effect blocked by hard dry-run: agent writable paths")
            return provider.structured(
                rendered,
                annotation=annotation,
                timeout=effective_timeout,
                filesystem=filesystem,
            )


def _require_agent_answer_type(annotation: Any, seen: set[int] | None = None) -> None:
    """Reject answer types that would let a model choose a host-prepared context."""
    from .prepared_context import PreparedContext

    visited = set() if seen is None else seen
    if id(annotation) in visited:
        return
    visited.add(id(annotation))
    if annotation is PreparedContext:
        raise AgentError("prepared contexts are issued by prepare_context(), not by agents")
    children = list(get_args(annotation))
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        children.extend(get_type_hints(annotation, include_extras=True).values())
    for child in children:
        _require_agent_answer_type(child, visited)


def _agent_working_directory(working_directory: str | Path | None, context: Any) -> str | Path:
    if working_directory is not None:
        return working_directory
    if context is None:
        raise AgentError("working_directory is required without a PreparedContext")
    return Path.cwd()


def _prepared_agent_context(
    rendered: str,
    readable_paths: tuple[str | Path, ...],
    context: Any,
) -> tuple[str, tuple[str | Path, ...], bool]:
    """Add the original instructions, or a prepared context that already embeds them."""
    from .instruction_context import current_instructions
    from .prepared_context import PreparedContext

    authority = current_instructions()
    if authority is not None:
        readable_paths = (*readable_paths, authority.directory)
    if context is None:
        if authority is not None:
            rendered += "\n\n" + authority.prompt(_INSTRUCTION_BUDGET - 2)
        return rendered, readable_paths, authority is not None
    if not isinstance(context, PreparedContext):
        raise AgentError("agent context must come from prepare_context()")
    if rendered != context.prompt:
        rendered += "\n\n" + context.prompt
    return rendered, (*readable_paths, context.directory), True


# Inline instruction bytes for calls without a PreparedContext; the full text stays readable.
_INSTRUCTION_BUDGET = 65536


# Match ``{name}`` where ``name`` is a Python identifier. Anything else
# (``{}``, ``{0}``, ``{a.b}``, ``{...prose...}``, JSON object literals)
# stays literal — a safer subset of ``str.format`` for prompts that often
# contain JSON examples or schema fragments with stray braces.
_SAFE_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _rendered_prompt(role: str, prompt: str, template_args: dict[str, Any]) -> str:
    unused = sorted(name for name in template_args if f"{{{name}}}" not in prompt)
    if unused:
        raise AgentError(
            f"agent {role!r} got template arguments with no {{name}} placeholder in "
            f"its prompt: {unused}; check for a misspelled agent() option"
        )
    return _safe_substitute(prompt, template_args)


def _safe_substitute(prompt: str, kwargs: dict[str, Any]) -> str:
    if not kwargs:
        return prompt

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(kwargs[key]) if key in kwargs else match.group(0)

    return _SAFE_PLACEHOLDER.sub(_sub, prompt)


def agent[T = Any](
    role: str,
    prompt: str,
    /,
    *,
    working_directory: str | Path | None = None,
    readable_paths: tuple[str | Path, ...] = (),
    writable_paths: tuple[str | Path, ...] = (),
    output: TypeForm[T] | None = None,
    timeout: float | None = None,
    context: Any | None = None,
    **template_args: object,
) -> T:
    """Invoke a declared agent role using the active registry.

    ``output`` is a type expression such as an ``@output_type`` class,
    ``list[Model]``, or ``Annotated[str, word_count(1, 50)]``; the call returns
    that type, or the response text when ``output`` is omitted. With ``context``,
    the caller's working directory remains the default and the bundle is readable;
    otherwise declare a working directory explicitly. Extra paths default to none.
    Writable paths must be existing directories, with the working directory inside one.
    Agents may change files beneath them except .git paths; agents never commit.
    Recipes run git with run_command(). One signature,
    rather than overloads, lets a type checker name the exact argument that is
    wrong.
    """
    if ROLE_RE.fullmatch(role) is None:
        raise AgentError(f"invalid agent role name: {role!r}")
    if role in BUILTIN_AGENTS:
        raise AgentError(f"provider name {role!r} cannot be used as a semantic role")
    return Agent(role)(
        prompt,
        working_directory=working_directory,
        readable_paths=readable_paths,
        writable_paths=writable_paths,
        output=output,
        timeout=timeout,
        context=context,
        **template_args,
    )
