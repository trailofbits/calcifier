"""Neutral compiler sources and shared pipeline setup."""

from __future__ import annotations

import textwrap
from pathlib import Path

from calcifier.agent_events import AgentActivity
from calcifier.calcify import (
    calcify_file,
)
from calcifier.recipe_draft import draft_response
from calcifier.registry import Registry
from tests.support.agents import StubProvider
from tests.support.compiler import (
    ScriptedCompiler,
    source_sequence,
)
from tests.support.recipes import (
    _REPAIR_SKILL,
)


def _draft_payload(source: str) -> str:
    return draft_response(source)


class _TwelveAndAHalfSecondTimer:
    @classmethod
    def start(cls) -> _TwelveAndAHalfSecondTimer:
        return cls()

    def elapsed(self) -> float:
        return 12.5


_COMMAND_RECIPE = textwrap.dedent(
    """
    from calcifier import recipe, run_command, step


    @step
    def show(target: str) -> str:
        return run_command(["git", "show", target], timeout_seconds=30, mutating=False).stdout


    @recipe(capabilities=("Bash",))
    def run(target: str = "HEAD") -> str:
        return show(target)


    if __name__ == "__main__":
        raise SystemExit(run.cli())
    """
)


_BASH_TOOL_RECIPE = _COMMAND_RECIPE.replace(
    "from calcifier import recipe, run_command, step",
    "from calcifier import recipe, step, tool",
).replace(
    'return run_command(["git", "show", target], timeout_seconds=30, mutating=False).stdout',
    'return str(tool("Bash", command=target))',
)


def _skill_allowing(tools: str) -> str:
    return f"---\nallowed-tools: {tools}\n---\n# skill\n"


_FOOTER = '\nif __name__ == "__main__":\n    raise SystemExit(r.cli())\n'


_TWO_ROLE_RECIPE = textwrap.dedent(
    """
    from __future__ import annotations

    from calcifier import Budget, agent, recipe, step


    @step
    def inspect(target: str) -> bool:
        return agent(
            "worker",
            "review {target}",
            working_directory=".",
            readable_paths=(".",),
            target=target,
        )


    @step
    def confirm(target: str) -> bool:
        return agent(
            "reviewer",
            "confirm {target}",
            working_directory=".",
            readable_paths=(".",),
            target=target,
        )


    @recipe(agents=("worker", "reviewer"), budget=Budget(4, 30))
    def writing_style(target: str = ".") -> bool:
        reviewed = inspect(target=target)
        confirmed = confirm(target=target)
        return reviewed and confirmed


    if __name__ == "__main__":
        raise SystemExit(writing_style.cli())
    """
).lstrip()


def _compile_file_drafts(
    tmp_path: Path, *drafts: str, activity: AgentActivity | None = None
) -> tuple[ScriptedCompiler, str]:
    """Compile through calcify_file, which runs Ruff and the dry run, with scripted drafts."""
    skill = tmp_path / "SKILL.md"
    skill.write_text(_REPAIR_SKILL)
    compiler = source_sequence(*drafts)
    registry = Registry()
    registry.register(compiler)
    for role in ("worker", "reviewer"):
        worker = StubProvider()
        worker.name = role
        registry.register(worker)
    return compiler, calcify_file(skill, registry, activity=activity).read_text()


def _helper_signature_recipe(call: str) -> str:
    return textwrap.dedent(
        f"""\
        from calcifier import emit, recipe, step

        @step
        def inspect(target: str) -> bool:
            {call}
            return True

        @recipe()
        def helper_signature(target: str = ".") -> bool:
            return inspect(target)

        if __name__ == "__main__":
            raise SystemExit(helper_signature.cli())
        """
    )
