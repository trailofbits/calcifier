"""Reusable recipe and skill fixtures for compiler tests."""

from __future__ import annotations

import hashlib
import json
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path


def write_skill(directory: Path, text: str) -> Path:
    skill = directory / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(text, encoding="utf-8", newline="\n")
    return skill


def metadata_recipe(
    path: Path,
    skill: str,
    resources: Sequence[Mapping[str, object]],
    body: str = (
        "from calcifier import recipe\n\n\n@recipe\ndef trusted() -> bool:\n    return True\n"
    ),
) -> Path:
    """Write a recipe with source and resource provenance for runtime tests."""
    path.write_text(
        body
        + "\n\n# --- calcifier metadata (skill provenance + sha256) ---\n"
        + f"_CALCIFIER_SKILL = {{'path': {skill!r}, 'sha256': "
        + repr(
            hashlib.sha256(Path(skill).read_bytes()).hexdigest()
            if Path(skill).is_file()
            else "skill"
        )
        + "}\n"
        + f"_CALCIFIER_RESOURCES = {resources!r}\n"
    )
    return path


_GOOD_RECIPE = textwrap.dedent(
    """
    from __future__ import annotations
    from calcifier import recipe, step

    @step
    def noop(target: str) -> bool:
        return bool(target)

    @recipe(version="0.1")
    def emitted(target: str = "x") -> bool:
        return noop(target=target)

    if __name__ == "__main__":
        raise SystemExit(emitted.cli())
    """
).strip()

_REPAIR_SKILL = "# skill"
_REPAIR_RECIPE = textwrap.dedent(
    """
    from __future__ import annotations
    from calcifier import Budget, agent, recipe, step

    @step
    def inspect(target: str) -> bool:
        return agent(
            "worker", "review {target}", working_directory=".",
            readable_paths=(".",), target=target,
        )

    @recipe(agents=("worker",), budget=Budget(2, 180))
    def writing_style(target: str = ".") -> bool:
        return inspect(target=target)

    if __name__ == "__main__":
        raise SystemExit(writing_style.cli())
    """
).strip()


def _distinct_failures(count: int) -> list[str]:
    """Drafts that each fail with a different diagnostic, so repairs never stall."""
    return [
        _REPAIR_RECIPE.replace("target=target,\n", f"target=missing_{index},\n")
        for index in range(count)
    ]


_STATIC_FAILURE_RECIPE = _REPAIR_RECIPE.replace(
    "from __future__ import annotations\n",
    "from __future__ import annotations\nfrom pathlib import Path\n",
).replace(
    "def inspect(target: str) -> bool:\n",
    'def inspect(target: str) -> bool:\n    Path("report.md").write_text("draft")\n',
)

_MISSING_FILESYSTEM_RECIPE = textwrap.dedent(
    """
    from calcifier import recipe, step

    @step
    def inspect() -> bool:
        return True

    @recipe(output_root="workspace")
    def writing_style() -> bool:
        return inspect()

    if __name__ == "__main__":
        raise SystemExit(writing_style.cli())
    """
).strip()

_REALIZED_FILESYSTEM_RECIPE = _MISSING_FILESYSTEM_RECIPE.replace(
    "from calcifier import recipe, step",
    "from calcifier import ensure_directory, recipe, step",
).replace(
    "def inspect() -> bool:\n    return True",
    'def inspect() -> bool:\n    ensure_directory(".")\n    return True',
)


_SKILL = "Count the entries in the supplied directory.\n"

_HEADER = "from calcifier import recipe, step\n\n\n"
_STEP = textwrap.dedent(
    '''
    @step
    def count_entries(target: str) -> bool:
        """Zählt die Einträge."""
        return bool(target)

    '''
).lstrip()
_UNANNOTATED_STEP = _STEP.replace("(target: str) -> bool:", "(target: str):")
_RECIPE = textwrap.dedent(
    """
    @recipe(version="2.5")
    def tally(target: str = ".") -> bool:
        return count_entries(target=target)

    """
).lstrip()
_WRITING_RECIPE = _RECIPE.replace(
    "    return count_entries", '    open("x.txt", "w")\n    return count_entries'
)
_FOOTER = 'if __name__ == "__main__":\n    raise SystemExit(tally.cli())\n'

_GOOD = _HEADER + _STEP + _RECIPE + _FOOTER
_UNANNOTATED_AND_WRITING = _HEADER + _UNANNOTATED_STEP + _WRITING_RECIPE + _FOOTER
_WRITING = _HEADER + _STEP + _WRITING_RECIPE + _FOOTER


def _draft(source: str) -> str:
    return json.dumps({"version": "1", "source": source}, ensure_ascii=False)
