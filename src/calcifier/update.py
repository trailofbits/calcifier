"""``calcifier update``: compile a skill again and diff the result against its recipe.

``update`` prepares a recipe exactly as ``calcify`` does, including the one dry
run on the host, and returns a unified diff from the recipe on disk to the
prepared one for review. It writes nothing to the destination tree. Whether a
recipe still matches its skill and packaged files is for ``lint --portable`` and
``exec --trust`` to say.
"""

from __future__ import annotations

import ast
import difflib
from pathlib import Path

from .agent_events import AgentActivity
from .calcify import prepare_artifact
from .metadata import interface_text, module_metadata
from .registry import Registry


def update_skill(
    skill_path: Path,
    *,
    existing_recipe: Path,
    registry: Registry,
    compiler: str = "compiler",
    timeout: int | None = None,
    compilation_timeout: int | None = None,
    max_compiler_calls: int | None = None,
    skill_catalog: dict[str, str] | None = None,
    activity: AgentActivity | None = None,
    python_version: str | None = None,
) -> str:
    """Return a unified diff from ``existing_recipe`` to a freshly prepared recipe.

    Use the same compiler options as prepare_artifact without modifying the skill
    package. Return an empty string when unchanged; a missing recipe raises
    FileNotFoundError. Compilation errors propagate.
    """
    if not existing_recipe.is_file():
        raise FileNotFoundError(f"no existing recipe at {existing_recipe}")
    current = existing_recipe.read_text(encoding="utf-8")
    module = ast.parse(current)
    artifact = prepare_artifact(
        skill_path,
        registry,
        compiler=compiler,
        output=existing_recipe,
        timeout=timeout,
        compilation_timeout=compilation_timeout,
        max_compiler_calls=max_compiler_calls,
        skill_catalog=skill_catalog,
        activity=activity,
        python_version=python_version,
        interface_requirements=interface_text(module_metadata(module)),
    )
    return "".join(
        difflib.unified_diff(
            current.splitlines(keepends=True),
            artifact.source.splitlines(keepends=True),
            fromfile=str(existing_recipe),
            tofile=f"{existing_recipe} (prepared)",
        )
    )
