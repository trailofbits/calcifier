"""A deterministic executable fallback that delegates the intact skill workflow."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, cast


def is_baseline(source: str) -> bool:
    from .metadata import module_metadata

    return module_metadata(ast.parse(source)).get("_CALCIFIER_COMPILATION") == "baseline_retained"


def catalog_context(catalog: dict[str, str], target: Path) -> tuple[str, tuple[str, ...]]:
    """Keep catalog skills and their resources available to the intact agent workflow."""
    from .metadata import parse_metadata_namespace, split_metadata

    instructions: list[str] = []
    files: set[Path] = set()
    visited: set[Path] = set()

    def visit(name: str, recipe: Path) -> None:
        recipe = recipe.resolve()
        if recipe in visited:
            return
        visited.add(recipe)
        source = recipe.read_text(encoding="utf-8")
        _, tail = split_metadata(source)
        metadata = cast("dict[str, Any]", parse_metadata_namespace(tail or ""))
        skill = metadata.get("_CALCIFIER_SKILL")
        if isinstance(skill, dict):
            path = (recipe.parent / skill["path"]).resolve()
            instructions.append(f"Catalog skill {name}:\n" + path.read_bytes().decode("utf-8"))
            files.add(path)
            for entry in metadata.get("_CALCIFIER_RESOURCES", []):
                files.add((path.parent / (entry.get("source") or entry["path"])).resolve())
        else:
            instructions.append(f"Catalog skill {name} is defined in {recipe.name}.")
            files.add(recipe)
        if is_baseline(source):
            constants = {
                node.targets[0].id: ast.literal_eval(node.value)
                for node in ast.parse(source).body
                if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {"ADDITIONAL_INSTRUCTIONS", "ADDITIONAL_REFERENCES"}
            }
            instructions.append(constants.get("ADDITIONAL_INSTRUCTIONS", ""))
            files.update(
                (recipe.parent / path).resolve()
                for path in constants.get("ADDITIONAL_REFERENCES", ())
            )
        for child in metadata.get("_CALCIFIER_SUBRECIPES", []):
            visit(child["name"], recipe.parent / child["path"])

    for name, path in sorted(catalog.items()):
        visit(name, Path(path))
    return "\n\n".join(instructions), tuple(
        Path(os.path.relpath(path, target.parent.resolve())).as_posix() for path in sorted(files)
    )


def baseline_source(
    skill_text: str,
    *,
    role: str,
    resources: tuple[str, ...],
    worker_timeout_seconds: int,
    wall_seconds: int | None = None,
    instructions: str = "",
    reference_paths: tuple[str, ...] = (),
) -> str:
    """Delegate the intact workflow once; headings do not define provider sessions."""
    from .metadata import metadata_literal

    attempts = 2
    wall_seconds = wall_seconds or worker_timeout_seconds * attempts
    return f'''"""Calcifier baseline: workflow delegated to an agent; not optimized."""
from __future__ import annotations

from pathlib import Path

from calcifier import Budget, agent, fail, output_type, recipe, resource, step, write_text

SKILL = {metadata_literal(skill_text)}
REFERENCES = {metadata_literal(resources)}
ADDITIONAL_INSTRUCTIONS = {metadata_literal(instructions)}
ADDITIONAL_REFERENCES = {metadata_literal(reference_paths)}


@output_type
class GeneratedFile:
    path: str
    content: str


@output_type
class SkillAnswer:
    answer: str
    files: list[GeneratedFile]
    blocked_reason: str


@output_type
class SkillResult:
    answer: str
    files: list[str]


@step
def perform(request: str, target: str) -> SkillAnswer:
    scope = Path(target).resolve()
    directory = scope if scope.is_dir() else scope.parent
    references = "\\n".join(f"{{name}}: {{resource(name)}}" for name in REFERENCES)
    additional = tuple((Path(__file__).parent / name).resolve() for name in ADDITIONAL_REFERENCES)
    prompt = (
        "Follow the entire original skill below to complete the user request. "
        "Preserve its conditions, ordering, required checks and output format. "
        "Headings include reference material; do not treat every heading as a task.\\n"
        f"Original skill (authoritative):\\n{{SKILL}}\\n\\n"
        f"Packaged references (read those required by the skill):\\n{{references}}\\n\\n"
        f"Catalog skills and interface instructions:\\n{{ADDITIONAL_INSTRUCTIONS}}\\n"
        f"Additional read-only references: {{additional}}\\n"
        f"User request: {{request}}\\nTarget: {{scope}}\\n"
        "Return the requested answer in answer. Return requested text files in files, "
        "with relative paths and their complete contents; Python will save them in "
        "the run's selected output directory. Do not prefix paths with that directory. "
        "Do not claim that you wrote files. "
        "If required work cannot be performed with the available tools or inputs, "
        "explain it in blocked_reason. Otherwise leave blocked_reason empty."
    )
    return agent(
        {role!r}, prompt, output=SkillAnswer,
        working_directory=directory, readable_paths=(scope, *additional),
    )


@step
def deliver(result: SkillAnswer) -> SkillResult:
    if result.blocked_reason:
        fail(result.blocked_reason)
    paths = [str(write_text(item.path, item.content)) for item in result.files]
    return SkillResult(result.answer, paths)


@recipe(
    agents=({role!r},), budget=Budget({attempts}, {wall_seconds}),
    output_root=".calcifier-output",
)
def run_skill(request: str, target: str = ".") -> SkillResult:
    return deliver(perform(request, target))


if __name__ == "__main__":
    raise SystemExit(run_skill.cli())
'''
