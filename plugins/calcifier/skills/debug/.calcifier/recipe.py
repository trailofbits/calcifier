from __future__ import annotations

from calcifier import CommandResult, emit, fail, recipe, run_command, step


@step
def run_debug(recipe_path: str, inputs: list[str], trust: bool) -> CommandResult:
    argv = ["calcifier", "debug", recipe_path]
    for value in inputs:
        argv.extend(("--input", value))
    if trust:
        argv.append("--trust")
    return run_command(argv, timeout_seconds=300, mutating=True)


@recipe(version="0.1")
def debug_recipe(recipe_path: str, inputs: str = "", trust: bool = False) -> bool:
    if not recipe_path:
        fail("recipe_path is required")
    pairs = [pair.strip() for pair in inputs.split(",") if pair.strip()]
    result = run_debug(recipe_path, pairs, trust)
    emit(result.stdout or result.stderr, name="transcript")
    if result.exit_code != 0:
        fail(f"calcifier debug exited {result.exit_code}")
    return True


if __name__ == "__main__":
    raise SystemExit(debug_recipe.cli())


# --- calcifier metadata (skill provenance + sha256) ---
_CALCIFIER_SKILL = {
    "path": "../SKILL.md",
    "sha256": ("6028f8fdc5bb0e21dd614db61ac71b7acce88bb188eeecce75ae27aa2af1efd2"),
}
_CALCIFIER_RESOURCES = []
_CALCIFIER_SUBRECIPES = []
