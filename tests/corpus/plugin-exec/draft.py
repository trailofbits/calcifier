from __future__ import annotations

from calcifier import CommandResult, emit, fail, recipe, run_command, step


@step
def execute_recipe(recipe_path: str, inputs: list[str], trust: bool) -> CommandResult:
    argv = ["calcifier", "exec", recipe_path]
    for value in inputs:
        argv.extend(("--input", value))
    if trust:
        argv.append("--trust")
    return run_command(argv, timeout_seconds=300, mutating=True)


@recipe(version="0.1")
def calcifier_exec(recipe_path: str, inputs: str = "", trust: bool = False) -> bool:
    if not recipe_path:
        fail("recipe_path is required")
    pairs = [pair.strip() for pair in inputs.split(",") if pair.strip()]
    if any("=" not in pair or not pair.split("=", 1)[0].strip() for pair in pairs):
        fail("inputs must be comma-separated key=value pairs")
    result = execute_recipe(recipe_path, pairs, trust)
    emit(result.stdout or result.stderr, name="transcript")
    if result.exit_code != 0:
        fail(f"calcifier exec exited {result.exit_code}")
    return True


if __name__ == "__main__":
    raise SystemExit(calcifier_exec.cli())
