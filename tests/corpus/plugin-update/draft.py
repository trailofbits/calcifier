from __future__ import annotations

from calcifier import CommandResult, emit, fail, recipe, run_command, step


@step
def run_update(skill_path: str, agent_config: str) -> CommandResult:
    argv = ["calcifier", "update", skill_path, "--agent-config", agent_config]
    return run_command(argv, timeout_seconds=300, mutating=False)


@recipe(version="0.1")
def update(skill_path: str, agent_config: str = "agents.json") -> bool:
    if not skill_path:
        fail("skill_path is required")
    result = run_update(skill_path, agent_config)
    emit(result.stdout or result.stderr, name="drift-report")
    if result.exit_code not in {0, 1}:
        fail(f"calcifier update exited {result.exit_code}")
    return result.exit_code == 0


if __name__ == "__main__":
    raise SystemExit(update.cli())
