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
    emit(result.stdout or result.stderr, name="recipe-diff")
    if result.exit_code != 0:
        fail(f"calcifier update exited {result.exit_code}")
    return True


if __name__ == "__main__":
    raise SystemExit(update.cli())


# --- calcifier metadata (skill provenance + sha256) ---
_CALCIFIER_SKILL = {
    "path": "../SKILL.md",
    "sha256": ("b52a91834d082ba0a4df6c87ac031ac68d257ab8750cdd738f215897c486f8a6"),
}
_CALCIFIER_RESOURCES = []
_CALCIFIER_SUBRECIPES = []
