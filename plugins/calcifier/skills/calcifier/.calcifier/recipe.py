from __future__ import annotations

from calcifier import CommandResult, emit, fail, recipe, run_command, step


@step
def run_calcify(skill: str, agent_config: str) -> CommandResult:
    argv = ["calcifier", "calcify", skill, "--agent-config", agent_config]
    return run_command(argv, timeout_seconds=300, mutating=True)


@recipe(version="0.1")
def calcify_skill(skill: str, agent_config: str = "agents.json") -> bool:
    if not skill:
        fail("skill is required")
    result = run_calcify(skill, agent_config)
    emit(result.stdout or result.stderr, name="summary")
    if result.exit_code != 0:
        fail(f"calcifier calcify exited {result.exit_code}")
    return True


if __name__ == "__main__":
    raise SystemExit(calcify_skill.cli())


# --- calcifier metadata (skill provenance + sha256) ---
_CALCIFIER_SKILL = {
    "path": "../SKILL.md",
    "sha256": ("663be84c7ff1491436478207aab807ce7a0cabed460614513c4f29aedab94876"),
}
_CALCIFIER_RESOURCES = []
_CALCIFIER_SUBRECIPES = []
