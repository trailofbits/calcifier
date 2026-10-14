---
name: exec
description: Execute a calcified .calcifier/recipe.py pipeline. Use when the user asks to "run a calcified skill", "exec a calcifier recipe", or wants deterministic playback of a pinned skill.
allowed-tools: Bash
---

# calcifier:exec — run a calcified recipe

Runs a calcified `.calcifier/recipe.py`. The recipe is a Python module with a single `@recipe`-decorated entry function whose body composes `@step`-decorated typed phases with ordinary Python control flow. The runtime captures a per-step trace (including agent calls and their typed outputs) and prints a pass/fail summary.

`calcifier exec` refuses to run a recipe whose complete-file sha256 isn't in the current user's Calcifier trust store. The first run on any recipe — and the first run after any edit — must pass `--trust` to record local consent.

## When to invoke

- The user asks to "run a calcified skill", "exec the recipe", "replay the calcifier pipeline".
- The user is operating on a SKILL.md that already has a `.calcifier/recipe.py` next to it.

## Inputs

- Path to a `recipe.py` (most commonly `<skill-dir>/.calcifier/recipe.py`).
- Any inputs the recipe declares as parameters of its `@recipe` function. Pass them via `--input key=value` (repeatable). If the recipe declares a parameter without a default and the user did not supply a value, ASK before running — do not invent values.

## Procedure

1. Run `calcifier --version` to confirm install.
2. Read the recipe and inspect the signature of the `@recipe`-decorated function. Each parameter is a recipe input; required inputs are those without a default.
3. Collect missing required inputs from the user, then run:
   ```
   calcifier exec <recipe-path> --input k1=v1 --input k2=v2
   ```
   The first run also needs `--trust` (after the user has reviewed the recipe).
   Agent-backed recipes require `--agent-config <agents.json>`.
4. Read the printed run summary. The summary lists every `@step` invocation in execution order with pass/fail, duration, and (on failure) the `fail()` reason or the exception type.
5. If the run failed, point the user at the failing step and suggest `calcifier debug <recipe>` for verbose logs.

## Notes

- Agents are read-only and return typed data. Deterministic declared host operations perform and verify generated filesystem mutations.
- Independent model fan-out uses `parallel()`. Workers do not mutate shared state; ordinary Python merges their results.
- Every agent-backed recipe declares whole-run call and wall budgets, and agent-config v1 declares per-role timeouts. Every attempted call, including a parse retry, is charged.
- The recipe file MUST end with `if __name__ == "__main__": raise SystemExit(<name>.cli())` so direct execution preserves a failing exit code.
- Generated filesystem mutations use declared primitives beneath `output_root`; `--dry-run hard` stops a mutating generated recipe before any agent call.

## Output

A line-by-line summary of the run, ending in a single verdict line (`OK` or `FAIL`).
