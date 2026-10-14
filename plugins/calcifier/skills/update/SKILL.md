---
name: update
description: Re-calcify a SKILL.md and unified-diff the result against the on-disk recipe. Use when the user asks to "check calcifier drift", "diff a recipe against its skill", or "update a calcified recipe".
allowed-tools: Bash
---

# calcifier:update — review a freshly compiled recipe

Re-runs the full calcify pipeline against the current SKILL.md without writing to the skill package, then diffs the prepared recipe against the existing `.calcifier/recipe.py`. The dry run executes the candidate once on the host.

## When to invoke

- The user says "check drift", "did the recipe drift", "update the calcifier recipe", or "is this recipe stale".
- After a SKILL.md edit, before running `calcifier exec` again.

## Procedure

1. Confirm the CLI: `calcifier --version`.
2. Locate the SKILL.md and its `.calcifier/recipe.py`. If the recipe is missing, tell the user there is nothing to update and recommend `calcifier:calcifier` instead.
3. Run:
   ```
   calcifier update <abs-path-to-SKILL.md> --compiler compiler --agent-config <agents.json>
   ```
   This operation generates only in a temporary staging directory and reads the
   on-disk recipe; it does not overwrite the recipe.
4. The CLI prints a unified diff of the complete recipe, or `no changes` when it matches. It exits 0 in either case and nonzero if compilation fails. Compiler activity and the dry-run result go to stderr.
5. If there is a diff, summarize the meaningful changes for the user. The diff is plain Python; common changes:
   - New / renamed `@step` functions.
   - Changed `@step` return annotations (the JSON schema sent to the LLM changes with the type).
   - Added or removed control flow inside the `@recipe` body.
   - Changed semantic roles, budgets, parallel fan-out, or subrecipe hashes.
6. Ask the user whether to overwrite. Do not overwrite unless they confirm. The overwrite is `calcifier calcify <skill>` again.

## Output

A unified diff or a no-changes message. Exit code 0 means compilation succeeded; a nonzero exit means it failed.
