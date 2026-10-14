---
name: calcifier
description: Convert a Claude/Codex SKILL.md into a typed Python recipe under .calcifier/recipe.py. Use when the user asks to "calcify a skill", convert a skill into a recipe, harden a skill into a deterministic pipeline, or pin agent control flow.
allowed-tools: Bash
---

# calcifier:calcifier — calcify a skill

Calcifies one SKILL.md into a typed Python recipe at `.calcifier/recipe.py` next to the SKILL.md. Structured outputs use closed `@output_type` records, model calls use semantic roles under declared budgets and read-only filesystem contracts, and one `@recipe` function composes typed steps with ordinary Python control flow.

## When to invoke

- The user says "calcify <skill-path>", "convert this skill into a calcifier recipe", or "calcify the X skill".
- The user explicitly wants typed Python control flow around an existing skill instead of LLM-interpreted prose.

## Inputs

The user must point at a `SKILL.md` file. If they only name a skill (e.g. "calcify sharp-edges"), search for `*/SKILL.md` matching that slug under their skills repos and confirm before proceeding.

## Procedure

1. **Verify the CLI is installed.** Run `calcifier --version`. If missing, instruct the user to `uv tool install /path/to/calcifier` and stop.
2. **Locate the SKILL.md.** Resolve the absolute path. If ambiguous, ask the user to disambiguate.
3. **Run the calcifier CLI.** Execute:
   ```
   calcifier calcify <abs-path-to-SKILL.md> --compiler compiler --agent-config <agents.json>
   ```
   The configured compiler returns Python implementing the skill. Before writing, Calcifier checks it with Ruff, ty, and the portable source gate, holds its declared roles and tools to the configured roles and the skill's allowed-tools, and runs it once against stub agents, all without importing agent output. If any check fails, it writes nothing.
4. **Show the user the summary** the CLI prints (recipe name, version, input parameters, list of `@step` functions) and offer to open the recipe for review.
5. **Surface the trust prompt.** `calcifier exec` refuses to run a recipe whose sha256 isn't in the per-user trust store. Calcify deliberately does NOT auto-record trust — the user MUST review the Python before granting consent. Tell them to run `calcifier exec <recipe.py> --trust` once they've read it.
6. **Suggest a smoke run.** Recommend `calcifier debug <recipe.py>` for the first execution so the user sees per-step traces.

## Hard requirements

- The recipe MUST define exactly one `@recipe`-decorated function. The acceptance gate enforces this; if you see a recipe that violates this after calcify, treat it as a tool bug and report it.
- Run `calcifier lint <recipe> --portable` before trust. Generated model calls must use literal semantic roles; provider identity belongs only in agent-config v1.
- Do NOT edit `.calcifier/recipe.py` by hand on the user's behalf without asking — drift between the SKILL.md and the recipe is detected by `calcifier:update`, and silent edits poison that signal.

## Outputs

- A path printed by the CLI (e.g. `wrote /path/to/skill/.calcifier/recipe.py`).
- A short summary: recipe name, version, inputs, list of `@step` functions.
- The "Run: `calcifier exec <recipe> --trust`" instruction the CLI emits.
