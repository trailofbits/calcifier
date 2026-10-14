---
name: debug
description: Execute a calcified recipe with verbose logging. Use when the user is iterating on a calcifier recipe, wants per-step transcripts, or asks to "debug a calcifier recipe".
allowed-tools: Bash
---

# calcifier:debug — author-friendly recipe execution

Same engine as `calcifier:exec` but with verbose Python logging. Per-step trace records show inputs, redacted outputs, durations, and any `fail()` reason. Intended for skill authors who are iterating on a calcified recipe.

## When to invoke

- The user is editing a recipe and wants to see what each `@step` returned.
- A previous `calcifier exec` run failed and the user wants more signal.

## Procedure

1. Run `calcifier --version` to confirm install.
2. Run:
   ```
   calcifier debug <recipe-path> --input k=v ...
   ```
   First run on the recipe also needs `--trust`.
   Agent-backed recipes also require `--agent-config <agents.json>`.
   Use `--trace-out <path>` to retain incremental trace v1 data if the run is
   interrupted or output parsing fails.
3. The CLI prints:
   - The full `RecipeRun` summary (every `@step` invocation with pass/fail and duration).
   - Verbose Python logging from `calcifier.*` loggers (recipe load, agent invocations, retries).
   - On failure: the `fail()` reason or the exception type and message.
4. Walk the user through any `[fail]` lines. Common author errors:
   - `@step` return annotation doesn't match what the LLM emits (rejected as schema mismatch — visible in `AgentError`).
   - An agent returned an undeclared output field (closed schemas reject it rather than caching false success).
   - A required artifact was not written or failed host verification.
   - The declared call or wall-clock budget was exhausted.
   - Recipe body references a step output before calling it.
   - Pre-condition check raised `fail("…")` early.

## Output

A long, line-by-line transcript of the run. Use it to refine the recipe; commit only after the run reaches `OK`.
