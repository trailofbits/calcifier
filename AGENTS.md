# Working on Calcifier

Calcifier turns a SKILL.md into a runnable Python workflow. Python handles
repeatable operations; agents handle understanding, interpretation, and uncertain
decisions. Preserve the skill's instructions, mandatory steps, and explicit
output requirements, including required fields and word counts.
Compiler behavior must stay independent of any particular skill or provider.

## Code organization

- `src/calcifier/` contains the compiler, runtime, and public package.
  [docs/architecture.md](docs/architecture.md) maps their module boundaries.
- `tests/` contains offline regression tests and neutral skill fixtures;
  shared fixtures belong in `tests/support/`.
- `scripts/` contains development tools, not runtime dependencies.
- `docs/` contains detailed usage and design notes. Keep the README focused on
  what Calcifier does, why someone would use it, and where to start.

## Changes

- Prefer deleting duplication to adding abstractions. Use existing helpers and
  module boundaries; avoid speculative frameworks and compatibility layers.
- Fix the general mechanism behind a failure. Do not add compiler rules, prompt
  sentences, or tests tied to one skill's wording or domain.
- When a recipe uses an API naturally, prefer supporting that use over adding a
  rejection rule. Before adding a check, see whether ty, the dry run, or repair
  already catches the failure.
- Keep `calcifier.__all__` small. Generated recipes import only those names;
  prefer extending an existing call over adding another public primitive.
- Preserve behavior and validation unless the task explicitly changes them.
  Do not weaken an assertion or replace a failed result with an unverified one
  to make tests pass. Compatibility work needs a supported API or format to serve.
- Keep rejection focused on behavioral defects: invalid code, certain runtime
  failures, unsafe effects, and races. Formatting preferences and type precision
  must not reject an otherwise runnable recipe.
- Always attempt the whole compiler dry run with stub agents and synthesized
  inputs. Preserve its limits and report coverage gaps as unverified, never passed.
- Keep instruction handling direct; do not build fidelity proof machinery as a
  substitute for compiling and running the recipe.

## Verification and workflow

- Run focused tests while editing, then `gmake verify`. This checks the lock,
  Ruff, ty, portable plugin recipes, and the test suite. CI reports mutation
  rates; do not run mutation testing locally.
- Tests must use stub providers and never call live Claude/Codex CLIs. Write
  byte-exact fixtures with explicit encoding/newline settings; cover LF and CRLF
  when preserving source or resource bytes.
- Live acceptance is owner-run. Do not inspect held-out fixtures, cohort lists,
  or results, and do not run them yourself. Test the acceptance runner only with
  neutral fixtures. Offline checks do not establish answer quality.
- Work in an isolated checkout, preserve the owner's local changes, keep
  generated evidence outside Git, and use normal pushes.
