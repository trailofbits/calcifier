# Changelog

## v0.1.0 — 2026-10-14

First public release of Calcifier. Write Python workflows that combine
repeatable operations with agent reasoning, or compile them from existing
skills. Requires Python 3.13 or newer.

### Added

- **Python recipes.** Author workflows directly with `@recipe` and typed `@step`
  functions, ordinary Python control flow, and async functions. Entry signatures
  define typed CLI inputs. `fail()` aborts; `emit()` exposes named artifacts.
- **Skill compilation.** Generate a reviewable Python recipe from a `SKILL.md`
  and packaged resources. Repeatable operations move into Python; reasoning
  stays with agents. Generation, static checks, whole-recipe stub execution,
  and repairs share a bounded call/time budget.
- **Runnable fallback.** Compilation starts from one agent session with the
  intact skill. Failed optimization retains that baseline and reports
  `baseline_retained`; `--require-compiled` preserves it but exits nonzero.
- **Typed agent responses.** Closed `@output_type` records enforce required
  fields, value types, and constraints such as allowed values and word counts.
  Invalid responses receive one parse retry. Codex receives schemas through its
  native structured-output channel; the host validates full Python types.
- **Agent roles and budgets.** Configure Claude, Codex, or custom argv adapters
  by semantic role. Steps, parallel workers, and children share call and wall
  limits. `--max-agent-calls` bounds charged attempts, including parse retries,
  without increasing declared limits.
  `Budget.remaining_wall_seconds()` exposes the active run's effective time left.
  Custom command adapters receive their effective timeout in the call contract.
- **Instruction delivery.** File-based recipes pin original instructions and
  references. `prepare_context()` bundles complete tasks and prior results
  without duplicating prompts or changing the project working directory.
- **Declared effects.** Contained, atomic report writes; artifact verification;
  host argv commands; and soft/hard dry-run modes. `--output-dir` redirects a
  declared root and protects existing files.
- **Typed command results.** `run_command(..., output=T)` validates successful
  JSON stdout. Compiler dry runs execute read-only commands and synthesize
  suppressed mutating results, reporting unavailable inputs as unverified.
- **Parallel checkpoints.** Serial `on_result` callbacks checkpoint completed
  workers; `reuse_existing=True` adopts byte-identical artifacts for updates.
- **Resources and composition.** Verified resource snapshots, pinned child
  recipes, shared budgets/output roots, and dependency-ordered graph builds
  with closed catalogs.
- **Diagnostics and traces.** Compiler telemetry and incremental versioned run
  traces record attempts, artifacts, effects, timing, budgets, and observed
  usage. Missing usage remains unknown. `profile` summarizes a trace.
- **CLI and plugin.** Compile, execute, debug, update-by-diff, portable lint,
  graph builds, and profiling; four Claude Code plugin skills with recipes.

### Security

- **Trust and integrity.** Execution requires consent for exact recipe bytes,
  resources, and child pins. Compilation does not grant trust; adjacent trust
  files have no authority. Recipes execute Python with host permissions; review
  the source and dependencies before granting trust.
- **Behavioral checks.** Source checks reject certain runtime failures, unsafe
  effects, model-controlled paths/commands, and races. Style and type precision
  do not reject runnable candidates. Stub execution reports coverage gaps
  without promoting them to passes.
- **Provider boundaries.** Host agents use explicit filesystem contracts and
  clean settings. Subprocesses run without a shell under time/output limits.
- **Strict records and redaction.** Malformed trace/trust data fails closed.
  Secret-like field names are redacted; free text still needs review.
- **Release provenance.** A signed-tag release workflow checks source and
  artifact versions, uses PyPI trusted publishing and build attestations, and
  builds without workflow caches.

See [usage](docs/usage.md), [architecture](docs/architecture.md), and
[providers](docs/providers.md) for contracts and limits. Compilation and offline
checks do not establish answer quality, instruction fidelity, or faster execution.
