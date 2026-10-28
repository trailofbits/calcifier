# Changelog

## Unreleased

### Fixed

- Codex calls preserve the `CODEX_API_KEY` non-interactive authentication override.
  A nonblank override bypasses stored credentials, including locked OS keyrings,
  while keeping the private temporary configuration home.
- Codex calls preserve OS-keyring authentication when isolating configuration,
  including `keyring` and `auto` credential storage. Credentials are staged only
  in the private temporary home and removed after the call.

## v0.2.0 — 2026-10-23

See the [release notes](docs/releases/v0.2.0.md) for contracts, validation limits,
and remaining generation work.

### Added

- Compilers can flag advisory source material with a suitability note for owner
  review. Only the final accepted candidate supplies the note to CLI output
  and compiler traces; terminal control characters are rejected (#16).
- Worker tool capabilities reach generation, repair, and compiler dry runs.
  Custom adapters can declare supported operations, and agent calls can require
  them before invocation without granting additional permissions (#7).
- Generated recipes and retained baselines include PEP 723 metadata declaring
  their Python minimum and exact Calcifier dependency, enabling script runners
  such as `uv run` to provision the environment (#19).

### Fixed

- Compiler scratch redirection preserves helper-returned paths when chained
  through writes and verification under absolute output roots.
- Predeclared outputs remain absent and explicitly unverified after suppressed
  compiler commands or writable stub agents, including wrapped verification
  errors and direct reads. Incorrect paths and materialized content failures
  remain errors. Missing default inputs remain identifiable through context errors.
- Inline script headers preserve encoding declarations on either permitted line,
  including LF and CRLF source files.
- Prepared context can gather source files and manifests with resolved readable
  roots, rejecting an escaping selection before reading any contents while
  retaining authorized resources, external inputs, and LF/CRLF content (#9).
  File bundles use bounded text chunks so spilled contents support line-paged reads.
  Path collections require tuples in generated code and reject bare strings at runtime.
- Artifact directory creation can preflight a complete subprocess output set,
  rejecting collisions, symlinked files, and excluded project roots before any
  mkdir or command launch. Explicit replacement respects output override
  protection (#10).
- Optional command workflows can distinguish launch failures and local timeouts
  through `RecipeError.command_failure`. Generation and repair receive this
  contract. Shared-budget exhaustion remains fatal; validation and output-limit
  failures remain errors (#8).

### Validation and remaining work

- Neutral offline workflows exercise complete inventory accounting and
  independent host checks, required-field rendering, saved delivery, accumulated
  imports, and explicit build effects (#11–#15, #18). These demonstrate supported
  authoring patterns; they do not establish that real compilers generate them.
- Live generation and end-to-end acceptance have not been performed for this
  release. The remaining generation issues stay open for follow-up; owner
  acceptance is not a prerequisite for publishing v0.2.0.

## v0.1.1 — 2026-10-14

Release metadata update for the first PyPI publication. The v0.1.0 GitHub
release used an unsigned lightweight tag and failed the signed-tag publishing
check. This release contains the features documented below for v0.1.0.

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
