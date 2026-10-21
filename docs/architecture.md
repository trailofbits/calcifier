# Calcifier architecture

A recipe's Python module defines the workflow. The compiler transports its
source in a `RecipeDraft` JSON envelope; that envelope does not describe control
flow. Each module has one `@recipe` entry point, typed steps, and a
`raise SystemExit(entry.cli())` footer.

File formats and protocols carry an explicit version. Current formats are version 1.

## Compilation pipeline

```text
SKILL.md + resources + configured roles
    → runnable baseline
    → RecipeDraft containing Python source
    → pinned provenance, formatting, and static checks
    → whole-recipe run with stub agents
    → bounded repairs
    → atomic write of the accepted bytes
```

1. `baseline.py` prepares one agent session with the intact skill and resources,
   a generic `request`/`target` interface, and text-file outputs under
   `.calcifier-output`. This is the starting point and recovery path.
2. `artifact.py` discovers resources and child recipes.
   `compiler_resources.py` stages a private read-only copy, checking hashes,
   sizes, aliases, and containment before a model call. Packaged scripts are
   supplied as source data. `source_quality.py` resolves the destination's
   Python version and Ruff settings.
3. `prompts.py` supplies the skill, resources, installed API, configured roles,
   child interfaces, and budget guidance. The compiler moves concrete,
   repeatable work into Python while keeping interpretation with agents.
4. `recipe_draft.py` checks envelope structure, grammar, size, depth, node kinds,
   and syntax warnings. `metadata.py` attaches runner requirements and skill,
   resource, and child hashes.
   Ruff formats the complete candidate and applies safe fixes. The
   gate collects independent Ruff, module-shape, ty, and semantic findings on
   that source so a repair can address them together.
5. `dry_run.py` attempts the whole recipe in a separate interpreter with a
   120-second subprocess limit. Inputs are synthesized in a temporary workspace;
   stub agents return host-schema values, tools return empty results, and
   declared writes use scratch storage. Mutating commands are suppressed.
6. `repair.py` returns the original prompt, latest source, and diagnostics to
   the compiler within the call/time budget. Three consecutive identical
   diagnostic sets stop repair. The final formatted bytes receive one gate
   pass, provenance checks, and the whole-recipe dry run before `artifact.py`
   writes them atomically. Compilation never grants trust.

The dry run always attempts the whole candidate. Exceptions and timeout end it;
the host does not deliberately stop it early. Its outcomes are:

| Outcome | Meaning |
| ------- | ------- |
| `passed` | The whole recipe completed on stub answers. |
| `failed` | A deterministic exception, timeout, or unreadable result needs repair. |
| `unverified` | An input cannot be synthesized, a stub fails a validator the schema cannot express, or a required default input file is absent. |

`unverified` reports a coverage gap and does not reject the candidate. A passing
stub run checks mechanics, not answer quality. Read-only commands execute during
the dry run; a typed command naming an unavailable synthetic input can record
a gap and return synthesized output so downstream steps still run. Other
nonzero exits and later deterministic failures reject the candidate. Command
input gaps receive one advisory repair when the call budget permits.

### Budgets and outcomes

File compilation defaults to five calls when a runnable baseline exists and
eight calls otherwise, with a shared 3,600-second deadline per recipe. Both CLI
and Python API use these defaults; source-only compilation has no baseline. Each call
gets at most the remaining shared time. [Usage](usage.md#compilation) describes
the overrides; the limits apply to `calcify`, `update`, and each graph node.

`repair.py` passes one frozen `SourceCompileContext` through generation and
validation. A separate mutable call counter enforces its budget; observer events
do not control retries. `PreparedArtifact` carries the selected outcome and
dry-run result directly, so a retained baseline cannot inherit a rejected
candidate's passing dry run.

A validated replacement records `compiled`; a failed optimization retains the
baseline and records `baseline_retained`. `--require-compiled` preserves that
fallback but exits nonzero. If one graph node falls back, the whole graph is
published as baselines. Neither compilation nor baseline recovery establishes
instruction fidelity, speed, or answer quality; those need live acceptance.

### What the gate rejects

The gate rejects behavioral defects: syntax and undefined names, ty rules that
fail whenever the line executes, dry-run crashes, unsafe effects, races, and
model-controlled paths or commands. Style, destination lint preferences, and
type precision never reject a candidate. Generated source cannot suppress
findings with noqa or type-ignore comments.

Semantic checks cover output annotations, imports, resources, roles,
capabilities, effects, child calls, async use, and taint. Agent and child output
cannot supply agent filesystem paths, tool paths, or command arguments and
working directories. A typed judgment may select between trusted values.
Top-level code is limited to imports, definitions, constants, and the footer.
The dry run executes trusted Python on the host; it is not a sandbox.

## Compiler module boundaries

| Modules | Responsibility |
| ------- | -------------- |
| `calcify.py` | Public compiler facade and trace writer. |
| `cli.py`, `cli_options.py` | Commands, configuration flags, input binding, and exit policies. |
| `artifact.py`, `baseline.py` | Preparation, final validation, and publication. |
| `repair.py`, `prompts.py`, `recipe_draft.py` | Generation, prompt construction, draft parsing, and repair. |
| `compiler_activity.py`, `compiler_resources.py` | Deadlines, telemetry, and verified resource copies. |
| `gate.py`, `recipe_ast.py`, `diagnostics.py` | Source checks, import resolution, and findings. |
| `source_quality.py`, `type_check.py` | Ruff and ty checks. |
| `output_annotations.py`, `runtime_conformance.py`, `effect_methods.py`, `taint.py`, `parallel_contract.py`, `graph_edges.py` | Annotation, API, effect, data-flow, parallel, and child-call checks. |
| `dry_run.py` | Whole-recipe stub execution. |
| `catalog.py`, `skill_package.py`, `skill_frontmatter.py`, `metadata.py`, `portable.py` | Child discovery, shared resource-path rules, declarations, provenance, and portable lint. |
| `graph.py`, `graph_build.py`, `update.py` | Graph inspection/builds and drift diffs. |

`update` stages its candidate and dry run outside the destination and prints a
diff. `build-graph` checks dependency edges and publishes a closed catalog.
Child calls and trust are detailed in [composition](recipes.md#composition).

## Runtime

Generated recipes import the standard library and names from
`calcifier.__all__`. The compiler's API reference omits host-only names such as
`Recipe`, `RecipeRun`, `load_recipe`, and `record_side_effect`.

`Recipe.run()` exposes the supported runtime options and delegates to the internal
`_execute()` method used by composition and dry runs. Execution creates a
context holding registries, permissions, output roots,
budget state, and trace records. Steps push frames with their return annotations;
agents use those annotations for strict parsing. Parent and child recipes share
the top-level budget. `parallel()` copies the context into isolated workers,
shares accounting, and merges records in input order. A plain executor does
not propagate the context.

| Modules | Responsibility |
| ------- | -------------- |
| `recipe.py`, `run_result.py`, `trace.py`, `trace_export.py`, `trace_profile.py` | Execution, decorators, CLI binding, records, export, and profiling. |
| `agent_config.py`, `agent_events.py`, `agent_providers.py`, `filesystem_contract.py`, `registry.py`, `agents.py` | Configuration, transport, telemetry, filesystem contracts, and routing. |
| `native_events.py` | One bounded JSONL decoder shared by live streams and saved provider output. |
| `instruction_context.py`, `prepared_context.py` | Pinned original instructions and task bundles. |
| `budget.py`, `clock.py`, `composition.py` | Shared limits, suspend-aware time, and child calls. |
| `tools.py`, `side_effects.py`, `mutations.py`, `subprocesses.py`, `windows_job.py` | Providers, declared effects, contained writes, and bounded processes. |
| `outputs.py`, `usage.py`, `trust.py` | Typed answers, wire schemas, stub synthesis, accounting, and consent. |
| `atomic_io.py`, `hashing.py`, `json_config.py`, `path_syntax.py`, `validation.py`, `exceptions.py` | Shared file writes, hashes, configuration parsing, portable path syntax, value checks, and error families. |
| `__init__.py` | The small public API available to generated recipes. |

[Recipe authoring](recipes.md) covers structured schemas, retries, budgets,
typed commands, and parallel callbacks. [Providers](providers.md) covers role
configuration and adapter protocols.

### Instruction delivery and prepared context

File-based recipes pin the skill and resources. At runtime, their bytes are
verified and copied into private read-only storage available to every agent
call. A changed pinned source fails execution. Original instructions fit
inline up to a 64 KiB budget; larger inputs are supplied through complete files.

Prepared contexts bundle the task, instructions, and prior results. The bundle
becomes readable without moving the working directory or granting access to
the target. Passing its prompt exactly avoids duplicate delivery. See
[agents and context](recipes.md#agents-and-context).

## Effects and provider boundaries

Report writes use declared primitives below an `output_root`; external commands
use argv without a shell. Effects are recorded before execution and marked
complete afterward. Hard dry-run blocks mutations and writable agents; soft
dry-run records host mutations without performing them. Configured agents,
custom providers, and recipe Python still run on the host.

Parallel workers cannot mutate shared state. A serial parent `on_result`
callback can persist completed work. Mutating child recipes receive private
lane roots. Writable agents require distinct declared directories; `.git`
remains read-only. Custom adapters must enforce their own filesystem contract.
See [effects](recipes.md#effects), [output protection](usage.md#output-directories),
and [provider access](providers.md#filesystem-access).

Packaged resources include standard resource directories and explicitly referenced
safe local files/directories. `resource()` returns a verified file snapshot;
absolute paths, traversal, and escaping symlinks fail. Child files likewise run
from the byte snapshots whose hashes were checked.

## Trust-on-first-use

`exec` and `debug` require user consent for the complete recipe hash, resources,
and child pins. Loading executes the admitted top-level code. `--trust` records
consent in a private per-user store; a changed pin requires new trust. Adjacent
trust files have no authority. `CALCIFIER_TRUST_STORE` selects an isolated store
for tests or CI. Direct `python recipe.py` execution bypasses this check.

## Trace

Traces contain step, agent, artifact, effect, error, timing, and usage records.
Bounded native transport diagnostics use `agent_transport`; only `emit()`
produces user artifacts.
Checkpoints coalesce routine updates and flush before blocking provider calls.
The terminal trace is validated against [trace schema v1](trace-schema.json)
and written atomically. See [trace usage](usage.md#traces).

Usage records sum observed values and retain `null` for unreported fields.
`usage_reported_attempts` counts final numeric reports per field, including
responses the host parser rejected. Partial usage from failed invocations
contributes observations without a final report. Provider fields have different
meanings; see [token accounting](usage.md#token-accounting).

### Redaction

Structured field names such as `token`, `password`, and `api_key` are redacted,
including dataclass fields and mapping keys. Free text and filenames are not
scanned. Agent-call records omit raw prompts/responses; parsed step outputs
remain. Review traces before sharing.

## Out of scope

Provider SDK transports, built-in remote tool backends, and VM isolation are
not implemented. Tests use offline stubs; see [contributing](../CONTRIBUTING.md#testing).
