# Recipe authoring

Import the public API from `calcifier`. A module has one `@recipe` entry point,
typed `@step` functions, and a `raise SystemExit(entry.cli())` footer:

```python
from calcifier import Budget, agent, emit, recipe, step


@step
def summarize(target: str) -> str:
    return agent(
        "worker",
        "Summarize the project at {target}.",
        working_directory=target,
        readable_paths=(target,),
        target=target,
    )


@recipe(version="0.1", agents=("worker",), budget=Budget(2, 120))
def project_summary(target: str) -> str:
    return emit(summarize(target), name="summary")


if __name__ == "__main__":
    raise SystemExit(project_summary.cli())
```

## Typed answers

An agent uses its enclosing step's return annotation, or an explicit `output=`,
to determine its answer type. `@output_type` defines a closed record:

```python
from typing import Annotated, Literal

from calcifier import output_type, word_count


@output_type
class Finding:
    file: str
    line: int
    severity: Literal["low", "medium", "high"]
    explanation: Annotated[str, word_count(10, 80)]
```

The host parses responses strictly and permits one retry on an invalid answer.
Extra record fields and incorrect value types fail. Constraints should reflect
the skill's requirements at the relevant stage. `emit(value, name="findings")`
exposes an artifact independently of the recipe's return value.

Provider schemas have object roots. Other roots use a `value` envelope; open
mappings and unconstrained values travel as encoded JSON that the host validates
against the full Python type. Codex receives the schema through its native
structured-output channel; Claude and custom adapters retain an inline directive.

## Agents and context

Declare permitted roles in `@recipe(agents=("worker", ...))` and call them with
`agent("worker", prompt, ...)`. Calls need a working directory and filesystem
paths; see [providers](providers.md#filesystem-access). File-based recipes pin
the original skill and references and supply them independently of stage prompts.

`prepare_context(task, instructions=..., prior_results=...)` saves a complete
task bundle. Pass it with `agent(..., context=context)`; using `context.prompt`
as the prompt sends the bundle once. Its working directory defaults to the
caller project, and target access still needs explicit paths. Include complete
records in `prior_results` when the agent must select from or refer to them.

Keep every source-required result field through rendering. Grouping records into
categories changes their presentation, not which conditions, qualifications, or
caller requirements the report retains. `model_dump()` preserves the complete
structured record; a custom renderer must select and display all fields required
by the source for every applicable record. Schema validation alone does not
check the rendered report.

For source files and manifests, use `prepare_context(..., files=selected_files,
readable_roots=(project,))`. It resolves and validates the entire selection
before reading any contents, rejecting files outside those roots, including
escaping symlinks. Internal symlinks remain valid. Include an external input's
root or exact path explicitly when the workflow authorizes it; pinned
`resource()` paths can be authorized the same way. Complete UTF-8 contents are
saved with their original line endings; the embedding budget does not truncate
the saved bundle. This helper enforces the declared selection, not a sandbox
around arbitrary trusted Python reads.

## Effects

Declare an `output_root` for report files and use `ensure_directory`, `write_text`,
or `write_json` beneath it. `verify_artifact` checks results without modifying
them. `resource("references/example.md")` resolves a pinned packaged file.

Before announcing saved delivery, call `verify_artifact` on the trusted declared
destination. Treat a worker's path as opaque report text. The helper checks a
regular, nonempty file and optionally its SHA-256; complete delivery also needs
an independent content check. Parse the saved report and compare its inventory
with the complete intended input set, including duplicates or omissions, or
compare against independently expected bytes with `sha256=`. A saved flag or
path string is insufficient. Failed verification must leave an explicit failure
or an alternate usable answer. Prefer host-owned writes when the worker can
return the report's structured content.

Before a subprocess writes artifacts, preflight every destination, including
intermediates, with `ensure_directory(..., output_files=("raw.json", "report.md"),
exclude_roots=(project,))`. File names are relative to the selected directory.
The complete set is checked before directory creation: destinations must remain
within that directory and the declared output root, avoid excluded roots and
symlinked files, and have no existing files or conflicting parents. Use
`overwrite=True` only when replacement is intended; it cannot bypass an
explicit output-directory override's existing-file protection. Calls without
`output_files` retain the ordinary directory helper's behavior. Preflight does
not sandbox a subprocess or protect against concurrent filesystem changes;
the supplied program must use those checked destinations. Verification after a
write checks the result and cannot undo an earlier overwrite.

Project modifications use a supplied program through `run_command()` or agent
`writable_paths`. Direct `Path.write_text()` calls are not admitted. Commands run
argv on the host without a shell; `mutating=True` declares an effect before
launch. Hard dry-run blocks declared mutations. Review these operations before
granting trust.

Classify commands by everything they can execute. An analysis wrapper that
compiles a target, runs project hooks, or loads executable configuration is
mutating even if its primary purpose is inspection. Make target builds an
explicit recipe input opt-in, and use an isolated environment prepared for the
target. Setting `mutating=True` declares effects and enables dry-run suppression;
it does not isolate the command. Keep a non-building or manual route when the
source workflow allows one.

`run_command(argv, timeout_seconds=..., output=ResultType)` requires exit zero
and validates stdout as strict JSON of that type. Without `output`, it returns
`CommandResult`. During compiler dry runs, read-only commands execute; mutating
commands are suppressed and their typed results synthesized. An unavailable
synthetic input can leave the dry run unverified. Synthesis neither verifies the
command nor creates its files. See [effect boundaries](architecture.md#effects-and-provider-boundaries).

When `ensure_directory(output_files=...)` predeclares a subprocess's outputs,
the compiler tracks those paths through suppression. If `verify_artifact` then
reaches a missing declared file, it reports an unverified coverage gap: no file
is fabricated, and content checks beyond that point remain unverified. An
undeclared or incorrect path, a missing output without a suppressed command,
an empty materialized file, or a hash mismatch remains a failure. Actual runtime
execution always requires the output file.
Caught verification gaps remain unverified even when an alternate answer is
produced; a later deterministic failure still fails the dry run.

For repeated stateful imports, inspect the supported dependency contract.
Aggregate compatible inputs into one import when each call replaces previous
state, or use an append API only when the installed supported release provides
it. Verify the final combined artifact against all input evidence. Per-input
success counters describe those operations and cannot establish that earlier
evidence survived later writes.

An unavailable executable or another process launch failure raises `RecipeError`
with `command_failure == "launch"`; a command's local timeout sets it to
`"timeout"`. Optional tools can fall back by catching `RecipeError` and checking
that attribute. Re-raise errors with `command_failure is None`: these include
invalid arguments, invalid typed output, nonzero typed-command exits, and output
limits. `BudgetExceeded` remains fatal when the shared deadline expires, even
if the command was optional. If a tool runs through an environment manager,
keep that complete argv prefix for both its availability probe and its actual
execution. A successful import in that environment does not make a bare command
available on the host.

## Budgets and parallel work

`Budget(max_agent_calls, max_wall_seconds)` limits attempts and elapsed time.
Include possible parse retries and child calls. Agent-backed recipes require a
budget. `fail("reason")` ends a recipe with a reported failure.

`Budget.remaining_wall_seconds()` returns the active run's effective time left,
including system suspension. Call it inside a budgeted recipe to allocate time
to remaining work; it raises `RecipeFailure` outside an active budgeted run.

`parallel(fn, items, on_result=callback)` propagates the run context and shared
budget, dispatches isolated workers, and returns results in input order.
Workers cannot mutate shared state. `on_result(item, result)` runs serially in
the parent context as workers finish, and may checkpoint outputs; async callbacks
are awaited. A worker failure stops further dispatch and cancels queued work;
successful in-flight results are checkpointed before the failure propagates.

`TransientAgentError` marks transport failures a recipe may retry in a bounded
loop; the host does not retry them automatically. Count those attempts in the
budget.

## Composition

Composition lets a parent recipe call a child recipe and receive its typed
output, emitted artifacts, and step records. The child must be a Calcifier
recipe with `_CALCIFIER_SKILL` provenance metadata; Calcifier does not load an
arbitrary Python module as a child. Parent and child share the top-level call
and wall budget.

### Calling children

`call_recipe(child_path, inputs, *, parent_trust=None)` runs one child from
inside a recipe or step body. A relative `child_path` resolves against the
parent recipe's directory. The parent path is inferred from the calling module;
if inference fails, the call fails. `parent_trust` names the parent recipe path
explicitly for tests and unusual embeddings.

`call_catalog_recipe(name, recipes, inputs)` selects a child from a closed,
literal name-to-path mapping. Generated parents use it for catalog families,
so an invented runtime name is rejected before any child is loaded.

Generation resolves explicit named-skill invocations through a version-1 skill
catalog and fails when a named dependency is unresolved. Generated Python calls
the child recipe rather than replacing its methodology with a generic prompt.

Both calls return a `SubrecipeResult` with `child_name`, `child_hash`,
`output` (the child's typed return value), `artifacts`, and `records`. Child
artifacts are not merged into the parent's automatically; the parent decides
what to `emit()`.

### Trust and pinning

Composition keeps trust-on-first-use. A child is Python that runs on the user's
machine once trusted, so the parent's trust entry must pin it. The pin lives in
the parent's entry in Calcifier's private per-user trust store, keyed by the
canonical parent path and complete-file digest, as a `dependencies` list of
`{path, sha256}` items. Each hash covers the complete child file, including its
metadata, computed from the single byte snapshot that is then executed.

- Generated parents carry `_CALCIFIER_SUBRECIPES` path/hash metadata from the
  generation-time catalog. `--trust` verifies those hashes and records them as
  dependencies. Repeatable `--subrecipe PATH` adds pins for hand-written
  parents.
- Pins belong to the exact parent body. Changing the parent requires new
  trust; changing a child invalidates the parent's authorization for it.
- At call time, an undeclared child fails before it is loaded, and a child (or
  any other listed dependency) whose current hash differs from its pin fails as
  dependency drift.
- A grandchild is authorized through the `_CALCIFIER_SUBRECIPES` metadata of
  its already-verified parent in the active call chain.
- Trust files next to a recipe are ignored: a publisher-controlled file cannot
  grant user consent.

The security boundary is review plus pinning. A pin proves that the reviewed
child body is the one being called; it does not make the child safe. Review
parent and child before recording trust.

### Run context and effects

A child runs with the parent's agent registry, tool registry, dry-run mode, and
shared budget state, so nesting cannot reset or evade the top-level limits. A
child that declares an `output_root` uses the parent's resolved output root
rather than a second root of its own, so its writes stay contained by the
top-level invocation.

Independent child calls may run through `parallel()`. Each mutating child call
in a parallel worker receives a private root at
`<output_root>/.calcifier-parallel/<lane>/call-NNNN-<child>`, so lanes cannot
race on shared files. Direct writes from parallel workers remain forbidden.

### Failures and cycles

If a child calls `fail()` or cannot be invoked safely, `call_recipe` raises
`SubrecipeError`, a `RecipeFailure`, so ordinary failure handling applies. If
the child crashed with an exception, it raises
`calcifier.composition.SubrecipeCrashError` instead; the parent then reports a
crash, and the compiler's dry run treats it as a defect to repair. The child's
original message is included in either error.

A context-local stack of recipe hashes tracks calls in flight. A child whose
hash is already on the stack (a self-call or a call back into an ancestor) is
rejected before it loads with an error starting `circular subrecipe call
detected`, followed by the call chain.

### Traces

Child step records nest at the call site. Inside a parent step, they are
appended to that step's `children`, and the step's `subrecipe` field records
the child's `name` and `hash`. Called outside a step, the child records attach
to the parent run's top-level records without a marker; call children inside a
step to keep the boundary visible. Every serialized step record has a
`subrecipe` field, `null` when the step is not a call site.
