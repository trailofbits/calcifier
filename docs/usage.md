# CLI usage

Each command's `--help` lists its flags.

## Quickstart

Save a role configuration as `agents.json`. The compiler writes Python; the
worker runs the skill's reasoning tasks. This example uses Codex for both:

```json
{
  "version": 1,
  "agents": {
    "compiler": {
      "target": {"builtin": "codex"},
      "runtime": {"kind": "host"},
      "timeout_seconds": 3600
    },
    "worker": {
      "target": {"builtin": "codex"},
      "runtime": {"kind": "host"},
      "timeout_seconds": 900
    }
  }
}
```

Compile a skill and review the generated file:

```bash
calcifier calcify path/to/SKILL.md --agent-config agents.json
less path/to/.calcifier/recipe.py
```

After review, grant trust and supply the inputs printed in the compilation
summary. Replace `name=value` with those inputs:

```bash
calcifier exec path/to/.calcifier/recipe.py --trust \
  --agent-config agents.json --input name=value --trace-out run.json
```

Subsequent runs omit `--trust` unless the recipe or pinned sources change. For
recipes that write reports, choose a fresh, dedicated folder with
`--output-dir ./calcifier-report` before running.

## Compilation

`calcify SKILL.md` writes `.calcifier/recipe.py` beside the skill by default;
`--output PATH` selects another destination. `update SKILL.md` uses the same
compiler and prints a diff against the existing recipe (`--recipe PATH`
overrides its location), without writing to the skill package. Both attempt a
host dry run of the candidate.

Both commands need `--agent-config PATH`, a configured compiler role (default
`compiler`), and at least one other runtime role. Generated recipes can use only
configured runtime role names. The Python compilation API permits an empty
role set for deterministic recipes.

SKILL.md `allowed-tools` records pre-approved tools and optional scopes such as
`Bash(git:*)`; it is not an exclusive allowlist. Malformed declarations fail
before a model call. Runtime authorization uses `@recipe(capabilities=...)`.

| Limit | Default | Override |
| ----- | ------- | -------- |
| Compiler calls, including repairs | 5 from the CLI | `--max-compiler-calls N` (1–8) |
| Shared deadline per recipe | 3,600 seconds | `--compilation-timeout SECONDS` |
| Per-call timeout | Compiler role's `timeout_seconds` | `--timeout SECONDS` |

`--timeout` takes precedence over `--agent-timeout ROLE=SECONDS`; neither extends
the shared deadline. Runtime workers have separate timeouts and a recipe budget.

A failed optimization retains a runnable baseline with outcome
`baseline_retained`; `--require-compiled` makes `calcify` exit nonzero while
preserving that fallback. If any `build-graph` node falls back, all nodes are
published as baselines. See [budgets and outcomes](architecture.md#budgets-and-outcomes).

`--compiler-trace PATH` saves attempt diagnostics and telemetry; `--verbose`
prints progress to stderr. The CLI reports calls, usage, time, and the recipe's
inputs, steps, and budget. Unreported usage is `null`, not zero.
`--python-version` overrides the destination's minimum Python version, normally
resolved from Ruff's `target-version` or the project's `requires-python`.

## Execution

`exec recipe.py` and `debug recipe.py` use the same inputs and permissions;
`debug` enables verbose logs. Repeat `--input NAME=VALUE` for recipe parameters.
Values accept JSON scalars, arrays, and objects, then undergo annotation-based
coercion. `--preflight-only` reports configuration and the output destination
without running the recipe.

`--agent-config PATH` is required when the recipe declares agents.
`--agent-timeout ROLE=SECONDS` overrides a role timeout. `--max-wall-seconds`
shortens the declared wall budget, or adds a wall limit to an unbudgeted
deterministic recipe; it adds no agent-call limit.

Recipes also run directly with `python recipe.py --parameter-name VALUE`.
Parameters become kebab-case flags; booleans have `--name` and `--no-name`.
Direct execution bypasses the CLI trust check. Use `calcifier exec` for reviewed
file-based execution.

### Trust

Review the complete Python file before using `--trust`. Consent pins the exact
file bytes, packaged resources, and declared child recipes in a private per-user
trust store. Changed files require renewed trust. Repeat `--subrecipe PATH` to
pin children of hand-written parents; generated parents carry their own pins.
Adjacent `recipe.py.trust.json` files have no authority.

`CALCIFIER_TRUST_STORE` overrides the store location for isolated tests or CI.
See [trust-on-first-use](architecture.md#trust-on-first-use) and
[composition](recipes.md#trust-and-pinning).

### Output directories

`--output-dir PATH` redirects a declared `output_root`; it cannot give an
otherwise read-only recipe permission to write. Relative output paths resolve
under that root, writes outside it fail, and `.git` paths are always rejected
case-insensitively. Nested recipes inherit the parent's root.

Use a dedicated, fresh directory: returned filenames may come from an agent.
With `--output-dir`, existing files are protected, while files created during
the run can be updated. Recipes can explicitly adopt byte-identical files with
`reuse_existing=True`; adoption is recorded without rewriting the file.
The output directory appears in preflight, run results, and traces.

### Dry runs

`--dry-run soft` records declared host mutations without performing them;
configured agents still run. `--dry-run hard` blocks declared mutations and
agent calls with writable paths. Mutating tools must declare their effect with
`record_side_effect()` before invocation. Trusted Python and custom providers
run on the host; dry-run modes are not sandboxes.

The compiler's separate dry run uses schema-stub agents and scratch files to
exercise the whole recipe. Its `passed`, `failed`, and `unverified` outcomes are
explained in the [compilation pipeline](architecture.md#compilation-pipeline).

### Traces

`--trace-out PATH` saves a version-1 JSON trace; `profile PATH` summarizes it.
The trace includes state (`running`, `completed`, `failed`, or `interrupted`),
steps, artifacts, effects, role routing, attempts, deadlines, and usage.
Agent-call records omit raw prompts and responses, but parsed outputs remain.

Checkpoints flush after attempts, artifacts, effects, and failures. Routine
progress coalesces over 250 ms; only the terminal trace undergoes schema
validation. Graceful interruption flushes state, while an uncatchable kill may
lose recent telemetry.

Secret-like field names are redacted, including `github_token`, `client-secret`,
`authorization`, `cookie`, and `session_id`. Free text and filenames are not
scanned. Review traces before sharing. See [redaction](architecture.md#redaction).

### Profiling

`calcifier profile run.json` ranks agent calls by time and reports attempts,
prompt and schema bytes, token use, and summed agent and tool time. Summed call
time can exceed wall time when calls overlap; it is not a critical path. When
some attempts lack final usage, the profile labels the total as observed and
shows report coverage, or says `unknown`. Missing usage is never a measured zero.

### Token accounting

Calcifier normalizes provider usage into `input_tokens`, `output_tokens`,
`cache_read_tokens`, `cache_write_tokens`, `nested_model_tokens`, and
`cost_usd`, but the providers define these fields differently. Do not add them
across providers, or within one provider, without applying these rules:

| | Codex | Claude |
| --- | --- | --- |
| `input_tokens` | Total input, including cached tokens | Ordinary (uncached) input only |
| `cache_read_tokens`, `cache_write_tokens` | Subsets of `input_tokens` | Additional input components |
| Total input | `input_tokens` | `input_tokens + cache_read_tokens + cache_write_tokens` |
| Fresh (non-cache-read) input | `input_tokens - cache_read_tokens` | `input_tokens + cache_write_tokens` |
| Ordinary input | `input_tokens - cache_read_tokens - cache_write_tokens` | `input_tokens` |
| `nested_model_tokens` | Reasoning tokens, already included in `output_tokens` | Thinking tokens, already included in `output_tokens` |
| `cost_usd` | Not reported (`null`) | CLI-reported list-price estimate |

Never add `nested_model_tokens` to `output_tokens`. A missing cache-write field
is unknown, not zero. Custom adapters report whatever their backend reports;
document their convention alongside any comparison.

Costs computed from API list prices, and the cost a CLI reports, are estimates.
They are not subscription invoices and do not verify quota consumption.
Label them as estimated list cost.
