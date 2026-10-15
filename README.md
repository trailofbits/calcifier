# calcifier

Calcifier turns Claude/Codex skills into executable Python workflows. Give it a
`SKILL.md`; it produces a recipe that you can read, review, run, and debug.
Python handles repeatable operations. Agents handle reasoning.

## Why use it?

A skill is a set of instructions that an agent interprets each time it runs.
That works well for exploration, but a recurring task often has a stable
procedure: gather files, run a script, ask for a judgment, then write a report.
Repeating that planning costs model calls and makes the procedure harder to
inspect.

Calcifier makes the procedure explicit. For example, a review skill can become
Python that selects files, calls an agent to assess them, validates the returned
findings, and writes the report. You can change the file selection or report
format without asking a model to invent the whole workflow again.

Use it for skills you run repeatedly, such as code reviews, data triage, and
report generation, when you want:

- **A workflow you own:** ordinary Python functions and control flow that humans
  can maintain.
- **Less repeated model work:** file operations, scripts, transformations, and
  rendering execute directly in Python.
- **Checked answers:** agent responses are parsed into typed values with explicit
  validation and a bounded retry.
- **Visible execution:** budgets, permissions, artifacts, and traces make it
  easier to see what ran and where it failed.

Agent answers remain nondeterministic. Compilation checks catch mechanical
errors; whether a recipe follows the skill and produces good answers needs
testing on real tasks. Recipes execute Python on your host,
so review generated code before granting trust.

## Try it

Install from PyPI. Requires Python 3.13+,
[uv](https://docs.astral.sh/uv/), and an authenticated Claude or Codex CLI
(or a custom agent adapter).

```bash
uv tool install calcifier
calcifier --version
```

Follow the [quickstart](docs/usage.md#quickstart) to configure agent roles,
compile a skill, review the generated recipe, and run it.

Compilation starts with a runnable baseline that gives one agent session the
intact skill and its resources. It then tries to move repeatable work into
Python. If that fails, it retains and labels the baseline; use
`--require-compiled` when you need compilation failure to exit nonzero.

The Claude Code plugin provides compilation, execution, debugging, and update
skills:

```text
/plugin marketplace add trailofbits/calcifier
/plugin install calcifier@calcifier
```

For local testing, use the checkout's absolute path as the marketplace.

## Commands and documentation

| Command | Purpose |
| ------- | ------- |
| `calcify SKILL.md` | Compile a skill and package its resources. |
| `exec recipe.py` | Run a reviewed, trusted recipe. |
| `debug recipe.py` | Run with verbose diagnostics. |
| `update SKILL.md` | Compile a candidate and print a diff without changing the package. |
| `lint recipe.py --portable` | Check recipe and dependency contracts. |
| `build-graph build.json` | Compile dependencies and publish a closed catalog. |
| `profile run.json` | Summarize timings, attempts, prompt sizes, and token use. |

Use `calcifier COMMAND --help` for flags.

- [Usage](docs/usage.md): setup, compilation, inputs, trust, outputs, and traces.
- [Recipe authoring](docs/recipes.md): typed answers, effects, budgets, parallel
  work, and composition.
- [Providers](docs/providers.md): agent roles and custom agent/tool contracts.
- [Architecture](docs/architecture.md): compiler and runtime internals.
- [Contributing](CONTRIBUTING.md): development, testing, and compiler replay.

## Development

```bash
uv --no-config sync --all-groups --locked
gmake verify
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SUPPORT.md](SUPPORT.md) for contribution
and support policies. Report vulnerabilities privately using
[SECURITY.md](SECURITY.md). Project interactions follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

[MIT](LICENSE), copyright Trail of Bits, Inc.
