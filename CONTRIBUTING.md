# Contributing to Calcifier

Thank you for helping improve Calcifier. Bug reports, documentation fixes, test
cases, and focused code changes are welcome.

By submitting a contribution, you confirm that you have the right to provide
it under the project's [MIT license](LICENSE). Calcifier does not currently
require a separate contributor license agreement or Developer Certificate of
Origin sign-off.

## Before starting

- Search existing issues before opening a new one.
- Open an issue before undertaking a large feature or public API change so the
  design can be discussed first.
- Report security vulnerabilities privately according to [SECURITY.md](SECURITY.md).
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md) in all project interactions.

## Development setup

Calcifier requires Python 3.13 or newer and
[`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/trailofbits/calcifier.git
cd calcifier
uv --no-config sync --all-groups --locked
gmake verify
```

## Testing

Run focused tests while editing, then `gmake verify`. It checks the lock, Ruff,
ty, portable plugin recipes, and the full test suite. CI tests Python 3.13 and
3.14 on Ubuntu and Python 3.13 on Windows.

Tests use stub providers and must not invoke real Claude or Codex processes.
`tests/conftest.py` blocks network connections and live provider executables and
gives each test a temporary trust store. Shared stubs and neutral compiler cases
live in `tests/support/`. `tests/corpus/` exercises scripted drafts through the
compiler gates, whole-recipe stub execution, portable lint, and runtime checks.
These tests check mechanics, not answer quality.

Use explicit UTF-8 and newline settings for byte-exact fixtures. Cover LF,
CRLF, and non-ASCII resources; hash the bytes actually written. Keep path tests
portable across POSIX and Windows. Reduce observed failures to neutral source
and data, with both positive and negative cases.

Run coverage serially; CI requires at least 90% combined statement/branch coverage:

```bash
uv run --frozen coverage run --branch -m pytest
uv run --frozen coverage report
```

Mutation testing runs in CI. Its sharded reports show
per-module kill rates without enforcing a threshold. Only executed failing
tests count as kills; timeouts and runner errors count as misses. A green report
job does not mean every module reached 90%.

Audit the locked dependencies with:

```bash
uv export --frozen --all-groups --format requirements-txt \
  --no-emit-project -o /tmp/calcifier-requirements.txt
uvx pip-audit -r /tmp/calcifier-requirements.txt --disable-pip --strict
```

For packaging changes, build into a fresh directory outside the checkout, then
check both distributions:

```bash
uv build --out-dir /tmp/calcifier-dist
uv run --frozen python scripts/verify_artifacts.py /tmp/calcifier-dist
uvx twine check /tmp/calcifier-dist/*
```

## Compiler replay

`scripts/compiler_replay.py` records compiler requests and exact replies so local
checks can be repeated without another model call. Keep recordings outside Git:

```bash
uv run --frozen python scripts/compiler_replay.py record path/to/SKILL.md \
  /tmp/compiler-packet --agent-config /path/to/agents.json
uv run --frozen python scripts/compiler_replay.py replay /tmp/compiler-packet \
  --output /tmp/compiler-replay
```

Recording calls the real compiler, preserves source/resource bytes and rejected
replies, and uses at most five compiler calls. `--compilation-timeout` sets the
shared deadline (default 3,600 seconds); `--timeout` limits each call (default
900). Replay constructs no live provider, runs static checks and the whole-recipe
stub dry run, and leaves the packet unchanged. Output directories must be new.

Changed prompts or schemas stop replay, except for the random source-delimiter
nonce. Use `--allow-request-changes` to apply old replies to intentionally changed
requests; the differences are recorded. `--stop-after N` checks a prefix.
Changed phases, missing replies, and incomplete responses always stop replay.

Results contain `result.json`, `trace.json`, and `requests.json`, plus accepted
`candidate.py` or rejected `rejected.py` when available. Exit zero means
`compiler_accepted` or `prefix_complete`. Acceptance means the local compiler
accepted those replies, not that the recipe produces correct answers. Replay
does not exercise baseline preparation, final publication, provider tools, or
historical latency and cost. Use `--help` for the remaining options.

## Live acceptance

Acceptance with real providers is an optional maintainer check, separate from
the offline suite. `scripts/acceptance.py` compiles and executes afresh on each
repetition using owner-provided fixtures:

```bash
uv run --frozen python scripts/acceptance.py \
  --agent-config /absolute/path/agents.json \
  --fixtures /absolute/path/fixtures --repeat 3 --timeout 600 \
  --output /tmp/calcifier-acceptance
```

Each immediate fixture directory contains `fixture.json` and optionally a
`workspace/`. The required fields are `skill` (an absolute `SKILL.md` path),
`request` (the task), and `check` (an argv array for an independent outcome check).
Optional `setup`, `path_prepend`, and `exec_timeout_seconds` configure preparation
and execution. Unknown fields fail. `{fixture}`, `{workspace}`, and `{case}` in
commands expand to absolute paths. Checks receive `CALCIFIER_RESULT`,
`CALCIFIER_WORKSPACE`, and `CALCIFIER_FIXTURE`. `--help` describes limits and
offline fallback mode. Keep output outside fixtures and skill packages.

## Pull requests

- Keep each pull request focused and explain the user-visible behavior it
  changes.
- Add regression tests for bug fixes and tests for new behavior.
- Update the README, architecture documentation, and `CHANGELOG.md` when a
  public contract changes.
- Run `gmake verify` before requesting review. Run the relevant coverage and
  package checks for security- or release-sensitive changes. CI reports the
  per-module mutation kill rate without enforcing a threshold.
- Keep compiler and runtime changes skill- and provider-agnostic: fix the
  general mechanism behind a failure rather than adding a rule, prompt sentence,
  or test keyed to one skill.
- Never commit credentials, user traces, trust stores, caches, generated
  workstation state, or benchmark result directories. Keep evidence outside Git.

The `main` ruleset requires an approving review, resolved conversations, and
the configured CI checks before merge.
