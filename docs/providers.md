# Agent and tool providers

## Agent roles

`--agent-config PATH` maps semantic roles to built-in Claude/Codex CLIs or custom
argv commands. Every referenced role needs a configuration entry; provider
names cannot serve as role names. All current runtimes are `host`.

```json
{
  "version": 1,
  "agents": {
    "compiler": {
      "target": {"builtin": "codex"},
      "runtime": {"kind": "host"},
      "timeout_seconds": 3600,
      "model": "gpt-5.4",
      "reasoning_effort": "xhigh"
    },
    "worker": {
      "target": {"builtin": "claude"},
      "runtime": {"kind": "host"},
      "timeout_seconds": 900
    },
    "verifier": {
      "target": {"command": ["local-agent-adapter", "--model", "qwen"]},
      "runtime": {"kind": "host"},
      "timeout_seconds": 900
    }
  }
}
```

Built-in routes may pin `model`; Codex also accepts `reasoning_effort`. These
settings are passed on every call, fingerprinted, and included in telemetry.
Omitted settings are reported as `provider-default`. Pin them for comparable
runs. Compiler timeouts are per call and also subject to the
[shared compilation deadline](usage.md#compilation).

### Filesystem access

Every agent call declares a working directory and readable paths. Verified
skill resources are added automatically. Calls are read-only unless they declare
`writable_paths`: existing directories that contain the working directory.
Writable paths are also readable; `.git` stays read-only, and agents never
commit. Recipes use `run_command()` for host git operations.

Paths must come from trusted inputs or deterministic results, not agent or child
output. Concurrent writable agents need separate directories. Prepared context
grants access to its bundle, not the target project.

Built-in CLIs are feature-checked before use and run with clean settings and no
session persistence. Claude uses read-only discovery tools in plan mode for
read-only calls; writable calls add scoped Edit/Write rules and deny `.git`
changes. Bash and permission prompts are unavailable. Codex uses `read-only` or
`workspace-write`, with writable roots mapped through `-C` and `--add-dir`.
Its network access is off by default and `/tmp` is excluded from writable roots;
the per-user temporary directory remains available.

Codex authentication is read from the original `CODEX_HOME` (default `~/.codex`).
Calcifier honors `cli_auth_credentials_store`: `file` copies `auth.json`, `keyring`
reads the home-specific OS credential entry, and `auto` tries the keyring before
falling back to the file. Keychain on macOS, Secret Service on Linux, and Windows
Credential Manager are supported. Only authentication is staged in a private
temporary home with a mode-0600 `auth.json`; the child uses file storage and the
temporary home is removed after the call. Other user settings are not copied.
`CODEX_API_KEY` is preserved for Codex's non-interactive API-key override. A
nonblank value takes precedence over stored credentials; Calcifier skips reading
the credential store and staging `auth.json` in that case. Empty or whitespace-only
values leave the stored-login path in use. `OPENAI_API_KEY` also remains in the
child environment for compatibility; use `CODEX_API_KEY` for `codex exec`.
Process-only `ephemeral` credentials cannot be transferred to a new CLI process.
Unlock the OS credential store and check `codex login status` if keyring access
fails. Token refreshes in the temporary home are not written back to the original
credential store.

Recipe Python, host commands, and tool providers still run with host permissions.
Custom adapters must enforce the filesystem contract themselves. VM isolation
is not implemented.

### Worker tool capabilities

Generation and repair receive each configured role's tool support separately
from the recipe's host tool declarations. Built-in Claude supports `read` and
scoped `write`, while built-in Codex also supports `commands`. Neither built-in
route promises online `research`. Writes still require `writable_paths` on the
call; a capability declaration does not change filesystem permissions.

For a custom command adapter, declare the tools it actually implements:

```json
{
  "target": {"command": ["research-adapter"]},
  "runtime": {"kind": "host"},
  "timeout_seconds": 900,
  "capabilities": ["read", "research"]
}
```

The supported capability names are `read`, `write`, `commands`, and `research`.
An omitted declaration means unknown support. Built-in capability declarations
cannot be overridden in configuration. Custom declarations are adapter
contracts, not enforcement or a grant of additional tools.

A recipe can require worker-owned work with
`agent(..., required_capabilities=("research",))`. Missing or unknown support
fails before invoking or charging the worker. Compiler dry runs use the actual
configured declarations on their schema stubs, so a requirement mismatch
reaches repair instead of receiving a synthetic success. Host `--tool-config`
providers do not grant tools to workers. Put repeatable commands in host Python,
and configure a compatible adapter when the worker must perform online research.

### Custom agent protocol

Commands run without a shell and receive prompts on stdin. The environment
includes `CALCIFIER_AGENT_PROTOCOL=1` and a JSON `CALCIFIER_AGENT_CONTRACT`:

```json
{
  "version": 1,
  "filesystem": {
    "mode": "read-only",
    "working_directory": "/absolute/target",
    "readable_paths": ["/absolute/target", "/absolute/skill/references"],
    "writable_paths": []
  },
  "schema": {"type": "object", "properties": {}, "required": []},
  "timeout_seconds": 300
}
```

`filesystem` and `schema` are `null` when absent. Write one JSON envelope to
stdout: `{"version": 1, "output": "...", "usage": {...}}`. `output` is response
text; optional usage keys are `input_tokens`, `output_tokens`,
`cache_read_input_tokens`, `cache_creation_input_tokens`, `nested_model_tokens`,
and `cost_usd`. Values must be nonnegative and finite. Missing or invalid values
become `null`; explicit zero remains zero.

`timeout_seconds` is the effective process timeout, after applying any per-call
override and remaining shared wall budget. Adapters can use it to give their
agents a deadline; Calcifier still enforces the timeout externally.

## Tool providers

Recipes authorize tools with `@recipe(capabilities=(...))` and call
`tool(name, **arguments)`. Declarations may include unused capabilities.
Preflight requires providers for possible source calls; indirect dispatch keeps
all declared provider requirements. `--skip-tool-preflight` bypasses this check.

Supply subprocess providers with `--tool-config PATH`:

```json
{
  "version": 1,
  "capabilities": {
    "Slack.postMessage": {
      "command": ["slack-post-message"],
      "timeout_seconds": 60,
      "redact_args": ["token"],
      "redact_result": false,
      "remediation": "install slack-post-message"
    }
  }
}
```

The command receives `{"capability":"Slack.postMessage","arguments":{...}}`
on stdin and writes exactly one JSON value to stdout. It runs without a shell,
inheriting the host directory and environment. Stdout and stderr are each
limited to 1 MiB. The last 4 KiB of stderr may appear interactively but is excluded
from persisted trace errors.

Declare mutating tools with `record_side_effect()` before calling them so
dry-run policy can apply. Host `run_command()` executes directly and counts as
Bash use during compilation; it needs no Bash provider unless the recipe also
calls `tool("Bash", ...)`.
