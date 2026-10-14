# Security policy

## Supported versions

Security fixes target `main` and the latest released minor version.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private
vulnerability-reporting form from the repository's **Security** tab when it is
available. Otherwise, use the confidential contact listed in Trail of Bits'
[security.txt](https://www.trailofbits.com/.well-known/security.txt) and identify
the report as concerning Calcifier.

Include the affected version or commit, operating system, reproduction steps,
impact, and any suggested remediation. Avoid including real credentials,
private source code, or customer data in the initial report.

Maintainers will acknowledge and triage reports as soon as practical, keep the
reporter informed of material status changes, and coordinate disclosure after a
fix is available. Calcifier does not promise a fixed response or remediation
SLA.

## Security model

Calcifier reduces accidental execution and integrity risks; it is not a Python
sandbox.

- A recipe is executable Python. Calcifier hashes the complete file and refuses
  to execute it until the user explicitly trusts those exact bytes. Trust means
  consent to run that Python with the invoking user's permissions.
- The static AST gate protects the generation workflow from executing agent
  output before review. It does not make a trusted recipe safe or restrict what
  that recipe can do after it is trusted.
- Packaged resources and declared child recipes are hash-pinned. A successful
  integrity check does not establish that their contents are benign.
- Host agent CLIs, configured tools, and packaged scripts are subprocesses. They
  inherit the host working directory and environment. Use an ephemeral Coop VM
  for agent isolation and separately constrain host-side tools and scripts.
- `--dry-run` covers mutations that a recipe declares with
  `record_side_effect()`. It is an authoring aid, not an enforcement boundary
  for arbitrary Python or undeclared external behavior.
- Trace redaction is based on structured secret-like field names. Free text,
  filenames, prompts, provider output, and custom fields can still disclose
  sensitive data. Review traces before sharing them.
- Trust-store and cache permission checks strengthen local isolation. They do
  not protect against a compromised host, invoking user, provider executable,
  or operating-system account.

Security-sensitive behavior and accepted trade-offs are described further in
[the architecture notes](docs/architecture.md).
