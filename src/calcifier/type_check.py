"""Type-check generated recipes with ty against the installed Calcifier API."""

from __future__ import annotations

import functools
import re
import sys
import tempfile
from pathlib import Path

from .diagnostics import Diagnostic
from .source_quality import DEFAULT_PYTHON_VERSION, SourceQualityError, python_feature_version
from .subprocesses import ProcessError, run_process

_TY_TIMEOUT_SECONDS = 60
_TY_OUTPUT_LIMIT = 4 * 1024 * 1024
_RECIPE_NAME = "recipe.py"
_TY_CONFIG = "[analysis]\nrespect-type-ignore-comments = false\n"
# These rules identify failures that occur whenever the affected line runs.
_BLOCKING_TY_RULES = frozenset(
    {
        "unresolved-reference",
        "unresolved-import",
        "call-non-callable",
        "missing-argument",
        "unknown-argument",
        "too-many-positional-arguments",
        "parameter-already-assigned",
        "positional-only-parameter-as-kwarg",
        "invalid-await",
        "invalid-syntax",
    }
)
# ty prints each finding relative to the checked file's directory, which keeps
# Windows drive letters out of the path.
_FINDING = re.compile(
    rf"^{re.escape(_RECIPE_NAME)}:(?P<line>\d+):\d+: "
    r"(?P<severity>error|warning|info)\[(?P<rule>[a-z-]+)\] (?P<message>.*)$"
)


def type_check_source(source: str, *, python_version: str | None = None) -> tuple[Diagnostic, ...]:
    """Report certain runtime failures identified by ty in a generated recipe.

    Check against the installed runtime and target Python version. Return blocking
    errors in ty's order, ignoring warnings. Tool failures raise SourceQualityError.
    """
    return _type_check(source, python_version, sys.executable)


@functools.lru_cache(maxsize=256)
def _type_check(
    source: str, python_version: str | None, interpreter: str
) -> tuple[Diagnostic, ...]:
    with tempfile.TemporaryDirectory(prefix="calcifier-ty-") as directory:
        root = Path(directory)
        (root / _RECIPE_NAME).write_text(source, encoding="utf-8", newline="\n")
        # A config file of our own keeps user and project ty settings out of the check,
        # and generated source may not silence its own findings with type-ignore comments.
        (root / "ty.toml").write_text(_TY_CONFIG, encoding="utf-8", newline="\n")
        argv = [
            _ty_bin(),
            "check",
            "--project",
            str(root),
            "--config-file",
            str(root / "ty.toml"),
            "--python",
            interpreter,
            "--output-format",
            "concise",
            "--no-progress",
        ]
        feature = python_feature_version(python_version) or DEFAULT_PYTHON_VERSION
        argv.extend(("--python-version", f"{feature[0]}.{feature[1]}"))
        argv.append(_RECIPE_NAME)
        try:
            result = run_process(
                argv,
                timeout_seconds=_TY_TIMEOUT_SECONDS,
                max_output_bytes=_TY_OUTPUT_LIMIT,
                cwd=str(root),
            )
        except ProcessError as exc:
            raise SourceQualityError(f"ty execution failed: {exc}") from exc
    output = result.stdout.decode(errors="replace")
    matches = [match for match in map(_FINDING.match, output.splitlines()) if match is not None]
    # Exit 1 means "diagnostics found" at any severity; anything else, or exit 1 with
    # nothing parsed, is a failure.
    if result.returncode not in (0, 1) or (result.returncode == 1 and not matches):
        detail = result.stderr_tail.decode(errors="replace").strip() or output.strip()
        raise SourceQualityError(f"ty failed with exit status {result.returncode}: {detail}")
    return tuple(
        Diagnostic(line=int(match["line"]), actual=f"ty[{match['rule']}]: {match['message']}")
        for match in matches
        if match["severity"] == "error" and match["rule"] in _BLOCKING_TY_RULES
    )


@functools.cache
def _ty_bin() -> str:
    """Resolve the ty binary bundled with the ``ty`` package."""
    from ty.__main__ import find_ty_bin

    return str(find_ty_bin())
