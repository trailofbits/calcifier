"""Ruff formatting and correctness linting for generated recipes."""

from __future__ import annotations

import ast
import functools
import io
import json
import re
import tokenize
import tomllib
import warnings
from collections.abc import Sequence
from pathlib import Path

from .diagnostics import Diagnostic, render_diagnostics
from .exceptions import CalcifierError, ErrorFamily
from .subprocesses import ProcessError, ProcessResult, run_process

PYTHON_VERSION_CHOICES = ("3.11", "3.12", "3.13", "3.14")
DEFAULT_PYTHON_VERSION = (3, 13)
_RUFF_TIMEOUT_SECONDS = 30
_RUFF_OUTPUT_LIMIT = 4 * 1024 * 1024
# Only runtime correctness findings reject generated recipes.
_CORRECTNESS_RULES = "E9,F63,F7,F82"


class SourceQualityError(CalcifierError, ValueError):
    """Raised when Ruff cannot produce and validate a deterministic artifact.

    Lint failures carry each remaining finding and the exact source Ruff checked,
    so callers can attribute findings to the functions that contain them.
    """

    family = ErrorFamily.COMPILATION

    def __init__(
        self,
        message: str,
        *,
        diagnostics: tuple[Diagnostic, ...] = (),
        source: str = "",
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics
        self.source = source


def python_feature_version(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    normalized = value.removeprefix("py")
    if "." in normalized:
        major, minor = normalized.split(".", 1)
    elif normalized.startswith("3") and len(normalized) == 3:
        major, minor = normalized[0], normalized[1:]
    else:
        raise SourceQualityError(f"unsupported minimum Python version: {value!r}")
    version = f"{major}.{minor}"
    if version not in PYTHON_VERSION_CHOICES:
        raise SourceQualityError(f"unsupported minimum Python version: {value!r}")
    return int(major), int(minor)


def resolve_python_version(path: str | Path, explicit: str | None = None) -> str | None:
    """Resolve an explicit version or the nearest Ruff target-version setting."""
    if explicit is not None:
        return _normalized_python_version(explicit)
    target = Path(path).resolve()
    for directory in target.parents:
        for name in (".ruff.toml", "ruff.toml", "pyproject.toml"):
            config = directory / name
            if not config.is_file():
                continue
            recognized, version = _python_version_from_config(
                config,
                pyproject=name == "pyproject.toml",
            )
            if recognized:
                return version
    return None


_RUFF_DEFAULT_LINE_LENGTH = 88


def resolve_line_length(path: str | Path) -> int:
    """Return the line length Ruff applies to ``path``.

    This is the ``line-length`` of the nearest Ruff configuration, found the same
    way as the target version, or Ruff's default when none sets it.
    """
    for directory in Path(path).resolve().parents:
        for name in (".ruff.toml", "ruff.toml", "pyproject.toml"):
            settings = _ruff_settings(directory / name)
            if settings is None:
                continue
            value = settings.get("line-length")
            valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
            return value if valid else _RUFF_DEFAULT_LINE_LENGTH
    return _RUFF_DEFAULT_LINE_LENGTH


def _ruff_settings(config: Path) -> dict[str, object] | None:
    """Return a file's Ruff settings, or None when it is not a Ruff configuration."""
    if not config.is_file():
        return None
    try:
        payload = tomllib.loads(config.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    if config.name != "pyproject.toml":
        return payload
    tool = payload.get("tool")
    ruff = tool.get("ruff") if isinstance(tool, dict) else None
    return ruff if isinstance(ruff, dict) else None


def _normalized_python_version(value: str) -> str:
    feature = python_feature_version(value)
    if feature is None:  # pragma: no cover - callers pass a concrete version
        raise SourceQualityError("minimum Python version is required")
    return f"{feature[0]}.{feature[1]}"


def _python_version_from_config(
    config: Path,
    *,
    pyproject: bool,
) -> tuple[bool, str | None]:
    try:
        payload = tomllib.loads(config.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return True, None
    tool = payload.get("tool")
    has_ruff_section = (
        pyproject and isinstance(tool, dict) and "ruff" in tool and isinstance(tool["ruff"], dict)
    )
    ruff = tool["ruff"] if has_ruff_section else payload if not pyproject else {}
    if isinstance(ruff, dict) and isinstance(ruff.get("target-version"), str):
        return True, _normalized_python_version(ruff["target-version"])
    if not pyproject:
        return True, None
    project = payload.get("project")
    requires = project.get("requires-python") if isinstance(project, dict) else None
    match = re.search(r">=\s*(3\.1[1-4])", requires) if isinstance(requires, str) else None
    if match is not None:
        return True, match.group(1)
    return has_ruff_section, None


def format_and_lint_source(
    source: str,
    path: str | Path,
    *,
    python_version: str | None = None,
) -> str:
    """Apply Ruff's safe fixes and formatting, then reject any remaining finding.

    Formatting follows the destination configuration; linting checks only
    runtime correctness, independently of the destination's lint policy.
    """
    formatted = format_source(source, path, python_version=python_version)
    diagnostics = lint_source_quality(formatted, path, python_version=python_version)
    if diagnostics:
        raise quality_error("Ruff validation failed", diagnostics, formatted)
    return formatted


def format_source(
    source: str,
    path: str | Path,
    *,
    python_version: str | None = None,
) -> str:
    """Apply destination-configured safe fixes and formatting."""
    target = Path(path).resolve()
    python_version = resolve_python_version(target, python_version)
    fixed = _ruff_filter(
        "check",
        source,
        target,
        python_version=python_version,
        extra=("--fix-only", "--no-unsafe-fixes"),
    )
    formatted = _ruff_filter("format", fixed, target, python_version=python_version)
    wrapped = _wrap_long_literals(formatted, resolve_line_length(target))
    if wrapped == formatted:
        return formatted
    return _ruff_filter("format", wrapped, target, python_version=python_version)


def _wrap_long_literals(source: str, line_length: int) -> str:
    """Split ordinary literals inside parentheses only when their AST is unchanged."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        try:
            original = ast.dump(ast.parse(source))
            edits = _literal_edits(source, line_length)
            if not edits:
                return source
            wrapped = source
            for start, end, replacement in reversed(edits):
                wrapped = wrapped[:start] + replacement + wrapped[end:]
            if ast.dump(ast.parse(wrapped)) == original:
                return wrapped
        except (
            SyntaxError,
            SyntaxWarning,
            tokenize.TokenError,
            ValueError,
            UnicodeError,
            RecursionError,
        ):
            pass  # Leave unsupported source for the existing validation/repair path.
    return source


def _literal_edits(source: str, line_length: int) -> list[tuple[int, int, str]]:
    lines = io.StringIO(source).readlines()
    offsets, position = [], 0
    for line in lines:
        offsets.append(position)
        position += len(line)
    edits, depth = [], 0
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.OP:
            depth += int(token.string in "([{") - int(token.string in ")]}")
        if not (
            token.type == tokenize.STRING
            and depth > 0
            and token.start[0] == token.end[0]
            and token.string.startswith(("'", '"'))
            and len(token.line.rstrip("\r\n").encode()) > line_length
        ):
            continue
        width = line_length - len(token.line[: token.start[1]].encode()) - 4
        if width < 8:
            continue
        parts = _literal_parts(ast.literal_eval(token.string), width)
        if len(parts) > 1:
            offset = offsets[token.start[0] - 1]
            edits.append(
                (
                    offset + token.start[1],
                    offset + token.end[1],
                    ("\n" + " " * token.start[1]).join(parts),
                )
            )
    return edits


def _literal_parts(value: str, width: int) -> list[str]:
    parts, current, size = [], "", 2
    for character in value:
        escaped = '\\"' if character == '"' else repr(character)[1:-1]
        length = len(escaped.encode())
        if current and size + length > width:
            parts.append('"' + current + '"')
            current, size = "", 2
        current += escaped
        size += length
    parts.append('"' + current + '"')
    return parts


def quality_error(title: str, diagnostics: Sequence[Diagnostic], source: str) -> SourceQualityError:
    """Render the first twenty findings, then count the rest, for one checked source."""
    return SourceQualityError(
        f"{title}: {render_diagnostics(diagnostics)}",
        diagnostics=tuple(diagnostics),
        source=source,
    )


def lint_source_quality(
    source: str,
    path: str | Path,
    *,
    python_version: str | None = None,
) -> list[Diagnostic]:
    """Report target-version syntax and runtime correctness diagnostics without writes."""
    target = Path(path).resolve()
    python_version = resolve_python_version(target, python_version)
    check = _run_ruff(
        "check",
        source,
        target,
        python_version=python_version,
        extra=("--output-format", "json"),
    )
    diagnostics = _parse_ruff_diagnostics(check.stdout)
    if check.returncode != 0 and not diagnostics:
        diagnostics.append(Diagnostic(line=1, actual=_process_failure_message(check)))
    return diagnostics


def _ruff_filter(
    command: str,
    source: str,
    target: Path,
    *,
    python_version: str | None,
    extra: tuple[str, ...] = (),
) -> str:
    result = _run_ruff(
        command,
        source,
        target,
        python_version=python_version,
        extra=extra,
    )
    if result.returncode != 0:
        raise SourceQualityError(_process_failure_message(result))
    try:
        return result.stdout.decode()
    except UnicodeDecodeError as exc:
        raise SourceQualityError("Ruff returned non-UTF-8 source") from exc


def _run_ruff(
    command: str,
    source: str,
    target: Path,
    *,
    python_version: str | None,
    extra: tuple[str, ...],
) -> ProcessResult:
    # --stdin-filename lets formatting discover the destination's configuration.
    argv = [_ruff_bin(), command, "--no-cache", "--stdin-filename", str(target), *extra]
    if command == "check" and "--fix-only" not in extra:
        # Generated source may not silence its own findings with noqa comments.
        argv.extend(("--isolated", "--select", _CORRECTNESS_RULES, "--ignore-noqa"))
    feature_version = python_feature_version(python_version) or DEFAULT_PYTHON_VERSION
    argv.extend(("--target-version", f"py{feature_version[0]}{feature_version[1]}", "-"))
    try:
        return _run_ruff_process(tuple(argv), source, str(_existing_parent(target)))
    except ProcessError as exc:
        detail = exc.stderr_tail.decode(errors="replace").strip()
        raise SourceQualityError(f"Ruff execution failed: {detail or exc}") from exc


@functools.cache
def _ruff_bin() -> str:
    """Resolve the bundled Ruff binary once; ``python -m ruff`` costs an extra interpreter."""
    from ruff.__main__ import find_ruff_bin

    return str(find_ruff_bin())


def _run_ruff_process(argv: tuple[str, ...], source: str, cwd: str) -> ProcessResult:
    # Let formatting rediscover the current configuration instead of caching old results.
    return run_process(
        argv,
        stdin=source.encode(),
        timeout_seconds=_RUFF_TIMEOUT_SECONDS,
        max_output_bytes=_RUFF_OUTPUT_LIMIT,
        cwd=cwd,
    )


def _parse_ruff_diagnostics(stdout: bytes) -> list[Diagnostic]:
    try:
        payload = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceQualityError("Ruff returned malformed JSON diagnostics") from exc
    if not isinstance(payload, list):
        raise SourceQualityError("Ruff returned non-list JSON diagnostics")
    diagnostics: list[Diagnostic] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        location = item.get("location")
        line = location.get("row", 1) if isinstance(location, dict) else 1
        code = item.get("code") or "Ruff"
        message = item.get("message") or "validation failure"
        diagnostics.append(Diagnostic(line=int(line), actual=f"{code}: {message}"))
    return diagnostics


def _process_failure_message(result: ProcessResult) -> str:
    stderr = result.stderr_tail.decode(errors="replace").strip()
    stdout = result.stdout.decode(errors="replace").strip()
    detail = stderr or stdout or f"exit status {result.returncode}"
    return f"Ruff validation failed: {detail}"


def _existing_parent(path: Path) -> Path:
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent
