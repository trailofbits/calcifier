from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from calcifier import source_quality
from calcifier.source_quality import (
    DEFAULT_PYTHON_VERSION,
    SourceQualityError,
    _parse_ruff_diagnostics,
    _wrap_long_literals,
    format_and_lint_source,
    lint_source_quality,
    python_feature_version,
    resolve_line_length,
    resolve_python_version,
)


def test_default_python_version_matches_the_runtime_requirement() -> None:
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    requires = tomllib.loads(project.read_text(encoding="utf-8"))["project"]["requires-python"]

    assert requires == f">={DEFAULT_PYTHON_VERSION[0]}.{DEFAULT_PYTHON_VERSION[1]}"


def test_default_python_target_accepts_pep701_fstrings(tmp_path: Path) -> None:
    source = "def run(value: str) -> str:\n    return f\"{value + '\\n'}\"\n"
    target = tmp_path / "recipe.py"

    assert resolve_python_version(target) is None
    assert format_and_lint_source(source, target) == source
    assert lint_source_quality(source, target) == []
    with pytest.raises(SourceQualityError, match="backslash"):
        format_and_lint_source(source, target, python_version="3.11")


def _ruff_config(root: Path, *, target: str = "py311") -> None:
    (root / "ruff.toml").write_text(
        f'target-version = "{target}"\nline-length = 80\n\n[lint]\nselect = ["E", "F", "I"]\n'
    )


def test_resolve_python_version_uses_nearest_target_configuration(tmp_path: Path):
    _ruff_config(tmp_path)
    nested = tmp_path / "skill" / ".calcifier" / "recipe.py"

    assert resolve_python_version(nested) == "3.11"
    assert resolve_python_version(nested, "3.13") == "3.13"
    assert python_feature_version("py312") == (3, 12)


def test_resolve_python_version_does_not_cross_a_nearer_ruff_config(tmp_path: Path):
    _ruff_config(tmp_path, target="py311")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ruff.toml").write_text("line-length = 88\n")

    assert resolve_python_version(nested / "recipe.py") is None


def test_format_and_lint_source_applies_the_destination_ruff_config(tmp_path: Path):
    _ruff_config(tmp_path)
    target = tmp_path / ".calcifier" / "recipe.py"
    raw = (
        "import os; import json\n"
        "VALUE = json.dumps(os.name)\n"
        f"_CALCIFIER_PLAN = {{'nodes': {list(range(30))!r}}}\n"
    )

    formatted = format_and_lint_source(raw, target)

    assert formatted.startswith("import json\nimport os\n")
    assert ";" not in formatted
    assert max(len(line) for line in formatted.splitlines()) <= 80
    assert "_CALCIFIER_PLAN = {" in formatted
    assert lint_source_quality(formatted, target) == []
    assert format_and_lint_source(formatted, target) == formatted


def test_safe_destination_fixes_preserve_runtime_values(tmp_path: Path) -> None:
    (tmp_path / "ruff.toml").write_bytes(
        b'target-version = "py311"\n[lint]\nselect = ["I", "UP017", "SIM102"]\n'
    )
    source = (
        "import json\nfrom datetime import timezone, datetime\n"
        "def run(value: bool) -> str:\n"
        "    if value:\n        if value is not None:\n"
        "            return json.dumps(str(timezone.utc))\n"
        "    return str(datetime(2020, 1, 1, tzinfo=timezone.utc))\n"
    )
    formatted = format_and_lint_source(source, tmp_path / "recipe.py")
    assert "timezone.utc" not in formatted
    before, after = {}, {}
    exec(source, before)  # noqa: S102 - execute this fixed, neutral semantic regression
    exec(formatted, after)  # noqa: S102 - compare Ruff's safe rewrite of the same fixed code
    for value in (True, False):
        assert before["run"](value) == after["run"](value)


def test_destination_cannot_enable_unsafe_fixes(tmp_path: Path) -> None:
    (tmp_path / "ruff.toml").write_bytes(
        b'unsafe-fixes = true\n[lint]\nselect = ["F841", "B905"]\n'
    )
    source = "def run():\n    unused = 1\n    return list(zip([1], [2]))\n"
    formatted = format_and_lint_source(source, tmp_path / "recipe.py")
    assert "unused = 1" in formatted
    assert "zip([1], [2])" in formatted
    assert lint_source_quality(formatted, tmp_path / "recipe.py") == []


@pytest.mark.parametrize(
    "source",
    [
        "import os\n",
        "def run():\n    unused = 1\n    return 2\n",
        "value = None\nresult = value == None\n",
        "identity = lambda value: value\n",
        "value = 1\nimport os\nresult = os.name\n",
        'value = f"plain text"\n',
        "l = 1\n",
    ],
    ids=["F401", "F841", "E711", "E731", "E402", "F541", "E741"],
)
def test_harmless_ruff_findings_do_not_reject_candidates(tmp_path: Path, source: str) -> None:
    target = tmp_path / "recipe.py"

    assert lint_source_quality(source, target) == []
    formatted = format_and_lint_source(source, target)
    assert lint_source_quality(formatted, target) == []


@pytest.mark.parametrize("source", ["result = missing\n", "def broken(:\n    pass\n"])
@pytest.mark.parametrize(
    "config",
    [
        None,
        b'[lint]\nselect = ["C901"]\n',
        b'[lint]\nselect = ["E501"]\n',
        b'[lint]\nselect = []\nignore = ["E9", "F821"]\n'
        b'[lint.per-file-ignores]\n"recipe.py" = ["ALL"]\n',
    ],
    ids=["no-destination-config", "complexity", "line-length", "ignored-correctness"],
)
def test_ruff_still_rejects_undefined_names_and_syntax_errors(tmp_path: Path, source: str, config):
    if config is not None:
        (tmp_path / "ruff.toml").write_bytes(config)
    target = tmp_path / "recipe.py"

    assert lint_source_quality(source, target)
    with pytest.raises(SourceQualityError):
        format_and_lint_source(source, target)


def test_format_and_lint_source_accepts_destination_complexity_findings(tmp_path: Path) -> None:
    (tmp_path / "ruff.toml").write_bytes(
        b'target-version = "py311"\n\n[lint]\nselect = ["C901"]\n'
        b"[lint.mccabe]\nmax-complexity = 1\n"
    )
    source = (
        "def generated(value: int) -> int:\n"
        "    if value > 0:\n"
        "        value += 1\n"
        "    if value > 1:\n"
        "        value += 1\n"
        "    return value\n"
    )

    assert lint_source_quality(source, tmp_path / "recipe.py") == []
    assert format_and_lint_source(source, tmp_path / "recipe.py") == source


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_adjacent_prompt_literals_wrap_without_changing_their_value(tmp_path, newline):
    (tmp_path / "ruff.toml").write_bytes(b'line-length = 100\n[lint]\nselect = ["E501"]\n')
    value = ("Read every input and preserve every independent judgment. " * 2)[:87]
    source = (
        "def prompt() -> str:\n"
        "    return format(\n"
        "        (\n"
        f"            {value!r}\n"
        f"            {value!r}\n"
        "        ),\n"
        '        "",\n'
        "    )\n"
    ).replace("\n", newline)
    before = ast.parse(source)
    assert max(len(line) for line in source.splitlines()) > 100
    assert lint_source_quality(source, tmp_path / "recipe.py") == []

    formatted = format_and_lint_source(source, tmp_path / "recipe.py")

    assert ast.dump(ast.parse(formatted)) == ast.dump(before)
    assert max(len(line) for line in formatted.splitlines()) <= 100
    assert format_and_lint_source(formatted, tmp_path / "recipe.py") == formatted


@given(st.text(min_size=1, max_size=200))
def test_literal_wrapping_preserves_unicode_escapes_and_whitespace(value):
    source = f"def prompt():\n    return (\n        {value!r}\n    )\n"
    wrapped = _wrap_long_literals(source, 60)
    assert ast.dump(ast.parse(wrapped)) == ast.dump(ast.parse(source))
    function = ast.parse(wrapped).body[0]
    assert isinstance(function, ast.FunctionDef)
    statement = function.body[0]
    assert isinstance(statement, ast.Return) and statement.value is not None
    assert ast.literal_eval(statement.value) == value


@pytest.mark.parametrize("prefix", ["r", "b", "f", "u"])
def test_literal_wrapping_leaves_prefixed_strings_for_validation(prefix):
    source = f'def prompt():\n    return ({prefix}"' + "word " * 30 + '")\n'
    assert _wrap_long_literals(source, 60) == source


def test_literal_wrapping_leaves_multiline_strings_and_short_widths_unchanged():
    source = 'def prompt():\n    return ("""' + "word " * 30 + '\nend""")\n'
    assert _wrap_long_literals(source, 60) == source
    ordinary = 'def prompt():\n    return ("' + "word " * 30 + '")\n'
    assert _wrap_long_literals(ordinary, 12) == ordinary


def test_literal_wrapping_cannot_silence_invalid_escape_warnings():
    source = 'def prompt():\n    return ("\\q' + "word " * 30 + '")\n'
    assert _wrap_long_literals(source, 60) == source


def test_literal_wrapping_refuses_a_changed_ast(monkeypatch):
    source = 'def prompt():\n    return ("original")\n'
    start = source.index('"original"')
    monkeypatch.setattr(
        source_quality, "_literal_edits", lambda *_: [(start, start + 10, '"invented"')]
    )
    assert _wrap_long_literals(source, 60) == source


def test_literal_wrapping_leaves_recursion_limits_to_validation(monkeypatch):
    def fail(tree):
        raise RecursionError("deep generated source")

    monkeypatch.setattr(ast, "dump", fail)
    source = 'def prompt():\n    return ("original")\n'
    assert _wrap_long_literals(source, 60) == source


def test_pre312_fstring_backslash_is_reported_not_rewritten(tmp_path: Path):
    _ruff_config(tmp_path, target="py311")
    target = tmp_path / "recipe.py"
    source = "import re\n\nvalue = \"x\"\nresult = f\"{re.sub(r'\\t', '    ', value)}\"\n"

    with pytest.raises(SourceQualityError) as caught:
        format_and_lint_source(source, target)

    assert caught.value.source is not None
    assert "re.sub(r'\\t'" in caught.value.source
    assert [item.line for item in caught.value.diagnostics] == [4]
    assert "backslash" in caught.value.diagnostics[0].actual
    assert format_and_lint_source(source, target, python_version="3.12") == source


def test_python_feature_version_rejects_unsupported_values():
    with pytest.raises(SourceQualityError, match="unsupported minimum Python"):
        python_feature_version("3.10")


@pytest.mark.parametrize("stdout", [b"not json", b"\xff", b'{"code": "E501"}'])
def test_malformed_ruff_diagnostics_fail_closed(stdout: bytes) -> None:
    with pytest.raises(SourceQualityError, match="Ruff returned"):
        _parse_ruff_diagnostics(stdout)


def test_ruff_runs_the_bundled_binary_directly() -> None:
    binary = Path(source_quality._ruff_bin())

    assert binary.name in {"ruff", "ruff.exe"}
    assert binary.is_file()


def test_identical_sources_ignore_destination_lint_policy(tmp_path: Path) -> None:
    (tmp_path / "ruff.toml").write_bytes(
        b'[lint]\nselect = ["B905"]\n[lint.per-file-ignores]\n"a.py" = ["B905"]\n'
    )
    source = "PAIRS = list(zip([1], [2]))\n"

    assert lint_source_quality(source, tmp_path / "a.py") == []
    assert lint_source_quality(source, tmp_path / "b.py") == []
    assert lint_source_quality(source, tmp_path / "a.py") == []


_UPPERCASE_STEP = (
    "{imports}\n\n\n"
    "@{decorator}\n"
    "def write_segment(K: str) -> str:\n"
    '    return f"segment_{{K}}.json"\n\n\n'
    "write_segment(K='a')\n"
)


_UNDEFINED = "VALUE = missing\n"


@pytest.mark.parametrize(
    "source",
    [
        _UNDEFINED.replace("\n", "  # noqa: F821\n"),
        _UNDEFINED.replace("\n", "  # noqa\n"),
        "# ruff: noqa: F821\n" + _UNDEFINED,
        "# flake8: noqa\n" + _UNDEFINED,
        "# ruff: disable[F821]\n" + _UNDEFINED + "# ruff: enable[F821]\n",
    ],
    ids=["line-code", "line-bare", "file-ruff", "file-flake8", "range"],
)
def test_generated_source_cannot_silence_its_own_findings(tmp_path: Path, source: str) -> None:
    (tmp_path / "ruff.toml").write_bytes(b'[lint]\nselect = ["E501"]\n')

    findings = lint_source_quality(source, tmp_path / "recipe.py")

    assert [item.actual.split(":")[0] for item in findings] == ["F821"]


def test_line_length_comes_from_the_nearest_ruff_configuration(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 120\n")
    nested = tmp_path / "pkg"
    nested.mkdir()
    # A pyproject without a Ruff section is not a Ruff configuration, so the search continues.
    (nested / "pyproject.toml").write_text("[project]\nname = 'pkg'\n")

    assert resolve_line_length(nested / "recipe.py") == 120
    (nested / "ruff.toml").write_text("target-version = 'py311'\n")
    assert resolve_line_length(nested / "recipe.py") == 88
    (nested / "ruff.toml").write_text("line-length = 79\n")
    assert resolve_line_length(nested / "recipe.py") == 79


def test_line_length_defaults_to_ruffs_when_nothing_is_configured(tmp_path: Path) -> None:
    (tmp_path / "ruff.toml").write_text("line-length = true\n")

    assert resolve_line_length(tmp_path / "recipe.py") == 88


@pytest.mark.parametrize(("setting", "expected"), [("0", 88), ("-1", 88), ("1", 1)])
def test_line_length_requires_a_positive_integer(tmp_path: Path, setting, expected) -> None:
    (tmp_path / "ruff.toml").write_bytes(f"line-length = {setting}\n".encode())
    assert resolve_line_length(tmp_path / "recipe.py") == expected


def test_line_length_defaults_without_any_configuration(tmp_path: Path, monkeypatch) -> None:
    # The suite installs a shared destination config; isolate the no-config case.
    monkeypatch.setattr(source_quality, "_ruff_settings", lambda _path: None)
    assert resolve_line_length(tmp_path / "recipe.py") == 88


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ("[tool.ruff]\nline-length = 100\n", None),
        ("[project]\nrequires-python = '>=3.12,<4'\n", "3.12"),
        ("[project]\nname = 'example'\n", "3.11"),
        ("[tool.ruff\n", None),
    ],
)
def test_python_version_respects_nearest_project_settings(tmp_path: Path, config, expected):
    _ruff_config(tmp_path, target="py311")
    nested = tmp_path / "package"
    nested.mkdir()
    (nested / "pyproject.toml").write_bytes(config.encode())
    assert resolve_python_version(nested / "recipe.py") == expected


@pytest.mark.parametrize(
    ("stdout", "stderr", "detail"),
    [
        (b"stdout detail", b"stderr detail\n", "stderr detail"),
        (b"stdout detail\n", b"", "stdout detail"),
        (b"", b"", "exit status 2"),
    ],
)
def test_ruff_process_failure_preserves_a_repair_diagnostic(
    tmp_path: Path, monkeypatch, stdout, stderr, detail
) -> None:
    from calcifier.subprocesses import ProcessResult

    monkeypatch.setattr(
        source_quality,
        "_run_ruff_process",
        lambda *_args: ProcessResult(("ruff",), 2, stdout, stderr),
    )
    with pytest.raises(SourceQualityError) as caught:
        format_and_lint_source("VALUE = 1\n", tmp_path / "recipe.py")
    assert str(caught.value) == f"Ruff validation failed: {detail}"


# Known-answer boundaries


def _format(source: str, tmp_path: Path, version: str = "3.13") -> str:
    return format_and_lint_source(source, tmp_path / "recipe.py", python_version=version)


def _undefined_names(count: int) -> str:
    return "".join(f"VALUE_{index:02d} = missing_{index:02d}\n" for index in range(count))


@pytest.mark.parametrize(
    ("count", "suffix"),
    [(19, ""), (20, ""), (21, "; 1 more"), (22, "; 2 more")],
)
def test_ruff_failure_lists_twenty_findings_then_counts_the_rest(
    count: int, suffix: str, tmp_path: Path
) -> None:
    listed = "; ".join(
        f"line {1 + index}: F821: Undefined name `missing_{index:02d}`"
        for index in range(min(count, 20))
    )

    with pytest.raises(SourceQualityError) as caught:
        _format(_undefined_names(count), tmp_path)

    assert str(caught.value) == f"Ruff validation failed: {listed}{suffix}"
    assert len(caught.value.diagnostics) == count


def test_pre312_fstring_backslash_literals_are_diagnosed_below_312(tmp_path: Path) -> None:
    source = "def f(a: str, b: str) -> str:\n" + """    return f"{a + '\\n'}{b + '\\t'}"\n"""

    assert _format(source, tmp_path, "3.12") == source
    with pytest.raises(SourceQualityError, match="line 2: invalid-syntax: .*backslash"):
        _format(source, tmp_path, "3.11")


@pytest.mark.parametrize("strict_first", [False, True])
def test_lint_ignores_changed_destination_policy(tmp_path: Path, strict_first: bool) -> None:
    config = tmp_path / "ruff.toml"
    target = tmp_path / "recipe.py"
    generated = "PAIRS = list(zip([1], [2]))\n"

    for strict in (strict_first, not strict_first):
        rules = b'"B905"' if strict else b'"F"'
        config.write_bytes(b"[lint]\nselect = [" + rules + b"]\n")
        diagnostics = lint_source_quality(generated, target)
        assert diagnostics == []
