"""ty type-checks generated recipes against the installed Calcifier API."""

from __future__ import annotations

import pytest

import calcifier.type_check as type_check
from calcifier.diagnostics import Diagnostic
from calcifier.source_quality import SourceQualityError, quality_error
from calcifier.subprocesses import ProcessError, ProcessResult
from calcifier.type_check import type_check_source

pytestmark = pytest.mark.semantic_offline


@pytest.fixture(autouse=True)
def _fresh_memo() -> object:
    """Keep faked ty results out of the memo other tests read."""
    type_check._type_check.cache_clear()
    yield
    type_check._type_check.cache_clear()


_STEPS = """\
from pathlib import Path

from calcifier import output_type, recipe, step, write_text


@output_type
class Summary:
    text: str


@step
def summarize(target: str) -> Summary:
    return Summary(text=target)


@step
def persist(summary: Summary, destination: str) -> Path:
    return write_text(f"{destination}/summary.md", summary.text)
"""


def test_a_type_correct_recipe_has_no_findings() -> None:
    source = _STEPS + (
        "\n\ndef run(target: str) -> Path:\n    return persist(summarize(target), '.')\n"
    )

    assert type_check_source(source) == ()


def test_step_arguments_are_checked_against_declared_parameters() -> None:
    source = _STEPS + "\n\ndef run(target: str) -> Path:\n    return persist(summarize(target))\n"

    (finding,) = type_check_source(source)

    assert finding.line == 22
    assert finding.actual.startswith("ty[missing-argument]: ")
    assert "`destination`" in finding.actual


def test_output_type_serializers_attached_at_runtime_are_not_findings() -> None:
    source = _STEPS + (
        "\n\ndef run(target: str) -> str:\n"
        "    summary = summarize(target)\n"
        "    return summary.model_dump_json() + str(summary.model_dump())\n"
    )

    assert type_check_source(source) == ()


def test_warnings_alone_are_not_findings() -> None:
    source = "def check(value: str) -> bool:\n    assert (value, 'always true')\n    return True\n"

    assert type_check_source(source) == ()


def test_the_target_python_version_limits_accepted_syntax() -> None:
    source = "type Alias = int\n"

    assert type_check_source(source, python_version="3.13") == ()
    (finding,) = type_check_source(source, python_version="3.11")
    assert finding.actual.startswith("ty[invalid-syntax]: ")


def test_default_python_target_accepts_pep701_fstrings() -> None:
    source = "def run(value: str) -> str:\n    return f\"{value + '\\n'}\"\n"

    assert type_check_source(source) == ()
    (finding,) = type_check_source(source, python_version="3.11")
    assert finding.actual.startswith("ty[invalid-syntax]: ")


def test_ty_findings_render_every_finding_with_the_checked_source() -> None:
    source = "def run() -> int:\n    return missing_one + missing_two\n"

    error = quality_error("ty validation failed", type_check_source(source), source)

    assert str(error) == (
        "ty validation failed: line 2: ty[unresolved-reference]: Name `missing_one` used when "
        "not defined; line 2: ty[unresolved-reference]: Name `missing_two` used when not defined"
    )
    assert [item.line for item in error.diagnostics] == [2, 2]
    assert error.source == source


@pytest.mark.parametrize(("count", "suffix"), [(20, ""), (21, "; 1 more"), (22, "; 2 more")])
def test_ty_findings_list_twenty_then_count_the_rest(count: int, suffix: str) -> None:
    source = "".join(f"value_{index} = missing_{index}\n" for index in range(count))

    error = quality_error("ty validation failed", type_check_source(source), source)

    assert len(error.diagnostics) == count
    assert str(error).endswith(f"used when not defined{suffix}")
    assert str(error).count("line ") == 20


def test_a_clean_recipe_has_no_ty_findings() -> None:
    assert type_check_source("def run() -> int:\n    return 1\n") == ()


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        ("value = missing  # type: ignore\n", "unresolved-reference"),
        ("import _calcifier_missing_module\n", "unresolved-import"),
        ("value = 1\nvalue()\n", "call-non-callable"),
        ("def run(value: int) -> int:\n    return value\nrun()\n", "missing-argument"),
        (
            "def run(value: int) -> int:\n    return value\nrun(value=1, extra=2)\n",
            "unknown-argument",
        ),
        (
            "def run(value: int) -> int:\n    return value\nrun(1, 2)\n",
            "too-many-positional-arguments",
        ),
        (
            "def total(xs: list[int]) -> int:\n    return sum(xs)\ntotal([1], xs=[2])\n",
            "parameter-already-assigned",
        ),
        (
            "def total(xs: list[int], /) -> int:\n    return sum(xs)\ntotal(xs=[1])\n",
            "positional-only-parameter-as-kwarg",
        ),
        (
            "from calcifier import step, write_text\n\n"
            "@step\nasync def persist() -> None:\n    await write_text('out.txt', 'value')\n",
            "invalid-await",
        ),
        ("def broken(:\n    pass\n", "invalid-syntax"),
    ],
    ids=lambda value: value if "\n" not in value else None,
)
def test_certain_runtime_failures_remain_blocking(source: str, rule: str) -> None:
    findings = type_check_source(source)

    assert findings
    assert any(item.actual.startswith(f"ty[{rule}]: ") for item in findings)


@pytest.mark.parametrize(
    "source",
    [
        "def lookup(value: object) -> object:\n    return value.get('answer')\n",
        "def upper(value: str | None) -> str:\n    return value.upper()\n",
        "def items(value: object) -> list[object]:\n    return [item for item in value]\n",
        "def run(value: int) -> int:\n    return value\nrun('text')\n",
        "value: int = 'text'\n",
    ],
    ids=["object-attribute", "possibly-none", "not-iterable", "argument-type", "assignment"],
)
def test_uncertain_type_findings_do_not_block(source: str) -> None:
    assert type_check_source(source) == ()


def _fake_ty(
    monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: bytes, stderr: bytes = b""
) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> ProcessResult:
        calls.append(list(argv))
        return ProcessResult(tuple(argv), returncode, stdout, stderr)

    monkeypatch.setattr(type_check, "run_process", run)
    type_check._type_check.cache_clear()
    return calls


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "detail"),
    [
        (2, b"", b"invalid configuration", "exit status 2: invalid configuration"),
        (1, b"All checks passed!\n", b"", "exit status 1: All checks passed!"),
        (101, b"panicked", b"", "exit status 101: panicked"),
    ],
)
def test_ty_failures_without_readable_findings_are_errors(
    monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: bytes, stderr: bytes, detail: str
) -> None:
    _fake_ty(monkeypatch, returncode, stdout, stderr)

    with pytest.raises(SourceQualityError, match=f"ty failed with {detail}"):
        type_check_source("value = 1\n")


def test_findings_are_parsed_from_the_recipe_file_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ty(
        monkeypatch,
        1,
        b"recipe.py:3:5: error[missing-argument] wrong\n"
        b"other.py:4:1: error[missing-argument] elsewhere\n"
        b"recipe.py:7:2: warning[missing-argument] noisy\n"
        b"recipe.py:9:1: info[missing-argument] note\n",
    )

    assert type_check_source("value = 1\n") == (
        Diagnostic(line=3, actual="ty[missing-argument]: wrong"),
    )


def test_ty_runs_isolated_against_the_running_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_ty(monkeypatch, 0, b"All checks passed!\n")

    type_check_source("value = 1\n", python_version="3.14")
    type_check_source("value = 2\n")

    with_version, without_version = calls
    assert with_version[1:2] == ["check"]
    assert with_version[with_version.index("--python") + 1] == type_check.sys.executable
    assert with_version[with_version.index("--output-format") + 1] == "concise"
    assert with_version[with_version.index("--python-version") + 1] == "3.14"
    assert without_version[without_version.index("--python-version") + 1] == "3.13"
    assert with_version[-1] == "recipe.py"
    config = with_version[with_version.index("--config-file") + 1]
    project = with_version[with_version.index("--project") + 1]
    assert config.startswith(project)


def test_results_are_memoized_per_source_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_ty(monkeypatch, 0, b"")

    type_check_source("value = 1\n")
    type_check_source("value = 1\n")
    type_check_source("value = 1\n", python_version="3.14")

    assert len(calls) == 2


def test_a_ty_that_cannot_run_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(argv: list[str], **_kwargs: object) -> ProcessResult:
        raise ProcessError("timed out", stderr_tail=b"")

    monkeypatch.setattr(type_check, "run_process", fail)
    type_check._type_check.cache_clear()

    with pytest.raises(SourceQualityError, match="ty execution failed: timed out"):
        type_check_source("value = 1\n")


_AGENT_CALLS = """\
from typing import Annotated

from calcifier import agent, output_type, word_count


@output_type
class Review:
    ok: bool


def reviewed(path: str) -> Review:
    return agent("w", "p", working_directory=path, readable_paths=(path,), output=Review)


def summary(path: str) -> str:
    bounded = Annotated[str, word_count(1, 9)]
    return agent("w", "p", working_directory=path, readable_paths=(path,), output=bounded)
"""


def test_agent_calls_return_their_output_type() -> None:
    misuse = _AGENT_CALLS + ("\n\ndef run(path: str) -> None:\n    reviewed(path)()\n")

    assert type_check_source(_AGENT_CALLS) == ()
    (finding,) = type_check_source(misuse)
    assert finding.actual.startswith("ty[call-non-callable]: ")
    assert "`Review`" in finding.actual


def test_an_extra_agent_argument_is_named_on_its_own_line() -> None:
    source = _AGENT_CALLS + (
        "\n\ndef prefixed(path: str) -> Review:\n"
        "    return agent(\n"
        '        "w",\n'
        '        "p",\n'
        "        path,\n"
        "        working_directory=path,\n"
        "        readable_paths=(path,),\n"
        "        output=Review,\n"
        "    )\n"
    )

    (finding,) = type_check_source(source)

    assert finding.line == source.splitlines().index("        path,") + 1
    assert finding.actual.startswith("ty[too-many-positional-arguments]: ")


def test_a_type_ignore_comment_does_not_hide_a_finding() -> None:
    (finding,) = type_check_source("value = missing  # type: ignore\n")

    assert finding.actual.startswith("ty[unresolved-reference]: ")


@pytest.mark.parametrize("output", ["", ", output=str"])
@pytest.mark.parametrize(
    "arguments,rule", [("", "missing-argument"), (", timeout=10", "unknown-argument")]
)
def test_command_argument_errors_are_blocking(output: str, arguments: str, rule: str) -> None:
    findings = type_check_source(
        f'from calcifier import run_command\nrun_command(["tool"]{arguments}{output})\n'
    )

    assert any(finding.actual.startswith(f"ty[{rule}]: ") for finding in findings)
