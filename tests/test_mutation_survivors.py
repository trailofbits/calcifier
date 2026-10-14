"""Public behavior regressions for mutation survivors triaged in issue #27."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from calcifier import emit, recipe
from calcifier.calcify import CalcifyError
from calcifier.gate import verify_source
from calcifier.outputs import OutputParseError, parse_response
from calcifier.recipe import coerce_input


def test_unclosed_response_fence_keeps_the_last_json_line() -> None:
    assert parse_response('```json\n{"a": 1}\n{"b": 2}', dict[str, int]) == {"b": 2}
    with pytest.raises(OutputParseError):
        parse_response("```", dict[str, int])


def test_bare_cli_footer_rejection_reports_its_source_line() -> None:
    source = textwrap.dedent(
        """\
        from calcifier import recipe, step

        @step
        def s(x: str) -> bool: return True

        @recipe
        def r(target: str = "x") -> bool:
            return s(x=target)

        if __name__ == "__main__":
            r.cli()
        """
    )
    footer_line = next(
        i for i, line in enumerate(source.splitlines(), 1) if line.startswith("if __name__ ==")
    )

    with pytest.raises(CalcifyError, match=rf"\(line {footer_line}\)"):
        verify_source(source)


def test_gate_rejects_non_string_capability() -> None:
    source = textwrap.dedent(
        """\
        from calcifier import recipe, step

        @step
        def s(x: str) -> bool: return True

        @recipe(capabilities=(42, "Read"))
        def r(target: str = "x") -> bool:
            return s(x=target)

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    )
    with pytest.raises(CalcifyError) as caught:
        verify_source(source)
    assert {item.code for item in caught.value.diagnostics} == {"tool_capabilities"}


def test_artifact_rendering_preserves_mapping_order_and_plain_text() -> None:
    @recipe
    def run() -> bool:
        emit({"z": 1, "a": 2}, name="mapping")
        emit("hello", name="text")
        return True

    rendered = run.run().render()
    assert "artifact mapping: {'z': 1, 'a': 2}" in rendered
    assert "artifact text: hello" in rendered


def test_recipe_cli_coerces_scalar_and_path_flags(capsys: pytest.CaptureFixture[str]) -> None:
    observed: list[tuple[str, Path]] = []

    @recipe
    def run(label: str, path: Path) -> bool:
        observed.append((label, path))
        return True

    assert run.cli(["--label", "7", "--path", "folder/item"]) == 0
    assert observed == [("7", Path("folder/item"))]
    capsys.readouterr()


def test_annotated_integer_input_converts_bool_and_string() -> None:
    assert coerce_input(42, int) == 42
    assert coerce_input(True, int) == 1
    assert type(coerce_input(True, int)) is int
    assert coerce_input(False, int) == 0
    assert coerce_input("7", int) == 7
