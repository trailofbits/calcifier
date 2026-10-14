"""String substitution needs no constructor proof; path moves stay rejected."""

from __future__ import annotations

import pytest

from calcifier.diagnostics import CalcifyError
from calcifier.gate import verify_source

pytestmark = pytest.mark.semantic_offline


@pytest.mark.parametrize(
    ("imports", "binding", "expression"),
    [
        ("from pathlib import Path as str", "", "str(value)"),
        ("from pathlib import Path", "str = Path", "str(value)"),
        ("from pathlib import Path", "local: object = Path", "local(value)"),
        ("from pathlib import Path as File", "", "File(value)"),
        ("", "", "value"),
    ],
)
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_one_destination_replace_is_rejected_before_execution(
    imports, binding, expression, newline
):
    source = newline.join(
        [
            imports,
            "from calcifier import recipe",
            "",
            "@recipe",
            "def workflow(value: str) -> None:",
            "    " + (binding or "pass"),
            f'    {expression}.replace("relocated")',
            "",
            'if __name__ == "__main__":',
            "    raise SystemExit(workflow.cli())",
            "",
        ]
    )
    with pytest.raises(CalcifyError, match="receiver method 'replace'"):
        verify_source(source, type_check=False)


@pytest.mark.parametrize(
    ("imports", "binding", "expression"),
    [
        ("", "", "value"),
        ("from builtins import str as Text", "", "Text(value)"),
        ("import builtins as native", "", "native.str(value)"),
        ("from pathlib import Path as File", "", "File(value).stem"),
        ("", "text = value", "text"),
    ],
)
def test_string_substitution_accepts_inputs_variables_and_aliases(imports, binding, expression):
    source = "\n".join(
        [
            imports,
            "from calcifier import recipe",
            "",
            "@recipe",
            "def workflow(value: str) -> str:",
            "    " + (binding or "pass"),
            f'    return {expression}.replace("old", "new")',
            "",
            'if __name__ == "__main__":',
            "    raise SystemExit(workflow.cli())",
            "",
        ]
    )
    verify_source(source)
