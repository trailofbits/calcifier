"""Compiled recipes use the declared public package surface, not runtime internals."""

from __future__ import annotations

import pytest

from calcifier.diagnostics import CalcifyError
from calcifier.gate import verify_source

pytestmark = pytest.mark.semantic_offline

SOURCE = """from calcifier import recipe

@recipe
def workflow() -> str:
    return "complete"

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""


@pytest.mark.parametrize(
    "declaration",
    [
        "from calcifier.mutations import run_command",
        "from calcifier.tools import tool",
        "from calcifier.trace import current",
        "from calcifier.recipe import step",
        "import calcifier.tools as tools",
        "from calcifier import trace",
    ],
)
def test_nonpublic_package_imports_are_rejected(declaration):
    source = SOURCE.replace(
        "from calcifier import recipe", declaration + "\nfrom calcifier import recipe"
    )
    with pytest.raises(CalcifyError, match="public Calcifier API"):
        verify_source(source, type_check=False)


def test_nonpublic_package_attributes_are_rejected_before_loading():
    source = SOURCE.replace(
        "from calcifier import recipe", "import calcifier as runtime\nfrom calcifier import recipe"
    ).replace('return "complete"', "return str(runtime.trace.current())")
    with pytest.raises(CalcifyError, match="public Calcifier API"):
        verify_source(source)


def test_function_local_private_import_is_rejected():
    source = SOURCE.replace(
        '    return "complete"',
        "    from calcifier.trace import current\n    return str(current())",
    )
    with pytest.raises(CalcifyError, match="public Calcifier API"):
        verify_source(source, type_check=False)


def test_public_package_aliases_remain_valid():
    source = SOURCE.replace("from calcifier import recipe", "import calcifier as runtime").replace(
        "@recipe", "@runtime.recipe"
    )
    verify_source(source)
