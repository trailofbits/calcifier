"""Static policies distinguish valid paths and child calls from forbidden effects."""

from __future__ import annotations

import ast

import pytest

from calcifier.diagnostics import CalcifyError
from calcifier.gate import (
    _check_subrecipe_call_contracts,
    _check_sync_primitive_awaits,
    _path_role_aliases,
    _recipe_string_defaults,
    _reject_target_rooted_output_paths,
)


@pytest.mark.parametrize(
    "expression",
    ['working_directory / "README.md"', '"reports" / output_path', '"a" / "b"'],
)
def test_ordinary_path_components_do_not_mix_target_and_output_roles(expression):
    module = ast.parse(f"def prepare(working_directory, output_path):\n    return {expression}\n")
    _reject_target_rooted_output_paths(module)


@pytest.mark.parametrize(
    "expression",
    [
        "working_directory / output_path",
        "output_path / working_directory",
        "ctx.working_directory / ctx.output_path",
    ],
)
def test_both_orderings_of_read_root_and_output_destination_are_rejected(expression):
    module = ast.parse(
        f"def prepare(ctx, working_directory, output_path):\n    return {expression}\n"
    )
    with pytest.raises(CalcifyError, match="incorrectly rooted beneath working_directory"):
        _reject_target_rooted_output_paths(module)


def test_path_aliases_reach_a_fixed_point_without_treating_attributes_as_local_names():
    module = ast.parse(
        """def prepare(working_directory, output_path):
    earlier = later
    later = working_directory
    destination: str = output_path
    holder.attribute = output_path
    return earlier / destination
"""
    )
    function = module.body[0]
    assert isinstance(function, ast.FunctionDef)
    aliases = _path_role_aliases(function)
    assert aliases["earlier"] == aliases["later"] == 1
    assert aliases["destination"] == 2
    assert "attribute" not in aliases
    with pytest.raises(CalcifyError, match="incorrectly rooted"):
        _reject_target_rooted_output_paths(module)


def test_only_string_defaults_participate_in_output_root_detection():
    function = ast.parse(
        'def run(required, /, number=1, path="reports", collection=(), '
        '*, flag=False, dest="out", missing): pass'
    ).body[0]
    assert isinstance(function, ast.FunctionDef)
    assert _recipe_string_defaults(function) == {"path": "reports", "dest": "out"}


@pytest.mark.parametrize("required", ["0", "None", "flag"])
def test_non_boolean_or_dynamic_child_required_policy_is_rejected(required):
    source = (
        f'from calcifier import call_recipe\ncall_recipe("child.py", {{}}, required={required})'
    )
    with pytest.raises(CalcifyError, match="invalid child recipe call"):
        _check_subrecipe_call_contracts(ast.parse(source))


def test_awaiting_a_synchronous_child_output_is_rejected_before_runtime():
    source = """from calcifier import call_recipe as invoke
async def worker():
    return await invoke("child.py", {}).output
"""
    with pytest.raises(CalcifyError, match="synchronous Calcifier primitive call_recipe"):
        _check_sync_primitive_awaits(ast.parse(source))
    _check_sync_primitive_awaits(ast.parse("async def worker(value):\n    return await value\n"))
