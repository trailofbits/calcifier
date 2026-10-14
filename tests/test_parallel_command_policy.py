"""Parallel command effects depend on an explicit literal read-only declaration."""

from __future__ import annotations

import ast

import pytest

from calcifier.parallel_contract import parallel_effect_errors


@pytest.mark.parametrize("callback", ['worker, ["input"]', 'fn=worker, items=["input"]'])
@pytest.mark.parametrize("option", ["", ", mutating=True", ", mutating=flag", ", mutating=0"])
def test_parallel_commands_need_literal_read_only_effects_even_through_helpers(option, callback):
    source = f"""from calcifier import parallel as fanout, run_command as command
def helper(item):
    return command(["tool", item]{option})
def worker(item):
    return helper(item)
fanout({callback})
"""
    errors = parallel_effect_errors(ast.parse(source))
    assert len(errors) == 1
    assert "calcifier.run_command at line 3" in errors[0].actual


def test_read_only_commands_are_legal_and_unrelated_calls_are_not_parallel_lanes():
    source = """from calcifier import parallel, run_command
def helper(item):
    return run_command(["tool", item], mutating=False)
def worker(item):
    return helper(item)
parallel(worker, ["input"])
run_command(["outside-lanes"])
"""
    assert parallel_effect_errors(ast.parse(source)) == []


def test_parallel_effect_analysis_leaves_missing_callback_to_call_validation():
    assert parallel_effect_errors(ast.parse("from calcifier import parallel\nparallel()")) == []
