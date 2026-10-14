"""Compiler context preserves interfaces and rejected drafts without extra calls."""

from __future__ import annotations

import json

import pytest

from calcifier.calcify import calcify_source
from calcifier.diagnostics import CalcifyError, Diagnostic
from calcifier.prompts import _catalog_prompt_payload, build_prompt
from calcifier.repair import _failure_key, _normalize_uniformly_indented_recipe_draft
from tests.support.compiler import ScriptedCompiler
from tests.support.recipes import _GOOD_RECIPE


def test_default_compiler_context_retains_draft_transport_and_installed_signatures():
    prompt = build_prompt("# Skill\nReview the declared scope.\n", python_version="3.11")
    assert "RecipeDraft JSON object" in prompt
    assert "prepare_context(" in prompt and "-> PreparedContext`" in prompt
    assert "prior_results" in prompt and "always saved in full" in prompt
    assert "Python 3.11" in prompt and "backslash" in prompt
    assert "choose one the skill's work can finish within" in prompt


def test_unspecified_tools_are_unrestricted_and_python_target_is_not_invented():
    prompt = build_prompt("# Skill\nReview the declared scope.\n")
    assert "has no allowed-tools field and does not restrict tools" in prompt
    assert "only the capabilities the recipe calls; run_command() needs Bash" in prompt
    assert "syntax accepted by the target repository's configured Python version" in prompt
    assert "Python None" not in prompt


def test_missing_catalog_file_has_no_invented_child_interface(tmp_path):
    assert _catalog_prompt_payload({"worker": "absent.py"}, base=tmp_path) == {
        "worker": {"path": "absent.py"}
    }


def test_relative_catalog_describes_nested_results_with_or_without_explicit_base(
    tmp_path, monkeypatch
):
    child = tmp_path / "child.py"
    child.write_text(
        """from calcifier import output_type, recipe
@output_type
class Evidence:
    locator: str
@output_type
class Result:
    evidence: list[Evidence]
@recipe
def child(subject: str) -> Result:
    return Result([])
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    without_base = _catalog_prompt_payload({"worker": "child.py"}, base=None)
    with_base = _catalog_prompt_payload({"worker": "child.py"}, base=tmp_path)
    assert without_base == with_base
    assert without_base["worker"]["output_types"] == {
        "Evidence": {"locator": "str"},
        "Result": {"evidence": "list[Evidence]"},
    }
    compiler = ScriptedCompiler([_GOOD_RECIPE], wrap_sources=True)
    monkeypatch.chdir(tmp_path.parent)
    calcify_source(
        "# Skill",
        compiler,
        source_path=tmp_path / "absent-output" / "recipe.py",
        skill_catalog={"worker": "../child.py"},
    )
    payload = json.loads(
        compiler.all_prompts[0]
        .split("## Catalogued child recipes\n", 1)[1]
        .split("\n\n## Packaged resources", 1)[0]
    )["worker"]
    assert payload["inputs"] == without_base["worker"]["inputs"]
    assert payload["returns"] == without_base["worker"]["returns"]
    assert payload["output_types"] == without_base["worker"]["output_types"]


def test_stall_detection_distinguishes_identical_diagnostics_at_different_bindings():
    def diagnostic(name, line):
        return Diagnostic(
            code="missing-field",
            phase="source",
            node=name,
            line=line,
            actual=f"required field missing at line {line}",
            expected="required field",
            safe_fix="restore the field",
        )

    first = CalcifyError("missing field", diagnostics=(diagnostic("first", 3),))
    moved = CalcifyError("missing field", diagnostics=(diagnostic("first", 50),))
    second = CalcifyError("missing field", diagnostics=(diagnostic("second", 3),))
    assert _failure_key(first) == _failure_key(moved)
    assert _failure_key(first) != _failure_key(second)


@pytest.mark.parametrize(
    "source",
    [" \n\t\n", "    if True:\n    return (\n", "  x = 1\n    y = 2\n"],
)
def test_invalid_drafts_are_preserved_when_uniform_indentation_cannot_fix_them(source):
    original = json.dumps({"version": "1", "source": source}, ensure_ascii=False)
    assert _normalize_uniformly_indented_recipe_draft(original, python_version="3.13") == original
