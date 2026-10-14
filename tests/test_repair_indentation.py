"""Indentation recovery must preserve literal task and report contents."""

from __future__ import annotations

import ast
import json

import pytest

from calcifier.calcify import calcify_source
from calcifier.recipe import load_recipe
from calcifier.repair import _normalize_uniformly_indented_recipe_draft
from tests.support.compiler import ScriptedCompiler
from tests.support.recipes import _draft

pytestmark = pytest.mark.semantic_offline


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("prefix", ["    ", "\t"], ids=["spaces", "tab"])
@pytest.mark.parametrize(
    "literal",
    [
        '"alpha\tbeta"',
        'b"alpha\tbeta"',
        '"""alpha\n    beta\tend\n    """',
        'f"""alpha\n    beta{value}\n    """',
    ],
    ids=["text-tab", "bytes-tab", "multiline", "formatted-multiline"],
)
def test_recovery_preserves_literal_values_and_draft_metadata(newline, prefix, literal):
    source = (
        f"{prefix}value = 7{newline}"
        f"{prefix}text = {literal.replace(chr(10), newline + prefix)}{newline}"
    )
    payload = {"version": "1", "source": source, "bindings": [{"key": "deliver_note"}]}
    response = json.dumps(payload, ensure_ascii=False)
    recovered = json.loads(
        _normalize_uniformly_indented_recipe_draft(response, python_version=None)
    )
    wrapped_source = "if True:\n" + source
    original = ast.parse(wrapped_source).body[0]
    assert isinstance(original, ast.If)
    actual = ast.parse(recovered["source"])
    # Compare literal values and f-string expressions as Python parsed them,
    # independently of the host's indentation transformation.
    assert ast.dump(actual) == ast.dump(ast.Module(body=original.body, type_ignores=[]))
    original_assignment = original.body[1]
    actual_assignment = actual.body[1]
    assert isinstance(original_assignment, ast.Assign)
    assert isinstance(actual_assignment, ast.Assign)
    assert ast.get_source_segment(recovered["source"], actual_assignment.value) == (
        ast.get_source_segment(wrapped_source, original_assignment.value)
    )
    assert recovered == {**payload, "source": recovered["source"]}


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_indented_recipe_compiles_once_and_delivers_unchanged_text(tmp_path, newline):
    source = (
        "from calcifier import recipe, step\n\n\n"
        "@step\n"
        "def prepare_note() -> str:\n"
        '    return """Heading\n    detail\tvalue\n    """\n\n\n'
        "@recipe\n"
        "def deliver_note() -> str:\n"
        "    return prepare_note()\n\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(deliver_note.cli())\n"
    )
    indented = "".join("    " + line for line in source.splitlines(keepends=True)).replace(
        "\n", newline
    )
    original = ast.parse("if True:\n" + indented)
    expected = next(
        node.value.value
        for node in ast.walk(original)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant)
    )
    compiler = ScriptedCompiler([_draft(indented)])
    path = tmp_path / "recipe.py"
    compiled = calcify_source(
        "Return the prepared note exactly.", compiler, agent_roles=(), source_path=path
    )
    path.write_bytes(compiled.encode())
    result = load_recipe(path).run()
    assert result.ok
    assert result.output == expected
    assert [kind for kind, _ in compiler.calls] == ["source"]


@pytest.mark.parametrize(
    "source",
    [
        'text = "alpha\tbeta"\n',
        '    text = "alpha"\nother = "beta"\n',
        '    text = """unfinished\n',
    ],
    ids=["already-valid", "mixed-module-indent", "malformed-literal"],
)
def test_recovery_keeps_valid_or_unrecoverable_responses_unchanged(source):
    response = _draft(source)
    assert _normalize_uniformly_indented_recipe_draft(response, python_version=None) == response
