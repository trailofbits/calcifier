"""Statically certain runtime-contract violations are rejected before execution."""

from __future__ import annotations

import ast
import textwrap

import pytest

from calcifier.runtime_conformance import (
    import_resolver,
    packaged_resource_errors,
    runtime_conformance_errors,
)

pytestmark = pytest.mark.semantic_offline

_IMPORTS = "from calcifier import call_recipe, resource\n"
_MANIFEST = frozenset({"references/guide.md", "references/notes.md"})


def _module(body: str) -> ast.Module:
    return ast.parse(_IMPORTS + textwrap.dedent(body))


def _resource_messages(body: str, packaged: frozenset[str] | None = _MANIFEST) -> list[str]:
    module = _module(body)
    return [
        item.actual for item in packaged_resource_errors(module, import_resolver(module), packaged)
    ]


@pytest.mark.parametrize("path", [".", "./", "references/", ""])
def test_resource_literals_that_name_a_directory_are_rejected(path: str) -> None:
    (message,) = _resource_messages(f"def load():\n    return resource({path!r})\n")

    assert f"packaged resource path {path!r} names a directory" in message


def test_resource_literals_must_name_a_manifest_file_when_the_manifest_is_known() -> None:
    body = (
        "def load():\n"
        "    resource('references/guide.md')\n"
        "    resource('./references/notes.md')\n"
        "    resource('references/missing.md')\n"
    )

    (message,) = _resource_messages(body)

    assert "'references/missing.md' is not in the skill package manifest" in message
    assert "['references/guide.md', 'references/notes.md']" in message
    assert _resource_messages(body, packaged=None) == []


def test_manifest_preview_is_bounded() -> None:
    packaged = frozenset(f"references/{index:02}.md" for index in range(30))

    (message,) = _resource_messages("def load():\n    resource('absent.md')\n", packaged)

    assert "references/19.md" in message
    assert "references/20.md" not in message


def test_non_literal_or_unrelated_calls_are_not_packaged_literals() -> None:
    body = (
        "def load(name):\n"
        "    resource(name)\n"
        "    resource(f'references/{name}.md')\n"
        "    resource()\n"
        "    print('.')\n"
    )

    assert _resource_messages(body) == []


def test_packaged_errors_name_the_enclosing_top_level_function() -> None:
    module = _module(
        "resource('.')\n"
        "def load():\n"
        "    def nested():\n"
        "        return resource('.')\n"
        "    return nested\n"
    )

    errors = packaged_resource_errors(module, import_resolver(module), _MANIFEST)

    assert [(item.line, item.node) for item in errors] == [(2, None), (5, "load")]


def test_import_resolver_mirrors_module_levelimport_bindings() -> None:
    module = ast.parse(
        "import calcifier\n"
        "import calcifier.recipe as runtime\n"
        "import os.path\n"
        "from calcifier import resource as packaged\n"
        "from . import sibling\n"
    )
    resolve = import_resolver(module)

    def target(source: str) -> str | None:
        expression = ast.parse(source, mode="eval").body
        assert isinstance(expression, ast.Call)
        return resolve(expression.func)

    assert target("calcifier.resource('a')") == "calcifier.resource"
    assert target("runtime.resource('a')") == "calcifier.recipe.resource"
    assert target("os.path.join('a')") == "os.path.join"
    assert target("packaged('a')") == "calcifier.resource"
    assert target("sibling()") == "sibling"
    assert target("unknown()") is None
    assert target("unknown.attribute()") is None
    assert target("factory()()") is None


def test_runtime_conformance_errors_are_sorted_by_line() -> None:
    module = _module(
        "def first():\n"
        "    return resource('.')\n"
        "def second():\n"
        "    return resource('references/absent.md')\n"
    )

    errors = runtime_conformance_errors(module, _MANIFEST)

    assert [(item.line, item.node) for item in errors] == [(3, "first"), (5, "second")]
    assert "names a directory" in errors[0].actual
    assert "not in the skill package manifest" in errors[1].actual


def test_conformance_errors_are_immutable_values() -> None:
    module = _module("def load():\n    return resource('.')\n")
    (error,) = runtime_conformance_errors(module, _MANIFEST)

    assert {error} == {__import__("dataclasses").replace(error)}
    with pytest.raises(AttributeError):
        error.line = 1  # ty: ignore[invalid-assignment]


def test_resource_paths_passed_by_keyword_follow_the_same_contract() -> None:
    (message,) = _resource_messages("def load():\n    return resource(rel_path='.')\n")

    assert "packaged resource path '.' names a directory" in message
    valid = "def load():\n    return resource(rel_path='references/guide.md')\n"
    assert _resource_messages(valid) == []


@pytest.mark.parametrize(
    "expression",
    [
        "Path('references/guide.md')",
        "Path(root) / 'references/guide.md'",
        "Path(root, 'references/guide.md')",
        "Path(root).joinpath('references/guide.md')",
        "root / 'references/guide.md'",
        "pathlib.Path(root) / './references/guide.md'",
        "FilePath('references/guide.md')",
        "os.path.join(root, 'references/guide.md')",
    ],
)
def test_packaged_paths_through_filesystem_construction_require_resource(expression: str) -> None:
    body = (
        "from pathlib import Path, Path as FilePath\n"
        "import pathlib\n"
        "import os.path\n"
        f"def load(root):\n    return {expression}\n"
    )

    (message,) = _resource_messages(body)

    assert "ships with the recipe" in message
    assert "resource(" in message and "references/guide.md" in message
    assert "remove any recipe input for it" in message
    assert "never ask the user to supply it" in message
    assert _resource_messages(body, packaged=None) == []
    assert _resource_messages(body, packaged=frozenset()) == []


def test_user_paths_and_packaged_text_do_not_trigger_filesystem_resource_diagnostics() -> None:
    body = (
        "from pathlib import Path\n"
        "def load(root):\n"
        "    note = 'references/guide.md'\n"
        "    user_file = Path(root) / 'document.md'\n"
        "    guide = resource('references/guide.md').read_text(encoding='utf-8')\n"
        "    return Path(resource('references/notes.md')), guide, user_file, note\n"
    )

    assert _resource_messages(body) == []
