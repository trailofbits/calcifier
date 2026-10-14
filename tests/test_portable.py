from __future__ import annotations

import ast
import functools
import hashlib
import tempfile
import textwrap
from functools import lru_cache
from pathlib import Path

import pytest

from calcifier import cli
from calcifier.portable import _pinned_text, lint_portable, lint_portable_source
from calcifier.source_quality import format_and_lint_source


@lru_cache
def _source(step_body: str = "return bool(target)", *, kind: str = "python") -> str:
    calcifier_imports = {"recipe", "step"}
    stdlib_imports: list[str] = []
    if kind == "agent" or "agent(" in step_body:
        calcifier_imports.add("agent")
    if kind == "agent":
        calcifier_imports.add("Budget")
    if kind == "subrecipe":
        calcifier_imports.add("call_recipe")
    if "write_text(" in step_body:
        calcifier_imports.add("write_text")
    if "record_side_effect(" in step_body:
        calcifier_imports.add("record_side_effect")
    if "Path(" in step_body:
        stdlib_imports.append("from pathlib import Path")
    if "os.system(" in step_body:
        stdlib_imports.append("import os")
    imports = ", ".join(sorted(calcifier_imports))
    import_block = "\n".join((*stdlib_imports, f"from calcifier import {imports}"))
    recipe_args = 'agents=("worker",), budget=Budget(2, 30)' if kind == "agent" else ""
    source = (
        import_block
        + "\n\n"
        + textwrap.dedent(
            f"""
        @step
        def work(target: str) -> bool:
            {step_body}

        @recipe({recipe_args})
        def run(target: str = ".") -> bool:
            return work(target)

        if __name__ == "__main__":
            raise SystemExit(run.cli())

        _CALCIFIER_SUBRECIPES = []
        """
        )
    )
    # Format under a neutral destination config, not this repository's stricter one.
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "ruff.toml").write_text("line-length = 100\n", encoding="utf-8")
        return format_and_lint_source(source, Path(directory) / "recipe.py")


def _replace_assignment(source: str, name: str, value: str | None) -> str:
    module = ast.parse(source)
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    lines = source.splitlines(keepends=True)
    replacement = "" if value is None else f"{name} = {value}\n"
    lines[assignment.lineno - 1 : assignment.end_lineno] = [replacement]
    return "".join(lines)


def _messages(path: Path, source: str) -> list[str]:
    path.write_text(source)
    return [item.actual for item in lint_portable(path)]


def test_portable_lint_accepts_closed_deterministic_recipe(tmp_path):
    assert _messages(tmp_path / "recipe.py", _source()) == []


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ("_CALCIFIER_RESOURCES = dict()", "invalid resource provenance"),
        ("_CALCIFIER_SKILL = dict()", "invalid skill provenance"),
        (
            "_CALCIFIER_SKILL = " + repr({"path": "missing.md", "sha256": "0" * 64}),
            "unresolved skill provenance path",
        ),
    ],
)
def test_portable_lint_reports_unreadable_provenance_without_crashing(tmp_path, metadata, message):
    assert message in _messages(tmp_path / "recipe.py", _source() + metadata + "\n")


def test_portable_lint_reports_nonlist_dependency_metadata(tmp_path):
    source = _source("return call_recipe('child.py', {'target': target}).ok", kind="subrecipe")
    source = _replace_assignment(source, "_CALCIFIER_SUBRECIPES", "{}")
    assert "subrecipe topology mismatch" in _messages(tmp_path / "recipe.py", source)


def test_portable_lint_reads_utf8_with_windows_default_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe_path = tmp_path / "recipe.py"
    recipe_path.write_bytes(_source().replace("target", "café").encode("utf-8"))
    original_read_text = Path.read_text

    def windows_read_text(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> str:
        return original_read_text(
            path, encoding=encoding or "cp1252", errors=errors, newline=newline
        )

    monkeypatch.setattr(Path, "read_text", windows_read_text)

    assert lint_portable(recipe_path) == []


def test_portable_lint_reports_non_utf8_source(tmp_path: Path) -> None:
    recipe_path = tmp_path / "recipe.py"
    recipe_path.write_bytes(("# Neutral recipe — café\n" + _source()).encode("cp1252"))

    diagnostics = lint_portable(recipe_path)

    assert len(diagnostics) == 1
    assert "cannot parse recipe" in diagnostics[0].actual
    assert "utf-8" in diagnostics[0].actual


def test_portable_lint_rejects_awaited_synchronous_agent_primitive(tmp_path):
    source = _source(
        'return await agent("worker", "review", working_directory=target, '
        "readable_paths=(target,))",
        kind="agent",
    )
    source = source.replace("def work(target: str)", "async def work(target: str)")
    source = source.replace("def run(target: str", "async def run(target: str")
    source = source.replace("return work(target)", "return await work(target)")

    messages = _messages(tmp_path / "recipe.py", source)

    assert any("synchronous Calcifier primitive agent() is awaited" in item for item in messages)


def test_portable_lint_rejects_obsolete_plan_metadata(tmp_path):
    source = _source() + "_CALCIFIER_PLAN = {}\n_CALCIFIER_PLAN_SHA256 = 'x'\n"

    messages = _messages(tmp_path / "recipe.py", source)

    assert any("obsolete _CALCIFIER_PLAN metadata" in item for item in messages)
    assert any("obsolete _CALCIFIER_PLAN_SHA256 metadata" in item for item in messages)


@pytest.mark.parametrize(
    ("path", "pinned", "trusted"),
    [
        ("../SKILL.md", True, True),
        ("../SKILL.md", False, False),
        ("../../outside/SKILL.md", True, False),
        ("<absolute>", True, False),
    ],
    ids=["contained-and-pinned", "hash-mismatch", "escaping", "absolute"],
)
def test_portable_lint_trusts_only_a_contained_hash_pinned_skill(
    tmp_path, path: str, pinned: bool, trusted: bool
):
    skill_dir = tmp_path / "skills" / "example"
    recipe = skill_dir / ".calcifier" / "recipe.py"
    for skill in (skill_dir / "SKILL.md", tmp_path / "outside" / "SKILL.md"):
        skill.parent.mkdir(parents=True, exist_ok=True)
        skill.write_bytes(b"---\nallowed-tools: Read\n---\n# skill\n")
    digest = (
        hashlib.sha256((skill_dir / "SKILL.md").read_bytes()).hexdigest() if pinned else "0" * 64
    )
    raw = (skill_dir / "SKILL.md").as_posix() if path == "<absolute>" else path

    text = _pinned_text(recipe, Path(raw), digest)

    assert (text == "---\nallowed-tools: Read\n---\n# skill\n") is trusted
    assert (text is None) is not trusted


def test_portable_lint_rejects_runtime_loader_contract_failures(tmp_path):
    missing_return = _source().replace("def work(target: str) -> bool:", "def work(target: str):")
    unavailable_import = ("import openpyxl\n" + _source()).replace(
        "return bool(target)", "return bool(target and openpyxl)"
    )

    assert any(
        "requires an explicit return annotation" in message
        for message in _messages(tmp_path / "missing-return.py", missing_return)
    )
    assert any(
        "unavailable top-level dependencies ['openpyxl']" in message
        for message in _messages(tmp_path / "unavailable-import.py", unavailable_import)
    )


def test_portable_lint_accepts_explicit_none_step_return(tmp_path: Path):
    source = (
        _source()
        .replace(
            "def work(target: str) -> bool:\n    return bool(target)",
            "def work(target: str) -> None:\n    return None",
        )
        .replace(
            'def run(target: str = ".") -> bool:',
            'def run(target: str = ".") -> None:',
        )
    )

    assert _messages(tmp_path / "none-return.py", source) == []


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("return bool(open(target, 'w'))", "arbitrary generated effect"),
        ("return os.system(target) == 0", "arbitrary generated effect"),
        (
            "return agent('worker', 'review')",
            "agent call requires working_directory or PreparedContext",
        ),
    ],
)
def test_portable_lint_rejects_provider_calls_writes_and_open_fs_contracts(
    tmp_path, replacement: str, message: str
):
    source = _source(replacement, kind="agent" if "agent(" in replacement else "python")
    if "os.system" in replacement:
        source = "import os\n" + source

    assert any(message in item for item in _messages(tmp_path / "recipe.py", source))


def test_portable_lint_rejects_open_output_records(tmp_path):
    source = (
        _source()
        .replace(
            "from calcifier import recipe, step",
            "from dataclasses import dataclass\nfrom calcifier import recipe, step",
        )
        .replace("@step\ndef work", "@dataclass\nclass Result:\n    ok: bool\n\n@step\ndef work")
    )

    assert any(
        "open dataclass output; use @output_type" in message
        for message in _messages(tmp_path / "recipe.py", source)
    )


def test_portable_lint_reports_one_authoritative_semantic_diagnostic(tmp_path: Path):
    messages = _messages(tmp_path / "recipe.py", _source("return bool(open(target, 'w'))"))

    assert sum("arbitrary generated effect" in message for message in messages) == 1
    assert not any("arbitrary generated write" in message for message in messages)


def test_checked_in_dogfood_recipes_pass_portable_gate():
    root = Path(__file__).parents[1] / "plugins" / "calcifier" / "skills"
    paths = sorted(root.glob("*/.calcifier/recipe.py"))

    assert paths
    assert {path: lint_portable(path) for path in paths} == {path: [] for path in paths}


def test_portable_lint_reports_missing_invalid_and_multiple_recipes(tmp_path):
    missing = tmp_path / "missing.py"
    assert lint_portable(missing)[0].actual.startswith("cannot parse recipe")
    assert _messages(tmp_path / "syntax.py", "not python !")[0].startswith("cannot parse recipe")
    assert any("no @recipe" in message for message in _messages(tmp_path / "none.py", "VALUE = 1"))
    doubled = _source().replace(
        "@recipe()\ndef run",
        "@recipe()\ndef other() -> bool:\n    return True\n\n@recipe()\ndef run",
    )
    assert any("multiple @recipe" in message for message in _messages(tmp_path / "two.py", doubled))


def test_portable_lint_rejects_syntax_warnings(tmp_path):
    warned = _source().replace(
        "return bool(target)",
        'pattern = "\\("\n    return bool(pattern)',
    )

    messages = _messages(tmp_path / "warning.py", warned)

    assert any(message.startswith("SyntaxWarning:") for message in messages)
    assert any("invalid escape sequence" in message for message in messages)

    bytecode_warning = _source().replace(
        "return bool(target)",
        'assert (target, "always true")\n    return bool(target)',
    )
    bytecode_messages = _messages(tmp_path / "bytecode-warning.py", bytecode_warning)
    assert any("assertion is always true" in message for message in bytecode_messages)


def test_portable_lint_reports_roles_budget_mutations_and_effects(tmp_path):
    agent_source = _source(
        'return agent("worker", "review", working_directory=target, readable_paths=(target,))',
        kind="agent",
    )
    provider_role = agent_source.replace('agents=("worker",)', 'agents=("codex",)')
    dynamic_role = agent_source.replace('agent("worker",', "agent(target,")
    missing_budget = agent_source.replace(
        'agents=("worker",), budget=Budget(2, 30)', 'agents=("worker",)'
    )
    mutation = _source("return bool(write_text(target, 'x'))").replace(
        "from calcifier import recipe, step",
        "from calcifier import recipe, step, write_text",
    )
    effect = _source("return bool(record_side_effect('other', target))").replace(
        "from calcifier import recipe, step",
        "from calcifier import recipe, record_side_effect, step",
    )

    assert any(
        "provider names cannot be semantic roles" in message
        for message in _messages(tmp_path / "provider.py", provider_role)
    )
    assert any(
        "dynamic agent role" in message
        for message in _messages(tmp_path / "dynamic.py", dynamic_role)
    )
    assert any(
        "agent-backed recipe requires Budget" in message
        for message in _messages(tmp_path / "budget.py", missing_budget)
    )
    assert any(
        "mutating recipe requires a concrete literal output_root" in message
        for message in _messages(tmp_path / "mutation.py", mutation)
    )
    assert any(
        "generated recipes must use deterministic mutation primitives" in message
        for message in _messages(tmp_path / "effect.py", effect)
    )


def test_portable_lint_distinguishes_string_and_path_replace(tmp_path: Path):
    string_source = _source('return bool(str(target).replace("\\t", "    "))')
    path_source = _source('return bool(Path(target).replace("other"))').replace(
        "from calcifier import recipe, step",
        "from pathlib import Path\n\nfrom calcifier import recipe, step",
    )

    assert not any(
        "arbitrary generated path receiver method 'replace'" in message
        for message in _messages(tmp_path / "string.py", string_source)
    )
    assert any(
        "arbitrary generated path receiver method 'replace'" in message
        for message in _messages(tmp_path / "path.py", path_source)
    )


def test_portable_lint_enforces_configured_minimum_python(tmp_path: Path):
    (tmp_path / "ruff.toml").write_text('target-version = "py311"\n')
    source = (
        _source()
        .replace(
            "from calcifier import recipe, step",
            "import re\n\nfrom calcifier import recipe, step",
        )
        .replace(
            "return bool(target)",
            "return bool(f\"{re.sub(r'\\t', '    ', target)}\")",
        )
    )

    messages = _messages(tmp_path / "recipe.py", source)

    assert any("invalid-syntax" in message and "Python 3.11" in message for message in messages)


def test_portable_lint_verifies_subrecipe_provenance(tmp_path):
    child = tmp_path / "child.py"
    child.write_text(_source())
    digest = hashlib.sha256(child.read_bytes()).hexdigest()
    source = _source("return bool(call_recipe('child.py', {}).output)", kind="subrecipe")
    no_metadata = _replace_assignment(source, "_CALCIFIER_SUBRECIPES", None)
    invalid = _replace_assignment(source, "_CALCIFIER_SUBRECIPES", "object()")
    invalid_entry = _replace_assignment(source, "_CALCIFIER_SUBRECIPES", "['invalid']")
    valid = format_and_lint_source(
        _replace_assignment(
            source,
            "_CALCIFIER_SUBRECIPES",
            repr([{"path": "child.py", "sha256": digest}]),
        ),
        tmp_path / "valid.py",
    )
    missing = valid.replace('"child.py"', '"absent-child.py"')
    drift = valid.replace(digest, "0" * 64)
    dynamic = valid.replace('call_recipe("child.py"', "call_recipe(target")
    extra = format_and_lint_source(
        _replace_assignment(
            source,
            "_CALCIFIER_SUBRECIPES",
            repr(
                [
                    {"path": "child.py", "sha256": digest},
                    {"path": "extra.py", "sha256": "0" * 64},
                ]
            ),
        ),
        tmp_path / "extra.py",
    )

    assert "unresolved subrecipes; provenance is missing" in _messages(
        tmp_path / "no-metadata.py", no_metadata
    )
    assert "invalid subrecipe provenance" in _messages(tmp_path / "invalid.py", invalid)
    assert "invalid subrecipe provenance" in _messages(tmp_path / "invalid-entry.py", invalid_entry)
    assert "unresolved subrecipe path" in _messages(tmp_path / "missing.py", missing)
    assert "subrecipe hash mismatch" in _messages(tmp_path / "drift.py", drift)
    assert "subrecipe path must be literal" in _messages(tmp_path / "dynamic.py", dynamic)
    assert "subrecipe topology mismatch" in _messages(tmp_path / "extra.py", extra)
    assert _messages(tmp_path / "valid.py", valid) == []


def test_portable_lint_recursively_rejects_invalid_pinned_child(tmp_path: Path):
    grandchild = tmp_path / "grandchild.py"
    grandchild.write_text(
        _source().replace("def work(target: str) -> bool:", "def work(target: str):")
    )
    grandchild_digest = hashlib.sha256(grandchild.read_bytes()).hexdigest()
    child = tmp_path / "child.py"
    child.write_text(
        _source("return bool(call_recipe('grandchild.py', {}).output)", kind="subrecipe").replace(
            "_CALCIFIER_SUBRECIPES = []",
            "_CALCIFIER_SUBRECIPES = "
            + repr(
                [
                    {
                        "name": "grandchild",
                        "path": "grandchild.py",
                        "sha256": grandchild_digest,
                    }
                ]
            ),
        )
    )
    child_digest = hashlib.sha256(child.read_bytes()).hexdigest()
    parent = tmp_path / "parent.py"
    parent.write_text(
        _source("return bool(call_recipe('child.py', {}).output)", kind="subrecipe").replace(
            "_CALCIFIER_SUBRECIPES = []",
            "_CALCIFIER_SUBRECIPES = "
            + repr([{"name": "child", "path": "child.py", "sha256": child_digest}]),
        )
    )

    messages = [diagnostic.actual for diagnostic in lint_portable(parent)]

    assert any(str(child) in message for message in messages)
    assert any(str(grandchild) in message for message in messages)
    assert any("requires an explicit return annotation" in message for message in messages)


def test_portable_lint_applies_selected_python_version_to_pinned_child(tmp_path: Path):
    child = tmp_path / "child.py"
    child.write_text(
        _source().replace(
            "return bool(target)",
            """return bool(f"{target.replace(r'\\t', ' ')}")""",
        )
    )
    digest = hashlib.sha256(child.read_bytes()).hexdigest()
    parent = tmp_path / "parent.py"
    parent.write_text(
        _source("return bool(call_recipe('child.py', {}).output)", kind="subrecipe").replace(
            "_CALCIFIER_SUBRECIPES = []",
            "_CALCIFIER_SUBRECIPES = "
            + repr([{"name": "child", "path": "child.py", "sha256": digest}]),
        )
    )

    messages = [diagnostic.actual for diagnostic in lint_portable(parent, python_version="3.11")]

    assert any(
        str(child) in message and "invalid-syntax" in message and "Python 3.11" in message
        for message in messages
    )


def test_portable_lint_and_hard_dry_run_agree_on_plugin_resource_source(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    plugin_root = tmp_path / "plugins" / "example"
    skill_root = plugin_root / "skills" / "report"
    recipe_path = skill_root / ".calcifier" / "recipe.py"
    recipe_path.parent.mkdir(parents=True)
    skill = skill_root / "SKILL.md"
    skill.write_text("# Report\n")
    script = plugin_root / "scripts" / "emit_summary.py"
    script.parent.mkdir()
    script.write_text("print('exported')\n")
    source = _source() + textwrap.dedent(
        f"""
        _CALCIFIER_SKILL = {{
            "path": "../SKILL.md",
            "sha256": {hashlib.sha256(skill.read_bytes()).hexdigest()!r},
        }}
        _CALCIFIER_RESOURCES = [
            {{
                "path": "scripts/emit_summary.py",
                "source": "../../scripts/emit_summary.py",
                "sha256": {hashlib.sha256(script.read_bytes()).hexdigest()!r},
            }}
        ]
        """
    )
    recipe_path.write_text(source)

    lint_rc = cli.main(["lint", str(recipe_path), "--portable"])
    exec_rc = cli.main(["exec", str(recipe_path), "--trust", "--dry-run", "hard"])

    output = capsys.readouterr()
    assert lint_rc == 0
    assert exec_rc == 0
    assert "portable: OK" in output.out

    escaped = source.replace(
        '"source": "../../scripts/emit_summary.py"',
        '"source": "../../../../outside/emit_summary.py"',
    )
    recipe_path.write_text(escaped)

    lint_rc = cli.main(["lint", str(recipe_path), "--portable"])
    exec_rc = cli.main(["exec", str(recipe_path), "--trust", "--dry-run", "hard"])

    output = capsys.readouterr()
    assert lint_rc != 0
    assert exec_rc != 0
    assert "source escapes its package" in output.err


# Known-answer boundaries


@functools.cache
def _portable_recipe() -> str:
    """Return a clean recipe whose last line is ``_CALCIFIER_SUBRECIPES = []``."""
    source = textwrap.dedent(
        """
        from calcifier import recipe, step

        @step
        def work(target: str) -> bool:
            return bool(target)

        @recipe()
        def run(target: str = ".") -> bool:
            return work(target)

        if __name__ == "__main__":
            raise SystemExit(run.cli())

        _CALCIFIER_SUBRECIPES = []
        """
    )
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "ruff.toml").write_text("line-length = 100\n", encoding="utf-8")
        return format_and_lint_source(source, Path(directory) / "recipe.py", python_version="3.13")


def _last_line(source: str | None = None) -> int:
    """The fixture's ``_CALCIFIER_SUBRECIPES`` line; text appended to it starts on the next."""
    return len((source or _portable_recipe()).splitlines())


def _lint(path: Path, source: str) -> list[tuple[int, str]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return [(item.line or 1, item.actual) for item in lint_portable(path, python_version="3.13")]


def test_portable_fixture_is_clean(tmp_path: Path) -> None:
    source = _portable_recipe()

    assert source.splitlines()[-1] == "_CALCIFIER_SUBRECIPES = []"
    assert _lint(tmp_path / "recipe.py", source) == []


@pytest.mark.parametrize(
    "provenance",
    ["[]", "{'path': 1, 'sha256': 'x'}", "{'path': 'SKILL.md', 'sha256': 1}"],
)
def test_skill_provenance_requires_a_mapping_with_string_path_and_hash(
    provenance: str, tmp_path: Path
) -> None:
    source = _portable_recipe() + f"_CALCIFIER_SKILL = {provenance}\n"

    assert _lint(tmp_path / "recipe.py", source) == [(_last_line() + 1, "invalid skill provenance")]


def _skill_block(path: str, sha256: str) -> str:
    """Render multi-line provenance; only an absolute path line can exceed the limit."""
    return (
        "_CALCIFIER_SKILL = {\n"
        f'    "path": "{path}",  # noqa: E501\n'
        f'    "sha256": "{sha256}",\n'
        "}\n"
    )


@pytest.mark.parametrize(
    ("recipe", "skill_path", "hash_ok", "message"),
    [
        ("recipe.py", "SKILL.md", True, None),
        (".calcifier/recipe.py", "../SKILL.md", True, None),
        ("nested/recipe.py", "../SKILL.md", True, None),
        ("recipe.py", "SKILL.md", False, "skill provenance hash mismatch"),
        ("recipe.py", "<absolute>", True, "skill provenance path must be relative"),
    ],
    ids=["contained", "calcifier-dir", "escaping", "hash-mismatch", "absolute"],
)
def test_skill_provenance_must_be_relative_and_hash_pinned(
    recipe: str, skill_path: str, hash_ok: bool, message: str | None, tmp_path: Path
) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text("Check the target.\n")
    digest = hashlib.sha256(skill.read_bytes()).hexdigest() if hash_ok else "0" * 64
    # A fixed absolute path, not the temporary skill's: a /tmp literal would also draw
    # Ruff S108 on Linux, and a Windows path's backslashes would need escaping.
    absolute = Path(tmp_path.anchor, "srv", "skills", "SKILL.md").as_posix()
    raw_path = absolute if skill_path == "<absolute>" else skill_path
    source = _portable_recipe() + _skill_block(raw_path, digest)

    expected = [] if message is None else [(_last_line() + 1, message)]
    assert _lint(tmp_path / recipe, source) == expected


@pytest.mark.parametrize("entries", ["['stage-a']", "[{'name': 'stage-a'}]"])
def test_malformed_subrecipe_entries_are_ignored_without_calls(
    entries: str, tmp_path: Path
) -> None:
    source = _portable_recipe().replace(
        "_CALCIFIER_SUBRECIPES = []", f"_CALCIFIER_SUBRECIPES = {entries}"
    )

    assert _lint(tmp_path / "recipe.py", source) == []


_CATALOG_CALL = "of function `call_catalog_recipe`"
_LITERAL_MAPPING = "catalog subrecipe mapping must be literal"
_TOPOLOGY = "subrecipe topology mismatch"


@pytest.mark.parametrize(
    ("arguments", "finding", "type_error"),
    [
        (
            "'stage-a'",
            _LITERAL_MAPPING,
            "ty[missing-argument]: No arguments provided for required parameters "
            f"`recipes`, `inputs` {_CATALOG_CALL}",
        ),
        (
            "'stage-a', {'stage-a': 'child.py'}",
            _TOPOLOGY,
            f"ty[missing-argument]: No argument provided for required parameter `inputs` "
            f"{_CATALOG_CALL}",
        ),
        ("'stage-a', {'stage-a': 'child.py'}, {'target': target}", _TOPOLOGY, None),
        (
            "'stage-a', {'stage-a': 1}",
            _LITERAL_MAPPING,
            f"ty[missing-argument]: No argument provided for required parameter `inputs` "
            f"{_CATALOG_CALL}",
        ),
        ("'stage-a', {'stage-a': target}, {'target': target}", _LITERAL_MAPPING, None),
    ],
    ids=["name-only", "literal-mapping", "literal-mapping-inputs", "int-path", "name-path"],
)
def test_catalog_call_mapping_is_the_second_positional_literal(
    arguments: str, finding: str, type_error: str | None, tmp_path: Path
) -> None:
    source = _portable_recipe().replace(
        "from calcifier import recipe, step",
        "from calcifier import call_catalog_recipe, recipe, step",
    )
    last = _last_line(source)
    source += (
        f"\n\ndef dispatch(target: str) -> object:\n    return call_catalog_recipe({arguments})\n"
    )
    call = last + 4
    # A literal-mapping finding sits on the call; a topology finding on the pins line.
    finding_line = call if finding == _LITERAL_MAPPING else last
    expected = [(finding_line, finding)]
    if type_error is not None:
        expected.append((call, f"ty validation failed: line {call}: {type_error}"))
    assert _lint(tmp_path / "recipe.py", source) == sorted(expected)


def test_aliased_module_subrecipe_call_is_recognized(tmp_path: Path) -> None:
    source = _portable_recipe().replace(
        "from calcifier import recipe, step",
        "import calcifier as cf\nfrom calcifier import recipe, step",
    )
    last = _last_line(source)
    source += (
        "\n\ndef dispatch(target: str) -> object:\n"
        "    return cf.call_recipe('child.py', {'target': target})\n"
    )

    assert _lint(tmp_path / "recipe.py", source) == [(last, _TOPOLOGY)]


@pytest.mark.parametrize(
    "declaration",
    [
        'CHILDREN: dict[str, str] = {"child": "child.py"}\nCHILDREN = {"child": "other.py"}',
        'CHILDREN: dict[str, str] = {"child": "child.py".strip()}',
        "CHILDREN: dict[str, str]",
    ],
)
def test_catalog_annotations_do_not_make_dynamic_or_ambiguous_mappings_literal(
    tmp_path: Path, declaration: str
) -> None:
    source = _portable_recipe().replace(
        "from calcifier import recipe, step",
        "from calcifier import call_catalog_recipe, recipe, step",
    )
    source += (
        f"\n{declaration}\n\ndef dispatch(target: str) -> object:\n"
        '    return call_catalog_recipe("child", CHILDREN, {"target": target})\n'
    )
    assert _LITERAL_MAPPING in _messages(tmp_path / "recipe.py", source)


@pytest.mark.parametrize(
    ("source", "line"), [("def broken(:\n", 1), ("x = 1\ny = 2\ndef broken(:\n", 3)]
)
def test_unparsable_recipe_reports_the_syntax_error_line(
    source: str, line: int, tmp_path: Path
) -> None:
    (diagnostic,) = lint_portable_source(source, tmp_path / "recipe.py", python_version="3.13")

    assert diagnostic.line == line
    assert diagnostic.actual.startswith("cannot parse recipe: ")
