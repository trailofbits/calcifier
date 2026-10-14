"""Catalog loading and child recipe prompt contracts."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from calcifier.catalog import _catalog_dependency_paths, _merge_skill_catalogs, load_skill_catalog
from calcifier.diagnostics import CalcifyError
from calcifier.prompts import _catalog_prompt_payload
from tests.support.recipes import _GOOD_RECIPE


def test_load_skill_catalog_resolves_relative_recipes_strictly(tmp_path: Path):
    child = tmp_path / "child.py"
    child.write_text("# child")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"version": 1, "recipes": {"child": "child.py"}}))

    assert load_skill_catalog(catalog) == {"child": str(child.resolve())}

    catalog.write_text(json.dumps({"version": 2, "recipes": {}}))
    with pytest.raises(CalcifyError, match="version must be 1"):
        load_skill_catalog(catalog)
    catalog.write_text(json.dumps({"version": 1, "recipes": {"missing": "missing.py"}}))
    with pytest.raises(CalcifyError, match="does not exist"):
        load_skill_catalog(catalog)
    catalog.write_text("not json")
    with pytest.raises(CalcifyError, match="failed to load"):
        load_skill_catalog(catalog)
    catalog.write_text('{"version": 1, "recipes": {"child": "child.py", "child": "child.py"}}')
    with pytest.raises(CalcifyError, match="duplicate key 'child'"):
        load_skill_catalog(catalog)


def test_catalog_prompt_describes_each_child_from_its_own_recipe(tmp_path: Path):
    structured = tmp_path / "structured.py"
    structured.write_text(
        textwrap.dedent(
            """
            from calcifier import output_type, recipe, step

            @output_type
            class Item:
                name: str

            @output_type
            class Envelope:
                items: list[Item]

            @step
            def produce() -> Envelope:
                return Envelope(items=[])

            @recipe()
            def structured_child() -> Envelope:
                return produce()

            if __name__ == "__main__":
                raise SystemExit(structured_child.cli())
            """
        )
    )
    serialized = tmp_path / "serialized.py"
    serialized.write_text(
        textwrap.dedent(
            """
            from calcifier import recipe, step

            @step
            def resolve_output_path(output_path: str) -> str:
                if output_path.startswith("/"):
                    raise ValueError("output_path must remain relative")
                return output_path

            @step
            def serialize(output_path: str) -> str:
                return "# Work Items\\n- **Recipe Key:** /analyze-real"

            @recipe()
            def serialized_child(output_path: str) -> str:
                resolved = resolve_output_path(output_path)
                rendered = serialize(resolved)
                return rendered

            if __name__ == "__main__":
                raise SystemExit(serialized_child.cli())
            """
        )
    )
    json_child = tmp_path / "json_child.py"
    json_child.write_text(
        textwrap.dedent(
            """
            import json
            from calcifier import recipe, step

            @step
            async def serialize_json() -> str:
                return json.dumps({"items": [], "total": 0})

            @recipe()
            async def json_child() -> str:
                return await serialize_json()

            if __name__ == "__main__":
                raise SystemExit(json_child.cli())
            """
        )
    )

    payload = _catalog_prompt_payload(
        {
            "structured": str(structured),
            "serialized": str(serialized),
            "json-child": str(json_child),
        },
        base=None,
    )

    assert payload["structured"]["output_types"] == {
        "Envelope": {"items": "list[Item]"},
        "Item": {"name": "str"},
    }
    # Only what each child's own recipe states: its inputs, return type, and budget.
    assert payload["serialized"] == {
        "path": str(serialized),
        "inputs": "output_path (required)",
        "returns": "str",
    }
    assert payload["json-child"] == {
        "path": str(json_child),
        "inputs": "(none)",
        "returns": "str",
    }


def test_catalog_prompt_resolves_child_relative_to_staged_destination(tmp_path: Path) -> None:
    (tmp_path / "child.py").write_text(_GOOD_RECIPE)

    payload = _catalog_prompt_payload({"child": "child.py"}, base=tmp_path)

    assert payload["child"]["path"] == "child.py"
    assert payload["child"]["returns"] == "bool"


def test_load_skill_catalog_rejects_ambiguous_and_cyclic_recipes(tmp_path: Path):
    marker = "# --- calcifier metadata (skill provenance + sha256) ---"
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text(f"# first\n{marker}\n_CALCIFIER_SUBRECIPES = [{{'path': 'second.py'}}]\n")
    second.write_text(f"# second\n{marker}\n_CALCIFIER_SUBRECIPES = [{{'path': 'first.py'}}]\n")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps({"version": 1, "recipes": {"first": "first.py", "second": "second.py"}})
    )

    with pytest.raises(CalcifyError, match="cyclic skill catalog dependency"):
        load_skill_catalog(catalog)

    catalog.write_text(
        json.dumps({"version": 1, "recipes": {"first": "first.py", "alias": "first.py"}})
    )
    with pytest.raises(CalcifyError, match="skill catalog is ambiguous"):
        load_skill_catalog(catalog)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not json", "failed to load skill catalog"),
        ("[]", "exactly version and recipes"),
        ('{"version": 1, "recipes": {}, "extra": true}', "exactly version and recipes"),
        ('{"version": 2, "recipes": {}}', "version must be 1"),
        ('{"version": 1, "recipes": []}', "version must be 1"),
        ('{"version": 1, "recipes": {"": "child.py"}}', "non-empty names"),
        ('{"version": 1, "recipes": {"alpha": 7}}', "non-empty names"),
        ('{"version": 1, "recipes": {"alpha": "missing.py"}}', "does not exist"),
        (
            '{"version": 1, "recipes": {"alpha": "a.py", "alpha": "b.py"}}',
            "duplicate key",
        ),
    ],
)
def test_load_skill_catalog_validation_matrix(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(payload)

    with pytest.raises(CalcifyError, match=message):
        load_skill_catalog(catalog)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ("_CALCIFIER_SUBRECIPES = {}", "invalid subrecipe metadata"),
        ("_CALCIFIER_SUBRECIPES = [1]", "invalid subrecipe metadata"),
        ("_CALCIFIER_SUBRECIPES = [{}]", "invalid subrecipe metadata"),
    ],
)
def test_catalog_dependency_metadata_validation_matrix(
    tmp_path: Path,
    metadata: str,
    message: str,
) -> None:
    child = tmp_path / "child.py"
    child.write_text("# --- calcifier metadata (skill provenance + sha256) ---\n" + metadata + "\n")

    with pytest.raises(CalcifyError, match=message):
        _catalog_dependency_paths(child)


def test_catalog_dependency_read_failure_is_actionable(tmp_path: Path) -> None:
    missing = tmp_path / "missing.py"

    with pytest.raises(CalcifyError, match="failed to inspect catalog recipe"):
        _catalog_dependency_paths(missing)


def test_explicit_catalog_merge_rejects_conflict_and_accepts_same_path(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("# first\n")
    second.write_text("# second\n")

    assert _merge_skill_catalogs({"alpha": str(first)}, {"alpha": str(first)}) == {
        "alpha": str(first)
    }
    with pytest.raises(CalcifyError, match="conflicts with discovered skill"):
        _merge_skill_catalogs({"alpha": str(first)}, {"alpha": str(second)})


def test_catalog_prompt_supports_bare_decorator_and_documented_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "child.py").write_text('''from calcifier import output_type, recipe
@output_type
class Report:
    """The consumer-visible report."""
    title: str
@recipe
def child() -> Report:
    return Report("done")
''')
    payload = _catalog_prompt_payload({"child": "child.py"}, base=None)
    assert payload["child"] == {
        "path": "child.py",
        "inputs": "(none)",
        "returns": "Report",
        "output_types": {"Report": {"title": "str"}},
    }
