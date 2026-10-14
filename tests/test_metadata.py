"""Recipe provenance metadata has an independent parsing boundary."""

from calcifier.metadata import (
    _METADATA_MARKER,
    parse_metadata_namespace,
    split_metadata,
)


def test_split_metadata_uses_last_full_marker():
    source = (
        f'PROMPT = "{_METADATA_MARKER}"\n'
        f"{_METADATA_MARKER}\n"
        '_CALCIFIER_SKILL = {"path": "SKILL.md", "sha256": "abc"}\n'
    )

    body, metadata = split_metadata(source)

    assert body == f'PROMPT = "{_METADATA_MARKER}"\n'
    assert metadata is not None and metadata.startswith(_METADATA_MARKER)


def test_parse_metadata_only_accepts_supported_literal_assignments():
    parsed = parse_metadata_namespace(
        '_CALCIFIER_SKILL = {"path": "SKILL.md", "sha256": "abc"}\n'
        '_CALCIFIER_RESOURCES = [{"path": "references/a.md"}]\n'
        '_CALCIFIER_SUBRECIPES = [{"path": "child.py", "sha256": "def"}]\n'
        '_CALCIFIER_PLAN = {"recipe_name": "obsolete"}\n'
        "IGNORED = 1\n"
    )

    # Plan metadata from before the plan stage was removed is not read.
    assert parsed == {
        "_CALCIFIER_SKILL": {"path": "SKILL.md", "sha256": "abc"},
        "_CALCIFIER_RESOURCES": [{"path": "references/a.md"}],
        "_CALCIFIER_SUBRECIPES": [{"path": "child.py", "sha256": "def"}],
    }


def test_parse_metadata_rejects_code_and_invalid_syntax():
    assert parse_metadata_namespace("_CALCIFIER_SKILL = make_value()\n") == {}
    assert parse_metadata_namespace("not valid python !") == {}
