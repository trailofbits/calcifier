"""Tests for reading a SKILL.md's allowed-tools field."""

from __future__ import annotations

import re

import pytest

from calcifier.skill_frontmatter import AllowedToolsError, declared_allowed_tools


def _skill(field: str) -> str:
    return f"---\nname: example\n{field}\n---\n# skill\n"


@pytest.mark.parametrize(
    ("field", "tools"),
    [
        ("allowed-tools: Read Grep Glob", ("Read", "Grep", "Glob")),
        ("allowed-tools: Read, Grep, Bash", ("Read", "Grep", "Bash")),
        ("allowed-tools: Read,Grep  Bash", ("Read", "Grep", "Bash")),
        ('allowed-tools: ["Bash", "Read", "Write"]', ("Bash", "Read", "Write")),
        ("allowed-tools:\n  - Read\n  - mcp__docs__search", ("Read", "mcp__docs__search")),
        ("allowed-tools: Slack.postMessage", ("Slack.postMessage",)),
        ("allowed-tools: Bash(git *) Read", ("Bash(git *)", "Read")),
        ('allowed-tools: ["Bash(git log:*)", "Read"]', ("Bash(git log:*)", "Read")),
        ("allowed-tools:\n  - Bash(python -m example:*)", ("Bash(python -m example:*)",)),
        ("allowed-tools: Bash(git:*), Bash(npm test)", ("Bash(git:*)", "Bash(npm test)")),
        ("allowed-tools:", ()),
        ("allowed-tools: []", ()),
    ],
)
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_space_comma_and_yaml_lists_are_read(
    field: str, tools: tuple[str, ...], newline: str
) -> None:
    assert declared_allowed_tools(_skill(field).replace("\n", newline)) == tools


@pytest.mark.parametrize(
    "text",
    [
        "---\nname: example\n---\n# skill\n",
        "# no frontmatter\nallowed-tools: Read\n",
        "---\nallowed-tools: Read, Grep\n# frontmatter never closes\n",
    ],
)
def test_a_missing_field_is_distinct_from_an_empty_one(text: str) -> None:
    assert declared_allowed_tools(text) is None


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("allowed-tools: Read Bash(git:*", "malformed allowed-tools entry"),
        ("allowed-tools: Bash(git log:*)), Read", "malformed allowed-tools entry"),
        ("allowed-tools: Read ~/bin/tool", "malformed allowed-tools entry '~/bin/tool'"),
        ("allowed-tools:\n  - Bash)", "malformed allowed-tools entry 'Bash)'"),
        ("allowed-tools: [Read, Grep", "malformed allowed-tools list"),
    ],
)
def test_malformed_entries_are_refused(field: str, message: str) -> None:
    with pytest.raises(AllowedToolsError, match=re.escape(message)):
        declared_allowed_tools(_skill(field))
