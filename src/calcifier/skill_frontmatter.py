"""Minimal SKILL.md frontmatter parsing for calcifier metadata."""

from __future__ import annotations

import re

from .exceptions import CalcifierError, ErrorFamily

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_ALLOWED_TOOLS_RE = re.compile(r"^allowed-tools[ \t]*:[ \t]*(?P<value>.*)$", re.MULTILINE)
# Built-in names (Read), dotted providers (Slack.postMessage), MCP tools (mcp__docs__search).
_TOOL_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]*")
# A tool narrowed to some of its uses, as in Bash(git:*).
_SCOPED_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]*\(.*\)")


class AllowedToolsError(CalcifierError, ValueError):
    """The skill's allowed-tools field cannot be read as a list of tool names."""

    family = ErrorFamily.COMPILATION


def declared_allowed_tools(skill_text: str) -> tuple[str, ...] | None:
    """Return pre-approved tool declarations, preserving any permission scopes.

    Entries may be separated by whitespace or commas, or written as a YAML list.
    Neither a missing field nor a declared one restricts the available tools.

    Args:
        skill_text: The complete SKILL.md.

    Returns:
        The declared names in order, or None when the frontmatter lacks the field.

    Raises:
        AllowedToolsError: A declaration is malformed.
    """
    frontmatter = _FRONTMATTER_RE.match(skill_text)
    if frontmatter is None:
        return None
    body = frontmatter.group("body")
    field = _ALLOWED_TOOLS_RE.search(body)
    if field is None:
        return None
    value = field.group("value").strip()
    entries = _inline_entries(value) if value else _yaml_list_entries(body[field.end() :])
    return tuple(_tool_name(entry) for entry in entries)


def _inline_entries(value: str) -> list[str]:
    if value.startswith("[") != value.endswith("]"):
        raise AllowedToolsError(f"malformed allowed-tools list: {value!r}")
    if value.startswith("["):
        value = value[1:-1]
    entries: list[str] = []
    current = ""
    depth = 0
    for character in value + " ":
        depth += {"(": 1, ")": -1}.get(character, 0)
        if depth < 0:
            raise AllowedToolsError(f"malformed allowed-tools entry: {value!r}")
        if depth == 0 and (character.isspace() or character == ","):
            entries.append(current)
            current = ""
        else:
            current += character
    if depth:
        raise AllowedToolsError(f"malformed allowed-tools entry: {value!r}")
    return [entry.strip("'\"") for entry in entries if entry.strip("'\"")]


def _yaml_list_entries(text: str) -> list[str]:
    entries: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        item = re.match(r"^\s*-\s+(.+?)\s*$", line)
        if item is None:
            break
        entries.append(item.group(1).strip("'\""))
    return entries


def _tool_name(entry: str) -> str:
    if entry.count("(") != entry.count(")") or not (
        _TOOL_NAME_RE.fullmatch(entry) or _SCOPED_RE.fullmatch(entry)
    ):
        raise AllowedToolsError(f"malformed allowed-tools entry {entry!r}")
    return entry
