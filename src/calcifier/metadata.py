"""Parse the provenance metadata appended to calcified recipes."""

from __future__ import annotations

import ast
import io
import os
import re
import tokenize
from collections.abc import Mapping
from pathlib import Path

from .diagnostics import CalcifyError
from .hashing import sha256_hex
from .skill_package import ResourceEntry
from .source_quality import python_feature_version

_METADATA_MARKER = "# --- calcifier metadata (skill provenance + sha256) ---"
_METADATA_RE = re.compile(rf"(?m)^{re.escape(_METADATA_MARKER)}\s*$")
_SCRIPT_METADATA_RE = re.compile(r"(?m)^# /// script\r?\n(?:#(?: .*)?\r?\n)*?# ///(?:\r?\n|$)")


def attach_script_metadata(source: str, python_version: str | None = None) -> str:
    """Stamp runner requirements for the installed recipe API, preserving source literals."""
    from . import __version__

    # Generated modules import only the standard library and Calcifier. External
    # programs and packaged scripts retain their own dependency environments.
    minimum = max((3, 13), python_feature_version(python_version) or (3, 13))
    comments = {
        token.start[0]
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and token.start[1] == 0
    }
    for match in reversed(list(_SCRIPT_METADATA_RE.finditer(source))):
        line = source.count("\n", 0, match.start()) + 1
        if all(line + offset in comments for offset in range(len(match.group().splitlines()))):
            source = source[: match.start()] + source[match.end() :]
    newline = "\r\n" if "\r\n" in source else "\n"
    header = newline.join(
        (
            "# /// script",
            f'# requires-python = ">={minimum[0]}.{minimum[1]}"',
            f'# dependencies = ["calcifier=={__version__}"]',
            "# ///",
            "",
            "",
        )
    )
    # Keep an executable script's shebang and encoding cookie in their original positions.
    lines = source.splitlines(keepends=True)
    prefix = 0
    if lines and lines[0].startswith("#!"):
        prefix = 1
    for index, line in enumerate(lines[:2]):
        if re.match(r"[ \t\f]*#.*coding[:=]\s*[-\w.]+", line):
            prefix = index + 1
    return "".join(lines[:prefix]) + header + "".join(lines[prefix:]).lstrip("\r\n")


def split_metadata(source: str) -> tuple[str, str | None]:
    """Return the normalized recipe body and its optional metadata block."""
    matches = list(_METADATA_RE.finditer(source))
    if not matches:
        return source.rstrip() + "\n", None
    index = matches[-1].start()
    return source[:index].rstrip() + "\n", source[index:]


def parse_metadata_namespace(metadata_text: str) -> dict[str, object]:
    """Safely read the supported constant assignments from metadata."""
    try:
        module = ast.parse(metadata_text)
    except SyntaxError:
        return {}
    return module_metadata(module)


def module_metadata(module: ast.Module, *, unique: bool = False) -> dict[str, object]:
    """Read the supported provenance constants from a parsed recipe.

    Raises:
        ValueError: ``unique`` is set and a constant is assigned more than once.
    """
    namespace: dict[str, object] = {}
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        target = statement.targets[0] if len(statement.targets) == 1 else None
        if unique and isinstance(target, ast.Name) and target.id in namespace:
            raise ValueError(f"duplicate provenance metadata: {target.id}")
        _parse_assignment(statement, namespace)
    return namespace


def interface_text(metadata: Mapping[str, object]) -> str:
    """Return pinned host interface text, or an empty string when none is pinned.

    Raises:
        ValueError: The interface record is malformed or its hash does not match.
    """
    interface = metadata.get("_CALCIFIER_INTERFACE")
    if interface is None:
        return ""
    if not isinstance(interface, dict) or set(interface) != {"text", "sha256"}:
        raise ValueError("invalid interface provenance")
    text = interface["text"]
    if not isinstance(text, str) or sha256_hex(text.encode("utf-8")) != interface["sha256"]:
        raise ValueError("interface source hash does not match")
    return text


def _parse_assignment(statement: ast.Assign, namespace: dict[str, object]) -> None:
    if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
        return
    name = statement.targets[0].id
    if name not in {
        "_CALCIFIER_COMPILATION",
        "_CALCIFIER_SKILL",
        "_CALCIFIER_RESOURCES",
        "_CALCIFIER_SUBRECIPES",
        "_CALCIFIER_INTERFACE",
    }:
        return
    try:
        namespace[name] = ast.literal_eval(statement.value)
    except (ValueError, TypeError):
        return


_METADATA_INLINE_STRING_LIMIT = 48
_METADATA_STRING_CHUNK_LIMIT = 60


def _resource_metadata(entries: list[ResourceEntry]) -> list[dict[str, object]]:
    return [
        {
            "path": entry.path,
            "sha256": entry.sha256,
            "size": entry.size,
            "source": entry.source,
        }
        for entry in sorted(entries, key=lambda item: item.path)
    ]


def metadata_assignment(name: str, value: object) -> str:
    return f"{name} = {metadata_literal(value)}\n"


def metadata_literal(  # noqa: C901 - recursive case per supported metadata value type
    value: object, *, indent: int = 0
) -> str:
    if isinstance(value, str):
        literal = repr(value)
        if len(literal) <= _METADATA_INLINE_STRING_LIMIT:
            return literal
        chunks = _split_metadata_string(value)
        padding = " " * (indent + 4)
        closing = " " * indent
        return "(\n" + "\n".join(f"{padding}{chunk!r}" for chunk in chunks) + f"\n{closing})"
    if value is None or isinstance(value, bool | int | float):
        return repr(value)
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = ["{"]
        child_indent = indent + 4
        padding = " " * child_indent
        for key, item in value.items():
            rendered = metadata_literal(item, indent=child_indent)
            first, *rest = rendered.splitlines()
            lines.append(f"{padding}{key!r}: {first}")
            lines.extend(rest)
            lines[-1] += ","
        lines.append(" " * indent + "}")
        return "\n".join(lines)
    if isinstance(value, list | tuple):
        if not value:
            return "[]" if isinstance(value, list) else "()"
        opening, closing = ("[", "]") if isinstance(value, list) else ("(", ")")
        lines = [opening]
        child_indent = indent + 4
        padding = " " * child_indent
        for item in value:
            rendered = metadata_literal(item, indent=child_indent)
            first, *rest = rendered.splitlines()
            lines.append(f"{padding}{first}")
            lines.extend(rest)
            lines[-1] += ","
        lines.append(" " * indent + closing)
        return "\n".join(lines)
    raise CalcifyError(f"unsupported metadata value type: {type(value).__name__}")


def _split_metadata_string(value: str) -> list[str]:
    chunks: list[str] = []
    current = ""
    for character in value:
        candidate = current + character
        if current and len(repr(candidate)) > _METADATA_STRING_CHUNK_LIMIT:
            chunks.append(current)
            current = character
        else:
            current = candidate
    chunks.append(current)
    return chunks


def attach_metadata(
    source: str,
    skill_path: Path,
    resources: list[ResourceEntry] | None = None,
    *,
    recipe_path: Path,
    subrecipes: dict[str, str] | None = None,
    skill_sha256: str | None = None,
    interface_requirements: str = "",
    outcome: str = "compiled",
) -> str:
    """Append a ``# calcifier metadata`` block: skill provenance, resources, and child pins.

    ``calcifier update`` reads the skill hash for drift detection, and trust pins
    each child recipe the source calls.
    """
    sha = sha256_hex(skill_path.read_bytes()) if skill_sha256 is None else skill_sha256
    recorded_path = Path(
        os.path.relpath(skill_path.resolve(), recipe_path.parent.resolve())
    ).as_posix()
    return (
        source.rstrip()
        + "\n\n\n"
        + _METADATA_MARKER
        + "\n"
        + metadata_assignment("_CALCIFIER_COMPILATION", outcome)
        + metadata_assignment("_CALCIFIER_SKILL", {"path": recorded_path, "sha256": sha})
        + metadata_assignment("_CALCIFIER_RESOURCES", _resource_metadata(resources or []))
        + (
            metadata_assignment(
                "_CALCIFIER_INTERFACE",
                {
                    "text": interface_requirements,
                    "sha256": sha256_hex(interface_requirements.encode("utf-8")),
                },
            )
            if interface_requirements
            else ""
        )
        + metadata_assignment(
            "_CALCIFIER_SUBRECIPES", _subrecipe_metadata(recipe_path, subrecipes or {})
        )
    )


def _subrecipe_metadata(recipe_path: Path, subrecipes: dict[str, str]) -> list[dict[str, str]]:
    result = []
    for name, raw_path in sorted(subrecipes.items()):
        child = Path(raw_path).resolve()
        relative = Path(os.path.relpath(child, recipe_path.parent.resolve())).as_posix()
        result.append(
            {
                "name": name,
                "path": relative,
                "sha256": sha256_hex(child.read_bytes()),
            }
        )
    return result


def skill_root_from_metadata(recipe_path: Path, skill: object) -> Path | None:
    """Resolve recorded skill provenance, falling back to the .calcifier layout."""
    if isinstance(skill, Mapping) and isinstance(skill.get("path"), str):
        skill_path = Path(skill["path"])
        if not skill_path.is_absolute():
            skill_path = recipe_path.parent / skill_path
        return skill_path.resolve().parent
    return recipe_path.parent.parent if recipe_path.parent.name == ".calcifier" else None
