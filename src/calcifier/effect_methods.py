"""Recognize direct filesystem mutation methods in generated recipes."""

from __future__ import annotations

import ast

FORBIDDEN_EFFECT_METHODS = frozenset(
    {
        "chmod",
        "mkdir",
        "open",
        "rename",
        "rmdir",
        "touch",
        "unlink",
        "write",
        "write_bytes",
        "write_text",
        "writelines",
    }
)


def forbidden_method(call: ast.Call) -> str | None:
    """Reject path replacement syntax; leave string substitution to normal Python.

    Path.replace takes one destination, while str.replace takes old and new values.
    This is a small syntax check, not an inference of every possible receiver type.
    """
    if not isinstance(call.func, ast.Attribute):
        return None
    method = call.func.attr
    if method in FORBIDDEN_EFFECT_METHODS:
        return method
    if method == "replace" and (
        len(call.args) == 1 or any(keyword.arg == "target" for keyword in call.keywords)
    ):
        return method
    return None
