"""Provider transport for model-written recipes and its syntax allowlist."""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
import warnings
from typing import Literal, cast

from .exceptions import CalcifierError, ErrorFamily
from .outputs import OutputParseError, output_type, parse_response
from .source_quality import python_feature_version


class RecipeDraftError(CalcifierError, ValueError):
    """Raised when a recipe draft is malformed or uses unsupported syntax."""

    family = ErrorFamily.COMPILATION


@output_type
class RecipeDraft:
    """One complete recipe module returned by the compiler model.

    The source is untrusted syntax, never evaluated or imported by the compiler.
    """

    version: Literal["1"]
    source: str
    suitability_note: str | None = None


# This is intentionally explicit. Adding Python syntax to generated recipes is
# a compiler policy change that requires tests.
_ALLOWED_NODE_KINDS = frozenset(
    {
        "Add",
        "And",
        "AnnAssign",
        "Assert",
        "Assign",
        "AsyncFor",
        "AsyncFunctionDef",
        "Attribute",
        "AugAssign",
        "Await",
        "BinOp",
        "BitAnd",
        "BitOr",
        "BoolOp",
        "Break",
        "Call",
        "ClassDef",
        "Compare",
        "Constant",
        "Continue",
        "Dict",
        "DictComp",
        "Div",
        "Eq",
        "ExceptHandler",
        "Expr",
        "FloorDiv",
        "For",
        "FormattedValue",
        "FunctionDef",
        "GeneratorExp",
        "Gt",
        "GtE",
        "If",
        "IfExp",
        "Import",
        "ImportFrom",
        "In",
        "Invert",
        "Is",
        "IsNot",
        "JoinedStr",
        "Lambda",
        "List",
        "ListComp",
        "Load",
        "Lt",
        "LtE",
        "MatMult",
        "Mod",
        "Module",
        "Mult",
        "Name",
        "NamedExpr",
        "Not",
        "NotEq",
        "NotIn",
        "Or",
        "Pass",
        "Pow",
        "Raise",
        "Return",
        "Set",
        "SetComp",
        "Slice",
        "Starred",
        "Store",
        "Sub",
        "Subscript",
        "Try",
        "Tuple",
        "TypeIgnore",
        "UAdd",
        "USub",
        "UnaryOp",
        "While",
        "With",
        "Yield",
        "YieldFrom",
        "alias",
        "arg",
        "arguments",
        "comprehension",
        "keyword",
        "withitem",
    }
)
_MAX_NODES = 20_000
_MAX_DEPTH = 200
_MAX_DRAFT_BYTES = 256 * 1024


def parse_recipe_draft(text: str, *, python_version: str | None = None) -> str:
    """Decode one bounded RecipeDraft response and return its source as written.

    Validate syntax for the target Python version and add a trailing newline.
    Invalid, oversized, or unsupported drafts raise RecipeDraftError. Never
    re-render the AST: that would collapse deliberately wrapped string literals.
    """
    return decode_recipe_draft(text, python_version=python_version).source


def decode_recipe_draft(text: str, *, python_version: str | None = None) -> RecipeDraft:
    """Validate source and retain the compiler's optional advisory assessment."""
    if len(text.encode()) > _MAX_DRAFT_BYTES:
        raise RecipeDraftError(f"recipe draft exceeds {_MAX_DRAFT_BYTES} bytes")
    try:
        # Strict JSON first: never salvage a draft from prose or tool output.
        payload = json.loads(text)
        draft = cast("RecipeDraft", parse_response(json.dumps(payload), RecipeDraft))
        if not draft.source.strip():
            raise RecipeDraftError("recipe draft source is empty; return a complete recipe module")
        with warnings.catch_warnings():
            warnings.simplefilter("error", SyntaxWarning)
            warnings.simplefilter("error", DeprecationWarning)
            module = ast.parse(draft.source, feature_version=python_feature_version(python_version))
    except (
        SyntaxError,
        SyntaxWarning,
        DeprecationWarning,
        OutputParseError,
        json.JSONDecodeError,
        RecursionError,
    ) as exc:
        raise RecipeDraftError(f"invalid or incomplete recipe draft: {exc}") from exc
    check_recipe_syntax(module)
    _reject_type_check_suppression(draft.source, module)
    return RecipeDraft(
        version=draft.version,
        source=draft.source if draft.source.endswith("\n") else draft.source + "\n",
        suitability_note=(draft.suitability_note or "").strip() or None,
    )


# ty's configuration can switch off type-ignore comments, but not its own directive.
_TY_IGNORE = re.compile(r"#\s*ty\s*:\s*ignore\b")


def _reject_type_check_suppression(source: str, module: ast.Module) -> None:
    """Reject the type-check suppressions that ty's configuration cannot switch off."""
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    if any(token.type == tokenize.COMMENT and _TY_IGNORE.search(token.string) for token in tokens):
        raise RecipeDraftError("`# ty: ignore` is not allowed; fix the type error it hides")
    for node in ast.walk(module):
        name = node.id if isinstance(node, ast.Name) else getattr(node, "attr", None)
        if name == "no_type_check":
            raise RecipeDraftError("`no_type_check` is not allowed; fix the type errors it hides")


def check_recipe_syntax(module: ast.Module) -> None:
    """Reject syntax outside the allowlist and trees beyond the size bounds.

    Args:
        module: Parsed recipe module.

    Raises:
        RecipeDraftError: A node kind is not allowed, or the tree is too large or deep.
    """
    pending: list[tuple[ast.AST, int]] = [(module, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > _MAX_NODES or depth > _MAX_DEPTH:
            raise RecipeDraftError("recipe draft exceeds node/depth limits")
        kind = type(node).__name__
        if kind not in _ALLOWED_NODE_KINDS:
            raise RecipeDraftError(f"unsupported Python syntax in recipe draft: {kind}")
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))


def draft_response(source: str) -> str:
    """Wrap complete module source as the RecipeDraft response a provider returns."""
    return json.dumps({"version": "1", "source": source}, ensure_ascii=False)


def recipe_draft_prompt_contract() -> str:
    """Describe the RecipeDraft transport and the accepted syntax."""
    kinds = ", ".join(sorted(_ALLOWED_NODE_KINDS))
    return (
        'Return one RecipeDraft JSON object: {"version":"1","source":"<complete Python module>"}. '
        "The optional suitability_note is null unless little meaningful executable work can be "
        "extracted, for example from ongoing behavioral guidance. In that case explain why owner "
        "review is needed and what remains agent-owned; preserve the guidance without inventing a "
        "procedure. This is an advisory assessment, not a pass/fail classifier or a measure of "
        "agent-call counts. Compiled source still goes through all checks. "
        "Put the complete, nonempty module in the source string of your FINAL structured response. "
        "Do not return an AST JSON tree, Markdown fences, a patch, or a file reference. "
        "Do not run Python-to-AST conversion commands, shell commands, or write temporary files. "
        "The host parses your source and checks it before all semantic, effect, portability, and "
        "load checks. Arbitrary imports and other syntax remain forbidden. "
        f"Only these AST node kinds are accepted: {kinds}. "
        f"The complete JSON response must fit within {_MAX_DRAFT_BYTES} UTF-8 bytes."
    )
