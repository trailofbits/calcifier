"""Static portability conformance checks for generated recipes."""

from __future__ import annotations

import ast
import contextlib
import warnings
from pathlib import Path
from typing import Any

from .diagnostics import Diagnostic
from .hashing import sha256_hex
from .recipe_ast import called_subrecipes, import_bindings, subrecipe_calls
from .skill_package import discover_resources
from .source_quality import lint_source_quality, python_feature_version, resolve_python_version


def lint_portable(
    path: str | Path,
    *,
    python_version: str | None = None,
) -> list[Diagnostic]:
    return _lint_portable_path(Path(path), python_version, {}, ())


def _lint_portable_path(
    recipe_path: Path,
    python_version: str | None,
    cache: dict[Path, tuple[Diagnostic, ...]],
    stack: tuple[Path, ...],
) -> list[Diagnostic]:
    resolved = recipe_path.resolve()
    if resolved in stack:
        cycle = " -> ".join(str(path) for path in (*stack, resolved))
        return [Diagnostic(line=1, actual=f"recursive subrecipe dependency cycle: {cycle}")]
    if resolved in cache:
        return list(cache[resolved])
    try:
        source = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [Diagnostic(line=1, actual=f"cannot parse recipe: {exc}")]
    diagnostics = lint_portable_source(
        source,
        resolved,
        python_version=python_version,
        _dependency_cache=cache,
        _dependency_stack=(*stack, resolved),
    )
    cache[resolved] = tuple(diagnostics)
    return diagnostics


def lint_portable_source(
    source: str,
    recipe_path: str | Path,
    *,
    python_version: str | None = None,
    _dependency_cache: dict[Path, tuple[Diagnostic, ...]] | None = None,
    _dependency_stack: tuple[Path, ...] = (),
) -> list[Diagnostic]:
    path = Path(recipe_path)
    python_version = resolve_python_version(path, python_version)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SyntaxWarning)
            module = ast.parse(
                source,
                feature_version=python_feature_version(python_version),
            )
            compile(module, "<portable-recipe>", "exec")
    except SyntaxError as exc:
        return [Diagnostic(line=exc.lineno or 1, actual=f"cannot parse recipe: {exc}")]
    diagnostics = [
        Diagnostic(line=item.line, actual=item.actual)
        for item in lint_source_quality(source, path, python_version=python_version)
    ]
    syntax_warnings = [item for item in caught if issubclass(item.category, SyntaxWarning)]
    diagnostics.extend(
        Diagnostic(line=line, actual=f"SyntaxWarning: {message}")
        for line, message in sorted(
            {(item.lineno or 1, str(item.message)) for item in syntax_warnings}
        )
    )
    diagnostics.extend(
        artifact_metadata_diagnostics(
            module, path, python_version, _dependency_cache, _dependency_stack
        )
    )
    diagnostics.extend(_lint_gate(module, source, path, python_version=python_version))
    return sorted(diagnostics, key=lambda item: (item.line or 1, item.actual))


def _lint_resource_metadata(path: Path, module: ast.Module) -> list[Diagnostic]:
    resources_assignment = _literal_assignment(module, "_CALCIFIER_RESOURCES")
    if resources_assignment is None:
        return []
    try:
        resources = ast.literal_eval(resources_assignment.value)
    except (ValueError, SyntaxError):
        return [Diagnostic(line=resources_assignment.lineno, actual="invalid resource provenance")]
    skill_assignment = _literal_assignment(module, "_CALCIFIER_SKILL")
    try:
        skill = ast.literal_eval(skill_assignment.value) if skill_assignment is not None else None
    except (ValueError, SyntaxError):
        skill = None
    from .recipe import RecipeError, _resources_from_namespace, _skill_root_from_namespace

    namespace = {"_CALCIFIER_SKILL": skill}
    skill_root = _skill_root_from_namespace(namespace, path)
    try:
        _resources_from_namespace({"_CALCIFIER_RESOURCES": resources}, skill_root)
    except RecipeError as exc:
        return [Diagnostic(line=resources_assignment.lineno, actual=str(exc))]
    return []


def _lint_unrecorded_resources(path: Path, module: ast.Module) -> list[Diagnostic]:
    """Report packaged files added beside the skill after the recipe recorded its manifest.

    The runtime verifies only the files the manifest names, so a new file would
    otherwise go unnoticed until the skill is compiled again.
    """
    assignment = _literal_assignment(module, "_CALCIFIER_RESOURCES")
    recorded = _recorded_resource_paths(module)
    skill = _recorded_skill(module)
    if assignment is None or recorded is None or skill is None:
        return []
    if _pinned_text(path, *skill) is None:
        return []
    try:
        current = discover_resources((path.parent / skill[0]).resolve())
    except OSError:
        return []
    return [
        Diagnostic(
            line=assignment.lineno,
            actual=f"packaged resource {entry.path!r} is not in the recipe's manifest; "
            "compile the skill again",
        )
        for entry in current
        if entry.path not in recorded
    ]


def _lint_skill_integrity(path: Path, module: ast.Module) -> list[Diagnostic]:
    assignment = _literal_assignment(module, "_CALCIFIER_SKILL")
    if assignment is None:
        return []
    try:
        entry = ast.literal_eval(assignment.value)
    except (ValueError, SyntaxError):
        return [Diagnostic(line=assignment.lineno, actual="invalid skill provenance")]
    if (
        not isinstance(entry, dict)
        or not isinstance(entry.get("path"), str)
        or not isinstance(entry.get("sha256"), str)
    ):
        return [Diagnostic(line=assignment.lineno, actual="invalid skill provenance")]
    raw_path = Path(entry["path"])
    if raw_path.is_absolute():
        return [Diagnostic(line=assignment.lineno, actual="skill provenance path must be relative")]
    skill_path = (path.parent / raw_path).resolve()
    try:
        content = skill_path.read_bytes()
    except OSError:
        return [Diagnostic(line=assignment.lineno, actual="unresolved skill provenance path")]
    if sha256_hex(content) != entry["sha256"]:
        return [Diagnostic(line=assignment.lineno, actual="skill provenance hash mismatch")]
    return []


# Metadata from before the plan stage was removed; such a recipe must be compiled again.
_OBSOLETE_METADATA = ("_CALCIFIER_PLAN", "_CALCIFIER_PLAN_SHA256", "_CALCIFIER_CONTRACT")


def _lint_obsolete_metadata(module: ast.Module) -> list[Diagnostic]:
    return [
        Diagnostic(
            line=assignment.lineno,
            actual=f"obsolete {name} metadata from an earlier Calcifier; compile the skill again",
        )
        for name in _OBSOLETE_METADATA
        if (assignment := _literal_assignment(module, name)) is not None
    ]


def _lint_gate(
    module: ast.Module, source: str, path: Path, *, python_version: str | None
) -> list[Diagnostic]:
    """Run the compile gate on the recorded source and resource manifest."""
    from .diagnostics import CalcifyError
    from .gate import verify_source

    try:
        verify_source(
            source, python_version=python_version, resource_paths=_recorded_resource_paths(module)
        )
    except CalcifyError as exc:
        line = next((item.line for item in exc.diagnostics if item.line), 1)
        return [Diagnostic(line=line, actual=str(exc))]
    return []


def _recorded_resource_paths(module: ast.Module) -> frozenset[str] | None:
    """Return the recorded packaged-resource manifest, or ``None`` when unreadable."""
    assignment = _literal_assignment(module, "_CALCIFIER_RESOURCES")
    if assignment is None:
        return None
    try:
        resources = ast.literal_eval(assignment.value)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(resources, list):
        return None
    paths = [item.get("path") for item in resources if isinstance(item, dict)]
    if len(paths) != len(resources) or not all(isinstance(path, str) for path in paths):
        return None
    return frozenset(paths)


def _lint_subrecipes(
    path: Path,
    module: ast.Module,
    *,
    python_version: str | None,
    dependency_cache: dict[Path, tuple[Diagnostic, ...]],
    dependency_stack: tuple[Path, ...],
) -> list[Diagnostic]:
    assignment = _literal_assignment(module, "_CALCIFIER_SUBRECIPES")
    imports = import_bindings(module)
    calls = subrecipe_calls(module, imports)
    if not calls:
        return []
    if assignment is None:
        return [Diagnostic(line=1, actual="unresolved subrecipes; provenance is missing")]
    try:
        entries = ast.literal_eval(assignment.value)
    except (ValueError, SyntaxError):
        return [Diagnostic(line=assignment.lineno, actual="invalid subrecipe provenance")]
    if not isinstance(entries, list):
        return [Diagnostic(line=assignment.lineno, actual="subrecipe topology mismatch")]
    diagnostics, declared_children, valid_children = _pinned_subrecipes(
        path, assignment.lineno, entries
    )
    for child in sorted(valid_children):
        child_diagnostics = _lint_portable_path(
            child,
            python_version,
            dependency_cache,
            dependency_stack,
        )
        diagnostics.extend(
            Diagnostic(
                line=assignment.lineno,
                actual=f"invalid pinned child {child}: {diagnostic.render()}",
            )
            for diagnostic in child_diagnostics
        )
    call_diagnostics, called_children = called_subrecipes(path, module, calls, imports)
    diagnostics.extend(call_diagnostics)
    if called_children != declared_children:
        diagnostics.append(Diagnostic(line=assignment.lineno, actual="subrecipe topology mismatch"))
    return diagnostics


def _pinned_subrecipes(
    path: Path,
    lineno: int,
    entries: list[Any],  # untyped `ast.literal_eval` provenance, validated per entry
) -> tuple[list[Diagnostic], set[Path], set[Path]]:
    """Return entry diagnostics, every declared child, and children whose hash matches."""
    diagnostics: list[Diagnostic] = []
    declared_children: set[Path] = set()
    valid_children: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            diagnostics.append(Diagnostic(line=lineno, actual="invalid subrecipe provenance"))
            continue
        child = (path.parent / entry["path"]).resolve()
        declared_children.add(child)
        try:
            actual = sha256_hex(child.read_bytes())
        except OSError:
            diagnostics.append(Diagnostic(line=lineno, actual="unresolved subrecipe path"))
            continue
        if actual != entry.get("sha256"):
            diagnostics.append(Diagnostic(line=lineno, actual="subrecipe hash mismatch"))
            continue
        valid_children.add(child)
    return diagnostics, declared_children, valid_children


def called_child_recipes(path: Path, module: ast.Module) -> set[Path]:
    """Return every child recipe that the recipe at ``path`` statically names.

    Literal ``call_recipe`` targets and every value of a literal
    ``call_catalog_recipe`` mapping count, so a dynamic selector pins all its choices.
    """
    imports = import_bindings(module)
    return called_subrecipes(path, module, subrecipe_calls(module, imports), imports)[1]


def _recorded_skill(module: ast.Module) -> tuple[Path, str] | None:
    """Return the recorded SKILL.md path, relative to the recipe, and its sha256."""
    skill = _literal_assignment(module, "_CALCIFIER_SKILL")
    try:
        entry = ast.literal_eval(skill.value) if skill is not None else None
    except (ValueError, SyntaxError):
        return None
    if not (
        isinstance(entry, dict)
        and isinstance(entry.get("path"), str)
        and isinstance(entry.get("sha256"), str)
    ):
        return None
    return Path(entry["path"]), entry["sha256"]


def _pinned_text(recipe: Path, raw_path: Path, sha256: str) -> str | None:
    skill_path = (recipe.parent / raw_path).resolve()
    containment_root = (
        recipe.parent.parent if recipe.parent.name == ".calcifier" else recipe.parent
    ).resolve()
    if raw_path.is_absolute() or not skill_path.is_relative_to(containment_root):
        return None
    try:
        skill_bytes = skill_path.read_bytes()
    except OSError:
        return None
    if sha256_hex(skill_bytes) != sha256:
        return None
    with contextlib.suppress(UnicodeDecodeError):
        return skill_bytes.decode()
    return None


def _literal_assignment(module: ast.Module, name: str) -> ast.Assign | None:
    return next(
        (
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ),
        None,
    )


def artifact_metadata_diagnostics(
    module: ast.Module,
    path: Path,
    python_version: str | None,
    _dependency_cache: dict[Path, tuple[Diagnostic, ...]] | None = None,
    _dependency_stack: tuple[Path, ...] = (),
) -> list[Diagnostic]:
    """Check provenance and every child dependency independently of source validation."""
    diagnostics: list[Diagnostic] = []
    diagnostics.extend(_lint_resource_metadata(path, module))
    diagnostics.extend(_lint_unrecorded_resources(path, module))
    diagnostics.extend(_lint_skill_integrity(path, module))
    diagnostics.extend(_lint_obsolete_metadata(module))
    dependency_cache = _dependency_cache if _dependency_cache is not None else {}
    dependency_stack = _dependency_stack or (path.resolve(),)
    diagnostics.extend(
        _lint_subrecipes(
            path,
            module,
            python_version=python_version,
            dependency_cache=dependency_cache,
            dependency_stack=dependency_stack,
        )
    )
    return diagnostics
