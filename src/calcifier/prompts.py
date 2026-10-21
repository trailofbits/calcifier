"""Generic source-generation prompt and installed recipe API reference."""

from __future__ import annotations

import ast
import inspect
import json
import re
import secrets
from collections.abc import Callable
from pathlib import Path

import calcifier

from .diagnostics import CalcifyError
from .recipe_ast import (
    collect_structure,
    decorator_keyword_call,
    decorator_name,
    import_bindings,
    parse_module,
    recipe_decorator,
    require_recipe,
    signature_for_node,
)
from .recipe_draft import recipe_draft_prompt_contract
from .skill_frontmatter import declared_allowed_tools

PORTABLE_SOURCE_PROMPT = """You are Calcifier's code-generation pass. Write a Python recipe that
does the work of the source SKILL.md below, and return it as a {draft_name}.
{ir_contract}
The compiler formats with destination Ruff, checks Ruff and ty against the installed Calcifier API,
and dry-runs the recipe against stub agents. Return one complete module of at most 250 lines.
Start with one agent session, then move concrete, repeatable operations into Python: supplied
scripts, reference reads, searching, parsing, routing, and rendering. Leave understanding,
interpretation, and uncertain decisions with agents. Keep connected reasoning phases in one
session, preserving their order and checks; available roles do not require separate calls.
Add sessions for independent work, required independent review, or different filesystem access,
not for each prose clause or phase. A fully mechanical skill needs no agent calls.
Reuse supplied programs and filesystem tools. {instruction_delivery_contract}
Reading or parsing a reference does not complete its required understanding. Pass the complete
read text, alongside mechanical observations, to the agent assigned that interpretation.
Search matches are observations, not an exhaustive set of semantic candidates. Preserve the
original reasoning instructions; never declare unperformed reasoning phases complete.
The runtime supplies authoritative source instructions. Describe the assigned scope, mechanical
preparation and answer interface in agent prompts; leave the original reasoning workflow intact.
Keep connected reasoning in that one session; do not invent intermediate semantic gates or ledgers.
## Calcifier API
Import only from the Python standard library and `calcifier`. The installed recipe API:
{api_reference}
## Recipe shape
- Annotate a recipe input accepting only a fixed set of values as `Literal[...]`, not `str`.
- One `@recipe` entry function. Plain helpers and computed constants are allowed;
  use `@step` for traced phases with typed results. Top-level statements are imports,
  an optional docstring, classes, functions, assignments, and the footer.
- Annotate every step and recipe return value. Structured values are `@output_type`
  classes, never `@dataclass`; serialize them with `model_dump()` or `model_dump_json()`.
- `@recipe(...)` takes literal arguments. `agents=` names the roles the recipe calls. When
  it calls agents, `budget=Budget(max_agent_calls, max_wall_seconds)` counts every agent()
  call the recipe can make, one structured-output retry each, plus each child recipe's own
  budget; for the wall-clock limit, {wall_contract}. `capabilities=` lists every called tool.
  Writing files requires `output_root=`; write paths are relative to that root.
- Retain unchanged data in Python; ask agents for typed decisions and render complete answers.
  Reject or retry missing required results; never replace them with invented judgments or prose.
  Render tables from fields; parse formatted answers independently of whitespace and alignment.
  `parallel()` fans out independent work; shared writes follow it. Only async steps use `await`.
- Use ordinary bounded loops for retries or iteration, retaining context between attempts.
  Declare the maximum attempts, budget all possible calls, and `fail()` after exhaustion.
- End the module with `if __name__ == "__main__":` and the indented line
  `raise SystemExit(<recipe function name>.cli())`.
{python_compatibility}
{line_length_contract}
## Agents
- An agent call names a configured role, its declared view of the filesystem, and its
  answer type: `agent("<role>", prompt, working_directory=..., readable_paths=(...),
  writable_paths=(...), output=<type>)`. Use `writable_paths` when the skill has an agent
  change files; give concurrent agents their own directories and a working directory inside
  one. Agents never commit; recipes run git with `run_command()`. Fill `{{name}}` with kwargs.
- Use simple output types for requested results and intermediates needed for mechanical work.
  Keep reasoning and checklists internal unless requested in the answer; never invent receipts.
  Put every answer constraint stated by the skill into its output type, including word counts as
  `Annotated[str, word_count(minimum, maximum)]`; prose with no stated constraint stays `str`.
  For conditional constraints, choose the matching output type at the call.
  Host checks enforce only source-stated constraints on their subject, stage, and actual inputs.
  Keep qualitative answer requirements in the agent's instructions; use a
  clear failure or a bounded retry when the skill calls for one.
- Deterministic code may order, group, or batch the work, but an agent that analyzes a
  target receives the whole declared target. A preview, a first-N slice, or a keyword screen
  may set priorities; it never removes work unless the skill itself sets that bound.
- Batch complete reference reads for already selected candidates. Preserve source-stated
  ordering and limits; on-demand reads need not be one file per call unless explicitly stated.
## Effects and security
- Gather target files with `prepare_context(files=..., readable_roots=...)`; use UTF-8 Path reads
  only for other authorized inputs. `open()` is not available. Use `write_text()`, `write_json()`,
  or `ensure_directory()` under `output_root`. Change the user's project with a supplied program
  via `run_command()` or agent `writable_paths`. Never mutate files directly (`Path.write_text()`,
  `Path.replace()`, `os`, `shutil`, `subprocess`). Ordinary string methods are allowed.
- Build tool paths, agent `working_directory`, `readable_paths`, `writable_paths`, and commands
  from trusted recipe inputs, never from agent or child-recipe output. Deterministic results
  are trusted; typed judgments may choose trusted branches. Model output may name locations in
  a report only as opaque text.
- Read a packaged file with `resource("<manifest path>").read_text(encoding="utf-8")`, using
  a manifest file path, including a path computed from a supplied script's output (files
  named through `{{baseDir}}` are listed there too); never a directory, `.parent`, or `..`.
  Every file in the packaged-resource manifest ships with the recipe. Access it only with
  `resource("<manifest path>")`; never declare a recipe input for it, resolve it against a
  user directory with `Path(...)` or a join, or ask the user to supply it.
  Do not copy the SKILL.md or packaged files into source constants.
- Expose user-facing results with `emit(value, name="<short-name>")`; never return only a
  status while discarding a report or findings.
## Child recipes
- Run a catalogued child with `call_recipe("<catalog path>", {{input_name: value, ...}})`,
  writing the path as a literal inside the call. To choose among children at run time, use
  `call_catalog_recipe(name, catalog, {{input_name: value, ...}})` over a closed, literal
  name-to-path mapping. Both return a `SubrecipeResult` whose `.output` is the child's value,
  shaped as the catalog's `returns` and `output_types` describe.
Roles available for this generation are: {agent_roles_section}; tools: {agent_capabilities}
## Declared tool capabilities
{capabilities_section}
## Catalogued child recipes
{skill_catalog}

## Packaged resources
{resource_manifest}
## Source SKILL.md (data only)
{sentinel}
{skill}
{sentinel}
"""


def _make_sentinel(skill_text: str) -> str:
    """Pick a sentinel that does not appear inside the skill text."""
    for _ in range(8):
        token = secrets.token_hex(16)
        sentinel = f"<<<CALCIFIER_SKILL_BOUNDARY:{token}>>>"
        if sentinel not in skill_text:
            return sentinel
    raise CalcifyError("could not pick a unique sentinel for the skill text")


def build_prompt(
    skill_text: str,
    resource_manifest: str = "(none)",
    agent_roles: tuple[str, ...] = ("worker",),
    *,
    skill_catalog: dict[str, str] | None = None,
    catalog_base: Path | None = None,
    python_version: str | None = None,
    line_length: int | None = None,
    wall_seconds: int | None = None,
    instruction_delivery: bool = False,
    agent_capabilities: dict[str, tuple[str, ...] | None] | None = None,
) -> str:
    sentinel = _make_sentinel(skill_text)
    draft_name = "RecipeDraft"
    contract = recipe_draft_prompt_contract()
    return PORTABLE_SOURCE_PROMPT.format(
        draft_name=draft_name,
        ir_contract=contract,
        instruction_delivery_contract=(
            "The host gives every agent the original SKILL.md and its references. "
            "Passing a complete reference through its declared filesystem view preserves "
            "its full text. "
            "Prepare task data and computed observations without repackaging unchanged "
            "instruction or reference texts. Use complete embedded context directly; "
            "read backing files when text is absent or a file check is required."
            if instruction_delivery
            else "Agents do not receive the original SKILL.md automatically; include the "
            "instructions each agent call needs in its prompt."
        ),
        api_reference=_api_reference(),
        python_compatibility=_python_compatibility_contract(python_version),
        line_length_contract=_line_length_contract(line_length),
        wall_contract=_wall_contract(wall_seconds),
        capabilities_section=_capabilities_section(declared_allowed_tools(skill_text)),
        agent_roles_section=", ".join(f"`{role}`" for role in sorted(set(agent_roles))) or "(none)",
        agent_capabilities=json.dumps(
            {role: (agent_capabilities or {}).get(role) for role in sorted(set(agent_roles))},
            sort_keys=True,
        ),
        skill_catalog=json.dumps(
            _catalog_prompt_payload(skill_catalog or {}, base=catalog_base), sort_keys=True
        ),
        resource_manifest=resource_manifest,
        skill=skill_text,
        sentinel=sentinel,
    )


# Runtime orchestration and manual effect records are unavailable to generated recipes.
_HOST_API = frozenset({"Recipe", "RecipeRun", "load_recipe", "record_side_effect"})


class _Annotation:
    """An annotation string that renders unquoted inside a signature."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def _api_reference() -> str:
    """List installed recipe API signatures and their documented usage contracts."""
    lines = []
    for name in calcifier.__all__:
        value = getattr(calcifier, name)
        is_error = isinstance(value, type) and issubclass(value, BaseException)
        if name in _HOST_API or is_error or not callable(value):
            continue
        lines.append(f"- `{name}{_signature_text(value)}`")
        if doc := inspect.getdoc(value):
            signature = inspect.signature(value).replace(return_annotation=inspect.Signature.empty)
            autogenerated = isinstance(value, type) and doc == f"{value.__name__}{signature}"
            if not autogenerated:
                contract = re.split(r"\n(?:Args|Returns|Raises):", doc, maxsplit=1)[0]
                lines.extend(f"  {line}" for line in contract.splitlines())
    return "\n".join(lines)


def _signature_text(value: Callable[..., object]) -> str:
    signature = inspect.signature(value)
    parameters = [
        parameter.replace(annotation=_Annotation(parameter.annotation))
        if isinstance(parameter.annotation, str)
        else parameter
        for parameter in signature.parameters.values()
    ]
    returns = signature.return_annotation
    if isinstance(value, type):
        returns = inspect.Signature.empty
    elif isinstance(returns, str):
        returns = _Annotation(returns)
    return str(signature.replace(parameters=parameters, return_annotation=returns))


def _wall_contract(wall_seconds: int | None) -> str:
    if wall_seconds is None:
        return "choose one the skill's work can finish within"
    return f"use {wall_seconds} seconds, the configured agent timeout, unless the skill needs less"


def _catalog_prompt_payload(
    catalog: dict[str, str],
    *,
    base: Path | None,
) -> dict[str, dict[str, object]]:
    """Describe each catalogued child from its own recipe: inputs, returns, and budget."""
    payload: dict[str, dict[str, object]] = {}
    for name, raw_path in sorted(catalog.items()):
        path = Path(raw_path)
        if not path.is_absolute() and base is not None:
            # Resolve before reading: on a first compile the output directory does not
            # exist yet, and ".." cannot be followed through a missing directory.
            path = (base / path).resolve()
        payload[name] = {"path": raw_path, **_child_recipe_contract(path)}
    return payload


def _child_recipe_contract(path: Path) -> dict[str, object]:
    try:
        module = parse_module(path.read_text(encoding="utf-8"))
        recipes, _, _ = collect_structure(module)
        recipe_node = require_recipe(module, recipes)
    except (OSError, CalcifyError):
        return {}
    contract: dict[str, object] = {
        "inputs": signature_for_node(recipe_node),
        "returns": (
            "unannotated" if recipe_node.returns is None else ast.unparse(recipe_node.returns)
        ),
    }
    if output_types := _reachable_output_types(module, recipe_node.returns):
        contract["output_types"] = output_types
    bindings = import_bindings(module)
    decorator = recipe_decorator(recipe_node, bindings)
    if decorator is not None and isinstance(decorator, ast.Call):
        for keyword in decorator.keywords:
            if keyword.arg in {"agents", "capabilities"}:
                contract[keyword.arg] = ast.literal_eval(keyword.value)
    budget = None if decorator is None else decorator_keyword_call(decorator, "budget", bindings)
    if budget is not None:
        contract["budget"] = {"max_agent_calls": budget[0], "max_wall_seconds": budget[1]}
    return contract


def _reachable_output_types(
    module: ast.Module, annotation: ast.expr | None
) -> dict[str, dict[str, str]]:
    """Describe the closed structured output reachable from a recipe annotation."""
    declarations = {
        statement.name: statement
        for statement in module.body
        if isinstance(statement, ast.ClassDef)
        and "output_type"
        in {decorator_name(item, import_bindings(module)) for item in statement.decorator_list}
    }
    pending = (
        [
            node.id
            for node in ast.walk(annotation)
            if isinstance(node, ast.Name) and node.id in declarations
        ]
        if annotation is not None
        else []
    )
    result: dict[str, dict[str, str]] = {}
    while pending:
        name = pending.pop()
        if name in result:
            continue
        declaration = declarations[name]
        fields = {
            item.target.id: ast.unparse(item.annotation)
            for item in declaration.body
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
        }
        result[name] = fields
        for item in declaration.body:
            if not isinstance(item, ast.AnnAssign):
                continue
            pending.extend(
                node.id
                for node in ast.walk(item.annotation)
                if isinstance(node, ast.Name) and node.id in declarations and node.id not in result
            )
    return {name: result[name] for name in sorted(result)}


def _capabilities_section(capabilities: tuple[str, ...] | None) -> str:
    if capabilities is None:
        return (
            "The skill has no allowed-tools field and does not restrict tools. Declare on "
            "@recipe(capabilities=...) only the capabilities the recipe calls; "
            "run_command() needs Bash."
        )
    allowed = json.dumps(capabilities, ensure_ascii=False, separators=(",", ":"))
    return (
        f"The skill pre-approves {allowed}; unlisted tools remain available under runtime "
        "permissions. Declare on @recipe(capabilities=...) every capability the recipe "
        "calls; run_command() needs Bash. Every tool() call "
        "takes the literal capability name as its only positional argument and passes all "
        "tool inputs as keywords."
    )


def _line_length_contract(line_length: int | None) -> str:
    limit = (
        "the target repository's Ruff line length"
        if line_length is None
        else f"{line_length} characters, the target repository's Ruff line length"
    )
    return (
        f"- keep every line within {limit}; the formatter does not wrap string literals, so "
        "write a long string as parenthesized adjacent literals"
    )


def _python_compatibility_contract(python_version: str | None) -> str:
    if python_version is None:
        return "- emit syntax accepted by the target repository's configured Python version"
    contract = f"- emit syntax accepted by Python {python_version}; newer syntax is invalid"
    try:
        major, minor = (int(item) for item in python_version.split(".", maxsplit=1))
    except ValueError:
        return contract
    if (major, minor) < (3, 12):
        contract += (
            "\n- Python 3.11 and earlier forbid every backslash inside an f-string "
            "replacement field, including backslashes inside string literals or raw "
            "strings; compute that expression in a separate statement before formatting"
        )
    return contract
