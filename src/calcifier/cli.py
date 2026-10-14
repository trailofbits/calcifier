"""Command-line entry points for compiling and running recipes."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from . import __version__
from .agent_config import AgentConfigError, load_agent_config, override_agent_timeouts
from .agent_events import AgentActivity
from .agent_providers import host_provider_check
from .budget import Budget
from .cli_options import (
    add_agent_flags,
    add_output_flag,
    add_tool_flags,
    resolve_output_directory,
    shorten_budget,
)
from .exceptions import CalcifierError, ErrorFamily
from .filesystem_contract import AgentError
from .hashing import sha256_hex
from .metadata import parse_metadata_namespace, split_metadata
from .recipe import (
    RecipeError,
    RecipeSnapshot,
    coerce_cli_inputs,
    load_recipe_snapshot,
    read_recipe_snapshot,
)
from .registry import Registry, configured_registry
from .side_effects import DryRunMode
from .source_quality import PYTHON_VERSION_CHOICES
from .subprocesses import ProcessError, ProcessResult, run_process
from .tools import (
    ToolConfigError,
    ToolRegistry,
    configured_tool_registry,
    render_tool_preflight,
    tool_preflight,
)
from .trace import ToolPreflightReport
from .trace_export import write_trace
from .trust import (
    TrustError,
    TrustStatus,
    check_trust_bytes,
    record_trust_bytes,
    render_trusted_message,
    render_untrusted_message,
)

# -- shared helpers -----------------------------------------------------------


class _CliExitError(CalcifierError):
    """A command already reported its failure; main selects the exit code."""

    def __init__(self, family: ErrorFamily) -> None:
        super().__init__(family.value)
        self.family = family


def _parse_inputs(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw in items or []:
        if "=" not in raw:
            raise SystemExit(f"--input expects key=value (got {raw!r})")
        k, v = raw.split("=", 1)
        out[k] = v
    return out


def _run_host_agent_preflight(providers: tuple[str, ...]) -> None:
    for provider in providers:
        if shutil.which(provider) is None:
            raise SystemExit(f"required agent executable is not on PATH: {provider}")
        command, required_output = _host_provider_check(provider)
        _run_agent_feature_check(command, required_output, context="host")


def _host_provider_check(provider: str) -> tuple[list[str], tuple[str, ...]]:
    try:
        return host_provider_check(provider)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _run_agent_feature_check(
    command: list[str],
    required_output: tuple[str, ...],
    *,
    context: str,
) -> ProcessResult:
    try:
        completed = run_process(
            command,
            timeout_seconds=20,
            max_output_bytes=1024 * 1024,
        )
    except ProcessError as exc:
        raise SystemExit(f"agent preflight failed in {context}: {exc}") from exc
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr_tail.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        detail = (stderr or stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise SystemExit(f"agent preflight failed in {context} for {command!r}{suffix}")
    output = f"{stdout}\n{stderr}"
    missing = [flag for flag in required_output if flag not in output]
    if missing:
        raise SystemExit(
            f"agent preflight failed in {context}: {command[0]!r} does not support "
            f"{', '.join(missing)}"
        )
    return completed


def _registry_for_args(args: argparse.Namespace, providers: tuple[str, ...]) -> Registry:
    config_path = getattr(args, "agent_config", None)
    if config_path is not None:
        try:
            configuration = override_agent_timeouts(
                load_agent_config(config_path),
                getattr(args, "agent_timeout", []),
            )
            registry = configured_registry(
                configuration,
                providers,
            )
        except AgentConfigError as exc:
            raise SystemExit(f"agent config failed: {exc}") from exc
        _preflight_configured_agents(registry, providers)
        return registry
    if providers:
        raise SystemExit("agent-backed recipes require --agent-config (version 1)")
    material = json.dumps(
        {
            "providers": sorted(providers),
            "runtime": {"kind": "host"},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return Registry(configuration_hash=sha256_hex(material.encode()))


def _preflight_configured_agents(registry: Registry, roles: tuple[str, ...]) -> None:
    for role in roles:
        provider = registry.get(role)
        if not provider.is_available():
            suffix = f"; {provider.remediation}" if provider.remediation else ""
            raise SystemExit(
                f"required executable for agent role {role!r} is not on PATH: "
                f"{provider.target_name}{suffix}"
            )
        if provider.target_type == "builtin":
            _run_host_agent_preflight((provider.target_name,))


def _agent_runtime_for_args(
    args: argparse.Namespace,
    registry: Registry | None = None,
    providers: tuple[str, ...] = (),
) -> dict[str, Any]:
    if getattr(args, "agent_config", None) is not None:
        if registry is None:
            try:
                configuration = load_agent_config(args.agent_config)
                registry = configured_registry(
                    configuration,
                    providers,
                )
            except AgentConfigError as exc:
                raise ValueError(f"agent config failed: {exc}") from exc
        result = registry.runtime_summary(providers)
        result["overrides"] = {
            "agent_timeouts": sorted(getattr(args, "agent_timeout", [])),
            "max_agent_calls": getattr(args, "max_agent_calls", None),
            "max_wall_seconds": getattr(args, "max_wall_seconds", None),
        }
        return result
    return {"mode": "host"}


def _enforce_trust(
    recipe_path: Path,
    granting: bool,
    subrecipes: list[Path] | None = None,
) -> tuple[int, RecipeSnapshot | None]:
    if subrecipes and not granting:
        print("error: --subrecipe requires --trust", file=sys.stderr)
        return 2, None
    if not recipe_path.is_file():
        print(f"error: {recipe_path} is not a file", file=sys.stderr)
        return 2, None
    try:
        snapshot = read_recipe_snapshot(recipe_path)
        result = check_trust_bytes(snapshot.path, snapshot.data)
        if granting:
            dependencies = [
                *_declared_subrecipe_pins(snapshot),
                *_subrecipe_pins(subrecipes or []),
            ]
            result = record_trust_bytes(
                snapshot.path,
                snapshot.data,
                recorded_by=_recorded_by(),
                dependencies=dependencies,
            )
            print(render_trusted_message(result), file=sys.stderr)
            return 0, snapshot
    except (OSError, TrustError) as exc:
        print(f"trust check failed: {exc}", file=sys.stderr)
        return 2, None
    if result.status is TrustStatus.TRUSTED:
        return 0, snapshot
    print(render_untrusted_message(result), file=sys.stderr)
    return 2, None


def _subrecipe_pins(paths: list[Path]) -> list[tuple[Path, str]]:
    pins: list[tuple[Path, str]] = []
    for path in paths:
        snapshot = read_recipe_snapshot(path)
        pins.append((snapshot.path, snapshot.sha256))
    return pins


def _declared_subrecipe_pins(snapshot: RecipeSnapshot) -> list[tuple[Path, str]]:
    try:
        _, metadata = split_metadata(snapshot.data.decode("utf-8"))
        raw = parse_metadata_namespace(metadata or "").get("_CALCIFIER_SUBRECIPES", [])
    except (UnicodeDecodeError, ValueError, SyntaxError) as exc:
        raise TrustError(f"invalid subrecipe metadata: {exc}") from exc
    if not isinstance(raw, list):
        raise TrustError("_CALCIFIER_SUBRECIPES must be a list")
    pins: list[tuple[Path, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TrustError("invalid subrecipe metadata entry")
        entry = cast("dict[str, object]", item)
        path = entry.get("path")
        sha = entry.get("sha256")
        if not isinstance(path, str) or not isinstance(sha, str):
            raise TrustError("invalid subrecipe metadata entry")
        child = (snapshot.path.parent / path).resolve()
        actual = sha256_hex(child.read_bytes())
        if actual != sha:
            raise TrustError(f"subrecipe drift detected before trust: {path}")
        pins.append((child, sha))
    return pins


def _recorded_by() -> str:
    import getpass
    import socket

    return f"{getpass.getuser()}@{socket.gethostname()}"


def _add_dry_run_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        nargs="?",
        const=DryRunMode.SOFT.value,
        default=DryRunMode.OFF.value,
        choices=[mode.value for mode in DryRunMode],
        help="declare side effects without completing them (soft) or block them (hard)",
    )


def _add_python_version_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--python-version",
        choices=PYTHON_VERSION_CHOICES,
        help=(
            "minimum generated-recipe Python version; by default Ruff project "
            "configuration for the output path applies"
        ),
    )


def _add_trust_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trust",
        action="store_true",
        help="record trust for this recipe's current sha256 before running",
    )
    parser.add_argument(
        "--subrecipe",
        action="append",
        type=Path,
        default=[],
        metavar="PATH",
        help="pin a declared child recipe when granting trust (repeatable)",
    )


def _add_trace_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trace-out", type=Path, help="write a JSON trace export to PATH")


def _prepare_tool_preflight(
    recipe_obj: Any, skip: bool, config_path: Path | None
) -> tuple[ToolRegistry, ToolPreflightReport]:
    registry = configured_tool_registry(recipe_obj._provider_capabilities, config_path)
    report = tool_preflight(registry, recipe_obj._provider_capabilities)
    if not report.ok and not skip:
        print(render_tool_preflight(report), file=sys.stderr)
    return registry, report


def _prepare_tools_or_report(
    recipe_obj: Any,
    args: argparse.Namespace,
) -> tuple[ToolRegistry, ToolPreflightReport] | None:
    try:
        return _prepare_tool_preflight(
            recipe_obj,
            args.skip_tool_preflight,
            args.tool_config,
        )
    except ToolConfigError as exc:
        print(f"tool config failed: {exc}", file=sys.stderr)
        return None


# -- subcommand: calcify ------------------------------------------------------


def _positive_compilation_timeout(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("compilation timeout must be a positive integer") from exc
    if seconds <= 0:
        raise argparse.ArgumentTypeError("compilation timeout must be a positive integer")
    return seconds


def _add_compilation_timeout_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compilation-timeout",
        type=_positive_compilation_timeout,
        help="total optimization seconds per recipe, including review (default: 3600)",
    )


def _add_compiler_call_limit_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-compiler-calls",
        type=int,
        choices=range(1, 9),
        help=("total calls per recipe, including repairs (default: 5 with a baseline, 8 without)"),
    )


def _add_calcify(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("calcify", help="compile a SKILL.md into .calcifier/recipe.py")
    p.add_argument("skill", type=Path, help="path to a SKILL.md file")
    p.add_argument("--compiler", default="compiler", help="semantic role for compilation")
    p.add_argument("--output", "-o", type=Path, help="override output path")
    p.add_argument(
        "--timeout",
        type=int,
        help="seconds per compiler call (default: role timeout_seconds; recommended: 3600)",
    )
    p.add_argument("--skill-catalog", type=Path, help="versioned named-skill recipe catalog")
    p.add_argument(
        "--require-compiled",
        action="store_true",
        help="exit nonzero when compilation retains a baseline; preserve the baseline and trace",
    )
    _add_compilation_timeout_flag(p)
    _add_compiler_call_limit_flag(p)
    p.add_argument(
        "--compiler-trace",
        type=Path,
        help="persist structured per-attempt compiler diagnostics and telemetry",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="report compiler stage progress to stderr while model calls run",
    )
    _add_python_version_flag(p)
    add_agent_flags(p, wall_budget=False)
    p.set_defaults(handler=_cmd_calcify)


def _require_runtime_roles(registry: Registry, compiler: str) -> None:
    from .calcify import CalcifyError

    if not (set(registry.known()) - {compiler}):
        raise CalcifyError(
            "configure at least one runtime semantic role in --agent-config "
            f"besides the compiler role {compiler!r}; no compiler calls made"
        )


def _cmd_calcify(args: argparse.Namespace) -> int:
    # Compiler modules load only for compiler commands; `exec`/`run` stay lean.
    from .calcify import (
        CalcifyError,
        calcify_file,
        load_skill_catalog,
        render_compiler_activity,
        render_dry_run,
        render_summary,
        write_compiler_trace,
    )

    if not args.skill.is_file():
        print(f"error: {args.skill} is not a file", file=sys.stderr)
        raise _CliExitError(ErrorFamily.SETUP)
    reg = _registry_for_args(args, (args.compiler,))

    def report_event(event: dict[str, object]) -> None:
        if args.verbose:
            print(
                "compiler progress: " + json.dumps(event, sort_keys=True, separators=(",", ":")),
                file=sys.stderr,
                flush=True,
            )
        if args.compiler_trace is not None:
            write_compiler_trace(args.compiler_trace, activity)

    activity = AgentActivity(event_callback=report_event)
    try:
        _require_runtime_roles(reg, args.compiler)
        catalog = {} if args.skill_catalog is None else load_skill_catalog(args.skill_catalog)
        target = calcify_file(
            args.skill,
            reg,
            compiler=args.compiler,
            output=args.output,
            timeout=args.timeout,
            compilation_timeout=args.compilation_timeout,
            max_compiler_calls=args.max_compiler_calls,
            skill_catalog=catalog,
            activity=activity,
            python_version=args.python_version,
        )
    except (CalcifyError, AgentError, RecipeError, OSError) as exc:
        if activity.outcome != "failed":
            activity.outcome = "failed"
            activity.record_event("compiler", "failed", detail=str(exc))
        print(f"calcify failed: {exc}", file=sys.stderr)
        print(render_compiler_activity(activity), file=sys.stderr)
        raise _CliExitError(ErrorFamily.COMPILATION) from exc
    finally:
        if args.compiler_trace is not None:
            write_compiler_trace(args.compiler_trace, activity)
    print(f"wrote {target}")
    print(render_compiler_activity(activity))
    print(render_dry_run(activity))
    print(render_summary(target))
    _check_compilation_acceptance(args.require_compiled, activity)
    print(
        "\nReview the recipe before granting trust. Once you've read it,"
        f"\nrun:  calcifier exec {target} --trust"
    )
    return 0


def _check_compilation_acceptance(required: bool, activity: AgentActivity) -> None:
    from .compiler_activity import compiler_activity_payload

    if required and compiler_activity_payload(activity)["outcome"] != "compiled":
        print("calcify failed acceptance: a compiled recipe was required", file=sys.stderr)
        raise _CliExitError(ErrorFamily.COMPILATION)


# -- recipe execution --------------------------------------------------------


def _add_run_command(
    sub: argparse._SubParsersAction,
    name: str,
    help_text: str,
    handler: Callable[[argparse.Namespace], int],
) -> None:
    parser = sub.add_parser(name, help=help_text)
    parser.add_argument("recipe", type=Path, help="path to a recipe.py")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--input", action="append", help="input as key=value (repeatable)")
    _add_trust_flag(parser)
    _add_trace_flag(parser)
    add_agent_flags(parser)
    add_tool_flags(parser)
    _add_dry_run_flag(parser)
    add_output_flag(parser)
    parser.set_defaults(handler=handler)


def _cmd_exec(args: argparse.Namespace) -> int:
    return _run_recipe(args, verbose=False)


def _write_trace_if_requested(run: Any, path: Path | None) -> None:
    if path is None:
        return
    try:
        write_trace(run, path)
    except OSError as exc:
        print(f"trace export failed: {exc}", file=sys.stderr)


def _cmd_debug(args: argparse.Namespace) -> int:
    return _run_recipe(args, verbose=True)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


def _run_recipe(args: argparse.Namespace, *, verbose: bool) -> int:  # noqa: C901
    _configure_logging(verbose)
    rc, snapshot = _enforce_trust(
        args.recipe,
        granting=getattr(args, "trust", False),
        subrecipes=getattr(args, "subrecipe", []),
    )
    if rc != 0:
        raise _CliExitError(ErrorFamily.SETUP)
    if snapshot is None:
        raise _CliExitError(ErrorFamily.SETUP)
    try:
        recipe_obj = load_recipe_snapshot(snapshot)
    except RecipeError as exc:
        print(f"recipe load failed: {exc}", file=sys.stderr)
        raise _CliExitError(ErrorFamily.SETUP) from exc
    try:
        output_directory = resolve_output_directory(recipe_obj.output_root, args.output_directory)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise _CliExitError(ErrorFamily.SETUP) from exc
    from .graph import inspect_graph

    graph = inspect_graph(args.recipe, entry_bytes=snapshot.data) if recipe_obj.composed else None
    roles = graph.agents if graph else recipe_obj.agent_names
    capabilities = graph.capabilities if graph else recipe_obj.capabilities
    reg = _registry_for_args(args, roles)
    prepared_tools = _prepare_tools_or_report(
        replace(
            recipe_obj,
            capabilities=capabilities,
            _tool_capabilities=(
                graph.provider_capabilities if graph else recipe_obj._provider_capabilities
            ),
        ),
        args,
    )
    if prepared_tools is None:
        raise _CliExitError(ErrorFamily.SETUP)
    tool_registry, tool_preflight = prepared_tools
    if not tool_preflight.ok and not args.skip_tool_preflight:
        raise _CliExitError(ErrorFamily.SETUP)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "output_directory": str(output_directory) if output_directory else None,
                    "agents": list(roles),
                    "capabilities": list(capabilities),
                    "unresolved": list(graph.unresolved) if graph else [],
                },
                indent=2,
            )
        )
        return 0
    raw_inputs = _parse_inputs(args.input)
    try:
        inputs = coerce_cli_inputs(recipe_obj, raw_inputs)
    except (ValueError, TypeError) as exc:
        print(f"input coercion failed: {exc}", file=sys.stderr)
        raise _CliExitError(ErrorFamily.SETUP) from exc
    run = recipe_obj.run(
        inputs,
        registry=reg,
        tool_registry=tool_registry,
        dry_run=DryRunMode(args.dry_run),
        budget=_budget_for_args(recipe_obj, args),
        trace_path=args.trace_out,
        output_directory=output_directory,
    )
    run.tool_preflight = tool_preflight
    run.agent_runtime = _agent_runtime_for_args(args, reg, roles)
    print(run.render())
    _write_trace_if_requested(run, args.trace_out)
    if not run.ok:
        raise _CliExitError(ErrorFamily.RUNTIME)
    return 0


def _budget_for_args(recipe_obj: Any, args: argparse.Namespace) -> Budget | None:
    try:
        return shorten_budget(
            recipe_obj.budget,
            getattr(args, "max_wall_seconds", None),
            getattr(args, "max_agent_calls", None),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


# -- subcommand: update -------------------------------------------------------


def _add_update(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "update",
        help=(
            "compile the skill again and print a diff against the existing recipe; like "
            "calcify, runs the candidate once on the host, and writes nothing"
        ),
    )
    p.add_argument("skill", type=Path)
    p.add_argument(
        "--recipe",
        type=Path,
        help="path to the existing recipe (defaults to skill/.calcifier/recipe.py)",
    )
    p.add_argument("--compiler", default="compiler")
    p.add_argument(
        "--timeout",
        type=int,
        help="seconds per compiler call (default: role timeout_seconds; recommended: 3600)",
    )
    p.add_argument("--skill-catalog", type=Path)
    _add_compilation_timeout_flag(p)
    _add_compiler_call_limit_flag(p)
    _add_python_version_flag(p)
    add_agent_flags(p, wall_budget=False)
    p.set_defaults(handler=_cmd_update)


def _add_lint(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("lint", help="statically check a calcified recipe")
    p.add_argument("recipe", type=Path)
    p.add_argument("--portable", action="store_true", required=True)
    _add_python_version_flag(p)
    p.set_defaults(handler=_cmd_lint)


def _cmd_lint(args: argparse.Namespace) -> int:
    from .portable import lint_portable

    diagnostics = lint_portable(args.recipe, python_version=args.python_version)
    if not diagnostics:
        print(f"portable: OK ({args.recipe})")
        return 0
    for diagnostic in diagnostics:
        print(f"{args.recipe}:{diagnostic.render()}", file=sys.stderr)
    raise _CliExitError(ErrorFamily.VERIFICATION)


def _add_profile(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("profile", help="summarize a recipe trace without running the recipe")
    p.add_argument("trace", type=Path, help="v1 JSON trace written by --trace-out")
    p.set_defaults(handler=_cmd_profile)


def _cmd_profile(args: argparse.Namespace) -> int:
    from jsonschema import ValidationError

    from .trace_profile import render_trace_profile

    try:
        payload = json.loads(args.trace.read_text(encoding="utf-8"))
        print(render_trace_profile(payload))
    except (OSError, ValueError, TypeError, ValidationError) as exc:
        print(f"trace profile failed: {exc}", file=sys.stderr)
        raise _CliExitError(ErrorFamily.VERIFICATION) from exc
    return 0


def _cmd_update(args: argparse.Namespace) -> int:
    from .calcify import CalcifyError, load_skill_catalog, render_compiler_activity, render_dry_run
    from .update import update_skill

    recipe_path = args.recipe or args.skill.parent / ".calcifier" / "recipe.py"
    reg = _registry_for_args(args, (args.compiler,))
    activity = AgentActivity()
    try:
        _require_runtime_roles(reg, args.compiler)
        diff = update_skill(
            args.skill,
            existing_recipe=recipe_path,
            registry=reg,
            compiler=args.compiler,
            timeout=args.timeout,
            compilation_timeout=args.compilation_timeout,
            max_compiler_calls=args.max_compiler_calls,
            skill_catalog=(
                {} if args.skill_catalog is None else load_skill_catalog(args.skill_catalog)
            ),
            activity=activity,
            python_version=args.python_version,
        )
    except (CalcifyError, AgentError, RecipeError, FileNotFoundError) as exc:
        print(f"update failed: {exc}", file=sys.stderr)
        print(render_compiler_activity(activity), file=sys.stderr)
        raise _CliExitError(ErrorFamily.COMPILATION) from exc
    # The diff alone goes to stdout, so it can be saved or applied as it is.
    if diff:
        print(diff, end="")
    else:
        print(f"no changes: the prepared recipe matches {recipe_path}")
    print(render_compiler_activity(activity), file=sys.stderr)
    print(render_dry_run(activity), file=sys.stderr)
    return 0


# -- top-level ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="calcifier", description=__doc__)
    p.add_argument("--version", action="version", version=f"calcifier {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    _add_calcify(sub)
    _add_run_command(sub, "exec", "run a calcified recipe", _cmd_exec)
    _add_run_command(sub, "debug", "run a recipe with verbose diagnostics", _cmd_debug)
    _add_update(sub)
    _add_lint(sub)
    _add_profile(sub)
    _add_graph_commands(sub)
    return p


def _add_graph_commands(sub: argparse._SubParsersAction) -> None:
    build = sub.add_parser("build-graph", help="compile a graph's skills in dependency order")
    build.add_argument("spec", type=Path)
    _add_compilation_timeout_flag(build)
    _add_compiler_call_limit_flag(build)
    add_agent_flags(build, call_budget=False)
    build.set_defaults(handler=_cmd_build_graph)


def _cmd_build_graph(args: argparse.Namespace) -> int:
    from .graph_build import build_graph

    print(
        build_graph(
            args.spec,
            _registry_for_args(args, ("compiler",)),
            compilation_timeout=args.compilation_timeout,
            max_compiler_calls=args.max_compiler_calls,
            runtime_wall_seconds=args.max_wall_seconds,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.getLogger().setLevel(logging.WARNING)
    handler = args.handler
    exit_codes = {
        ErrorFamily.SETUP: 2,
        ErrorFamily.RUNTIME: 1,
        ErrorFamily.COMPILATION: 1,
        ErrorFamily.VERIFICATION: 1,
    }
    try:
        return handler(args)
    except _CliExitError as exc:
        return exit_codes[exc.family]
    except CalcifierError as exc:
        label = "recipe setup" if isinstance(exc, RecipeError) else exc.family.value
        print(f"{label} failed: {exc}", file=sys.stderr)
        return exit_codes[exc.family]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
