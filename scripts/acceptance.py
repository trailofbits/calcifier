"""Measure offline baseline reliability or fixture-driven live recipe acceptance.

Use the repository's Python, including on Windows. Offline mode samples ``--pool`` skills
with scripted agents and keeps produced/lint/exec counts. Live mode requires
``--agent-config`` and ``--fixtures DIR``: each sorted fixture is freshly compiled
``--repeat N`` times, bound by the worker role, executed in a workspace copy, and checked
by its command. Acceptance requires compiled output within LIVE_LIMITS, successful real
execution, and a passing fixture check. Lint and dry-run status are diagnostics only.
Logs, bound inputs, run-result.json, results.jsonl and SUMMARY.json stay in --output.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calcifier.agent_config import load_agent_config
from calcifier.exceptions import CalcifierError
from calcifier.filesystem_contract import FilesystemContract
from calcifier.recipe_ast import import_resolver
from calcifier.registry import configured_registry
from calcifier.validation import is_positive_number

EXCLUDED = {
    ".git",
    ".venv",
    ".calcifier",
    ".calcifier-output",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".hypothesis",
    ".calcifier-runtime",
}
# Live acceptance: the compiler's own recipe is kept within these limits, and enough skills pass.
LIVE_LIMITS = {"model_calls": 3, "wall_seconds": 300}
REQUIRED_SHARE = 7 / 8
STUB = """import json
import os
import sys

if os.environ.get("CALCIFIER_AGENT_PROTOCOL") != "1":
    raise SystemExit("expected agent protocol 2")
sys.stdin.read()
output = "deliberately invalid compiler reply"
if sys.argv[1] == "worker":
    output = json.dumps({"answer": "Offline stub response.", "files": [], "blocked_reason": ""})
print(json.dumps({"version": 1, "output": output, "usage": {}}))
"""


def sample_skills(pools: list[Path], count: int, seed: int) -> list[Path]:
    skills: set[Path] = set()
    for pool in pools:
        if not pool.exists():
            raise ValueError(f"pool does not exist: {pool}")
        candidates = [pool] if pool.is_file() else pool.rglob("SKILL.md")
        for candidate in candidates:
            relative = candidate.relative_to(pool) if pool.is_dir() else Path(candidate.name)
            if (
                candidate.name == "SKILL.md"
                and candidate.is_file()
                and not EXCLUDED.intersection(relative.parts)
            ):
                skills.add(candidate.resolve())
    if not 1 <= count <= len(skills):
        raise ValueError(f"count must be between 1 and pool size {len(skills)}")
    return random.Random(seed).sample(sorted(skills, key=str), count)  # noqa: S311


def package_root(skill: Path) -> Path:
    for parent in skill.parents:
        if parent.name == "skills" and parent.parent.name not in {".codex", ".claude", ".agents"}:
            return parent.parent
    return skill.parent


def run_command(argv: list[str], directory: Path, name: str, timeout: int) -> bool:
    environment = dict(os.environ, CALCIFIER_TRUST_STORE=str(directory / "trust.json"))
    with (
        (directory / f"{name}.stdout").open("wb") as stdout,
        (directory / f"{name}.stderr").open("wb") as stderr,
    ):
        try:
            return (
                subprocess.run(
                    argv,
                    cwd=directory,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout,
                    check=False,
                ).returncode
                == 0
            )
        except (subprocess.TimeoutExpired, OSError) as error:
            stderr.write(str(error).encode("utf-8"))
            return False


def stage_package(skill: Path, directory: Path) -> Path:
    """Copy the skill's package into a new case directory and return the copied SKILL.md."""
    directory.mkdir()
    source = package_root(skill)
    shutil.copytree(source, directory / "package", ignore=shutil.ignore_patterns(*EXCLUDED))
    return directory / "package" / skill.relative_to(source)


def run_case(skill: Path, directory: Path, config: Path, timeout: int) -> tuple[bool, bool, bool]:
    copied_skill = stage_package(skill, directory)
    target = directory / "target"
    target.mkdir()
    (target / "input.txt").write_text("Offline acceptance target.\n", encoding="utf-8")
    recipe = directory / "recipe.py"
    cli = [sys.executable, "-m", "calcifier.cli"]
    produced = (
        run_command(
            [
                *cli,
                "calcify",
                str(copied_skill),
                "--output",
                str(recipe),
                "--agent-config",
                str(config),
                "--timeout",
                "10",
            ],
            directory,
            "calcify",
            timeout,
        )
        and recipe.is_file()
    )
    if not produced:
        return False, False, False
    linted = run_command([*cli, "lint", str(recipe), "--portable"], directory, "lint", timeout)
    executed = run_command(
        [
            *cli,
            "exec",
            str(recipe),
            "--trust",
            "--agent-config",
            str(config),
            "--input",
            "request=Perform the offline stub check.",
            "--input",
            f"target={target}",
            "--max-wall-seconds",
            "30",
        ],
        directory,
        "exec",
        timeout,
    )
    return True, linted, executed


def recipe_inputs(recipe: Path) -> list[str]:
    """Name the inputs of the recipe entry point without loading its code."""
    try:
        return [parameter["name"] for parameter in recipe_signature(recipe)["inputs"]]
    except (OSError, SyntaxError, ValueError):
        return []


def recipe_signature(recipe: Path) -> dict[str, Any]:
    """Read parameter names, annotation/default source text, and the entry docstring."""
    source = recipe.read_bytes().decode("utf-8-sig")
    module = ast.parse(source)
    resolve = import_resolver(module)
    for node in module.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and any(
            resolve(decorator.func if isinstance(decorator, ast.Call) else decorator)
            == "calcifier.recipe"
            for decorator in node.decorator_list
        ):
            positional = [*node.args.posonlyargs, *node.args.args]
            defaults = [None] * (len(positional) - len(node.args.defaults)) + node.args.defaults
            return {
                "name": node.name,
                "docstring": ast.get_docstring(node),
                "inputs": [
                    {
                        "name": argument.arg,
                        "annotation": (
                            ast.get_source_segment(source, argument.annotation)
                            if argument.annotation is not None
                            else None
                        ),
                        "default": (
                            ast.get_source_segment(source, default) if default is not None else None
                        ),
                        "required": default is None,
                    }
                    for argument, default in zip(
                        [*positional, *node.args.kwonlyargs],
                        [*defaults, *node.args.kw_defaults],
                        strict=True,
                    )
                ],
            }
    raise ValueError("recipe entry point not found")


def run_live_case(skill: Path, directory: Path, config: Path, timeout: int) -> dict[str, object]:
    """Compile one skill with the configured roles and collect what acceptance judges."""
    copied_skill = stage_package(skill, directory)
    recipe, trace = directory / "recipe.py", directory / "compiler-trace.json"
    cli = [sys.executable, "-m", "calcifier.cli"]
    run_command(
        [*cli, "calcify", str(copied_skill), "--output", str(recipe)]
        + ["--agent-config", str(config), "--compiler-trace", str(trace)],
        directory,
        "calcify",
        timeout,
    )
    try:
        payload = json.loads(trace.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    dry_run = next(
        (e["status"] for e in reversed(payload.get("events", [])) if e["phase"] == "dry_run"),
        "not run",
    )
    linted = recipe.is_file() and run_command(
        [*cli, "lint", str(recipe), "--portable"], directory, "lint", timeout
    )
    return {
        "skill": str(skill),
        "outcome": payload.get("outcome", "failed"),
        "model_calls": payload.get("model_calls"),
        "wall_seconds": payload.get("wall_seconds"),
        "lint": linted,
        "dry_run": dry_run,
        "inputs": recipe_inputs(recipe),
    }


def live_accepted(result: dict[str, object]) -> bool:
    calls, seconds = result["model_calls"], result["wall_seconds"]
    return (
        result["outcome"] == "compiled"
        and isinstance(calls, int)
        and calls <= LIVE_LIMITS["model_calls"]
        and isinstance(seconds, int | float)
        and seconds <= LIVE_LIMITS["wall_seconds"]
        and result["exec_ok"] is True
        and result["check_exit"] == 0
    )


@dataclass(frozen=True)
class Fixture:
    directory: Path
    skill: Path
    request: str
    check: list[str]
    setup: list[str] | None
    path_prepend: list[str]
    exec_timeout_seconds: int | float


def fixture_strings(value: object, name: str, *, command: bool = False) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(part, str) for part in value):
        raise ValueError(f"{name} must be an array of strings")
    if command and (not value or not value[0]):
        raise ValueError(f"{name} must contain a command")
    return value


def load_fixture(directory: Path) -> Fixture:
    payload = json.loads((directory / "fixture.json").read_bytes())
    required = {"skill", "request", "check"}
    allowed = required | {"setup", "path_prepend", "exec_timeout_seconds"}
    if not isinstance(payload, dict) or required - payload.keys():
        raise ValueError(f"fixture {directory.name}: skill, request and check are required")
    if unknown := payload.keys() - allowed:
        raise ValueError(f"fixture {directory.name}: unknown keys {sorted(unknown)}")
    skill, request = payload["skill"], payload["request"]
    if not isinstance(skill, str) or not Path(skill).is_absolute() or not Path(skill).is_file():
        raise ValueError(f"fixture {directory.name}: skill must be an absolute file path")
    if not isinstance(request, str):
        raise ValueError(f"fixture {directory.name}: request must be a string")
    seconds = payload.get("exec_timeout_seconds", 900)
    if not is_positive_number(seconds, finite=True):
        raise ValueError(
            f"fixture {directory.name}: exec_timeout_seconds must be positive and finite"
        )
    return Fixture(
        directory=directory,
        skill=Path(skill).resolve(),
        request=request,
        check=fixture_strings(payload["check"], "check", command=True),
        setup=(
            fixture_strings(payload["setup"], "setup", command=True) if "setup" in payload else None
        ),
        path_prepend=fixture_strings(payload.get("path_prepend", []), "path_prepend"),
        exec_timeout_seconds=seconds,
    )


def expand_fixture(parts: list[str], fixture: Fixture, case: Path) -> list[str]:
    replacements = {
        "{fixture}": str(fixture.directory),
        "{workspace}": str(case / "workspace"),
        "{case}": str(case),
    }
    expanded = []
    for part in parts:
        for placeholder, value in replacements.items():
            part = part.replace(placeholder, value)
        expanded.append(part)
    return expanded


def fixture_command(
    argv: list[str], case: Path, name: str, timeout: int | float, environment: dict[str, str]
) -> int | None:
    """Run an argv command in the workspace, preserving its exit code and both streams."""
    with (
        (case / f"{name}.stdout").open("wb") as stdout,
        (case / f"{name}.stderr").open("wb") as stderr,
    ):
        try:
            return subprocess.run(
                argv,
                cwd=case / "workspace",
                env=environment,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            ).returncode
        except (subprocess.TimeoutExpired, OSError) as error:
            stderr.write(str(error).encode("utf-8"))
            return None


def bind_inputs(
    fixture: Fixture, case: Path, config: Path, signature: dict[str, Any]
) -> dict[str, Any]:
    """Ask the configured worker for a structured input mapping using the existing transport."""
    prompt = (
        "Bind the user's request to the recipe entry point below. Return a JSON object "
        "mapping input names to JSON values. Include every required input; optional inputs "
        "may be omitted to use their defaults. Use only declared input names. Resolve workspace "
        "paths against the absolute workspace path.\n"
        + json.dumps(
            {
                "request": fixture.request,
                "workspace": str(case / "workspace"),
                "entry_point": signature,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    (case / "binding-prompt.txt").write_text(prompt, encoding="utf-8")
    worker = configured_registry(load_agent_config(config), ("worker",)).get("worker")
    return worker.structured(
        prompt,
        annotation=dict[str, Any],
        filesystem=FilesystemContract.build(case / "workspace", (case,)),
    )


def validate_binding(signature: dict[str, Any], bound: dict[str, Any]) -> None:
    names = {parameter["name"] for parameter in signature["inputs"]}
    required = {parameter["name"] for parameter in signature["inputs"] if parameter["required"]}
    if unknown := bound.keys() - names:
        raise ValueError(f"unknown inputs: {sorted(unknown)}")
    if missing := required - bound.keys():
        raise ValueError(f"missing required inputs: {sorted(missing)}")


def run_result(case: Path, exit_code: int | None) -> dict[str, Any]:
    """Extract the return value and named emits from the execution trace, including failures."""
    try:
        payload = json.loads((case / "run-trace.json").read_bytes())
        result = {
            "ok": payload["ok"] is True,
            "error": payload["error"],
            "output": payload["output"],
            "artifacts": {artifact["name"]: artifact["value"] for artifact in payload["artifacts"]},
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {
            "ok": False,
            "error": f"exec exit={exit_code}; could not read run trace: {error}",
            "output": None,
            "artifacts": {},
        }
    (case / "run-result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def execute_fixture(
    fixture: Fixture, case: Path, config: Path, timeout: int, result: dict[str, Any]
) -> None:
    """Populate setup, binding, execution and check results, stopping at the first failure."""
    result["failed_stage"] = "setup"
    workspace = case / "workspace"
    source = fixture.directory / "workspace"
    if source.exists():
        shutil.copytree(source, workspace)
    else:
        workspace.mkdir()
    environment = dict(os.environ, CALCIFIER_TRUST_STORE=str(case / "trust.json"))
    environment["PATH"] = os.pathsep.join(
        [*expand_fixture(fixture.path_prepend, fixture, case), environment.get("PATH", "")]
    )
    if (
        fixture.setup is not None
        and fixture_command(
            expand_fixture(fixture.setup, fixture, case), case, "setup", timeout, environment
        )
        != 0
    ):
        return
    result["failed_stage"] = "binding"
    signature = recipe_signature(case / "recipe.py")
    bound = bind_inputs(fixture, case, config, signature)
    result["bound_inputs"] = bound
    validate_binding(signature, bound)
    result["failed_stage"] = "exec"
    command = [
        sys.executable,
        "-m",
        "calcifier.cli",
        "exec",
        str(case / "recipe.py"),
        "--trust",
        "--agent-config",
        str(config),
        "--max-wall-seconds",
        str(fixture.exec_timeout_seconds),
        "--trace-out",
        str(case / "run-trace.json"),
    ]
    for name, value in bound.items():
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        command.extend(["--input", f"{name}={raw}"])
    started = time.monotonic()
    exit_code = fixture_command(command, case, "exec", fixture.exec_timeout_seconds, environment)
    result["exec_seconds"] = time.monotonic() - started
    execution = run_result(case, exit_code)
    result["exec_ok"] = exit_code == 0 and execution["ok"]
    if not result["exec_ok"]:
        result["error"] = execution["error"] or f"exec exit={exit_code}"
        return
    result["failed_stage"] = "check"
    environment.update(
        CALCIFIER_RESULT=str(case / "run-result.json"),
        CALCIFIER_WORKSPACE=str(workspace),
        CALCIFIER_FIXTURE=str(fixture.directory),
    )
    result["check_exit"] = fixture_command(
        expand_fixture(fixture.check, fixture, case), case, "check", timeout, environment
    )
    if result["check_exit"] == 0:
        result["failed_stage"] = None if live_accepted(result) else "compile"


def run_fixture_case(
    fixture: Fixture, case: Path, config: Path, timeout: int, repeat: int
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "fixture": fixture.directory.name,
        "repeat": repeat,
        "failed_stage": "compile",
        "bound_inputs": {},
        "exec_seconds": None,
        "exec_ok": False,
        "check_exit": None,
        "outcome": "failed",
        "model_calls": None,
        "wall_seconds": None,
    }
    try:
        result.update(run_live_case(fixture.skill, case, config, timeout))
        if result["outcome"] == "compiled":
            execute_fixture(fixture, case, config, timeout, result)
    except (CalcifierError, OSError, ValueError, SyntaxError) as error:
        result["error"] = str(error)
    result["accepted"] = live_accepted(result)
    return result


def run_live_acceptance(
    fixtures: Path, *, output: Path, config: Path, timeout: int, repeat: int = 1
) -> list[dict[str, object]]:
    """Freshly compile and run every sorted fixture on each repeat, retaining all evidence."""
    fixtures, output, config = fixtures.resolve(), output.resolve(), config.resolve()
    selected = [load_fixture(path) for path in sorted(fixtures.iterdir()) if path.is_dir()]
    if not selected:
        raise ValueError("fixtures directory has no fixture directories")
    if repeat < 1:
        raise ValueError("--repeat must be positive")
    if output.is_relative_to(fixtures) or any(
        output.is_relative_to(package_root(fixture.skill)) for fixture in selected
    ):
        raise ValueError("output must be outside fixtures and source packages")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    with (output / "results.jsonl").open("w", encoding="utf-8") as log:
        for fixture in selected:
            for iteration in range(1, repeat + 1):
                case = output / f"sample-{len(results) + 1:04d}"
                result = run_fixture_case(fixture, case, config, timeout, iteration)
                log.write(json.dumps(result) + "\n")
                log.flush()
                results.append(result)
                verdict = "ACCEPTED" if result["accepted"] else "rejected"
                print(
                    f"{verdict} {result['fixture']} repeat={iteration}: "
                    f"outcome={result['outcome']} "
                    f"calls={result['model_calls']} seconds={result['wall_seconds']} "
                    f"lint={result.get('lint')} dry_run={result.get('dry_run')} "
                    f"failed_stage={result['failed_stage']}",
                    flush=True,
                )
    summary = {
        "accepted": sum(bool(result["accepted"]) for result in results),
        "total": len(results),
        "fixtures": {
            fixture.directory.name: {
                "accepted": sum(
                    bool(result["accepted"])
                    for result in results
                    if result["fixture"] == fixture.directory.name
                ),
                "total": repeat,
            }
            for fixture in selected
        },
        "failed_stage": {
            str(stage) if stage is not None else "null": sum(
                result["failed_stage"] == stage for result in results
            )
            for stage in ("compile", "binding", "setup", "exec", "check", None)
        },
        "baseline_fallbacks": sum(result["outcome"] == "baseline_retained" for result in results),
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return results


def run_acceptance(
    pools: list[Path], *, count: int, seed: int, output: Path | None, timeout: int
) -> tuple[Path, dict[str, int]]:
    selected = sample_skills([pool.expanduser().resolve() for pool in pools], count, seed)
    if output is None:
        output = Path(tempfile.mkdtemp(prefix="calcifier-acceptance-")).resolve()
    else:
        output = output.expanduser().resolve()
        if any(output.is_relative_to(package_root(skill)) for skill in selected):
            raise ValueError("output must be outside selected source packages")
        output.mkdir(parents=True, exist_ok=False)
    if any(output.is_relative_to(package_root(skill)) for skill in selected):
        raise ValueError("output must be outside selected source packages")
    stub = output / "agent_stub.py"
    stub.write_text(STUB, encoding="utf-8")
    config = output / "agents.json"
    agents = {
        role: {
            "target": {"command": [sys.executable, str(stub), role]},
            "runtime": {"kind": "host"},
            "timeout_seconds": 10,
        }
        for role in ("compiler", "worker")
    }
    config.write_text(json.dumps({"version": 1, "agents": agents}), encoding="utf-8")
    counts = {"produced": 0, "lint": 0, "exec": 0}
    for index, skill in enumerate(selected, 1):
        directory = output / f"sample-{index:04d}"
        try:
            result = run_case(skill, directory, config, timeout)
        except OSError as error:
            print(f"{skill}: {error}", file=sys.stderr)
            result = False, False, False
        for key, passed in zip(counts, result, strict=True):
            counts[key] += passed
        if not all(result):
            print(f"{skill}: produced/lint/exec={result}; logs: {directory}", file=sys.stderr)
    return output, counts


def positive_repeat(value: str) -> int:
    repeat = int(value)
    if repeat < 1:
        raise argparse.ArgumentTypeError("--repeat must be positive")
    return repeat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        type=Path,
        action="append",
        help="offline directory or SKILL.md path (repeatable; required offline)",
    )
    parser.add_argument("--count", "-n", type=int, help="offline sample size (default: 10)")
    parser.add_argument("--seed", type=int, help="repeatable sample seed; random if omitted")
    parser.add_argument(
        "--output", type=Path, help="new directory; defaults to a retained temp folder"
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="seconds per compile/lint/setup/check subprocess"
    )
    parser.add_argument(
        "--agent-config", type=Path, help="compile live with these roles instead of stubs"
    )
    parser.add_argument("--fixtures", type=Path, help="fixture directory (required live)")
    parser.add_argument(
        "--repeat", type=positive_repeat, help="fresh compilations per live fixture (default: 1)"
    )
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    if args.agent_config is not None:
        if args.fixtures is None:
            parser.error("--fixtures is required with --agent-config")
        if args.pool is not None or args.count is not None or args.seed is not None:
            parser.error("--pool, --count/-n and --seed are offline-only")
        return _live_main(parser, args)
    if args.fixtures is not None or args.repeat is not None:
        parser.error("--fixtures and --repeat require --agent-config")
    if args.pool is None:
        parser.error("--pool is required in offline mode")
    args.count = args.count if args.count is not None else 10
    seed = args.seed if args.seed is not None else random.SystemRandom().getrandbits(64)
    try:
        output, counts = run_acceptance(
            args.pool, count=args.count, seed=seed, output=args.output, timeout=args.timeout
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Offline baseline reliability: seed={seed}; output={output}")
    print(f"sampled={args.count} " + " ".join(f"{key}={value}" for key, value in counts.items()))
    return int(any(value != args.count for value in counts.values()))


def _live_main(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        output = (
            Path(tempfile.mkdtemp(prefix="calcifier-live-acceptance-"))
            if args.output is None
            else args.output.expanduser().resolve()
        )
        if args.output is None:
            output.rmdir()
        results = run_live_acceptance(
            args.fixtures.expanduser().resolve(),
            output=output,
            config=args.agent_config.expanduser().resolve(),
            timeout=args.timeout,
            repeat=args.repeat if args.repeat is not None else 1,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    accepted = sum(bool(result["accepted"]) for result in results)
    fallbacks = sum(result["outcome"] == "baseline_retained" for result in results)
    print(
        f"Live acceptance: accepted={accepted}/{len(results)}; "
        f"baseline_fallbacks={fallbacks}; output={output}"
    )
    return int(accepted < REQUIRED_SHARE * len(results))


if __name__ == "__main__":
    raise SystemExit(main())
