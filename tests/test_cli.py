"""Tests for the ``calcifier`` CLI surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import stat
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path

import pytest

from calcifier import cli
from calcifier.budget import Budget
from calcifier.recipe import Recipe, load_recipe, read_recipe_snapshot
from calcifier.registry import Registry
from calcifier.tools import configured_tool_registry
from calcifier.trust import TrustError
from tests.support.agents import StubProvider
from tests.support.compiler import ScriptedCompiler as _ScriptedCompiler
from tests.support.recipes import _GOOD, _SKILL

_GOOD_RECIPE = textwrap.dedent(
    """
    from __future__ import annotations
    from calcifier import recipe, step

    @step
    def noop(target: str) -> bool:
        return bool(target)

    @recipe(version="0.1")
    def emitted(target: str = "x") -> bool:
        return noop(target=target)

    if __name__ == "__main__":
        raise SystemExit(emitted.cli())
    """
).strip()

_REPORT_RECIPE = textwrap.dedent(
    """
    from __future__ import annotations
    from calcifier import emit, recipe, step

    @step
    def draft_report(target: str) -> str:
        return f"# Final Report\\n\\nTarget: {target}"

    @recipe(version="0.1")
    def emitted(target: str = "x") -> bool:
        report = draft_report(target=target)
        if not report:
            return False
        emit(report, name="report")
        return True

    if __name__ == "__main__":
        raise SystemExit(emitted.cli())
    """
).strip()


_CAPABILITY_RECIPE = textwrap.dedent(
    """
    from __future__ import annotations
    from calcifier import recipe, step, tool

    @step
    def noop(send: bool = False) -> bool:
        if send:
            tool("Slack.postMessage", text="complete")
        return True

    @recipe(version="0.1", capabilities=("Slack.postMessage",))
    def emitted() -> bool:
        return noop()

    if __name__ == "__main__":
        raise SystemExit(emitted.cli())
    """
).strip()

_TOOL_RECIPE = textwrap.dedent(
    """
    from __future__ import annotations
    from calcifier import recipe, step, tool

    @step
    def read() -> dict[str, object]:
        return tool("Read", path="README.md")

    @recipe(version="0.1", capabilities=("Read",))
    def emitted() -> bool:
        return read()["ok"] is True

    if __name__ == "__main__":
        raise SystemExit(emitted.cli())
    """
).strip()

_SIDE_EFFECT_RECIPE = textwrap.dedent(
    """
    from __future__ import annotations
    from calcifier import recipe, record_side_effect, step

    @step
    def publish() -> bool:
        record_side_effect("file_write", "file:/tmp/report.md", idempotency_key="file-key")
        record_side_effect("publish", "slack:#general/msg-123", idempotency_key="slack-key")
        return True

    @recipe(version="0.1")
    def emitted() -> bool:
        return publish()

    if __name__ == "__main__":
        raise SystemExit(emitted.cli())
    """
).strip()


def _patch_registry(
    monkeypatch, default_text: str = _GOOD_RECIPE, *, responses=None, skill_text="# skill\n"
) -> None:
    registry = Registry()
    registry.register(
        _ScriptedCompiler(
            [default_text] if responses is None else responses,
            wrap_sources=responses is None,
            repeat_last=True,
        )
    )
    worker = StubProvider()
    worker.name = "worker"
    registry.register(worker)
    monkeypatch.setattr(cli, "_registry_for_args", lambda args, providers: registry)


# -- argparse plumbing -------------------------------------------------------


def test_help_exits_zero():
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_calcify_help_has_no_compilation_mode_choice(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["calcify", "--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "--scaffolding" not in help_text
    assert "--no-scaffolding" not in help_text


def test_top_level_requires_subcommand():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_unknown_subcommand_errors():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["bogus"])


@pytest.mark.parametrize("command", ["calcify", "update"])
@pytest.mark.parametrize("builtin", ["claude", "codex"])
@pytest.mark.parametrize(
    ("configured_timeout", "flags", "expected_timeout"),
    [
        (3600, [], 3600),
        (600, [], 600),
        (600, ["--timeout", "3600"], 3600),
        (600, ["--agent-timeout", "builder=3600"], 3600),
        (3600, ["--agent-timeout", "builder=7200", "--timeout", "300"], 300),
    ],
)
def test_compiler_timeout_uses_config_and_honors_explicit_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    builtin: str,
    configured_timeout: int,
    flags: list[str],
    expected_timeout: int,
) -> None:
    from tests.support.agents import capture_run as _capture_run

    config_path = tmp_path / "agents.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "agents": {
                    "builder": {
                        "target": {"builtin": builtin},
                        "runtime": {"kind": "host"},
                        "timeout_seconds": configured_timeout,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_preflight_configured_agents", lambda registry, roles: None)
    calls = _capture_run(monkeypatch)
    args = cli.build_parser().parse_args(
        [
            command,
            "SKILL.md",
            "--compiler",
            "builder",
            "--agent-config",
            str(config_path),
            *flags,
        ]
    )
    provider = cli._registry_for_args(args, (args.compiler,)).get(args.compiler)

    provider.invoke("neutral compiler request", timeout=args.timeout)

    assert len(calls) == 1
    assert calls[0][2:] == (expected_timeout, "builder")


def test_subcommand_flag_matrix_is_explicit():
    parser = cli.build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, cli.argparse._SubParsersAction)
    )
    expected = {
        "calcify": {
            "--require-compiled",
            "--help",
            "--compiler",
            "--output",
            "--timeout",
            "--compilation-timeout",
            "--max-compiler-calls",
            "--skill-catalog",
            "--compiler-trace",
            "--verbose",
            "--python-version",
            "--agent-config",
            "--agent-timeout",
        },
        "exec": {
            "--output-dir",
            "--preflight-only",
            "--help",
            "--input",
            "--trust",
            "--subrecipe",
            "--trace-out",
            "--agent-config",
            "--agent-timeout",
            "--max-wall-seconds",
            "--tool-config",
            "--skip-tool-preflight",
            "--dry-run",
        },
        "debug": {
            "--output-dir",
            "--preflight-only",
            "--help",
            "--input",
            "--trust",
            "--subrecipe",
            "--trace-out",
            "--agent-config",
            "--agent-timeout",
            "--max-wall-seconds",
            "--tool-config",
            "--skip-tool-preflight",
            "--dry-run",
        },
        "update": {
            "--help",
            "--recipe",
            "--compiler",
            "--timeout",
            "--compilation-timeout",
            "--max-compiler-calls",
            "--skill-catalog",
            "--python-version",
            "--agent-config",
            "--agent-timeout",
        },
        "lint": {"--help", "--portable", "--python-version"},
        "profile": {"--help"},
        "build-graph": {
            "--help",
            "--agent-config",
            "--agent-timeout",
            "--max-wall-seconds",
            "--compilation-timeout",
            "--max-compiler-calls",
        },
    }

    actual = {
        name: {
            option
            for action in subparser._actions
            for option in action.option_strings
            if option.startswith("--")
        }
        for name, subparser in subparsers.choices.items()
    }

    assert actual == expected


# -- command-line input text -------------------------------------------------


@pytest.mark.parametrize("raw", ["007", "1e3", "true", "null", '"quoted"', "[1, 2]"])
def test_parse_inputs_preserves_text_for_recipe_annotations(raw):
    assert cli._parse_inputs([f"value={raw}"]) == {"value": raw}


def test_parse_inputs_splits_on_first_equals():
    parsed = cli._parse_inputs(["target=foo=bar"])
    assert parsed == {"target": "foo=bar"}


def test_parse_inputs_rejects_missing_equals():
    with pytest.raises(SystemExit):
        cli._parse_inputs(["badform"])


# -- subcommand: calcify -----------------------------------------------------


@pytest.mark.parametrize("compiler_role", ["compiler", "builder"])
def test_cmd_calcify_rejects_compiler_only_config_before_model_calls(
    tmp_path: Path, capsys, monkeypatch, compiler_role: str
) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text("Analyze the input.")
    registry = Registry()
    provider = StubProvider()
    provider.name = compiler_role
    registry.register(provider)
    monkeypatch.setattr(cli, "_registry_for_args", lambda *_: registry)

    def unexpected_call(*args, **kwargs):
        pytest.fail("empty runtime roles must fail before a compiler call")

    monkeypatch.setattr(provider, "structured", unexpected_call)
    assert cli.main(["calcify", str(skill), "--compiler", compiler_role]) == 1
    error = capsys.readouterr().err
    assert "configure at least one runtime semantic role" in error
    assert "model_calls=0" in error
    assert not (tmp_path / ".calcifier").exists()


def test_cmd_calcify_catalog_preflight_is_retained_without_a_provider_attempt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.support.compiler import one_draft

    skill = tmp_path / "SKILL.md"
    skill.write_text("Invoke /alpha to inspect the target.")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "version": 1,
                "recipes": {"alpha": "missing/.calcifier/recipe.py"},
            }
        )
    )
    trace = tmp_path / "compiler-trace.json"
    compiler = one_draft(_GOOD_RECIPE)

    def unexpected_call(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("catalog preflight must fail before a compiler call")

    monkeypatch.setattr(compiler, "structured", unexpected_call)
    worker = StubProvider()
    worker.name = "worker"
    registry = Registry()
    registry.register(compiler)
    registry.register(worker)
    monkeypatch.setattr(cli, "_registry_for_args", lambda *_: registry)

    assert (
        cli.main(
            [
                "calcify",
                str(skill),
                "--skill-catalog",
                str(catalog),
                "--compiler-trace",
                str(trace),
            ]
        )
        == 1
    )

    error = capsys.readouterr().err
    payload = json.loads(trace.read_text())
    assert "skill catalog recipe does not exist" in error
    assert "compiler activity: model_calls=0" in error
    assert payload["model_calls"] == 0
    assert payload["usage"] == {
        "cache_read_tokens": None,
        "cache_write_tokens": None,
        "cost_usd": None,
        "input_tokens": None,
        "nested_model_tokens": None,
        "output_tokens": None,
    }
    assert payload["usage_reported_attempts"] == {}
    assert payload["outcome"] == "failed"
    assert len(payload["events"]) == 1
    assert payload["events"][0]["status"] == "failed"
    assert not (tmp_path / ".calcifier").exists()


def test_cmd_calcify_writes_recipe_and_prints_summary(tmp_path: Path, capsys, monkeypatch):
    skill = tmp_path / "SKILL.md"
    skill.write_text("# my skill\n")
    _patch_registry(monkeypatch, skill_text="# my skill\n")
    rc = cli.main(["calcify", str(skill)])
    captured = capsys.readouterr()
    out, err = captured.out, captured.err
    assert rc == 0, f"stdout={out!r} stderr={err!r}"
    assert "compiler activity: model_calls=" in out
    assert "wall_seconds=" in out
    assert "input_tokens" in out
    assert "\ndry run: passed\ndry-run writes: none\n" in out
    # The recipe is NOT auto-trusted; calcify must instruct the user how to grant trust.
    assert "--trust" in out


def test_cmd_calcify_errors_when_skill_missing(tmp_path: Path, capsys):
    rc = cli.main(["calcify", str(tmp_path / "nope.md")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "not a file" in err


def test_cmd_calcify_fallback_exposes_rejected_compiler_activity(
    tmp_path: Path, capsys, monkeypatch
):
    skill = tmp_path / "SKILL.md"
    skill.write_text("# my skill\n")
    _patch_registry(
        monkeypatch,
        responses=["invalid — Python"],
    )

    rc = cli.main(["calcify", str(skill)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "baseline retained" in out
    assert "source repair made no progress" in out
    assert "compiler activity: model_calls=3" in out
    assert "max_model_calls=5" in out
    assert 'model="provider-default"' in out
    assert 'reasoning_effort="provider-default"' in out
    assert "wall_seconds=" in out
    assert "input_tokens" in out
    assert (tmp_path / ".calcifier/recipe.py").is_file()


def test_required_compilation_keeps_a_baseline_but_fails_acceptance(
    tmp_path: Path, capsys, monkeypatch
):
    skill = tmp_path / "SKILL.md"
    trace = tmp_path / "compiler-trace.json"
    skill.write_text("# my skill\n")
    _patch_registry(monkeypatch, responses=["invalid"])
    rc = cli.main(["calcify", str(skill), "--require-compiled", "--compiler-trace", str(trace)])
    assert rc == 1
    assert (tmp_path / ".calcifier/recipe.py").is_file()
    assert json.loads(trace.read_text())["outcome"] == "baseline_retained"
    captured = capsys.readouterr()
    assert "a compiled recipe was required" in captured.err
    assert "baseline retained" in captured.out


def test_cmd_calcify_persists_verbose_per_attempt_diagnostics(tmp_path: Path, capsys, monkeypatch):
    skill = tmp_path / "SKILL.md"
    trace = tmp_path / "compiler-trace.json"
    skill.write_text("# my skill\n")
    _patch_registry(
        monkeypatch,
        responses=["invalid — Python"],
    )

    rc = cli.main(
        [
            "calcify",
            str(skill),
            "--verbose",
            "--compiler-trace",
            str(trace),
        ]
    )

    assert rc == 0
    payload = json.loads(trace.read_text())
    assert payload["model_calls"] == 3
    assert payload["max_model_calls"] == 5
    failures = [event for event in payload["events"] if event["status"] == "validation_failed"]
    assert [event["phase"] for event in failures] == [
        "source.initial",
        "source.repair_1",
        "source.repair_2",
    ]
    diagnostic = json.loads(failures[0]["detail"])["diagnostic"]
    assert "invalid RecipeDraft" in diagnostic
    assert (
        failures[0]["detail"] and "invalid — Python" in json.loads(failures[0]["detail"])["draft"]
    )
    assert payload["events"][-1]["phase"] == "baseline"
    assert payload["events"][-1]["status"] == "retained"
    assert "compiler progress:" in capsys.readouterr().err


def test_cmd_calcify_with_container_flag_errors(tmp_path: Path, capsys, monkeypatch):
    skill = tmp_path / "SKILL.md"
    skill.write_text("# skill\n")
    _patch_registry(monkeypatch)
    with pytest.raises(SystemExit):
        cli.main(["calcify", str(skill), "--container", "alpine:3"])
    assert "--container" in capsys.readouterr().err


# -- subcommand: exec --------------------------------------------------------


def test_cmd_exec_runs_trusted_recipe(tmp_path: Path, capsys, monkeypatch):
    recipe_path = tmp_path / "r.py"
    recipe_path.write_text(_GOOD_RECIPE)
    rc = cli.main(["exec", str(recipe_path), "--trust", "--input", "target=go"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out


def test_cmd_exec_exposes_report_artifact_when_recipe_returns_status(
    tmp_path: Path, capsys, monkeypatch
):
    recipe_path = tmp_path / "r.py"
    recipe_path.write_text(_REPORT_RECIPE)
    rc = cli.main(["exec", str(recipe_path), "--trust", "--input", "target=repo"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "artifact report:" in out
    assert "# Final Report" in out
    assert "Target: repo" in out
    assert "output: True" in out


def test_cmd_exec_errors_when_recipe_missing(tmp_path: Path, capsys):
    rc = cli.main(["exec", str(tmp_path / "nope.py")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "not a file" in err


def test_cmd_exec_reports_recipe_setup_error_without_traceback(tmp_path: Path, capsys, monkeypatch):
    recipe_path = tmp_path / "r.py"
    recipe_path.write_text(_GOOD_RECIPE)

    def fail_setup(*args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        raise cli.RecipeError("packaged resource drift")

    monkeypatch.setattr(Recipe, "run", fail_setup)

    rc = cli.main(["exec", str(recipe_path), "--trust"])
    captured = capsys.readouterr()

    assert rc == 2
    assert "recipe setup failed: packaged resource drift" in captured.err
    assert "Traceback" not in captured.err


def test_cmd_exec_unknown_kwargs_passed_through(tmp_path: Path, capsys, monkeypatch):
    """The recipe's argparse errors loudly when given a flag it doesn't expose."""
    recipe_path = tmp_path / "r.py"
    recipe_path.write_text(_GOOD_RECIPE)
    rc = cli.main(["exec", str(recipe_path), "--trust", "--input", "target=go"])
    assert rc == 0


def test_cli_exec_capability_preflight_reports_missing(tmp_path: Path, capsys, monkeypatch):
    recipe_path = tmp_path / "r.py"
    recipe_path.write_text(_CAPABILITY_RECIPE)
    rc = cli.main(["exec", str(recipe_path), "--trust"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "tool preflight" in captured.err
    assert "Slack.postMessage" in captured.err
    assert "OK" not in captured.out


def test_cli_exec_capability_preflight_skipped_with_flag(tmp_path: Path, capsys, monkeypatch):
    recipe_path = tmp_path / "r.py"
    recipe_path.write_text(_CAPABILITY_RECIPE)
    rc = cli.main(["exec", str(recipe_path), "--trust", "--skip-tool-preflight"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "OK" in captured.out
    assert "tool preflight" not in captured.err


def test_agent_free_recipe_has_an_empty_registry_without_config() -> None:
    registry = cli._registry_for_args(argparse.Namespace(agent_config=None), ())

    assert registry.known() == []
    assert len(registry.configuration_hash) == 64


def test_agent_config_builds_role_registry_and_runtime_summary(tmp_path: Path):
    config = tmp_path / "agents.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "agents": {
                    "worker": {
                        "target": {"command": [cli.sys.executable, "adapter.py"]},
                        "runtime": {"kind": "host"},
                        "timeout_seconds": 30,
                    }
                },
            }
        )
    )
    args = cli.argparse.Namespace(agent_config=config)

    registry = cli._registry_for_args(args, ("worker",))
    runtime = cli._agent_runtime_for_args(args, registry, ("worker",))
    config.write_text("changed after registry construction")

    assert registry.known() == ["worker"]
    assert runtime["mode"] == "configured"
    assert len(runtime["configuration_hash"]) == 64
    assert runtime["agents"]["worker"] == {
        "target_type": "command",
        "target_name": cli.sys.executable,
        "runtime": {"kind": "host"},
        "configured_model": None,
        "reasoning_effort": None,
        "model_identity_source": "unavailable",
        "timeout_seconds": 30,
    }


def _tool_recipe_with_provider(tmp_path: Path) -> tuple[Path, Path]:
    """Write the tool-using recipe and a tool config whose Read provider answers ok."""
    import sys

    recipe_path = tmp_path / "r.py"
    provider_path = tmp_path / "provider.py"
    config_path = tmp_path / "tools.json"
    recipe_path.write_text(_TOOL_RECIPE)
    provider_path.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "assert request == {'capability': 'Read', 'arguments': {'path': 'README.md'}}\n"
        "json.dump({'ok': True}, sys.stdout)\n"
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "capabilities": {"Read": {"command": [sys.executable, str(provider_path)]}},
            }
        )
    )
    return recipe_path, config_path


def test_a_recipe_footer_runs_a_tool_through_its_tool_config(tmp_path: Path) -> None:
    import subprocess
    import sys

    recipe_path, config_path = _tool_recipe_with_provider(tmp_path)

    result = subprocess.run(
        [sys.executable, str(recipe_path), "--tool-config", str(config_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_a_recipe_footer_without_a_tool_provider_stops_at_preflight(tmp_path: Path, capsys) -> None:
    recipe_path, _ = _tool_recipe_with_provider(tmp_path)
    recipe_obj = load_recipe(recipe_path)

    assert recipe_obj.cli([]) == 2
    assert "- Read (configure a provider for Read)" in capsys.readouterr().err
    # Skipping the preflight runs the recipe, and the unconfigured tool fails the run.
    assert recipe_obj.cli(["--skip-tool-preflight"]) == 1


def test_cli_exec_tool_config_runs_declared_capability(tmp_path: Path, capsys, monkeypatch):
    recipe_path, config_path = _tool_recipe_with_provider(tmp_path)

    rc = cli.main(["exec", str(recipe_path), "--trust", "--tool-config", str(config_path)])

    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_tool_registry_includes_configured_child_capabilities(tmp_path: Path):
    config_path = tmp_path / "tools.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "capabilities": {
                    "Read": {"command": [cli.sys.executable, "provider.py"]},
                },
            }
        )
    )
    registry = configured_tool_registry(("Write",), config_path)

    assert registry.known() == ("Read", "Write")
    # A declared capability with no configured provider reports exactly False.
    assert registry.get_provider("Write").available("Write") is False


def test_cli_exec_tool_config_renders_configured_remediation(tmp_path: Path, capsys, monkeypatch):
    recipe_path = tmp_path / "r.py"
    config_path = tmp_path / "tools.json"
    recipe_path.write_text(_TOOL_RECIPE)
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "capabilities": {
                    "Read": {
                        "command": [str(tmp_path / "missing-provider")],
                        "remediation": "install the read-provider package",
                    }
                },
            }
        )
    )

    rc = cli.main(["exec", str(recipe_path), "--trust", "--tool-config", str(config_path)])
    err = capsys.readouterr().err

    assert rc == 2
    assert "install the read-provider package" in err


def test_cli_exec_renders_side_effect_summary(tmp_path: Path, capsys, monkeypatch):
    recipe_path = tmp_path / "r.py"
    recipe_path.write_text(_SIDE_EFFECT_RECIPE)
    rc = cli.main(["exec", str(recipe_path), "--trust"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "side effects:" in out
    assert "file_write file:/tmp/report.md" in out
    assert "publish slack:#general/msg-123" in out
    assert "completed=True" in out


def test_cli_exec_dry_run_soft_marks_effects_uncompleted(tmp_path: Path, capsys, monkeypatch):
    recipe_path = tmp_path / "r.py"
    recipe_path.write_text(_SIDE_EFFECT_RECIPE)
    rc = cli.main(["exec", str(recipe_path), "--trust", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "side effects:" in out
    assert out.count("completed=False") == 2


# -- subcommand: debug -------------------------------------------------------


def test_cmd_debug_runs_with_verbose_logging(tmp_path: Path, capsys, monkeypatch):
    recipe_path = tmp_path / "r.py"
    recipe_path.write_text(_GOOD_RECIPE)
    rc = cli.main(["debug", str(recipe_path), "--trust", "--input", "target=go"])
    assert rc == 0


# -- subcommand: update ------------------------------------------------------


def test_cmd_update_default_recipe_path(tmp_path: Path, capsys, monkeypatch):
    skill = tmp_path / "SKILL.md"
    skill.write_text("# skill\n")
    _patch_registry(monkeypatch)
    rc = cli.main(["calcify", str(skill)])
    assert rc == 0
    capsys.readouterr()

    # The same draft prepares the same recipe, so there is nothing to diff.
    _patch_registry(monkeypatch)
    rc = cli.main(["update", str(skill)])
    captured = capsys.readouterr()
    recipe = tmp_path / ".calcifier" / "recipe.py"
    assert rc == 0
    assert captured.out == f"no changes: the prepared recipe matches {recipe}\n"
    assert "dry run: passed" in captured.err


def test_cmd_update_prints_only_the_diff_and_succeeds(tmp_path: Path, capsys, monkeypatch):
    skill = tmp_path / "SKILL.md"
    skill.write_text("# skill\n")
    _patch_registry(monkeypatch)
    cli.main(["calcify", str(skill)])
    capsys.readouterr()
    recipe = tmp_path / ".calcifier" / "recipe.py"
    written = recipe.read_bytes()

    # Replace the registry so update produces a different body.
    new_body = _GOOD_RECIPE.replace("emitted", "renamed")
    _patch_registry(monkeypatch, new_body)
    rc = cli.main(["update", str(skill)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.startswith(f"--- {recipe}\n+++ {recipe} (prepared)\n")
    assert "+def renamed(" in captured.out
    assert "compiler activity:" not in captured.out
    assert recipe.read_bytes() == written


def test_cmd_update_missing_recipe_errors(tmp_path: Path, capsys, monkeypatch):
    skill = tmp_path / "SKILL.md"
    skill.write_text("# skill\n")
    _patch_registry(monkeypatch)
    rc = cli.main(["update", str(skill)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no existing recipe" in err


def test_runtime_imports_do_not_load_the_compiler() -> None:
    """Generated recipes and `exec`/`run` import calcifier; keep compiler modules lazy."""
    compiler = (
        "calcifier.calcify",
        "calcifier.plan",
        "calcifier.portable",
        "calcifier.skill_intent",
        "calcifier.update",
    )
    probe = (
        "import sys, calcifier, calcifier.cli; "
        f"print(sorted(m for m in {compiler!r} if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def _recipe_with_subrecipes(tmp_path: Path, subrecipes: str) -> Path:
    recipe = tmp_path / "parent.py"
    recipe.write_text(
        "x = 1\n"
        "# --- calcifier metadata (skill provenance + sha256) ---\n"
        f"_CALCIFIER_SUBRECIPES = {subrecipes}\n"
    )
    return recipe


def test_trust_pins_declared_subrecipes_by_current_digest(tmp_path: Path) -> None:
    child = tmp_path / "child.py"
    child.write_bytes(b"y = 2\n")
    digest = hashlib.sha256(b"y = 2\n").hexdigest()
    recipe = _recipe_with_subrecipes(tmp_path, f"[{{'path': 'child.py', 'sha256': '{digest}'}}]")

    pins = cli._declared_subrecipe_pins(read_recipe_snapshot(recipe))

    assert pins == [(child.resolve(), digest)]


def test_trust_rejects_declared_subrecipe_drift(tmp_path: Path) -> None:
    (tmp_path / "child.py").write_bytes(b"y = 3\n")
    stale = hashlib.sha256(b"y = 2\n").hexdigest()
    recipe = _recipe_with_subrecipes(tmp_path, f"[{{'path': 'child.py', 'sha256': '{stale}'}}]")

    with pytest.raises(TrustError, match="drift detected before trust: child.py"):
        cli._declared_subrecipe_pins(read_recipe_snapshot(recipe))


@pytest.mark.parametrize(
    ("subrecipes", "message"),
    [
        ("{'path': 'child.py'}", "must be a list"),
        ("['child.py']", "invalid subrecipe metadata entry"),
        ("[{'path': 'child.py'}]", "invalid subrecipe metadata entry"),
        ("[{'path': 1, 'sha256': 'x'}]", "invalid subrecipe metadata entry"),
    ],
)
def test_trust_rejects_malformed_subrecipe_metadata(
    tmp_path: Path, subrecipes: str, message: str
) -> None:
    recipe = _recipe_with_subrecipes(tmp_path, subrecipes)

    with pytest.raises(TrustError, match=message):
        cli._declared_subrecipe_pins(read_recipe_snapshot(recipe))


def test_trust_without_subrecipe_metadata_pins_nothing(tmp_path: Path) -> None:
    recipe = tmp_path / "plain.py"
    recipe.write_text("x = 1\n")

    assert cli._declared_subrecipe_pins(read_recipe_snapshot(recipe)) == []


def test_exec_trust_refuses_to_record_drifted_subrecipes(tmp_path: Path, capsys) -> None:
    (tmp_path / "child.py").write_bytes(b"y = 3\n")
    stale = hashlib.sha256(b"y = 2\n").hexdigest()
    recipe = _recipe_with_subrecipes(tmp_path, f"[{{'path': 'child.py', 'sha256': '{stale}'}}]")

    status, snapshot = cli._enforce_trust(recipe, granting=True)

    assert (status, snapshot) == (2, None)
    assert "subrecipe drift detected before trust" in capsys.readouterr().err


class _BudgetedRecipe:
    def __init__(self, budget: Budget | None) -> None:
        self.budget = budget


def test_wall_clock_override_only_tightens_a_declared_budget() -> None:
    declared = Budget(max_agent_calls=4, max_wall_seconds=60)
    recipe = _BudgetedRecipe(declared)

    assert cli._budget_for_args(recipe, argparse.Namespace()) is declared
    assert cli._budget_for_args(recipe, argparse.Namespace(max_wall_seconds=90)) == declared
    assert cli._budget_for_args(recipe, argparse.Namespace(max_wall_seconds=30)) == Budget(
        max_agent_calls=4, max_wall_seconds=30
    )


@pytest.mark.parametrize(
    ("has_budget", "override", "message"),
    [
        (True, 0, "must be positive"),
        (False, 0, "must be positive"),
    ],
)
def test_wall_clock_override_rejects_invalid_use(
    has_budget: bool, override: float, message: str
) -> None:
    args = argparse.Namespace(max_wall_seconds=override)
    budget = Budget(max_agent_calls=1, max_wall_seconds=5) if has_budget else None

    with pytest.raises(SystemExit, match=message):
        cli._budget_for_args(_BudgetedRecipe(budget), args)


# -- CLI known answers ---------------------------------------------------------

_RECIPE = textwrap.dedent(
    """
    from calcifier import recipe, step


    @step
    def check_target(target: str) -> bool:
        return bool(target)


    @recipe(version="0.1")
    def inspect(target: str = "x") -> bool:
        return check_target(target=target)


    if __name__ == "__main__":
        raise SystemExit(inspect.cli())
    """
).lstrip()
_BUDGETED_RECIPE = _RECIPE.replace(
    "from calcifier import recipe, step", "from calcifier import Budget, recipe, step"
).replace('@recipe(version="0.1")', '@recipe(version="0.1", budget=Budget(1, 60))')
_KNOWN_CAPABILITY_RECIPE = _RECIPE.replace(
    '@recipe(version="0.1")', '@recipe(version="0.1", capabilities=("Slack.postMessage",))'
)
_MISSING_CAPABILITY = [
    "tool preflight: missing capabilities:",
    "- Slack.postMessage (configure a provider for Slack.postMessage)",
]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _suite(path: Path) -> Path:
    case = {"name": "target-go", "inputs": {"target": "go"}, "expectations": {"ok": True}}
    return _write(path, json.dumps({"cases": [case]}))


def _stderr_lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return capsys.readouterr().err.splitlines()


# -- argument parsing ----------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["lint", "recipe.py"],
            "calcifier lint: error: the following arguments are required: --portable",
        ),
    ],
)
def test_required_flags_are_enforced_by_the_parser(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(argv)

    assert caught.value.code == 2
    assert _stderr_lines(capsys)[-1] == message


# -- calcify / lint ------------------------------------------------------------


def _calcify_with_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str
) -> tuple[int, Path]:
    skill = _write(tmp_path / "neutral" / "SKILL.md", _SKILL)
    registry = Registry()
    registry.register(_ScriptedCompiler([_GOOD], wrap_sources=True))
    worker = StubProvider()
    worker.name = "worker"
    registry.register(worker)
    monkeypatch.setattr(cli, "_registry_for_args", lambda args, providers: registry)
    return cli.main(["calcify", str(skill), *extra]), skill.parent / ".calcifier" / "recipe.py"


@pytest.mark.parametrize("command", ["calcify", "update", "build-graph"])
@pytest.mark.parametrize("limit", ["0", "9", "invalid"])
def test_invalid_compiler_call_budget_is_rejected_before_setup(command, limit, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid budgets must not configure or invoke providers")

    monkeypatch.setattr(cli, "_registry_for_args", forbidden)
    with pytest.raises(SystemExit) as caught:
        cli.main([command, "absent-input", "--max-compiler-calls", limit])
    assert caught.value.code == 2


@pytest.mark.parametrize("command", ["update", "build-graph"])
def test_compiler_commands_forward_an_explicit_call_budget(tmp_path, monkeypatch, command):
    registry = Registry()
    compiler = _ScriptedCompiler([])
    registry.register(compiler)
    worker = StubProvider()
    worker.name = "worker"
    registry.register(worker)
    monkeypatch.setattr(cli, "_registry_for_args", lambda *args: registry)
    budgets = []

    def prepare(*args, **kwargs):
        budgets.append(kwargs["max_compiler_calls"])
        return "" if command == "update" else tmp_path / "catalog.json"

    target = (
        "calcifier.update.update_skill"
        if command == "update"
        else "calcifier.graph_build.build_graph"
    )
    monkeypatch.setattr(target, prepare)
    assert cli.main([command, str(tmp_path / "input"), "--max-compiler-calls", "6"]) == 0
    assert budgets == [6]
    assert compiler.calls == []


def test_calcify_explicit_call_budget_is_recorded_in_compiler_trace(tmp_path, monkeypatch):
    trace = tmp_path / "compiler-trace.json"
    rc, target = _calcify_with_stub(
        tmp_path, monkeypatch, "--max-compiler-calls", "6", "--compiler-trace", str(trace)
    )
    assert rc == 0
    assert target.is_file()
    payload = json.loads(trace.read_bytes())
    assert payload["max_model_calls"] == 6
    assert payload["model_calls"] == 1
    assert payload["outcome"] == "compiled"


_PROGRESS_PREFIX = [
    'compiler progress: {"model_calls":0,"phase":"baseline","sequence":1,"status":"prepared"}',
    'compiler progress: {"model_calls":0,"phase":"compiler","sequence":2,"status":"started"}',
    'compiler progress: {"model_calls":1,"phase":"source.initial","sequence":4,'
    '"status":"request_started"}',
    'compiler progress: {"model_calls":1,"phase":"source.initial","sequence":5,'
    '"status":"response_received"}',
]


def test_calcify_reports_exact_progress_summary_and_trust_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, target = _calcify_with_stub(tmp_path, monkeypatch, "--verbose")

    captured = capsys.readouterr()
    assert rc == 0
    progress = captured.err.splitlines()
    budget = json.loads(progress.pop(2).removeprefix("compiler progress: "))
    assert budget["phase"] == "compiler.budget"
    assert json.loads(budget["detail"]) == {"max_model_calls": 5, "max_wall_seconds": 3600}
    assert progress[:3] == _PROGRESS_PREFIX[:3]
    response = json.loads(progress[3].removeprefix("compiler progress: "))
    assert response == json.loads(_PROGRESS_PREFIX[3].removeprefix("compiler progress: "))
    lines = captured.out.splitlines()
    assert lines[0] == f"wrote {target}"
    assert lines[1].startswith("compiler activity: model_calls=1 max_model_calls=5 ")
    phases = [line.split(": ", 1)[0] for line in lines if line.startswith("compiler phase ")]
    assert phases == []
    lines = [line for line in lines if not line.startswith("compiler phase ")]
    assert lines[2:] == [
        "dry run: passed",
        "dry-run writes: none",
        "name: tally",
        "version: 2.5",
        "inputs: target='.'",
        "steps: ['count_entries']",
        "budget: none declared",
        "",
        "Review the recipe before granting trust. Once you've read it,",
        f"run:  calcifier exec {target} --trust",
    ]


def test_lint_reports_exact_portable_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, clean = _calcify_with_stub(tmp_path, monkeypatch)
    dirty = _write(clean.with_name("dirty.py"), "VALUE = missing\n" + clean.read_text())
    capsys.readouterr()

    assert rc == 0
    assert cli.main(["lint", str(clean), "--portable"]) == 0
    assert capsys.readouterr().out == f"portable: OK ({clean})\n"
    assert cli.main(["lint", str(dirty), "--portable"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == [
        f"{dirty}:line 1: F821: Undefined name `missing`",
        f"{dirty}:line 1: ty validation failed: line 1: "
        "ty[unresolved-reference]: Name `missing` used when not defined",
    ]


# -- exec / debug ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "level"), [("exec", logging.WARNING), ("debug", logging.DEBUG)]
)
def test_exec_and_debug_configure_exact_log_levels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, level: int
) -> None:
    recipe = _write(tmp_path / "recipe.py", _RECIPE)
    configured: list[object] = []

    def record_basic_config(**kwargs: object) -> None:
        configured.append(kwargs["level"])

    monkeypatch.setattr(logging, "basicConfig", record_basic_config)

    assert cli.main([command, str(recipe), "--trust", "--input", "target=go"]) == 0
    assert configured == [level]


def test_exec_without_registered_agents_runs_deterministic_recipe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipe = _write(tmp_path / "recipe.py", _RECIPE)
    trace = tmp_path / "trace.json"

    argv = ["exec", str(recipe), "--trust", "--trace-out", str(trace), "--input", "target=go"]
    assert cli.main(argv) == 0
    assert "output: True" in capsys.readouterr().out.splitlines()
    payload = json.loads(trace.read_text())
    assert payload["agent_preflight"] == {"agents": [], "ok": True}
    assert payload["agent_runtime"] == {"mode": "host"}
    assert payload["output"] is True


def test_exec_reports_unloadable_trusted_recipe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipe = _write(tmp_path / "recipe.py", "VALUE = 1\n")

    assert cli.main(["exec", str(recipe), "--trust"]) == 2
    assert _stderr_lines(capsys) == [
        f"recorded trust for {recipe.resolve()}:",
        f"  current sha256:  {hashlib.sha256(recipe.read_bytes()).hexdigest()}",
        "  declared capabilities: (unavailable)",
        "  agent roles: (unavailable)",
        f"recipe load failed: Recipe {recipe} defines no @recipe-decorated function "
        "(import calcifier.recipe and decorate one with @recipe).",
    ]


@pytest.mark.parametrize(
    ("recipe_text", "seconds", "message"),
    [
        (_BUDGETED_RECIPE, "0", "--max-wall-seconds must be positive"),
        (_BUDGETED_RECIPE, "-1", "--max-wall-seconds must be positive"),
        (_RECIPE, "0", "--max-wall-seconds must be positive"),
    ],
)
def test_wall_clock_override_rejections_are_exact(
    tmp_path: Path, recipe_text: str, seconds: str, message: str
) -> None:
    recipe = _write(tmp_path / "recipe.py", recipe_text)

    with pytest.raises(SystemExit) as caught:
        cli.main(["exec", str(recipe), "--trust", f"--max-wall-seconds={seconds}"])

    assert caught.value.code == message


@pytest.mark.parametrize("seconds", ["1", "0.5"])
@pytest.mark.parametrize("recipe_text", [_RECIPE, _BUDGETED_RECIPE])
def test_positive_wall_clock_override_runs_recipe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], seconds: str, recipe_text: str
) -> None:
    recipe = tmp_path / "recipe.py"
    recipe.write_bytes(recipe_text.encode("utf-8"))

    argv = ["exec", str(recipe), "--trust", f"--max-wall-seconds={seconds}", "--input", "target=go"]
    assert cli.main(argv) == 0
    assert "output: True" in capsys.readouterr().out.splitlines()


# Runtime fingerprint of a host run with no agents, no tools, and no wall override:
# sha256 of {"agent_runtime":{"mode":"host"},"max_wall_seconds_override":null,
# "providers":[],"tools":...} serialized with sorted keys and compact separators.


# -- configured agent preflight --------------------------------------------------

_CLAUDE_HELP = "--no-session-persistence --safe-mode --tools"
_ROLES_ONLY_COMPILER = (
    "calcify failed: configure at least one runtime semantic role in --agent-config "
    "besides the compiler role 'compiler'; no compiler calls made"
)


_POSIX_FAKE_CLIS = pytest.mark.skipif(
    sys.platform == "win32", reason="fake agent CLIs are POSIX shell scripts"
)


def _fake_executable(bin_dir: Path, name: str, body: str) -> None:
    script = _write(bin_dir / name, f"#!/bin/sh\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)


def _calcify_with_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: Mapping[str, object],
    claude_help: str = _CLAUDE_HELP,
) -> int:
    bin_dir = tmp_path / "bin"
    _fake_executable(bin_dir, "claude", f"printf '%s\\n' '{claude_help}'")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    config = _write(
        tmp_path / "agents.json",
        json.dumps({"version": 1, "agents": {"compiler": {**route, "timeout_seconds": 30}}}),
    )
    skill = _write(tmp_path / "SKILL.md", _SKILL)
    return cli.main(["calcify", str(skill), "--agent-config", str(config)])


@pytest.mark.parametrize(
    "route",
    [
        {"target": {"builtin": "claude"}, "runtime": {"kind": "host"}},
        {"target": {"command": [cli.sys.executable, "adapter.py"]}, "runtime": {"kind": "host"}},
    ],
)
@_POSIX_FAKE_CLIS
def test_passing_agent_preflight_reaches_runtime_role_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    route: dict[str, object],
) -> None:
    assert _calcify_with_config(tmp_path, monkeypatch, route) == 1
    assert _stderr_lines(capsys)[0] == _ROLES_ONLY_COMPILER


@_POSIX_FAKE_CLIS
def test_host_agent_preflight_rejects_missing_cli_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = {"target": {"builtin": "claude"}, "runtime": {"kind": "host"}}

    with pytest.raises(SystemExit) as caught:
        _calcify_with_config(tmp_path, monkeypatch, route, claude_help="usage: claude")

    assert caught.value.code == (
        "agent preflight failed in host: 'claude' does not support "
        "--no-session-persistence, --safe-mode, --tools"
    )
