"""Declared agent writes retain provider containment and work in parallel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from calcifier import Budget, agent, recipe, step
from calcifier.agent_providers import CommandCli, claude_command, codex_command
from calcifier.dry_run import dry_run_recipe
from calcifier.filesystem_contract import AgentError, FilesystemContract, _filesystem_prompt
from calcifier.gate import verify_source
from calcifier.recipe import load_recipe
from calcifier.registry import Registry
from calcifier.trace import DryRunMode
from calcifier.trace_export import serialize_run, validate_trace


def _claude_absolute_path(path: Path) -> str:
    posix = path.as_posix()
    if sys.platform == "win32":
        return f"/{path.drive[0].lower()}{posix[2:]}"
    return posix


@pytest.mark.parametrize("writable", [False, True])
def test_contract_and_prompt_describe_both_modes(tmp_path: Path, writable: bool) -> None:
    references = tmp_path / "references"
    references.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    paths = (project, project) if writable else ()
    contract = FilesystemContract.build(project, (references,), paths)
    expected = (project,) if writable else ()
    assert contract.writable_paths == expected
    assert contract.readable_paths == (references, project)
    assert contract.as_dict() == {
        "mode": "read-write" if writable else "read-only",
        "working_directory": str(project),
        "readable_paths": [str(path) for path in contract.readable_paths],
        "writable_paths": [str(path) for path in expected],
    }
    prompt = _filesystem_prompt(contract, "Apply the task", host_delivery=True)
    assert f"Filesystem contract: {'read-write' if writable else 'read-only'}." in prompt
    assert f"Declared readable paths:\n- {references}" in prompt
    if writable:
        assert "create, modify, and delete files only under the declared writable paths" in prompt
        assert "Everything else is read-only." in prompt
        assert f"Declared writable paths:\n- {project}" in prompt
        assert ".git paths stay read-only." in prompt
        assert "Agents never commit; recipes run git with run_command()." in prompt
    else:
        assert "Do not write, modify, or delete files." in prompt
        assert "Declared writable paths:\n(none)" in prompt


def test_writable_path_must_exist(tmp_path: Path) -> None:
    with pytest.raises(AgentError, match="writable path must be an existing directory"):
        FilesystemContract.build(tmp_path, (), (tmp_path / "missing",))


def test_writable_file_path_is_rejected_even_with_a_valid_directory(tmp_path: Path) -> None:
    file = tmp_path / "file.txt"
    file.write_bytes(b"original\n")
    with pytest.raises(AgentError, match="writable path must be an existing directory"):
        FilesystemContract.build(tmp_path, (), (tmp_path, file))


@pytest.mark.parametrize("cwd_name", [".", "project-other"])
def test_working_directory_must_be_inside_a_writable_directory(tmp_path: Path, cwd_name: str):
    writable = tmp_path / "project"
    writable.mkdir()
    cwd = tmp_path / cwd_name
    cwd.mkdir(exist_ok=True)
    with pytest.raises(AgentError, match="working directory must be inside a writable path"):
        FilesystemContract.build(cwd, (), (writable,))


def test_claude_writes_are_scoped_and_readable_paths_get_no_write_grant(tmp_path: Path) -> None:
    writable = tmp_path / "editable dir"
    writable.mkdir()
    readable = tmp_path / "reference"
    readable.mkdir()
    contract = FilesystemContract.build(writable, (readable,), (writable,))
    command = claude_command("claude", filesystem=contract, schema={"type": "boolean"})
    assert command.argv == [
        "claude",
        "-p",
        "--output-format=stream-json",
        "--verbose",
        "--no-session-persistence",
        "--safe-mode",
        "--restricted",
        "--permission-prompts",
        "none",
        "--permission-mode",
        "dontAsk",
        "--strict-mcp-config",
        "--disallowedTools",
        "Edit(//**/.git/**)",
        "Edit(//**/.git)",
        "--tools",
        "Read,Glob,Grep,Edit,Write",
        "--allowedTools",
        "Read,Glob,Grep",
        f"Edit(/{_claude_absolute_path(writable)}/**)",
        "--max-turns",
        "12",
        "--add-dir",
        str(readable),
        "--json-schema",
        '{"type": "boolean"}',
    ]
    assert all("Bash" not in argument for argument in command.argv)
    assert f"Edit(/{_claude_absolute_path(readable)}/**)" not in command.argv
    assert f"Edit(/{_claude_absolute_path(tmp_path)}/**)" not in command.argv
    assert "Edit" not in command.argv and "Write" not in command.argv


def test_claude_path_glob_characters_are_literal(tmp_path: Path) -> None:
    name, escaped = ("literal[1]", "literal\\[1\\]")
    if sys.platform != "win32":
        name += "*?"
        escaped += "\\*\\?"
    directory = tmp_path / name
    directory.mkdir()
    contract = FilesystemContract.build(directory, (), (directory,))
    command = claude_command("claude", filesystem=contract)
    expected = f"Edit(/{_claude_absolute_path(tmp_path)}/{escaped}/**)"
    assert expected in command.argv


@pytest.mark.parametrize("nested_cwd", [False, True])
def test_codex_writable_roots_exclude_readable_only_paths(tmp_path: Path, nested_cwd: bool):
    project = tmp_path / "project"
    project.mkdir()
    cwd = project / "nested" if nested_cwd else project
    cwd.mkdir(exist_ok=True)
    other = tmp_path / "other writable"
    other.mkdir()
    readable = tmp_path / "reference"
    readable.mkdir()
    contract = FilesystemContract.build(cwd, (readable,), (project, other))
    schema_path = tmp_path / "schema.json"
    command = codex_command("codex", filesystem=contract, schema_path=schema_path)
    assert command.argv == [
        "codex",
        "exec",
        "--color",
        "never",
        "--ephemeral",
        "--json",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "workspace-write",
        "--skip-git-repo-check",
        "--disable",
        "multi_agent",
        "--disable",
        "plugins",
        "--disable",
        "apps",
        "--disable",
        "browser_use",
        "--disable",
        "computer_use",
        "-c",
        "project_doc_max_bytes=0",
        "-c",
        'cli_auth_credentials_store="file"',
        "-C",
        str(cwd),
        "--add-dir",
        str(other),
        "-c",
        "sandbox_workspace_write.exclude_slash_tmp=true",
        "--output-schema",
        str(schema_path),
        "-",
    ]
    assert str(readable) not in command.argv
    assert command.argv.count("--add-dir") == 1


def test_custom_adapter_request_includes_writable_paths(monkeypatch, tmp_path: Path) -> None:
    from calcifier import agent_providers
    from calcifier.agent_events import AgentResult

    contract = FilesystemContract.build(tmp_path, (), (tmp_path,))
    captured = {}

    def invoke(argv, stdin, **kwargs):  # noqa: ANN001
        del argv, stdin
        captured.update(kwargs)
        return AgentResult("true", "true", "", 0, "worker")

    monkeypatch.setattr(agent_providers, "_run_cli", invoke)
    CommandCli("worker", ("adapter",), default_timeout=5).invoke(
        "fix", filesystem=contract, schema={"type": "boolean"}
    )
    assert captured["env"]["CALCIFIER_AGENT_PROTOCOL"] == "1"
    assert json.loads(captured["env"]["CALCIFIER_AGENT_CONTRACT"]) == {
        "version": 1,
        "filesystem": contract.as_dict(),
        "schema": {"type": "boolean"},
        "timeout_seconds": 5,
    }


_RECIPE = """from pathlib import Path
from calcifier import Budget, agent, parallel, recipe, step

@step
def fix(directory: str) -> str:
    return agent("worker", "Fix target.py", working_directory=directory,
                 writable_paths=(directory,))

@recipe(agents=("worker",), budget=Budget(2, 30))
def workflow(project: str) -> list[str]:
    paths = [str(Path(project) / name) for name in ("first", "second")]
    return parallel(fix, paths, max_workers=2)

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""

_ADAPTER = """import json, os, sys
from pathlib import Path
assert os.environ["CALCIFIER_AGENT_PROTOCOL"] == "1"
request = json.loads(os.environ["CALCIFIER_AGENT_CONTRACT"])
contract = request["filesystem"]
assert contract["mode"] == "read-write"
assert len(contract["writable_paths"]) == 1
path = Path(contract["writable_paths"][0])
assert str(path) in contract["readable_paths"]
assert "Filesystem contract: read-write" in sys.stdin.read()
(path / "target.py").write_bytes(b"fixed by adapter\\n")
print(json.dumps({"version": 1, "output": json.dumps({"value": path.name}), "usage": {}}))
"""


def test_parallel_agents_edit_their_declared_files_and_trace_them(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    directories = [project / name for name in ("first", "second")]
    targets = [directory / "target.py" for directory in directories]
    for directory, target in zip(directories, targets, strict=True):
        directory.mkdir()
        target.write_bytes(b"original\n")
    reference = project / "reference.txt"
    reference.write_bytes(b"unchanged\n")
    script = tmp_path / "adapter.py"
    script.write_text(_ADAPTER, encoding="utf-8", newline="\n")
    entry = tmp_path / "recipe.py"
    entry.write_text(_RECIPE, encoding="utf-8", newline="\n")
    verify_source(_RECIPE)
    registry = Registry()
    registry.register(CommandCli("worker", (sys.executable, str(script)), default_timeout=10))
    result = load_recipe(entry).run({"project": str(project)}, registry=registry)
    assert result.ok, result.error
    assert result.output == [directory.name for directory in directories]
    assert [target.read_bytes() for target in targets] == [b"fixed by adapter\n"] * 2
    assert reference.read_bytes() == b"unchanged\n"
    payload = serialize_run(result)
    validate_trace(payload)
    calls = [call for record in payload["steps"] for call in record["agent_calls"]]
    assert [call["filesystem"]["writable_paths"] for call in calls] == [
        [str(directory)] for directory in directories
    ]
    assert all(call["filesystem"]["mode"] == "read-write" for call in calls)


@pytest.mark.parametrize("writable", [False, True])
def test_hard_dry_run_blocks_writable_agents_before_invocation(tmp_path, writable):
    from tests.support.agents import StubProvider

    registry = Registry()
    provider = StubProvider(default='{"value": "read"}')
    provider.name = "worker"
    registry.register(provider)

    @step
    def inspect() -> str:
        return agent(
            "worker",
            "Inspect",
            working_directory=tmp_path,
            writable_paths=(tmp_path,) if writable else (),
        )

    @recipe(agents=("worker",), budget=Budget(2, 30))
    def workflow() -> str:
        return inspect()

    result = workflow.run(registry=registry, dry_run=DryRunMode.HARD)
    assert result.ok is (not writable)
    call = result.records[0].agent_calls[0]
    assert call.attempts == (0 if writable else 1)
    assert call.filesystem is not None
    assert call.filesystem["writable_paths"] == ([str(tmp_path)] if writable else [])
    if writable:
        assert call.error == "SideEffectBlocked"
        assert result.error is not None
        assert "side effect blocked by hard dry-run: agent writable paths" in result.error
    else:
        assert result.output == "read"


def test_compiler_soft_dry_run_executes_the_whole_recipe_and_agents_write_nothing(tmp_path):
    source = """from calcifier import Budget, agent, recipe, step
@step
def inspect(project: str) -> str:
    return agent("worker", "Apply changes", working_directory=project,
                 writable_paths=(project,))
@recipe(agents=("worker",), budget=Budget(2, 30))
def workflow(project: str) -> str:
    inspect(project)
    return "finished after the writable agent"
"""
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nDispatch a worker to update the supplied project.\n")
    result = dry_run_recipe(source, recipe_path=tmp_path / "recipe.py", skill_path=skill)
    assert result.status == "passed", result
    assert result.writes == ()
    assert set(tmp_path.iterdir()) == {skill}
