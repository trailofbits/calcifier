"""Offline sampling, CLI stage counts, and failure status."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import acceptance


def make_skill(directory: Path) -> Path:
    directory.mkdir(parents=True)
    skill = directory / "SKILL.md"
    skill.write_text("# Task\nReview the supplied target.\n", encoding="utf-8")
    return skill


def test_sampling_is_reproducible_and_deduplicates_overlapping_pools(tmp_path):
    skills = [make_skill(tmp_path / f"skill-{number}") for number in range(8)]
    make_skill(tmp_path / ".calcifier" / "ignored")
    first = acceptance.sample_skills([tmp_path, skills[0]], 5, 123)
    assert first == acceptance.sample_skills([skills[0], tmp_path], 5, 123)
    assert first != acceptance.sample_skills([tmp_path], 5, 124)
    assert len(set(first)) == 5
    assert set(first) <= set(skills)
    with pytest.raises(ValueError, match="pool size 8"):
        acceptance.sample_skills([tmp_path], 9, 123)


def test_lint_failure_does_not_erase_successful_production_or_execution(
    tmp_path, monkeypatch, capsys
):
    skill = make_skill(tmp_path / "source")

    def run(argv, directory, name, timeout):
        if name == "calcify":
            (directory / "recipe.py").write_text("# generated\n", encoding="utf-8")
        return name != "lint"

    monkeypatch.setattr(acceptance, "run_command", run)
    status = acceptance.main(
        ["--pool", str(skill), "-n", "1", "--seed", "123", "--output", str(tmp_path / "output")]
    )
    assert status == 1
    assert "sampled=1 produced=1 lint=0 exec=1" in capsys.readouterr().out


def test_command_preserves_nonzero_exit_and_logs(tmp_path):
    assert not acceptance.run_command(
        [sys.executable, "-c", "import sys; print('diagnostic'); sys.exit(7)"],
        tmp_path,
        "failed",
        10,
    )
    assert (tmp_path / "failed.stdout").read_text(encoding="utf-8").strip() == "diagnostic"


def test_command_timeout_is_a_failure(tmp_path, monkeypatch):
    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(acceptance.subprocess, "run", timeout)
    assert not acceptance.run_command([sys.executable], tmp_path, "timeout", 1)
    assert "timed out" in (tmp_path / "timeout.stderr").read_text(encoding="utf-8")


def test_copy_keeps_sibling_resources_and_materializes_links(tmp_path, monkeypatch):
    pool = tmp_path / "pool"
    skill = make_skill(pool / "plugin" / "skills" / "example")
    reference = pool / "plugin" / "references" / "instructions.md"
    reference.parent.mkdir()
    reference.write_text("Shared reference.\n", encoding="utf-8")
    link = skill.parent / "reference.md"
    try:
        link.symlink_to(reference)
    except OSError as error:
        pytest.skip(f"test filesystem cannot create symlinks: {error}")
    frozen = skill.parent / ".calcifier" / "recipe.py"
    frozen.parent.mkdir()
    frozen.write_text("Frozen recipe.\n", encoding="utf-8")
    monkeypatch.setattr(acceptance, "run_command", lambda *args: False)
    directory = tmp_path / "case"
    assert acceptance.run_case(skill, directory, tmp_path / "agents.json", 10) == (False,) * 3
    copied = directory / "package" / "skills" / "example"
    assert (copied / "reference.md").read_bytes() == reference.read_bytes()
    assert not (copied / "reference.md").is_symlink()
    assert (directory / "package" / "references" / "instructions.md").is_file()
    assert not (copied / ".calcifier").exists()
    assert frozen.read_text(encoding="utf-8") == "Frozen recipe.\n"


def test_real_cli_counts_only_production_lint_and_execution(tmp_path):
    skill = make_skill(tmp_path / "pool" / "example")
    source = skill.read_bytes()
    output, counts = acceptance.run_acceptance(
        [skill.parent],
        count=1,
        seed=9817,
        output=tmp_path / "output",
        timeout=90,
    )
    assert counts == {"produced": 1, "lint": 1, "exec": 1}
    assert skill.read_bytes() == source
    configuration = json.loads((output / "agents.json").read_text(encoding="utf-8"))
    assert all("command" in role["target"] for role in configuration["agents"].values())
    diagnostic = (output / "sample-0001" / "calcify.stderr").read_text(encoding="utf-8")
    assert "Compiler improvement rejected; retaining baseline" in diagnostic
    assert "invalid RecipeDraft" in diagnostic
    assert not (output / "manifest.json").exists()
    assert not (output / "summary.json").exists()
    with pytest.raises(FileExistsError):
        acceptance.run_acceptance([skill], count=1, seed=9817, output=output, timeout=90)


_ACCEPTED = {
    "outcome": "compiled",
    "model_calls": 2,
    "wall_seconds": 120.5,
    "lint": True,
    "dry_run": "passed",
    "exec_ok": True,
    "check_exit": 0,
}


@pytest.mark.parametrize(
    ("change", "accepted"),
    [
        ({}, True),
        ({"model_calls": 3, "wall_seconds": 300}, True),
        ({"outcome": "baseline_retained"}, False),
        ({"model_calls": 4}, False),
        ({"model_calls": None}, False),
        ({"wall_seconds": 300.01}, False),
        ({"lint": False}, True),
        ({"dry_run": "unverified"}, True),
        ({"dry_run": "not run"}, True),
        ({"exec_ok": False}, False),
        ({"check_exit": 1}, False),
        ({"check_exit": None}, False),
    ],
)
def test_live_acceptance_requires_the_compiled_recipe_within_limits(change, accepted):
    assert acceptance.live_accepted({**_ACCEPTED, **change}) is accepted


def _fake_live_cli(outcome: str):
    def run(argv, directory, name, timeout):
        del argv, timeout
        if name == "calcify":
            (directory / "recipe.py").write_text(
                "from calcifier import recipe\n\n@recipe\ndef run(target: str, *, depth: int = 1):"
                "\n    return target\n",
                encoding="utf-8",
            )
            trace = {
                **_ACCEPTED,
                "outcome": outcome,
                "events": [{"phase": "dry_run", "status": "passed"}],
            }
            (directory / "compiler-trace.json").write_text(json.dumps(trace), encoding="utf-8")
        return True

    return run


def test_live_case_reads_the_compiler_trace_and_recipe_inputs(tmp_path, monkeypatch):
    skill = make_skill(tmp_path / "source")
    monkeypatch.setattr(acceptance, "run_command", _fake_live_cli("compiled"))
    result = acceptance.run_live_case(skill, tmp_path / "case", tmp_path / "agents.json", 10)
    assert result == {
        **{key: value for key, value in _ACCEPTED.items() if key not in {"exec_ok", "check_exit"}},
        "skill": str(skill),
        "inputs": ["target", "depth"],
    }


@pytest.mark.parametrize(("rejected", "status"), [(1, 0), (2, 1)])
def test_live_main_requires_seven_of_eight_accepted(
    tmp_path, monkeypatch, capsys, rejected, status
):
    fixtures = [make_fixture(tmp_path, f"fixture-{number}") for number in range(8)]
    outcomes = iter(["baseline_retained"] * rejected + ["compiled"] * (8 - rejected))
    original = _fake_live_cli

    def run(argv, directory, name, timeout):
        if name == "calcify":
            return original(next(outcomes))(argv, directory, name, timeout)
        return True

    monkeypatch.setattr(acceptance, "run_command", run)

    def execute(fixture, case, config, timeout, result):
        result.update(exec_ok=True, check_exit=0, failed_stage=None)

    monkeypatch.setattr(acceptance, "execute_fixture", execute)
    arguments = ["--fixtures", str(tmp_path / "fixtures")]
    arguments += ["--output", str(tmp_path / "out"), "--agent-config", str(tmp_path / "a.json")]
    assert acceptance.main(arguments) == status
    assert f"accepted={8 - rejected}/8" in capsys.readouterr().out
    lines = (tmp_path / "out" / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8 and {json.loads(line)["fixture"] for line in lines} == {
        fixture.name for fixture in fixtures
    }
    summary = json.loads((tmp_path / "out" / "SUMMARY.json").read_bytes())
    assert summary["baseline_fallbacks"] == rejected
    assert summary["failed_stage"]["compile"] == rejected


_COUNT_RECIPE = '''\
from pathlib import Path

from calcifier import Budget, agent, emit, output_type, recipe, step


@output_type
class Count:
    count: int
    label: str
    text: str
    labels: list[str] | None
    options: dict[str, int] | None
    marker: int | None


@step
def count_file(path: str) -> int:
    return len(Path(path).read_text(encoding="utf-8").splitlines())


@step
def label_count(workspace: str) -> str:
    return agent("worker", "Label the line count.", working_directory=workspace)


@recipe(agents=("worker",), budget=Budget(2, 30))
def line_count(
    workspace: str,
    *,
    filename: str = "example.txt",
    labels: list[str] | None = None,
    options: dict[str, int] | None = None,
    marker: int | None = None,
    text: str = "001",
) -> Count:
    """Count lines in the requested file and label the result."""
    count = count_file(str(Path(workspace) / filename))
    label = label_count(workspace)
    emit(count, name="count")
    return Count(count, label, text, labels, options, marker)


if __name__ == "__main__":
    raise SystemExit(line_count.cli())
'''

_CHECK = """\
import json
import os
import sys
from pathlib import Path

result = json.loads(Path(os.environ["CALCIFIER_RESULT"]).read_bytes())
assert result == {
    "ok": True,
    "error": None,
    "output": {
        "count": 3, "label": "counted", "text": "001",
        "labels": ["a", "b"], "options": {"delta": 0}, "marker": None,
    },
    "artifacts": {"count": 3},
}, result
assert Path.cwd() == Path(os.environ["CALCIFIER_WORKSPACE"])
assert (Path(os.environ["CALCIFIER_FIXTURE"]) / "fixture.json").is_file()
sys.stdout.buffer.write(b"fixture passed\\n")
sys.stderr.buffer.write(b"check diagnostic\\n")
"""

_AGENTS = """\
import json
import os
import sys
from pathlib import Path

from calcifier.recipe_draft import draft_response

prompt = sys.stdin.read()
data = json.loads(Path(sys.argv[2]).read_bytes())
contract = json.loads(os.environ["CALCIFIER_AGENT_CONTRACT"])
assert os.environ["CALCIFIER_AGENT_PROTOCOL"] == "1"
phase = "compiler" if sys.argv[1] == "compiler" else (
    "binder" if prompt.startswith("Bind the user's request") else "worker"
)
log = Path(data["log"])
previous = (
    [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    if log.exists() else []
)
with log.open("a", encoding="utf-8", newline="\\n") as stream:
    stream.write(json.dumps({
        "phase": phase, "prompt": prompt, "contract": contract,
        "cwd": str(Path.cwd()), "path": os.environ["PATH"],
    }) + "\\n")
if phase == "compiler":
    number = sum(call["phase"] == "compiler" for call in previous)
    sources = data["sources"]
    response = draft_response(sources[number % len(sources)])
elif phase == "binder":
    details = json.JSONDecoder().raw_decode(prompt[prompt.index("{\\n"):])[0]
    required = next(item["name"] for item in details["entry_point"]["inputs"] if item["required"])
    bound = data.get("binding", {
        required: details["workspace"], "labels": ["a", "b"], "options": {"delta": 0},
        "marker": None, "text": "001",
    })
    response = json.dumps({"value": json.dumps(bound), "_calcifier_json": True})
else:
    response = json.dumps({"value": "counted"})
print(json.dumps({"version": 1, "output": response, "usage": {}}))
"""


def make_fixture(root: Path, name: str = "count", **changes) -> Path:
    fixture = root / "fixtures" / name
    fixture.mkdir(parents=True)
    skill = make_skill(root / "sources" / name)
    skill.write_bytes(b"Count lines in the requested file and label the result.\n")
    (fixture / "workspace").mkdir()
    (fixture / "workspace" / "example.txt").write_bytes(b"one\ntwo\nthree\n")
    (fixture / "check.py").write_text(_CHECK, encoding="utf-8", newline="\n")
    (fixture / "fixture.json").write_text(
        json.dumps(
            {
                "skill": str(skill.resolve()),
                "request": "Count the lines in example.txt in this workspace and label the count.",
                "check": [sys.executable, "{fixture}/check.py"],
                **changes,
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    return fixture


def scripted_agents(root: Path, **changes) -> Path:
    stub = root / "agents.py"
    stub.write_text(_AGENTS, encoding="utf-8", newline="\n")
    data = root / "agent-data.json"
    data.write_text(
        json.dumps({"log": str(root / "calls.jsonl"), "sources": [_COUNT_RECIPE], **changes}),
        encoding="utf-8",
    )
    config = root / "agents.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "agents": {
                    role: {
                        "target": {"command": [sys.executable, str(stub), role, str(data)]},
                        "runtime": {"kind": "host"},
                        "timeout_seconds": 15,
                    }
                    for role in ("compiler", "worker")
                },
            }
        ),
        encoding="utf-8",
    )
    return config


def run_fixtures(root: Path, config: Path, repeat: int = 1) -> list[dict[str, object]]:
    return acceptance.run_live_acceptance(
        root / "fixtures", output=root / "out", config=config, timeout=90, repeat=repeat
    )


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_fixture_runs_real_cli_binds_values_and_checks_return_and_emits(tmp_path, capsys, newline):
    fixture = make_fixture(tmp_path)
    skill = Path(json.loads((fixture / "fixture.json").read_bytes())["skill"])
    skill.write_bytes(skill.read_bytes().replace(b"\n", newline))
    skill_source = skill.read_bytes()
    (fixture / "workspace" / "example.txt").write_bytes(
        b"one\ntwo\nthree\n".replace(b"\n", newline)
    )
    source = (fixture / "workspace" / "example.txt").read_bytes()
    config = scripted_agents(tmp_path)

    (result,) = run_fixtures(tmp_path, config)

    assert result["accepted"] is True, result
    assert result["failed_stage"] is None and result["check_exit"] == 0
    assert isinstance(result["exec_seconds"], float) and result["exec_seconds"] > 0
    case = tmp_path / "out" / "sample-0001"
    assert result["bound_inputs"] == {
        "workspace": str(case / "workspace"),
        "labels": ["a", "b"],
        "options": {"delta": 0},
        "marker": None,
        "text": "001",
    }
    assert (case / "check.stdout").read_bytes() == b"fixture passed\n"
    assert (case / "check.stderr").read_bytes() == b"check diagnostic\n"
    assert (fixture / "workspace" / "example.txt").read_bytes() == source
    assert (case / "workspace" / "example.txt").read_bytes() == source
    assert skill.read_bytes() == (case / "package" / "SKILL.md").read_bytes() == skill_source
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_bytes().splitlines()]
    assert [call["phase"] for call in calls] == ["compiler", "binder", "worker"]
    binder = calls[1]
    assert binder["contract"]["filesystem"]["working_directory"] == str(case / "workspace")
    assert binder["contract"]["filesystem"]["readable_paths"] == [
        str(case),
        str(case / "workspace"),
    ]
    assert binder["contract"]["schema"]["type"] == "object"
    assert '"request": "Count the lines' in binder["prompt"]
    assert '"annotation": "str"' in binder["prompt"]
    assert "Count lines in the requested file" in binder["prompt"]
    assert str(fixture) not in capsys.readouterr().out
    rows = [
        json.loads(line) for line in (tmp_path / "out" / "results.jsonl").read_bytes().splitlines()
    ]
    assert rows == [result]
    summary = json.loads((tmp_path / "out" / "SUMMARY.json").read_bytes())
    assert summary["accepted"] == summary["total"] == 1
    assert summary["fixtures"] == {"count": {"accepted": 1, "total": 1}}
    assert summary["failed_stage"]["null"] == 1


@pytest.mark.parametrize(
    ("stage", "changes"),
    [
        ("check", {"check": [sys.executable, "-c", "import sys; print('wrong'); sys.exit(7)"]}),
        ("setup", {"setup": [sys.executable, "-c", "import sys; print('setup'); sys.exit(8)"]}),
        ("exec", {}),
        ("binding", {"binding": {"unknown": 1}}),
        ("binding", {"binding": {}}),
    ],
)
def test_fixture_stage_failures_are_rejected_and_skip_later_work(tmp_path, stage, changes):
    binding = {"binding": changes["binding"]} if "binding" in changes else {}
    fixture = make_fixture(tmp_path, **{k: v for k, v in changes.items() if k != "binding"})
    if stage == "exec":
        (fixture / "workspace" / "example.txt").unlink()
    config = scripted_agents(tmp_path, **binding)

    (result,) = run_fixtures(tmp_path, config)

    assert result["outcome"] == "compiled", result
    assert result["accepted"] is False and result["failed_stage"] == stage, result
    case = tmp_path / "out" / "sample-0001"
    if stage != "check":
        assert result["check_exit"] is None and not (case / "check.stdout").exists()
    else:
        assert result["exec_ok"] is True and result["check_exit"] == 7
        assert (case / "check.stdout").read_text(encoding="utf-8").strip() == "wrong"
    if stage == "exec":
        execution = json.loads((case / "run-result.json").read_bytes())
        assert execution["ok"] is False and execution["error"]
    if stage == "binding":
        assert result["bound_inputs"] == binding["binding"]
        assert isinstance(result["error"], str) and "inputs" in result["error"]
        assert not (case / "exec.stdout").exists()
    if stage == "setup":
        assert (case / "setup.stdout").read_text(encoding="utf-8").strip() == "setup"
        assert result["bound_inputs"] == {}
    summary = json.loads((tmp_path / "out" / "SUMMARY.json").read_bytes())
    assert summary["accepted"] == 0 and summary["failed_stage"][stage] == 1


def test_repeat_freshly_compiles_sorted_fixtures_and_binds_changed_names(tmp_path, capsys):
    for name in ("zeta", "alpha"):
        make_fixture(tmp_path, name)
    config = scripted_agents(
        tmp_path, sources=[_COUNT_RECIPE, _COUNT_RECIPE.replace("workspace", "directory")]
    )

    assert (
        acceptance.main(
            [
                "--agent-config",
                str(config),
                "--fixtures",
                str(tmp_path / "fixtures"),
                "--repeat",
                "2",
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )

    rows = [
        json.loads(line) for line in (tmp_path / "out" / "results.jsonl").read_bytes().splitlines()
    ]
    assert [(row["fixture"], row["repeat"]) for row in rows] == [
        ("alpha", 1),
        ("alpha", 2),
        ("zeta", 1),
        ("zeta", 2),
    ]
    assert all(row["accepted"] for row in rows), rows
    assert [row["inputs"][0] for row in rows] == ["workspace", "directory"] * 2
    assert all(row["inputs"][0] in row["bound_inputs"] for row in rows)
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_bytes().splitlines()]
    assert [call["phase"] for call in calls] == ["compiler", "binder", "worker"] * 4
    printed = capsys.readouterr().out
    assert "accepted=4/4" in printed and str(tmp_path / "sources") not in printed
    summary = json.loads((tmp_path / "out" / "SUMMARY.json").read_bytes())
    assert summary["fixtures"] == {name: {"accepted": 2, "total": 2} for name in ("alpha", "zeta")}


@pytest.mark.parametrize("has_workspace", [True, False])
def test_placeholder_expansion_setup_and_shared_path_with_spaces(tmp_path, has_workspace):
    root = tmp_path / "paths with spaces"
    root.mkdir()
    arguments = ["{fixture}", "{workspace}", "{case}"]
    fixture = make_fixture(
        root,
        setup=[sys.executable, "{fixture}/setup.py", *arguments],
        check=[sys.executable, "{fixture}/check.py", *arguments],
        path_prepend=["{workspace}/.venv/bin", "{fixture}/bin", "{case}/bin"],
        exec_timeout_seconds=17,
    )
    setup = """\
import os
import sys
from pathlib import Path
fixture, workspace, case = map(Path, sys.argv[1:])
assert Path.cwd() == workspace == case / "workspace"
assert os.environ["PATH"].split(os.pathsep)[:3] == [
    f"{workspace}/.venv/bin", f"{fixture}/bin", f"{case}/bin",
]
(workspace / "example.txt").write_bytes(b"first\\nsecond\\nthird\\n")
(case / "setup-marker").write_bytes(b"setup done")
"""
    (fixture / "setup.py").write_text(setup, encoding="utf-8", newline="\n")
    (fixture / "check.py").write_text(
        _CHECK + setup.split('(workspace / "example.txt")')[0], encoding="utf-8", newline="\n"
    )
    if not has_workspace:
        shutil.rmtree(fixture / "workspace")
    config = scripted_agents(root)

    (result,) = run_fixtures(root, config)

    assert result["accepted"] is True, result
    case = root / "out" / "sample-0001"
    assert (case / "setup-marker").read_bytes() == b"setup done"
    assert (case / "workspace" / "example.txt").read_bytes() == b"first\nsecond\nthird\n"
    if has_workspace:
        assert (fixture / "workspace" / "example.txt").read_bytes() == b"one\ntwo\nthree\n"
    else:
        assert not (fixture / "workspace").exists()
    calls = [json.loads(line) for line in (root / "calls.jsonl").read_bytes().splitlines()]
    assert calls[-1]["path"].split(os.pathsep)[0] == f"{case / 'workspace'}/.venv/bin"
    assert calls[-1]["cwd"] == str(case / "workspace")
    trace = json.loads((case / "run-trace.json").read_bytes())
    assert trace["budget"]["max_wall_seconds"] == 17


def test_baseline_fallback_stops_before_setup_or_binding(tmp_path):
    make_fixture(tmp_path, setup=[sys.executable, "-c", "raise AssertionError('must not run')"])
    config = scripted_agents(tmp_path, sources=["invalid recipe"])

    (result,) = run_fixtures(tmp_path, config)

    assert result["outcome"] == "baseline_retained" and result["failed_stage"] == "compile"
    assert result["accepted"] is False and result["bound_inputs"] == {}
    case = tmp_path / "out" / "sample-0001"
    assert not (case / "workspace").exists() and not (case / "setup.stdout").exists()
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_bytes().splitlines()]
    assert all(call["phase"] == "compiler" for call in calls)


def test_unverified_dry_run_still_requires_a_passing_real_execution(tmp_path):
    make_fixture(tmp_path)
    config = scripted_agents(
        tmp_path,
        sources=[
            _COUNT_RECIPE.replace(
                "options: dict[str, int] | None = None", "options: dict[str, int]"
            )
        ],
    )

    (result,) = run_fixtures(tmp_path, config)

    assert result["outcome"] == "compiled" and result["dry_run"] == "unverified", result
    assert result["accepted"] is True and result["exec_ok"] is True and result["check_exit"] == 0
    trace = json.loads((tmp_path / "out" / "sample-0001" / "compiler-trace.json").read_bytes())
    event = next(event for event in trace["events"] if event["phase"] == "dry_run")
    assert "required recipe input cannot be synthesized" in event["detail"]


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_signature_reads_alias_annotations_defaults_and_docstring_without_loading(
    tmp_path, newline
):
    recipe = tmp_path / "recipe.py"
    source = (
        b"from calcifier import recipe as workflow\n"
        b"@workflow\n"
        b"def run(target: 'str', count: int = 2, *, labels: list[str] = ['a'], flag=None):\n"
        b"    '''Entry documentation.'''\n"
        b"    raise RuntimeError('must not execute')\n"
    )
    recipe.write_bytes(source.replace(b"\n", newline))
    assert acceptance.recipe_signature(recipe) == {
        "name": "run",
        "docstring": "Entry documentation.",
        "inputs": [
            {"name": "target", "annotation": "'str'", "default": None, "required": True},
            {"name": "count", "annotation": "int", "default": "2", "required": False},
            {"name": "labels", "annotation": "list[str]", "default": "['a']", "required": False},
            {"name": "flag", "annotation": None, "default": "None", "required": False},
        ],
    }
    assert recipe.read_bytes() == source.replace(b"\n", newline)


@pytest.mark.parametrize(
    "changes",
    [
        {"unexpected": True},
        {"skill": "relative/SKILL.md"},
        {"request": 3},
        {"check": []},
        {"setup": "bash setup.sh"},
        {"path_prepend": [3]},
        {"exec_timeout_seconds": 0},
        {"exec_timeout_seconds": True},
    ],
)
def test_invalid_fixture_is_rejected_before_compilation(tmp_path, changes, monkeypatch):
    make_fixture(tmp_path, **changes)

    def unexpected(*args):
        pytest.fail("invalid fixtures must not compile")

    monkeypatch.setattr(acceptance, "run_live_case", unexpected)
    with pytest.raises(ValueError):
        run_fixtures(tmp_path, tmp_path / "unused.json")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("missing", ["skill", "request", "check"])
def test_required_fixture_keys(tmp_path, missing):
    fixture = make_fixture(tmp_path)
    payload = json.loads((fixture / "fixture.json").read_bytes())
    del payload[missing]
    (fixture / "fixture.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="required"):
        acceptance.load_fixture(fixture)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--agent-config", "a.json"],
        ["--agent-config", "a.json", "--fixtures", "fixtures", "--pool", "pool"],
        ["--agent-config", "a.json", "--fixtures", "fixtures", "-n", "1"],
        ["--agent-config", "a.json", "--fixtures", "fixtures", "--seed", "1"],
        ["--agent-config", "a.json", "--fixtures", "fixtures", "--repeat", "0"],
        ["--pool", "pool", "--fixtures", "fixtures"],
        ["--pool", "pool", "--repeat", "1"],
    ],
)
def test_mode_specific_arguments_are_required_and_not_silently_ignored(arguments):
    with pytest.raises(SystemExit) as error:
        acceptance.main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("failure", ["timeout", "missing"])
def test_fixture_command_failures_preserve_diagnostics(tmp_path, monkeypatch, failure):
    (tmp_path / "workspace").mkdir()

    def run(argv, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        raise FileNotFoundError("executable missing")

    monkeypatch.setattr(acceptance.subprocess, "run", run)
    assert acceptance.fixture_command(["command"], tmp_path, "exec", 1, {}) is None
    assert (tmp_path / "exec.stderr").read_bytes()
    result = acceptance.run_result(tmp_path, None)
    assert result["ok"] is False and "could not read run trace" in result["error"]
