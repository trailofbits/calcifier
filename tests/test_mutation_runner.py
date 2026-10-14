"""Mutation scores must come from executed tests, not runner setup failures."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import pytest_gremlins.plugin as gremlins
from pytest_gremlins.node_id_file import command_with_node_ids_file

from tests.conftest import (
    _COVERAGE_BOOTSTRAP,
    _COVERAGE_COLLECTION_TIMEOUT_SECONDS,
    _MUTATION_TIMEOUT_SECONDS,
    _mutation_runner_script,
    pytest_configure,
)

pytestmark = pytest.mark.semantic_offline


def test_mutation_subprocess_executes_fixtures_parameters_and_real_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setattr(request.config.option, "gremlins", True, raising=False)
    pytest_configure(request.config)
    assert gremlins._get_lightweight_runner_script is _mutation_runner_script
    source = ast.parse(
        'def identity(value):\n    return value + 1 if __gremlin_active__ == "fault" else value\n'
    )
    instrumented = gremlins._write_instrumented_sources(
        {str(tmp_path / "alpha.py"): source}, tmp_path, {str(tmp_path / "alpha.py"): []}
    )
    test_file = tmp_path / "test_alpha.py"
    test_file.write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "from alpha import identity\n"
        "@pytest.fixture(autouse=True)\n"
        "def offline_guard():\n"
        '    with Path("fixture-runs.txt").open("a") as stream:\n'
        '        stream.write("guard\\n")\n'
        "@pytest.fixture\n"
        "def offset():\n"
        "    return 10\n"
        '@pytest.mark.parametrize("value", [1, 2])\n'
        "def test_identity(value, offset):\n"
        "    assert identity(value + offset) == value + offset\n"
    )
    command = command_with_node_ids_file(
        gremlins._build_test_command(instrumented),
        ["test_alpha.py::test_identity[1]", "test_alpha.py::test_identity[2]"],
        instrumented,
    )
    env = {**os.environ, "PYTEST_GREMLINS_SOURCES_FILE": str(instrumented / "sources.json")}
    try:
        baseline = subprocess.run(
            command,
            cwd=tmp_path,
            env={**env, "ACTIVE_GREMLIN": ""},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert baseline.returncode == 0, baseline.stdout + baseline.stderr
        assert "2 passed" in baseline.stdout
        assert (tmp_path / "fixture-runs.txt").read_text() == "guard\nguard\n"
        fault = subprocess.run(
            command,
            cwd=tmp_path,
            env={**env, "ACTIVE_GREMLIN": "fault"},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert fault.returncode == 1, fault.stdout + fault.stderr
        assert "1 failed" in fault.stdout
        assert "assert 12 == (1 + 10)" in fault.stdout
        # The active failure is followed by the same two-case passing control.
        assert (tmp_path / "fixture-runs.txt").read_text() == "guard\n" * 5
    finally:
        gremlins._cleanup_instrumented_dir(instrumented)


def test_coverage_and_mutation_timeouts_leave_other_subprocess_limits_unchanged(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    calls: list[tuple[list[str], float | None]] = []

    def record(command: list[str], *args: object, **kwargs: object) -> str:
        del args
        timeout = kwargs.get("timeout")
        calls.append((command, timeout if isinstance(timeout, int | float) else None))
        return "completed"

    monkeypatch.setattr(request.config.option, "gremlins", True, raising=False)
    pytest_configure(request.config)
    monkeypatch.setattr(subprocess, "run", record)
    coverage_command = [sys.executable, "-m", "coverage", "run", "-p", _COVERAGE_BOOTSTRAP]

    assert gremlins.subprocess.run(coverage_command, timeout=120) == "completed"
    gremlins.subprocess.run(coverage_command, timeout=7200)
    gremlins.subprocess.run(coverage_command)
    gremlins.subprocess.run(["pytest", "-q"], timeout=30)
    gremlins.subprocess.run(["pytest"], timeout=30, env={"ACTIVE_GREMLIN": "mutation"})
    gremlins.subprocess.run(["pytest"], timeout=7200, env={"ACTIVE_GREMLIN": "mutation"})
    gremlins.subprocess.run(["pytest"], env={"ACTIVE_GREMLIN": "mutation"})
    gremlins.subprocess.run(["pytest"], timeout=30, env={"ACTIVE_GREMLIN": ""})

    assert [timeout for _, timeout in calls] == [
        _COVERAGE_COLLECTION_TIMEOUT_SECONDS,
        7200,
        None,
        30,
        _MUTATION_TIMEOUT_SECONDS,
        7200,
        None,
        30,
    ]
    assert gremlins.subprocess.TimeoutExpired is subprocess.TimeoutExpired


def test_parallel_mutation_pool_uses_the_same_bounded_timeout(monkeypatch, request):
    import pytest_gremlins.parallel.pool as worker_pool

    from tests.mutation_worker import run_logged_test

    monkeypatch.setattr(request.config.option, "gremlins", True, raising=False)
    pytest_configure(request.config)
    pool = gremlins.WorkerPool(max_workers=3, timeout=30)
    assert pool.timeout == _MUTATION_TIMEOUT_SECONDS
    assert pool.max_workers == 3
    assert gremlins.WorkerPool(max_workers=1, timeout=7200).timeout == 7200
    assert worker_pool._run_gremlin_test is run_logged_test


@pytest.mark.parametrize("broken_control", [False, True])
@pytest.mark.parametrize("selector", ["test_value", "test_absent"])
@pytest.mark.parametrize("exception", ["ValueError", "SystemExit"])
def test_import_mutants_need_an_executed_failing_probe_and_passing_control(
    tmp_path, monkeypatch, request, broken_control, selector, exception
):
    monkeypatch.setattr(request.config.option, "gremlins", True, raising=False)
    pytest_configure(request.config)
    source = ast.parse(
        f'if __gremlin_active__ == "import-fault":\n    raise {exception}("mutant import broke")\n'
        + ('raise ValueError("baseline import broke")\n' if broken_control else "value = 1\n")
    )
    instrumented = gremlins._write_instrumented_sources(
        {str(tmp_path / "alpha.py"): source}, tmp_path
    )
    test = tmp_path / "test_alpha.py"
    test.write_text("import alpha\ndef test_value():\n    assert alpha.value == 1\n")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(instrumented / "gremlin_lightweight_runner.py"),
                f"test_alpha.py::{selector}",
            ],
            cwd=tmp_path,
            env={
                **os.environ,
                "PYTEST_GREMLINS_SOURCES_FILE": str(instrumented / "sources.json"),
                "ACTIVE_GREMLIN": "import-fault",
            },
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if broken_control or selector == "test_absent":
            assert result.returncode in {2, 3, 4}, result.stdout + result.stderr
            assert "Mutation import probe (active)" not in result.stdout
        else:
            assert result.returncode == 1, result.stdout + result.stderr
            assert "Mutation import probe (control)" in result.stdout
            assert "1 passed" in result.stdout
            assert "Mutation import probe (active)" in result.stdout
            assert "1 failed" in result.stdout
            assert "mutant import broke" in result.stdout
    finally:
        gremlins._cleanup_instrumented_dir(instrumented)


def test_internal_runner_error_with_successful_import_probes_is_not_a_kill(
    tmp_path, monkeypatch, request
):
    monkeypatch.setattr(request.config.option, "gremlins", True, raising=False)
    pytest_configure(request.config)
    source = ast.parse('value = 2 if __gremlin_active__ == "fault" else 1\n')
    instrumented = gremlins._write_instrumented_sources(
        {str(tmp_path / "alpha.py"): source}, tmp_path
    )
    (tmp_path / "test_alpha.py").write_bytes(
        b"import alpha\nimport pytest\ndef test_value():\n"
        b'    pytest.exit("unrelated runner failure", returncode=3)\n'
    )
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(instrumented / "gremlin_lightweight_runner.py"),
                "test_alpha.py::test_value",
            ],
            cwd=tmp_path,
            env={
                **os.environ,
                "ACTIVE_GREMLIN": "fault",
                "PYTEST_GREMLINS_SOURCES_FILE": str(instrumented / "sources.json"),
            },
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 3, result.stdout + result.stderr
        assert "Mutation import probe (control)" in result.stdout
        assert "Mutation import probe (active)" in result.stdout
        assert result.stdout.count("1 passed") == 2
        assert "1 failed" not in result.stdout
    finally:
        gremlins._cleanup_instrumented_dir(instrumented)


@pytest.mark.parametrize("failure", ["missing-argument", "exception", "string-exit"])
def test_unexpected_launcher_errors_are_not_mutation_kills(tmp_path, failure):
    from tests import mutation_runner

    bootstrap = tmp_path / "bootstrap.py"
    bootstrap.write_text(
        'raise SystemExit("broken bootstrap")\n'
        if failure == "string-exit"
        else 'raise RuntimeError("broken bootstrap")\n'
    )
    arguments = [] if failure == "missing-argument" else [str(bootstrap)]
    result = subprocess.run(
        [sys.executable, mutation_runner.__file__, *arguments],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    if failure != "string-exit":
        message = "IndexError" if failure == "missing-argument" else "broken bootstrap"
        assert message in result.stderr


def test_import_probe_rejects_missing_and_ambiguous_mutant_id():
    from tests.mutation_runner import _unique_mutant_module

    assert _unique_mutant_module({"a": "value = 1"}, "fault") is None
    assert _unique_mutant_module({"a": 'value = "fault"'}, "fault") == "a"
    assert _unique_mutant_module({"a": 'value = "fault"', "b": 'value = "fault"'}, "fault") is None


@pytest.mark.parametrize(
    ("body", "failed", "valid"),
    [
        ('<testcase name="test_selected_modules_import"/>', False, True),
        ('<testcase name="test_selected_modules_import"><failure/></testcase>', True, True),
        ('<testcase name="test_selected_modules_import"><error/></testcase>', True, False),
        ('<testcase name="test_selected_modules_import"><skipped/></testcase>', False, False),
        ('<testcase name="different"><failure/></testcase>', True, False),
        ("", False, False),
    ],
)
def test_probe_requires_exactly_one_real_test_outcome(tmp_path, body, failed, valid):
    from tests.mutation_runner import _probe_result

    report = tmp_path / "result.xml"
    report.write_text(f"<testsuite>{body}</testsuite>")
    assert _probe_result(report, failed=failed) is valid
    assert not _probe_result(tmp_path / "missing", failed=failed)
    report.write_text("malformed")
    assert not _probe_result(report, failed=failed)


@pytest.mark.parametrize("failure", ["timeout", "launch"])
@pytest.mark.parametrize("phase", ["selection", "control", "active"])
def test_probe_timeout_or_runner_failure_never_counts_as_a_kill(
    tmp_path, monkeypatch, capsys, failure, phase
):
    from tests.mutation_runner import _import_failure_is_tested

    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"alpha": {"source": 'value = "fault"'}}))
    monkeypatch.setenv("ACTIVE_GREMLIN", "fault")
    monkeypatch.setenv("PYTEST_GREMLINS_SOURCES_FILE", str(sources))
    selected = tmp_path / "test_selected.py"
    selected.write_text("def test_ok():\n    pass\n")

    attempts = []

    def run_probe(command, **kwargs):
        attempts.append(command)
        current = (
            "selection"
            if "--collect-only" in command
            else ("active" if kwargs["env"]["ACTIVE_GREMLIN"] else "control")
        )
        if current == phase:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            raise OSError("probe launcher failed")
        if current != "selection":
            report = Path(
                next(
                    arg.removeprefix("--junitxml=")
                    for arg in command
                    if arg.startswith("--junitxml=")
                )
            )
            report.write_text(
                '<testsuite><testcase name="test_selected_modules_import"/></testsuite>',
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0, "test_selected.py::test_ok", "")

    monkeypatch.setattr(subprocess, "run", run_probe)
    assert not _import_failure_is_tested(tmp_path / "bootstrap.py", [str(selected)])
    assert len(attempts) == {"selection": 1, "control": 2, "active": 3}[phase]
    diagnostic = capsys.readouterr().err
    assert "could not finish" in diagnostic
    assert ("timed out" if failure == "timeout" else "probe launcher failed") in diagnostic
    monkeypatch.delenv("ACTIVE_GREMLIN")
    assert not _import_failure_is_tested(tmp_path / "bootstrap.py", [str(selected)])


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group isolation")
def test_mutation_launcher_has_a_separate_process_group(tmp_path):
    launcher = tmp_path / "runner.py"
    launcher.write_text(_mutation_runner_script(), encoding="utf-8")
    (tmp_path / "gremlin_bootstrap.py").write_text(
        "import json, os\nprint(json.dumps([os.getpid(), os.getpgrp(), os.getsid(0)]))\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(launcher)],
        capture_output=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    )
    pid, group, session = json.loads(result.stdout)
    assert pid == group == session
    assert group != os.getpgrp()


def test_parallel_worker_logs_start_and_outcome_without_changing_the_result(monkeypatch, capsys):
    from pytest_gremlins.parallel.pool import WorkerResult
    from pytest_gremlins.reporting.results import GremlinResultStatus

    from tests import mutation_worker

    expected = WorkerResult("example", GremlinResultStatus.TIMEOUT)
    attempt_paths = []

    def run(*args):
        gremlin, command, root, env, timeout = args
        assert (gremlin, command, root, timeout) == ("example", ["pytest"], "workspace", 600)
        assert env["fixture"] == "value"
        assert env["TMPDIR"] == env["TMP"] == env["TEMP"]
        scratch = Path(env["TMPDIR"])
        assert scratch.is_dir()
        (scratch / "retained-fixture").write_bytes(b"fixture bytes")
        attempt_paths.append(scratch)
        assert "example started" in capsys.readouterr().out
        return expected

    monkeypatch.setattr(mutation_worker, "_run_gremlin_test", run)
    assert (
        mutation_worker.run_logged_test(
            "example", ["pytest"], "workspace", {"fixture": "value"}, 600
        )
        is expected
    )
    assert "example finished timeout" in capsys.readouterr().out
    assert attempt_paths and not attempt_paths[0].exists()


def test_parallel_worker_logs_unexpected_failure_and_reraises(monkeypatch, capsys):
    from tests import mutation_worker

    attempt_paths = []

    def run(*args):
        scratch = Path(args[3]["TMPDIR"])
        (scratch / "partial-fixture").write_bytes(b"partial bytes")
        attempt_paths.append(scratch)
        raise RuntimeError("worker failure")

    monkeypatch.setattr(mutation_worker, "_run_gremlin_test", run)
    with pytest.raises(RuntimeError, match="worker failure"):
        mutation_worker.run_logged_test("example", [], "workspace", {}, 600)
    assert "example raised RuntimeError: worker failure" in capsys.readouterr().out
    assert attempt_paths and not attempt_paths[0].exists()


@pytest.mark.parametrize("returncode", [0, 1, 3])
def test_worker_removes_native_child_scratch_without_changing_outcome(tmp_path, returncode):
    from pytest_gremlins.reporting.results import GremlinResultStatus

    from tests.mutation_worker import run_logged_test

    recorded = tmp_path / "attempt-paths.json"
    command = [
        sys.executable,
        "-c",
        "import json, pathlib, sys, tempfile; "
        "root = pathlib.Path(tempfile.gettempdir()); "
        "fixture = pathlib.Path(tempfile.mkdtemp(prefix='retained-fixture-')); "
        "(fixture / 'data').write_bytes(b'x' * 1024 * 1024); "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps([str(root), str(fixture)])); "
        "sys.exit(int(sys.argv[2]))",
        str(recorded),
        str(returncode),
    ]
    attempts = []
    for _ in range(2):
        result = run_logged_test("example", command, str(tmp_path), {"TEMP": "old"}, 30)
        root, fixture = map(Path, json.loads(recorded.read_text()))
        attempts.append(root)
        assert not root.exists()
        assert not fixture.exists()
        assert (
            result.status
            is {
                0: GremlinResultStatus.SURVIVED,
                1: GremlinResultStatus.ZAPPED,
                3: GremlinResultStatus.ERROR,
            }[returncode]
        )
    assert attempts[0] != attempts[1]
    assert recorded.is_file()


def test_worker_keeps_result_when_temporary_cleanup_is_busy(tmp_path, monkeypatch):
    import errno
    import shutil
    import tempfile

    from pytest_gremlins.parallel.pool import WorkerResult
    from pytest_gremlins.reporting.results import GremlinResultStatus

    from tests import mutation_worker

    expected = WorkerResult("example", GremlinResultStatus.SURVIVED)

    def busy_cleanup(path, *, onexc):
        # Inject the filesystem error at rmtree's callback boundary. Patching
        # os.rmdir globally exercises different recursive retries on each OS.
        try:
            raise OSError(errno.ENOTEMPTY, "child still writes scratch", path)
        except OSError as error:
            onexc(os.rmdir, path, error)

    with monkeypatch.context() as patch:
        patch.setattr(tempfile, "tempdir", str(tmp_path))
        patch.setattr(mutation_worker, "_run_gremlin_test", lambda *args: expected)
        patch.setattr(shutil, "rmtree", busy_cleanup)
        assert mutation_worker.run_logged_test("example", [], str(tmp_path), {}, 30) is expected
        scratch = list(tmp_path.glob("cmw-*"))
        assert len(scratch) == 1
    scratch[0].rmdir()


def test_parallel_worker_entry_point_runs_in_a_spawned_interpreter(tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from pytest_gremlins.reporting.results import GremlinResultStatus

    from tests.mutation_worker import run_logged_test

    # Exercise worker startup/pickling with a plain command, without generating
    # or running any mutations locally.
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        result = pool.submit(
            run_logged_test, "example", [sys.executable, "-c", "pass"], str(tmp_path), {}, 30
        ).result(timeout=60)
    assert result.gremlin_id == "example"
    assert result.status is GremlinResultStatus.SURVIVED


def test_import_probes_use_their_own_pytest_config(tmp_path, monkeypatch):
    import tempfile

    from tests.mutation_runner import _run_import_probes

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (tmp_path / "pytest.ini").write_bytes(
        b"[pytest]\nrequired_plugins = nonexistent-probe-plugin\n"
    )
    selected = checkout / "test_subject.py"
    selected.write_bytes(
        b"import os\n"
        b"if os.environ.get('ACTIVE_GREMLIN') == 'fault':\n"
        b"    raise ValueError('mutant import fault')\n"
    )
    bootstrap = checkout / "bootstrap.py"
    bootstrap.write_bytes(
        b"import os, sys\nfrom pathlib import Path\nimport pytest\n"
        b"os.chdir(Path(__file__).parent)\n"
        b"raise SystemExit(pytest.main(sys.argv[1:]))\n"
    )
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    assert _run_import_probes(bootstrap, [str(selected)], "fault")
