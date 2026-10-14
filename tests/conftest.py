"""Shared test isolation."""

from __future__ import annotations

import hashlib
import re
import shutil
import socket
import subprocess
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import pytest
from hypothesis import settings

from calcifier.agent_providers import ClaudeCli, CodexCli, CommandCli

settings.register_profile(
    "calcifier_deterministic",
    derandomize=True,
    deadline=None,
    max_examples=250,
)
settings.load_profile("calcifier_deterministic")


def pytest_make_parametrize_id(val: object) -> str | None:
    """Keep source snippets out of coverage-context delimiters and test selectors."""
    if not isinstance(val, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", val):
        return None
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", val).strip("-")[:48]
    digest = hashlib.sha256(val.encode()).hexdigest()[:12]
    return f"{readable}-{digest}"


def _mutation_runner_script() -> str:
    """Use real pytest, including fixture setup and parametrization, for every mutant."""
    runner = str(Path(__file__).with_name("mutation_runner.py"))
    return f"""from pathlib import Path
import os
import runpy
import sys

# Keep process-group signals in tests away from the CI runner and other mutants.
if os.name == "posix" and os.getpgrp() != os.getpid():
    os.setsid()

sys.argv = [
    {runner!r},
    str(Path(__file__).with_name("gremlin_bootstrap.py")),
    *sys.argv[1:],
]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


# pytest-gremlins (through 1.9.0) collects its per-test coverage map by running the
# whole selected suite once under coverage, with a hard-coded 120-second timeout.
# When the suite outgrows that, the map comes back empty and every mutant silently
# runs the entire suite. Upstream counts timeouts as kills; our threshold does not.
_COVERAGE_BOOTSTRAP = "pytest_gremlins.coverage.subprocess_bootstrap"
_COVERAGE_COLLECTION_TIMEOUT_SECONDS = 1800
# Import-time mutations can select the entire suite (~150s before instrumentation).
# Give real pytest room to finish; the threshold never treats a timeout as a kill.
_MUTATION_TIMEOUT_SECONDS = 600


def _mutation_worker_pool(**kwargs: Any) -> Any:
    from pytest_gremlins.parallel.pool import WorkerPool

    kwargs["timeout"] = max(kwargs.get("timeout", 30), _MUTATION_TIMEOUT_SECONDS)
    return WorkerPool(**kwargs)


class _CoverageCollectionSubprocess:
    """``subprocess`` for the gremlins plugin, with room for coverage collection."""

    def __getattr__(self, name: str) -> Any:
        return getattr(subprocess, name)

    @staticmethod
    def run(command: Any, *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if _COVERAGE_BOOTSTRAP in command and kwargs.get("timeout") is not None:
            kwargs["timeout"] = max(kwargs["timeout"], _COVERAGE_COLLECTION_TIMEOUT_SECONDS)
        elif (kwargs.get("env") or {}).get("ACTIVE_GREMLIN") and kwargs.get("timeout") is not None:
            kwargs["timeout"] = max(kwargs["timeout"], _MUTATION_TIMEOUT_SECONDS)
        return subprocess.run(command, *args, **kwargs)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("gremlins", default=False):
        import pytest_gremlins.parallel.pool as pool
        import pytest_gremlins.plugin as gremlins

        from tests.mutation_sources import write_sources
        from tests.mutation_worker import run_logged_test

        # The upstream shortcut calls test functions without pytest, treating
        # missing fixtures/parameterized names as kills. Run actual tests and
        # physical instrumented packages, including in native child processes.
        patch = pytest.MonkeyPatch()
        # Coverage's sysmon core can retain only the first test context for a
        # line on Python 3.14. Mutation selection needs every covering test.
        patch.setenv("COVERAGE_CORE", "ctrace")
        patch.setattr(gremlins, "_get_lightweight_runner_script", _mutation_runner_script)
        patch.setattr(gremlins, "_write_instrumented_sources", write_sources)
        patch.setattr(gremlins, "subprocess", _CoverageCollectionSubprocess())
        patch.setattr(gremlins, "WorkerPool", _mutation_worker_pool)
        patch.setattr(pool, "_run_gremlin_test", run_logged_test)
        config.add_cleanup(patch.undo)


@pytest.fixture(autouse=True)
def _isolated_trust_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALCIFIER_TRUST_STORE", str(tmp_path / "user-config" / "trust.json"))


@pytest.fixture(scope="session", autouse=True)
def _destination_ruff_config(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Give every temporary destination one known Ruff configuration.

    Ruff reads the configuration governing each recipe path, so without this the
    results would depend on Ruff's defaults and on any user-level Ruff config.
    A test's own closer configuration still takes precedence.
    """
    config = tmp_path_factory.getbasetemp() / "ruff.toml"
    config.write_text("line-length = 100\n", encoding="utf-8")


@pytest.fixture(autouse=True)
def _offline_semantic_compiler_tests(
    _offline_test_processes: None,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed if a semantic compiler test reaches a real provider/network."""
    if request.node.get_closest_marker("semantic_offline") is None:
        return

    def blocked(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("semantic compiler tests must not invoke a real provider or network")

    for provider in (
        ClaudeCli,
        CodexCli,
        CommandCli,
    ):
        monkeypatch.setattr(provider, "invoke", blocked)
    monkeypatch.setattr(socket, "socket", blocked)


@pytest.fixture(autouse=True)
def _offline_test_processes(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Block actual provider executables and network connections throughout the suite."""
    popen = subprocess.Popen
    fixture_root = tmp_path_factory.getbasetemp().resolve()
    socketpair = socket.socketpair
    connect = socket.socket.connect
    creating_socketpair = ContextVar("creating_socketpair", default=False)

    def guarded(command: Any, *args: Any, **kwargs: Any) -> Any:
        executable = command[0] if isinstance(command, list | tuple) and command else command
        if isinstance(executable, str | Path) and Path(executable).name.casefold() in {
            "claude",
            "codex",
            "claude.exe",
            "codex.exe",
        }:
            resolved = Path(shutil.which(str(executable)) or executable).resolve()
            # Preflight tests install neutral fake CLIs in pytest's private workspace.
            if not resolved.is_relative_to(fixture_root):
                raise AssertionError("offline tests must not start live Claude/Codex CLIs")
        return popen(command, *args, **kwargs)

    def no_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("offline tests must not connect to the network")

    def guarded_connect(*args: Any, **kwargs: Any) -> Any:
        if creating_socketpair.get():
            return connect(*args, **kwargs)
        return no_network(*args, **kwargs)

    def local_socketpair(*args: Any, **kwargs: Any) -> Any:
        # Windows implements this local IPC primitive using a loopback TCP connection.
        # Keep the exception scoped to this call and context, including worker threads.
        token = creating_socketpair.set(True)
        try:
            return socketpair(*args, **kwargs)
        finally:
            creating_socketpair.reset(token)

    monkeypatch.setattr(subprocess, "Popen", guarded)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "socketpair", local_socketpair)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
