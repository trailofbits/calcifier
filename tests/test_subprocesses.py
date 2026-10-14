"""Tests for bounded process-group execution."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from math import inf, nan
from pathlib import Path
from typing import cast

import pytest

from calcifier import subprocesses
from calcifier.subprocesses import (
    MAX_TIMEOUT_SECONDS,
    ProcessOutputLimitError,
    ProcessTimeoutError,
    run_process,
)


def test_run_process_captures_output_and_stderr_tail():
    result = run_process(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'ok\\n'); sys.stderr.buffer.write(b'detail\\n')",
        ],
        timeout_seconds=10,
        max_output_bytes=1024,
        stderr_tail_bytes=4,
    )

    assert result.returncode == 0
    assert result.stdout == b"ok\n"
    assert result.stderr_tail == b"ail\n"


def test_run_process_preserves_one_byte_stdout_and_stderr_limits():
    result = run_process(
        [sys.executable, "-c", "import os; os.write(1, b'o'); os.write(2, b'e')"],
        timeout_seconds=10,
        max_output_bytes=1,
        stderr_tail_bytes=1,
    )

    assert (result.returncode, result.stdout, result.stderr_tail) == (0, b"o", b"e")


@pytest.mark.parametrize(("tail", "expected"), [(False, b"abcd"), (True, b"ghij")])
def test_chunked_output_retains_the_requested_prefix_or_tail(tail, expected):
    capture = subprocesses._Capture(limit=9, retain=4, tail=tail)
    for chunk in (b"abc", b"def", b"ghi"):
        capture.consume(chunk)
    assert capture.total == 9
    assert not capture.exceeded.is_set()
    capture.consume(b"j")
    assert capture.total == 10
    assert capture.exceeded.is_set()
    assert bytes(capture.data) == expected


@pytest.mark.parametrize("argv", [[], [""], ["command", ""], ["command", 1]])
def test_run_process_rejects_invalid_arguments_before_start(monkeypatch, argv):
    def unexpected_start(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid arguments must be rejected before starting a process")

    monkeypatch.setattr(subprocesses, "_start_process", unexpected_start)
    with pytest.raises(ValueError, match="argv must contain non-empty strings"):
        run_process(argv, timeout_seconds=1, max_output_bytes=100)


@pytest.mark.parametrize(("output_limit", "tail_limit"), [(0, 1), (1, 0)])
def test_run_process_rejects_zero_limits_before_start(monkeypatch, output_limit, tail_limit):
    def unexpected_start(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid limits must be rejected before starting a process")

    monkeypatch.setattr(subprocesses, "_start_process", unexpected_start)
    with pytest.raises(ValueError, match="process limits must be positive"):
        run_process(
            [sys.executable, "-c", "pass"],
            timeout_seconds=1,
            max_output_bytes=output_limit,
            stderr_tail_bytes=tail_limit,
        )


def test_run_process_enforces_output_limit():
    with pytest.raises(ProcessOutputLimitError):
        run_process(
            [sys.executable, "-c", "print('x' * 10000)"],
            timeout_seconds=10,
            max_output_bytes=100,
        )


def test_run_process_enforces_timeout():
    with pytest.raises(ProcessTimeoutError):
        run_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            timeout_seconds=0.05,
            max_output_bytes=1024,
        )


def test_wait_bounded_counts_suspend_time_when_monotonic_pauses(monkeypatch):
    class SuspendedProcess:
        def poll(self) -> None:
            return None

        def wait(self, *, timeout: float) -> None:
            del timeout
            raise AssertionError("deadline should expire before process.wait()")

    wall_times = iter((1_000.0, 15_010.0))
    monkeypatch.setattr("calcifier.clock._CONTINUOUS_CLOCK", None)
    monkeypatch.setattr("calcifier.clock.time.monotonic", lambda: 100.0)
    monkeypatch.setattr("calcifier.clock.time.time", lambda: next(wall_times))

    failure = subprocesses._wait_bounded(
        cast("subprocess.Popen[bytes]", SuspendedProcess()),
        subprocesses._Capture(limit=1024),
        subprocesses._Capture(limit=1024),
        [],
        timeout_seconds=1_200,
        max_output_bytes=1024,
    )

    assert isinstance(failure, ProcessTimeoutError)
    assert str(failure) == "process timed out after 1200s"


@pytest.mark.parametrize("remaining", [0.0, 0.5])
def test_wait_bounded_distinguishes_expired_and_positive_deadlines(monkeypatch, remaining):
    class FixedTimer:
        def elapsed(self) -> float:
            return 1.0 - remaining

    class CompletingProcess:
        def __init__(self) -> None:
            self.finished = False
            self.waits: list[float] = []

        def poll(self) -> int | None:
            return 0 if self.finished else None

        def wait(self, *, timeout: float) -> int:
            assert remaining > 0, "an expired deadline must not wait for a running process"
            self.waits.append(timeout)
            self.finished = True
            return 0

    monkeypatch.setattr(subprocesses.SuspendAwareTimer, "start", FixedTimer)
    process = CompletingProcess()
    failure = subprocesses._wait_bounded(
        cast("subprocess.Popen[bytes]", process),
        subprocesses._Capture(limit=1),
        subprocesses._Capture(limit=1),
        [],
        timeout_seconds=1,
        max_output_bytes=1,
    )

    if remaining == 0:
        assert isinstance(failure, ProcessTimeoutError)
        assert process.waits == []
    else:
        assert failure is None
        assert len(process.waits) == 1
        assert 0 < process.waits[0] <= remaining


@pytest.mark.parametrize("timeout", [inf, nan, MAX_TIMEOUT_SECONDS + 1])
def test_run_process_rejects_unbounded_timeout(timeout: float):
    with pytest.raises(ValueError, match="timeout_seconds must be finite"):
        run_process(
            [sys.executable, "-c", "pass"],
            timeout_seconds=timeout,
            max_output_bytes=1024,
        )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_run_process_timeout_escalates_for_descendant_ignoring_sigterm(
    tmp_path: Path,
    monkeypatch,
):
    pid_path = tmp_path / "child.pid"
    child = (
        "import os, pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(10)"
    )
    parent = (
        "import pathlib, subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        f"pid_path=pathlib.Path({str(pid_path)!r}); "
        "deadline=time.monotonic() + 2; "
        "exec('while not pid_path.exists() and time.monotonic() < deadline:\\n time.sleep(0.01)'); "
        "time.sleep(10)"
    )
    signals: list[int] = []
    real_killpg = subprocesses.os.killpg
    real_wait = subprocesses._wait_bounded

    def wait_after_child_is_ready(*args, **kwargs):  # noqa: ANN002, ANN003
        # Test escalation, not whether two Python interpreters start within
        # 200 ms under coverage/mutation load. Begin the real timeout only
        # after the child has installed its SIGTERM handler.
        deadline = time.monotonic() + 10
        while not pid_path.exists():
            if time.monotonic() >= deadline:
                pytest.fail("descendant did not install its signal handler")
            time.sleep(0.01)
        return real_wait(*args, **kwargs)

    def recorded_killpg(pgid: int, sig: int) -> None:
        signals.append(sig)
        real_killpg(pgid, sig)

    monkeypatch.setattr(subprocesses, "_TERMINATION_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(subprocesses.os, "killpg", recorded_killpg)
    monkeypatch.setattr(subprocesses, "_wait_bounded", wait_after_child_is_ready)

    with pytest.raises(ProcessTimeoutError):
        run_process(
            [sys.executable, "-c", parent],
            timeout_seconds=0.2,
            max_output_bytes=1024,
        )

    assert subprocesses.signal.SIGKILL in signals
    assert int(pid_path.read_text()) > 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_run_process_drains_descendant_output_after_group_leader_exits():
    child = "import time; time.sleep(1.1); print('late output')"
    parent = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}], stderr=subprocess.DEVNULL)"
    )

    result = run_process(
        [sys.executable, "-c", parent],
        # This asserts pipe draining, not interpreter startup latency under
        # concurrent mutation/coverage load. Timeout enforcement is tested above.
        timeout_seconds=10,
        max_output_bytes=1024,
    )

    assert result.stdout == b"late output\n"


def test_run_process_interrupt_terminates_process(monkeypatch):
    started: list[subprocess.Popen[bytes]] = []
    real_start = subprocesses._start_process

    def capture_start(*args, **kwargs):  # noqa: ANN002, ANN003
        process = real_start(*args, **kwargs)
        started.append(process)
        return process

    def interrupt(*args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocesses, "_start_process", capture_start)
    monkeypatch.setattr(subprocesses, "_wait_bounded", interrupt)

    with pytest.raises(KeyboardInterrupt):
        run_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            timeout_seconds=2,
            max_output_bytes=1024,
        )

    assert len(started) == 1
    assert started[0].poll() is not None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX detached session")
def test_timeout_cleanup_does_not_wait_for_detached_pipe_holder(tmp_path: Path, monkeypatch):
    ready = tmp_path / "child.pid"
    child = (
        "import os,time,pathlib; "
        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(15)"
    )
    parent = (
        "import subprocess,sys; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}], start_new_session=True)"
    )
    real_wait = subprocesses._wait_bounded
    elapsed = []

    def wait_ready(*args, **kwargs):
        deadline = time.monotonic() + 10
        while not ready.exists():
            if time.monotonic() >= deadline:
                pytest.fail("detached child did not start")
            time.sleep(0.01)
        elapsed.append(time.monotonic())
        return real_wait(*args, **kwargs)

    monkeypatch.setattr(subprocesses, "_wait_bounded", wait_ready)
    try:
        with pytest.raises(ProcessTimeoutError):
            run_process([sys.executable, "-c", parent], timeout_seconds=0.1, max_output_bytes=1024)
        assert time.monotonic() - elapsed[0] < 3
    finally:
        if ready.exists():
            os.kill(int(ready.read_text()), subprocesses.signal.SIGKILL)


def test_timeout_cleanup_stops_a_backpressured_stdin_writer():
    started = time.monotonic()
    with pytest.raises(ProcessTimeoutError):
        run_process(
            [sys.executable, "-c", "import time; time.sleep(15)"],
            stdin=b"x" * (2 * 1024 * 1024),
            timeout_seconds=0.1,
            max_output_bytes=1024,
        )
    assert time.monotonic() - started < 5


def test_run_process_waits_for_leader_after_both_pipes_close():
    result = run_process(
        [sys.executable, "-c", "import os, time; os.close(1); os.close(2); time.sleep(0.2)"],
        timeout_seconds=10,
        max_output_bytes=1024,
    )

    assert (result.returncode, result.stdout, result.stderr_tail) == (0, b"", b"")


def test_run_process_reads_without_poll_support(monkeypatch):
    """Windows pipes lack poll(2); readers fall back to sleeping between reads."""
    monkeypatch.delattr(subprocesses.select, "poll", raising=False)

    payload = b"x" * (512 * 1024)  # larger than any pipe buffer: the writer must retry
    script = "import sys, time; time.sleep(0.1); print(len(sys.stdin.buffer.read()), flush=True)"

    result = run_process(
        [sys.executable, "-c", script],
        stdin=payload,
        timeout_seconds=10,
        max_output_bytes=1024,
    )

    assert result.stdout.strip() == str(len(payload)).encode()
