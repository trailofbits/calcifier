"""Bounded process-group execution shared by runtime subprocess surfaces."""

from __future__ import annotations

import os
import select
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .clock import SuspendAwareTimer
from .exceptions import CalcifierError
from .validation import is_positive_number

MAX_TIMEOUT_SECONDS = 86_400.0
_TERMINATION_GRACE_SECONDS = 2.0
# Pipe workers wake on readiness/EOF via poll(2) where available; the timeout
# only bounds how long a worker waits before rechecking ``stop`` (a descendant
# can keep an inherited pipe open after the leader exits). Windows pipes are not
# pollable, so workers there sleep between non-blocking attempts instead.
_PIPE_POLL_MILLISECONDS = 50
_PIPE_SLEEP_SECONDS = 0.01


class ProcessError(CalcifierError, RuntimeError):
    """Base class for bounded subprocess failures."""

    def __init__(
        self,
        message: str,
        *,
        stdout: bytes = b"",
        stderr_tail: bytes = b"",
    ) -> None:
        self.stdout = stdout
        self.stderr_tail = stderr_tail
        super().__init__(message)


class ProcessStartError(ProcessError):
    """Raised when a process cannot be launched."""


class ProcessTimeoutError(ProcessError):
    """Raised after terminating a timed-out process group."""


class ProcessOutputLimitError(ProcessError):
    """Raised after terminating a process group that exceeded an output limit."""


@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr_tail: bytes


class _Capture:
    def __init__(
        self,
        *,
        limit: int,
        tail: bool = False,
        retain: int | None = None,
        consumer: Callable[[bytes], None] | None = None,
    ) -> None:
        self.limit = limit
        self.tail = tail
        self.retain = limit if retain is None else min(retain, limit)
        self.total = 0
        self.data = bytearray()
        self.exceeded = threading.Event()
        self.consumer = consumer
        self.failure: ProcessError | None = None

    def consume(self, chunk: bytes) -> None:
        self.total += len(chunk)
        if self.consumer is None and self.total > self.limit:
            self.exceeded.set()
        if self.consumer is not None and self.failure is None:
            try:
                self.consumer(chunk)
            except ProcessError as exc:
                self.failure = exc
            except Exception as exc:  # noqa: BLE001 - wake the supervisor on decoder failures
                self.failure = ProcessError(f"stdout consumer failed: {type(exc).__name__}")
        if self.tail:
            self.data.extend(chunk)
            if len(self.data) > self.retain:
                del self.data[: len(self.data) - self.retain]
            return
        remaining = self.retain - len(self.data)
        if remaining > 0:
            self.data.extend(chunk[:remaining])


def run_process(
    argv: list[str] | tuple[str, ...],
    *,
    stdin: bytes | None = None,
    timeout_seconds: float,
    max_output_bytes: int,
    stderr_tail_bytes: int = 4096,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    stdout_consumer: Callable[[bytes], None] | None = None,
) -> ProcessResult:
    """Run explicit argv with bounded output and process-group cleanup.

    A stdout consumer owns its stream limits and receives chunks incrementally;
    only a diagnostic tail is captured. Stderr keeps its aggregate limit.
    Consumer failures terminate and reap the same process group as other failures.
    """
    _validate_process_args(argv, timeout_seconds, max_output_bytes, stderr_tail_bytes)
    process = _start_process(argv, stdin=stdin, cwd=cwd, env=env)
    return _collect_process(
        process,
        argv,
        stdin=stdin,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        stderr_tail_bytes=stderr_tail_bytes,
        stdout_consumer=stdout_consumer,
    )


def _validate_process_args(
    argv: list[str] | tuple[str, ...],
    timeout_seconds: float,
    max_output_bytes: int,
    stderr_tail_bytes: int,
) -> None:
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ValueError("argv must contain non-empty strings")
    if not is_positive_number(timeout_seconds, finite=True, maximum=MAX_TIMEOUT_SECONDS):
        raise ValueError(
            f"timeout_seconds must be finite, positive, and at most {MAX_TIMEOUT_SECONDS:g}"
        )
    if max_output_bytes <= 0 or stderr_tail_bytes <= 0:
        raise ValueError("process limits must be positive")


def _start_process(
    argv: list[str] | tuple[str, ...],
    *,
    stdin: bytes | None,
    cwd: str | Path | None,
    env: dict[str, str] | None,
) -> subprocess.Popen[bytes]:
    try:
        if os.name == "nt":
            from .windows_job import start_process

            return start_process(argv, stdin=stdin, cwd=cwd, env=env)
        return subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        raise ProcessStartError(f"could not start {argv[0]!r}: {exc}") from exc


def _collect_process(
    process: subprocess.Popen[bytes],
    argv: list[str] | tuple[str, ...],
    *,
    stdin: bytes | None,
    timeout_seconds: float,
    max_output_bytes: int,
    stderr_tail_bytes: int,
    stdout_consumer: Callable[[bytes], None] | None = None,
) -> ProcessResult:
    stdout_capture = _Capture(
        limit=max_output_bytes,
        tail=stdout_consumer is not None,
        retain=stderr_tail_bytes if stdout_consumer is not None else None,
        consumer=stdout_consumer,
    )
    stderr_capture = _Capture(
        limit=max_output_bytes,
        tail=True,
        retain=stderr_tail_bytes,
    )
    stop = threading.Event()
    readers = [
        threading.Thread(
            target=_read_stream,
            args=(process.stdout, stdout_capture, stop),
            daemon=True,
        ),
        threading.Thread(
            target=_read_stream,
            args=(process.stderr, stderr_capture, stop),
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()
    writer = None
    if stdin is not None:
        writer = threading.Thread(
            target=_write_stdin,
            args=(process.stdin, stdin, stop),
            daemon=True,
        )
        writer.start()

    try:
        failure = _wait_bounded(
            process,
            stdout_capture,
            stderr_capture,
            readers,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
    finally:
        # This also handles BaseException (notably KeyboardInterrupt). Always
        # address the process group before joining pipe workers: a descendant
        # can outlive the group leader while retaining an inherited pipe.
        try:
            _terminate_process_group(process)
        finally:
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=_TERMINATION_GRACE_SECONDS)
            stop.set()
            if writer is not None:
                writer.join(timeout=_TERMINATION_GRACE_SECONDS)
            for reader in readers:
                reader.join(timeout=_TERMINATION_GRACE_SECONDS)
    if failure is None and stdout_capture.failure is not None:
        failure = stdout_capture.failure
    if failure is None and (stdout_capture.exceeded.is_set() or stderr_capture.exceeded.is_set()):
        failure = _output_limit_error(stdout_capture, stderr_capture, max_output_bytes)
    if failure is not None:
        failure.stdout = bytes(stdout_capture.data)
        failure.stderr_tail = bytes(stderr_capture.data)
        raise failure
    return ProcessResult(
        argv=tuple(argv),
        returncode=process.returncode,
        stdout=bytes(stdout_capture.data),
        stderr_tail=bytes(stderr_capture.data),
    )


def _wait_bounded(
    process: subprocess.Popen[bytes],
    stdout: _Capture,
    stderr: _Capture,
    readers: list[threading.Thread],
    *,
    timeout_seconds: float,
    max_output_bytes: int,
) -> ProcessError | None:
    timer = SuspendAwareTimer.start()
    while True:
        if stdout.failure is not None:
            return stdout.failure
        if stdout.exceeded.is_set() or stderr.exceeded.is_set():
            return _output_limit_error(stdout, stderr, max_output_bytes)
        process_done = process.poll() is not None
        if process_done and all(not reader.is_alive() for reader in readers):
            return None
        remaining = timeout_seconds - timer.elapsed()
        if remaining <= 0:
            return ProcessTimeoutError(f"process timed out after {timeout_seconds:g}s")
        # Pipe EOF usually signals exit first and wakes a reader immediately;
        # Popen.wait(timeout) polls waitpid with exponential sleeps instead.
        alive = next((reader for reader in readers if reader.is_alive()), None)
        if alive is not None:
            alive.join(timeout=min(0.05, remaining))
        else:  # both pipes closed, leader still running
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=min(0.05, remaining))


def _output_limit_error(
    stdout: _Capture,
    stderr: _Capture,
    max_output_bytes: int,
) -> ProcessOutputLimitError:
    stream = "stdout" if stdout.exceeded.is_set() else "stderr"
    return ProcessOutputLimitError(f"process {stream} exceeded {max_output_bytes} bytes")


def _read_stream(stream: BinaryIO | None, capture: _Capture, stop: threading.Event) -> None:
    if stream is None:
        return
    try:
        fd = stream.fileno()
        os.set_blocking(fd, False)
        poller = _pipe_poller(fd, writable=False)
        while not stop.is_set():
            if poller is not None and not poller.poll(_PIPE_POLL_MILLISECONDS):
                continue
            try:
                chunk = os.read(fd, 64 * 1024)
            except BlockingIOError:
                if poller is None:
                    stop.wait(_PIPE_SLEEP_SECONDS)
                continue
            if not chunk:
                break
            capture.consume(chunk)
            if capture.failure is not None:
                break
    finally:
        stream.close()


def _pipe_poller(fd: int, *, writable: bool) -> select.poll | None:
    if not hasattr(select, "poll"):
        return None
    poller = select.poll()
    poller.register(fd, select.POLLOUT if writable else select.POLLIN)
    return poller


def _write_stdin(stream: BinaryIO | None, data: bytes, stop: threading.Event) -> None:
    if stream is None:
        return
    try:
        fd = stream.fileno()
        os.set_blocking(fd, False)
        poller = _pipe_poller(fd, writable=True)
        remaining = memoryview(data)
        while remaining and not stop.is_set():
            if poller is not None and not poller.poll(_PIPE_POLL_MILLISECONDS):
                continue
            try:
                written = os.write(fd, remaining[: 64 * 1024])
            except BlockingIOError:
                if poller is None:
                    stop.wait(_PIPE_SLEEP_SECONDS)
                continue
            remaining = remaining[written:]
    except BrokenPipeError:
        pass
    finally:
        stream.close()


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        _terminate_windows_process(process)
        return

    _signal_process_group(process, signal.SIGTERM)
    if _wait_for_process_group_exit(process, _TERMINATION_GRACE_SECONDS):
        return
    _signal_process_group(process, signal.SIGKILL)
    _wait_for_process_group_exit(process, _TERMINATION_GRACE_SECONDS)
    if process.poll() is None:
        with suppress(OSError):
            process.kill()


def _terminate_windows_process(process: subprocess.Popen[bytes]) -> None:
    from .windows_job import close_process

    try:
        close_process(process)
    except OSError as exc:
        raise ProcessError(f"could not terminate Windows process tree: {exc}") from exc


def _signal_process_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        if process.poll() is not None:
            return
        method = process.terminate if sig is signal.SIGTERM else process.kill
        with suppress(OSError):
            method()


def _wait_for_process_group_exit(
    process: subprocess.Popen[bytes],
    timeout_seconds: float,
) -> bool:
    timer = SuspendAwareTimer.start()
    while True:
        # Reap the leader before probing the group so its zombie does not keep
        # an otherwise-empty process group looking alive.
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        if timer.elapsed() >= timeout_seconds:
            return False
        time.sleep(0.01)
