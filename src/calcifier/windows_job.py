"""Windows job ownership established before a suspended process starts running.

Jobs retain descendants after the leader exits. No breakaway flag is enabled.
The suspended primary thread is reopened through documented Toolhelp APIs;
Popen closes the creation thread handle before returning.
"""

from __future__ import annotations

import ctypes as c
import subprocess
from contextlib import suppress
from pathlib import Path
from typing import Any, BinaryIO, cast

CREATE_SUSPENDED = 0x00000004
_HANDLE = c.c_void_p
_DWORD = c.c_uint32
_BOOL = c.c_int32
_windows_ctypes: Any = c  # WinDLL/WinError exist only on Windows; loading stays lazy.


class _BasicLimits(c.Structure):
    _fields_ = [
        ("process_time", c.c_int64),
        ("job_time", c.c_int64),
        ("flags", _DWORD),
        ("minimum_working_set", c.c_size_t),
        ("maximum_working_set", c.c_size_t),
        ("active_process_limit", _DWORD),
        ("affinity", c.c_size_t),
        ("priority_class", _DWORD),
        ("scheduling_class", _DWORD),
    ]


class _ExtendedLimits(c.Structure):
    _fields_ = [
        ("basic", _BasicLimits),
        ("io_counters", c.c_uint64 * 6),
        ("process_memory", c.c_size_t),
        ("job_memory", c.c_size_t),
        ("peak_process_memory", c.c_size_t),
        ("peak_job_memory", c.c_size_t),
    ]


class _ThreadEntry(c.Structure):
    _fields_ = [
        ("size", _DWORD),
        ("usage", _DWORD),
        ("thread_id", _DWORD),
        ("owner_pid", _DWORD),
        ("base_priority", c.c_int32),
        ("delta_priority", c.c_int32),
        ("flags", _DWORD),
    ]


def _kernel() -> Any:
    library = _windows_ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([_HANDLE, c.c_wchar_p], _HANDLE),
        "SetInformationJobObject": ([_HANDLE, c.c_int32, _HANDLE, _DWORD], _BOOL),
        "AssignProcessToJobObject": ([_HANDLE, _HANDLE], _BOOL),
        "TerminateJobObject": ([_HANDLE, _DWORD], _BOOL),
        "OpenProcess": ([_DWORD, _BOOL, _DWORD], _HANDLE),
        "CreateToolhelp32Snapshot": ([_DWORD, _DWORD], _HANDLE),
        "Thread32First": ([_HANDLE, c.POINTER(_ThreadEntry)], _BOOL),
        "Thread32Next": ([_HANDLE, c.POINTER(_ThreadEntry)], _BOOL),
        "OpenThread": ([_DWORD, _BOOL, _DWORD], _HANDLE),
        "ResumeThread": ([_HANDLE], _DWORD),
        "CloseHandle": ([_HANDLE], _BOOL),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(library, name)
        function.argtypes = arguments
        function.restype = result
    return library


def _error(operation: str) -> OSError:
    return OSError(f"{operation}: {_windows_ctypes.WinError()}")


class WindowsJob:
    """Own a native process tree without changing command arguments or stdio."""

    def __init__(self) -> None:
        self.api = _kernel()
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise _error("CreateJobObjectW")
        limits = _ExtendedLimits()
        limits.basic.flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            error = _error("SetInformationJobObject")
            self.close()
            raise error

    def attach_and_resume(self, pid: int) -> None:
        process = self.api.OpenProcess(0x0101, False, pid)  # SET_QUOTA | TERMINATE
        if not process:
            raise _error("OpenProcess")
        try:
            if not self.api.AssignProcessToJobObject(self.handle, process):
                raise _error("AssignProcessToJobObject")
        finally:
            self._close_handle(process)
        self._resume(pid)

    def _resume(self, pid: int) -> None:
        snapshot = self.api.CreateToolhelp32Snapshot(0x00000004, 0)  # SNAPTHREAD
        if snapshot == _HANDLE(-1).value:
            raise _error("CreateToolhelp32Snapshot")
        try:
            entry = _ThreadEntry()
            entry.size = c.sizeof(entry)
            available = self.api.Thread32First(snapshot, c.byref(entry))
            while available:
                if entry.owner_pid == pid:
                    self._resume_thread(entry.thread_id)
                    return
                entry.size = c.sizeof(entry)
                available = self.api.Thread32Next(snapshot, c.byref(entry))
            raise OSError("suspended process has no discoverable primary thread")
        finally:
            self._close_handle(snapshot)

    def _resume_thread(self, thread_id: int) -> None:
        thread = self.api.OpenThread(0x0002, False, thread_id)  # SUSPEND_RESUME
        if not thread:
            raise _error("OpenThread")
        try:
            # Exactly one suspension was requested at process creation.
            if self.api.ResumeThread(thread) != 1:
                raise _error("ResumeThread")
        finally:
            self._close_handle(thread)

    def _close_handle(self, handle: int) -> None:
        if not self.api.CloseHandle(handle):
            raise _error("CloseHandle")

    def close(self) -> None:
        if self.handle is not None:
            handle, self.handle = self.handle, None
            try:
                if not self.api.TerminateJobObject(handle, 1):
                    raise _error("TerminateJobObject")
            finally:
                # Kill-on-close also covers host exit and termination failures.
                self._close_handle(handle)


class _JobProcess(subprocess.Popen[bytes]):
    _calcifier_job: WindowsJob


def start_process(
    argv: list[str] | tuple[str, ...],
    *,
    stdin: bytes | None,
    cwd: str | Path | None,
    env: dict[str, str] | None,
    stdout: int | BinaryIO = subprocess.PIPE,
    stderr: int | BinaryIO = subprocess.PIPE,
) -> subprocess.Popen[bytes]:
    job = WindowsJob()
    process = None
    try:
        process = _JobProcess(
            list(argv),
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            cwd=cwd,
            env=env,
            creationflags=CREATE_SUSPENDED,
        )
        # Keep ownership on the exact Popen object through pipe collection.
        process._calcifier_job = job
        job.attach_and_resume(process.pid)
        return process
    except BaseException:
        try:
            job.close()
        finally:
            if process is not None:
                with suppress(OSError):
                    process.kill()
                try:
                    process.wait(timeout=2)
                finally:
                    for stream in (process.stdin, process.stdout, process.stderr):
                        if stream is not None:
                            stream.close()
        raise


def close_process(process: subprocess.Popen[bytes]) -> None:
    cast(_JobProcess, process)._calcifier_job.close()
