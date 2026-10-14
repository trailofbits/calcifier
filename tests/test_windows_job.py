"""Atomic Windows job setup, failure cleanup, and native descendant lifetimes."""

from __future__ import annotations

import ctypes
import io
import os
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from calcifier import subprocesses, windows_job


@pytest.fixture
def kernel(monkeypatch):
    calls = []
    api = SimpleNamespace()
    threads = iter(((999, 9), (100, 10)))

    def enumerate_thread(snapshot, entry):
        assert snapshot == 33
        value = next(threads, None)
        if value is None:
            return 0
        assert entry._obj.size == 28
        entry._obj.owner_pid, entry._obj.thread_id = value
        return 1

    def function(name, result):
        def invoke(*args):
            calls.append((name, args))
            return result(*args) if callable(result) else result

        return Mock(side_effect=invoke)

    for name, result in {
        "CreateJobObjectW": 11,
        "SetInformationJobObject": 1,
        "AssignProcessToJobObject": 1,
        "TerminateJobObject": 1,
        "OpenProcess": 22,
        "CreateToolhelp32Snapshot": 33,
        "Thread32First": enumerate_thread,
        "Thread32Next": enumerate_thread,
        "OpenThread": 44,
        "ResumeThread": 1,
        "CloseHandle": 1,
    }.items():
        setattr(api, name, function(name, result))
    monkeypatch.setattr(ctypes, "WinDLL", Mock(return_value=api), raising=False)
    monkeypatch.setattr(ctypes, "WinError", lambda: OSError("native failure"), raising=False)
    return api, calls


def test_job_is_configured_and_assigned_before_suspended_thread_resumes(kernel):
    api, calls = kernel
    job = windows_job.WindowsJob()
    job.attach_and_resume(100)
    job.close()
    job.close()
    names = [name for name, _ in calls]
    assert names.index("SetInformationJobObject") < names.index("AssignProcessToJobObject")
    assert names.index("AssignProcessToJobObject") < names.index("ResumeThread")
    limits = api.SetInformationJobObject.call_args.args[2]._obj
    assert limits.basic.flags == 0x2000
    assert api.SetInformationJobObject.call_args.args[1] == 9
    assert ctypes.sizeof(limits) == (144 if ctypes.sizeof(ctypes.c_void_p) == 8 else 112)
    api.OpenProcess.assert_called_once_with(0x101, False, 100)
    api.OpenThread.assert_called_once_with(2, False, 10)
    api.CreateToolhelp32Snapshot.assert_called_once_with(4, 0)
    assert [args[0] for name, args in calls if name == "CloseHandle"] == [22, 44, 33, 11]
    api.TerminateJobObject.assert_called_once_with(11, 1)
    assert api.CreateJobObjectW.restype is ctypes.c_void_p


@pytest.mark.parametrize(
    ("operation", "failure"),
    [
        ("CreateJobObjectW", 0),
        ("SetInformationJobObject", 0),
        ("OpenProcess", 0),
        ("AssignProcessToJobObject", 0),
        ("CreateToolhelp32Snapshot", ctypes.c_void_p(-1).value),
        ("Thread32First", 0),
        ("Thread32Next", 0),
        ("OpenThread", 0),
        ("ResumeThread", 0xFFFFFFFF),
        ("ResumeThread", 0),
        ("ResumeThread", 2),
        ("CloseHandle", 0),
    ],
)
def test_setup_fails_closed_at_each_native_failure(kernel, operation, failure):
    api, _ = kernel
    getattr(api, operation).side_effect = lambda *args: failure
    job = None
    try:
        with pytest.raises(OSError):
            job = windows_job.WindowsJob()
            job.attach_and_resume(100)
        if operation not in {"ResumeThread", "CloseHandle"}:
            api.ResumeThread.assert_not_called()
    finally:
        api.CloseHandle.side_effect = None
        api.CloseHandle.return_value = 1
        if job is not None:
            job.close()


def test_termination_failure_still_closes_kill_on_close_job(kernel):
    api, _ = kernel
    job = windows_job.WindowsJob()
    api.TerminateJobObject.side_effect = lambda *args: 0
    with pytest.raises(OSError, match="TerminateJobObject"):
        job.close()
    api.CloseHandle.assert_called_once_with(11)


@pytest.mark.parametrize("stage", ["spawn", "attach", "resume", "success"])
def test_process_start_preserves_arguments_and_cleans_failed_setup(kernel, monkeypatch, stage):
    api, _ = kernel
    process = Mock(pid=100, stdin=io.BytesIO(), stdout=io.BytesIO(), stderr=io.BytesIO())
    spawn = Mock(return_value=process)
    if stage == "spawn":
        spawn.side_effect = OSError("cannot spawn")
    elif stage == "attach":
        api.AssignProcessToJobObject.side_effect = lambda *args: 0
    elif stage == "resume":
        api.ResumeThread.side_effect = KeyboardInterrupt
    monkeypatch.setattr(windows_job, "_JobProcess", spawn)
    if stage == "success":
        result = windows_job.start_process(
            ["command", "a b"], stdin=b"input", cwd="working directory", env={"key": "value"}
        )
        assert result is process
        process.kill.assert_not_called()
        windows_job.close_process(result)
    else:
        with pytest.raises((OSError, KeyboardInterrupt)):
            windows_job.start_process(
                ["command", "a b"], stdin=b"input", cwd="working directory", env={"key": "value"}
            )
        if stage != "spawn":
            process.kill.assert_called_once()
            process.wait.assert_called_once_with(timeout=2)
            assert all(stream.closed for stream in (process.stdin, process.stdout, process.stderr))
    assert spawn.call_args.args == (["command", "a b"],)
    assert spawn.call_args.kwargs == {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "cwd": "working directory",
        "env": {"key": "value"},
        "creationflags": 4,
    }
    api.TerminateJobObject.assert_called_once_with(11, 1)


@pytest.mark.skipif(os.name != "nt", reason="native Windows Job Objects")
@pytest.mark.parametrize("mode", ["timeout", "leader-exit", "output-limit"])
def test_native_job_terminates_descendants_after_leader_exit(tmp_path, monkeypatch, mode):
    windows_ctypes: Any = ctypes
    api = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    for name, arguments, result in (
        ("OpenProcess", [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32], ctypes.c_void_p),
        ("WaitForSingleObject", [ctypes.c_void_p, ctypes.c_uint32], ctypes.c_uint32),
        ("TerminateProcess", [ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int32),
        ("CloseHandle", [ctypes.c_void_p], ctypes.c_int32),
    ):
        getattr(api, name).argtypes = arguments
        getattr(api, name).restype = result
    paths = [tmp_path / "child.pid", tmp_path / "grandchild.pid"]
    output = "print('x' * 10000, flush=True); " if mode == "output-limit" else ""
    grandchild = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(paths[1])!r}).write_text(str(os.getpid())); "
        f"{output}time.sleep(120)"
    )
    child = (
        "import os,pathlib,subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
        f"pathlib.Path({str(paths[0])!r}).write_text(str(os.getpid())); time.sleep(120)"
    )
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); "
    parent += "pass" if mode == "leader-exit" else "time.sleep(120)"
    handles = []
    real_wait = subprocesses._wait_bounded

    def wait_ready(*args, **kwargs):
        deadline = time.monotonic() + 30
        for path in paths:
            while not path.exists() or not path.stat().st_size:
                if time.monotonic() >= deadline:
                    pytest.fail("Windows descendant did not start")
                time.sleep(0.01)
            handle = api.OpenProcess(0x100001, False, int(path.read_text()))
            assert handle, windows_ctypes.WinError()
            handles.append(handle)
        return real_wait(*args, **kwargs)

    monkeypatch.setattr(subprocesses, "_wait_bounded", wait_ready)
    expected = (
        subprocesses.ProcessOutputLimitError
        if mode == "output-limit"
        else subprocesses.ProcessTimeoutError
    )
    try:
        with pytest.raises(expected):
            subprocesses.run_process(
                [sys.executable, "-c", parent], timeout_seconds=0.2, max_output_bytes=1024
            )
        assert len(handles) == 2
        assert all(api.WaitForSingleObject(handle, 5000) == 0 for handle in handles)
    finally:
        for handle in handles:
            api.TerminateProcess(handle, 99)
            api.CloseHandle(handle)


def test_windows_runner_selects_owned_launch_and_reports_cleanup_failure(monkeypatch):
    process = Mock()
    start = Mock(return_value=process)
    monkeypatch.setattr(subprocesses, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(windows_job, "start_process", start)
    result = subprocesses._start_process(["tool"], stdin=None, cwd=None, env=None)
    assert result is process
    start.assert_called_once_with(["tool"], stdin=None, cwd=None, env=None)
    close = Mock(side_effect=OSError("native cleanup failure"))
    monkeypatch.setattr(windows_job, "close_process", close)
    with pytest.raises(subprocesses.ProcessError, match="Windows process tree"):
        subprocesses._terminate_process_group(process)
    close.assert_called_once_with(process)


def test_missing_standard_input_uses_devnull(kernel, monkeypatch):
    process = Mock(pid=100)
    spawn = Mock(return_value=process)
    monkeypatch.setattr(windows_job, "_JobProcess", spawn)
    result = windows_job.start_process(["tool"], stdin=None, cwd=None, env=None)
    assert spawn.call_args.kwargs["stdin"] == subprocess.DEVNULL
    windows_job.close_process(result)


@pytest.mark.parametrize("setup_fails", [False, True])
def test_redirected_log_stays_owned_by_caller(kernel, monkeypatch, tmp_path, setup_fails):
    api, _ = kernel
    if setup_fails:
        api.AssignProcessToJobObject.side_effect = lambda *args: 0
    process = Mock(pid=100, stdin=None, stdout=None, stderr=None)
    spawn = Mock(return_value=process)
    monkeypatch.setattr(windows_job, "_JobProcess", spawn)
    path = tmp_path / "worker.log"
    with path.open("wb") as log:
        kwargs = {"stdin": None, "cwd": tmp_path, "env": None, "stdout": log, "stderr": log}
        if setup_fails:
            with pytest.raises(OSError, match="AssignProcessToJobObject"):
                windows_job.start_process(["worker"], **kwargs)
            process.kill.assert_called_once()
            process.wait.assert_called_once_with(timeout=2)
        else:
            result = windows_job.start_process(["worker"], **kwargs)
            windows_job.close_process(result)
        assert spawn.call_args.kwargs["stdout"] is log
        assert spawn.call_args.kwargs["stderr"] is log
        assert not log.closed
        log.write(b"caller still owns log\n")
    assert path.read_bytes() == b"caller still owns log\n"
