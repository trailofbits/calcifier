"""Parallel completion observers checkpoint results before slower workers finish."""

from __future__ import annotations

import asyncio
import importlib
import json
import threading
from concurrent.futures import Future

import pytest

from calcifier import parallel, recipe, step, write_json, write_text
from calcifier.trace import DryRunMode


@pytest.mark.parametrize("asynchronous", [False, True])
def test_completion_callback_is_serial_and_preserves_return_order(tmp_path, asynchronous):
    saved = threading.Event()
    owner = threading.get_ident()
    observed = []

    @step
    def worker(item: int) -> int:
        if item == 0:
            assert saved.wait(5), "first result was not checkpointed during the other worker"
        return item * 10

    @step
    def checkpoint(item: int, result: int) -> None:
        assert threading.get_ident() == owner
        write_json(f"{item}.json", result)
        observed.append(item)
        saved.set()

    @step
    async def async_checkpoint(item: int, result: int) -> None:
        await asyncio.sleep(0)
        checkpoint(item, result)

    @recipe(output_root="output")
    def run() -> list[int]:
        callback = async_checkpoint if asynchronous else checkpoint
        return parallel(worker, [0, 1], max_workers=2, on_result=callback)

    result = run.run(output_directory=tmp_path / "output")
    assert result.ok, result.error
    assert result.output == [0, 10]
    assert observed == [1, 0]
    assert json.loads((tmp_path / "output/1.json").read_text()) == 10


class _QueuedPool:
    _max_workers = 2

    def __enter__(self):
        self.queued = []
        return self

    def submit(self, fn, worker):
        future = Future()
        if not self.queued:
            future.set_result(fn(worker))
        self.queued.append((future, fn, worker))
        return future

    def __exit__(self, *_exc):
        for future, fn, worker in self.queued:
            if not future.done() and future.set_running_or_notify_cancel():
                future.set_result(fn(worker))


def test_callback_failure_cancels_queued_workers(monkeypatch):
    called = []
    started = []
    monkeypatch.setattr(
        importlib.import_module("calcifier.recipe"), "ThreadPoolExecutor", lambda **_: _QueuedPool()
    )

    def worker(item):
        started.append(item)
        return item

    def observe(item, result):
        called.append((item, result))
        raise RuntimeError("checkpoint failed")

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        parallel(worker, [1, 2], max_workers=2, on_result=observe)
    assert len(called) == 1
    assert started == [1]


def test_worker_failure_stops_scheduling_new_work():
    started = []

    def worker(item):
        started.append(item)
        raise ValueError("worker failed")

    with pytest.raises(ValueError, match="worker failed"):
        parallel(worker, range(10), max_workers=1, on_result=lambda *_: None)
    assert started == [0]


@pytest.mark.parametrize("callback_fails", [False, True])
def test_running_results_are_checkpointed_after_worker_failure(monkeypatch, callback_fails):
    running = threading.Barrier(2)
    failure_observed = threading.Event()
    saved = []
    cancel = Future.cancel

    def observe_cancellation(future):
        failure_observed.set()
        return cancel(future)

    monkeypatch.setattr(Future, "cancel", observe_cancellation)

    def worker(item):
        running.wait(5)
        if item == 0:
            raise ValueError("worker failed")
        assert failure_observed.wait(5)
        return item

    def observe(item, result):
        saved.append(result)
        if callback_fails:
            raise RuntimeError("checkpoint failed")

    with pytest.raises(RuntimeError if callback_fails else ValueError) as caught:
        parallel(worker, [0, 1], max_workers=2, on_result=observe)
    assert saved == [1]
    if callback_fails:
        assert isinstance(caught.value.__cause__, ValueError)
        assert str(caught.value.__cause__) == "worker failed"


def test_resume_accepts_identical_artifacts_but_never_changes_protected_contents(tmp_path):
    @step
    def save(value: int) -> str:
        return str(write_json("checkpoint.json", {"value": value}, reuse_existing=True))

    @recipe(output_root="output")
    def run(value: int) -> str:
        return save(value)

    output = tmp_path / "output"
    assert run.run({"value": 1}, output_directory=output).ok
    path = output / "checkpoint.json"
    before = path.stat().st_mtime_ns
    resumed = run.run({"value": 1}, output_directory=output)
    assert resumed.ok
    assert path.stat().st_mtime_ns == before
    (effect,) = resumed.records[0].side_effects
    assert effect.kind == "file_write" and effect.target == str(path) and effect.completed
    assert not run.run({"value": 2}, output_directory=output).ok
    assert json.loads(path.read_text()) == {"value": 1}


@pytest.mark.parametrize("existing,replacement", [(b"line\r\n", "line\n"), (b"line\n", "line\r\n")])
def test_reuse_rejects_different_line_endings(tmp_path, existing, replacement):
    path = tmp_path / "checkpoint.txt"
    path.write_bytes(existing)

    @recipe(output_root="output")
    def run() -> None:
        write_text("checkpoint.txt", replacement, reuse_existing=True)

    result = run.run(output_directory=tmp_path)
    assert not result.ok
    assert result.error is not None
    assert "refusing to replace an existing file" in result.error
    assert path.read_bytes() == existing


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("dry_run", [DryRunMode.OFF, DryRunMode.SOFT])
def test_reused_artifact_can_be_updated_in_the_same_run(tmp_path, resume, dry_run):
    path = tmp_path / "status.json"
    if resume:
        path.write_bytes(b'{\n  "phase": 1\n}\n')

    @step
    def update() -> None:
        write_json("status.json", {"phase": 1}, reuse_existing=True)
        write_json("status.json", {"phase": 2})

    @recipe(output_root="output")
    def run() -> None:
        update()

    result = run.run(output_directory=tmp_path, dry_run=dry_run)
    assert result.ok, result.error
    if dry_run is DryRunMode.OFF:
        assert json.loads(path.read_text()) == {"phase": 2}
    elif resume:
        assert json.loads(path.read_text()) == {"phase": 1}
    else:
        assert not path.exists()
    assert len(result.records[0].side_effects) == 2
    assert all(
        effect.completed == (dry_run is DryRunMode.OFF) for effect in result.records[0].side_effects
    )
