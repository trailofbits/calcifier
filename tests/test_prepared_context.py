"""Offline regressions for concurrent content-addressed context bundles."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from calcifier import parallel, prepare_context, recipe
from calcifier.agents import _prepared_agent_context
from calcifier.prepared_context import PreparedContext, prepared_context_run

pytestmark = pytest.mark.semantic_offline


def test_passing_context_prompt_directly_does_not_duplicate_the_assignment(tmp_path):
    context = PreparedContext("complete original assignment", str(tmp_path))
    prompt, paths, _ = _prepared_agent_context(context.prompt, (), context)
    assert prompt == context.prompt
    assert paths == (str(tmp_path),)
    prompt, _, _ = _prepared_agent_context("Additional request", (), context)
    assert prompt == "Additional request\n\n" + context.prompt


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("windows_replace", [False, True])
def test_concurrent_prepared_context_bundles_are_complete_and_identical(
    monkeypatch: pytest.MonkeyPatch, newline: str, windows_replace: bool
) -> None:
    workers = 32
    barrier = threading.Barrier(workers)
    replace_lock = threading.Lock()
    replace = os.replace
    collisions: list[Path] = []
    task = ("complete assignment café" + newline) * 100
    instructions = "retain" + newline + "every record"
    prior = "first" + newline + "last"
    expected = (
        f"Task:\n{task}\n\nInstructions:\n{instructions}\n\nPrior results:\n{prior}\n"
    ).encode()

    def refuse_existing_destination(source: str | Path, destination: str | Path) -> None:
        # Model the existence check and rename as one atomic Windows operation.
        with replace_lock:
            path = Path(destination)
            if path.exists():
                collisions.append(path)
                raise PermissionError("simulated Windows destination in use")
            replace(source, destination)

    if windows_replace:
        monkeypatch.setattr(os, "replace", refuse_existing_destination)

    def prepare(_: int) -> PreparedContext:
        barrier.wait(timeout=10)
        context = prepare_context(
            task, instructions=instructions, prior_results=prior, byte_budget=1024
        )
        directory = Path(context.directory)
        assert list(directory.iterdir()) == [directory / "task.txt"]
        assert (directory / "task.txt").read_bytes() == expected
        return context

    @recipe
    def workflow() -> str:
        contexts = parallel(prepare, range(workers), max_workers=workers)
        assert len(contexts) == workers
        assert all(context == contexts[0] for context in contexts)
        directory = Path(contexts[0].directory)
        assert list(directory.parent.iterdir()) == [directory]
        return str(directory)

    run = workflow.run()
    assert run.ok, run.error
    assert isinstance(run.output, str)
    assert not Path(run.output).exists()
    if windows_replace:
        assert collisions == [Path(run.output)] * (workers - 1)


def test_context_publication_error_without_a_winner_propagates_and_cleans_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replace = os.replace
    staging: list[Path] = []

    def refuse_directory(source: str | Path, destination: str | Path) -> None:
        path = Path(source)
        if path.is_dir():
            staging.append(path)
            raise PermissionError("publication denied")
        replace(source, destination)

    monkeypatch.setattr(os, "replace", refuse_directory)
    with prepared_context_run():
        with pytest.raises(PermissionError, match="publication denied"):
            prepare_context("complete assignment")
        assert len(staging) == 1
        assert list(staging[0].parent.iterdir()) == []
