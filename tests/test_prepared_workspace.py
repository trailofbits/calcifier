"""Preparing a prompt must not hide the caller's task workspace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calcifier import agent, prepare_context
from calcifier.prepared_context import prepared_context_run
from calcifier.registry import Registry
from calcifier.trace import active_run
from tests.support.agents import StubProvider


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("budget", [1024, 65536])
@pytest.mark.parametrize("explicit_directory", [False, True])
def test_preparing_context_preserves_workspace_and_explicit_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    newline: str,
    budget: int,
    explicit_directory: bool,
) -> None:
    project = tmp_path / "project"
    override = tmp_path / "override"
    for directory in (project, override):
        directory.mkdir()
        records = newline.join(f"{directory.name} record {index} café" for index in range(9))
        (directory / "records.txt").write_bytes((records + newline).encode())
    extra = tmp_path / "supplement.txt"
    extra.write_bytes(("additional source" + newline).encode())
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(project)
    expected = override if explicit_directory else project
    registry = Registry()
    observed = []
    task = ("Read every record in records.txt from the task workspace." + newline) * 100

    class Reader(StubProvider):
        name = "worker"

        def invoke(self, prompt, *, filesystem=None, **kwargs):
            assert filesystem is not None
            observed.append(filesystem)
            bundle = Path(context.directory)
            assert bundle in filesystem.readable_paths
            assert extra in filesystem.readable_paths
            assert unrelated not in filesystem.readable_paths
            assert not filesystem.writable_paths
            assert task.encode() in (bundle / "task.txt").read_bytes()
            if explicit_directory:
                assert project not in filesystem.readable_paths
            source = filesystem.working_directory / "records.txt"
            return self._make(json.dumps(source.read_bytes().decode()))

    registry.register(Reader())
    with prepared_context_run(), active_run(registry=registry):
        context = prepare_context(task, byte_budget=budget)
        result = agent(
            "worker",
            context.prompt,
            context=context,
            working_directory=override if explicit_directory else None,
            readable_paths=(extra,),
            output=str,
        )
        assert result.encode() == (expected / "records.txt").read_bytes()
        assert len(observed) == 1
        assert observed[0].working_directory == expected
    assert not Path(context.directory).exists()
