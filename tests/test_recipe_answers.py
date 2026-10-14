"""Recipe-runtime known answers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from calcifier import (
    Budget,
    RecipeError,
    agent,
    emit,
    load_recipe,
    recipe,
    step,
)
from calcifier.agent_events import AgentProvider, AgentResult
from calcifier.agent_providers import ClaudeCli, CodexCli, CommandCli
from calcifier.filesystem_contract import FilesystemContract
from calcifier.registry import Registry
from calcifier.side_effects import DryRunMode
from calcifier.trace import AgentPreflightEntry, AgentPreflightReport
from tests.support.agents import StubProvider
from tests.support.recipes import metadata_recipe as _metadata_recipe
from tests.support.trust import record_trust

_ZERO_SHA = "0" * 64
_RESOURCE = "references/a.txt"
_ABSOLUTE = Path(Path.cwd().anchor, "abs", "a.txt").as_posix()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _elapsed(seconds: float) -> bool:
    return 0.0 <= seconds < 60.0


# -- recipe runtime -------------------------------------------------------------


class _Worker(StubProvider):
    name = "worker"


def _rendered(run: Any) -> list[str]:
    return [re.sub(r"\d+\.\d{2}s", "<t>s", line) for line in run.render().splitlines()]


def test_unconfigured_declared_role_has_an_exact_unavailable_preflight_entry() -> None:
    @recipe(agents=("reviewer",), budget=Budget(1, 30))
    def review() -> bool:
        return True

    result = review.run(registry=Registry())

    assert result.error is None
    assert result.agent_preflight == AgentPreflightReport(
        agents=(
            AgentPreflightEntry(
                role="reviewer",
                target_type="command",
                target_name="reviewer",
                runtime={"kind": "host"},
                available=False,
                remediation="configure this declared agent role",
            ),
        )
    )


@pytest.mark.parametrize("provider_kind", ["claude", "codex", "command"])
@pytest.mark.parametrize("available", [True, False], ids=["installed", "missing"])
def test_configured_provider_preflight_reports_executable_presence_without_dispatch(
    tmp_path, monkeypatch, provider_kind, available
):
    binary = sys.executable if available else str(tmp_path / "absent-provider")
    if provider_kind == "command":
        provider = CommandCli("reviewer", (binary,))
    else:
        provider_class = ClaudeCli if provider_kind == "claude" else CodexCli
        provider = provider_class(binary=binary, role="reviewer")

    def forbidden(*args, **kwargs):
        pytest.fail("availability reporting must not invoke a provider")

    monkeypatch.setattr(provider, "invoke", forbidden)

    @recipe(agents=("reviewer",), budget=Budget(1, 30))
    def review() -> bool:
        return True

    registry = Registry()
    registry.register(provider)
    run = review.run(registry=registry)
    assert run.ok
    assert run.agent_preflight is not None
    assert run.agent_preflight.ok is available
    (entry,) = run.agent_preflight.agents
    assert entry.role == "reviewer"
    assert entry.available is available
    assert entry.target_type == ("command" if provider_kind == "command" else "builtin")


def test_in_process_provider_inherits_available_preflight_and_typed_dispatch(
    tmp_path: Path,
) -> None:
    class InProcessProvider(AgentProvider):
        name = "reviewer"
        default_timeout = 30

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def invoke(
            self,
            prompt: str,
            *,
            timeout: float | None = None,
            filesystem: FilesystemContract | None = None,
            schema: dict[str, Any] | None = None,
        ) -> AgentResult:
            assert filesystem is not None
            assert filesystem.working_directory == tmp_path
            self.prompts.append(prompt)
            return AgentResult("true", "true", "", 0, self.name)

    @step
    def judge() -> bool:
        return agent(
            "reviewer",
            "Return a typed verdict.",
            working_directory=tmp_path,
            readable_paths=(tmp_path,),
            output=bool,
        )

    @recipe(agents=("reviewer",), budget=Budget(1, 30))
    def review() -> bool:
        return judge()

    provider = InProcessProvider()
    registry = Registry()
    registry.register(provider)
    result = review.run(registry=registry)
    assert result.ok, result.error
    assert result.output is True
    assert result.agent_preflight is not None and result.agent_preflight.ok
    (entry,) = result.agent_preflight.agents
    assert entry.role == "reviewer" and entry.available
    assert len(provider.prompts) == 1


def test_cli_coerces_integer_flags_before_the_recipe_body(
    capsys: pytest.CaptureFixture[str],
) -> None:
    @recipe
    def double(count: int) -> int:
        return count * 2

    assert double.cli(["--count", "21"]) == 0

    output = capsys.readouterr().out.splitlines()
    assert output[0] == "=== Calcifier run :: double :: OK ==="
    assert output[2:] == ["output: 42"]


def _fallback_sigterm(signum: int, frame: object) -> None:
    """Fail loudly, instead of killing pytest, if a run leaves SIGTERM unhandled."""
    del signum, frame
    raise RuntimeError("SIGTERM reached the test's fallback handler")


def test_sigterm_interrupts_a_main_thread_run_with_its_signal_name(tmp_path: Path) -> None:
    if os.name == "nt" or threading.current_thread() is not threading.main_thread():
        pytest.skip("SIGTERM conversion applies to the POSIX main thread only")
    trace_path = tmp_path / "trace.json"

    @step
    def wait() -> bool:
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)
        return True

    @recipe
    def run() -> bool:
        return wait()

    previous = signal.signal(signal.SIGTERM, _fallback_sigterm)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            run.run(trace_path=trace_path)
        restored = signal.getsignal(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)

    assert str(raised.value) == "SIGTERM"
    assert restored is _fallback_sigterm
    written = json.loads(trace_path.read_text())
    assert (written["state"], written["error"]) == ("interrupted", "SIGTERM: interrupted")


def test_run_from_a_worker_thread_leaves_signal_handlers_alone() -> None:
    before = signal.getsignal(signal.SIGTERM)
    results: list[Any] = []

    @recipe
    def run() -> str:
        return threading.current_thread().name

    worker = threading.Thread(target=lambda: results.append(run.run()), name="recipe-worker")
    worker.start()
    worker.join(timeout=30)

    assert [(item.error, item.output) for item in results] == [(None, "recipe-worker")]
    assert signal.getsignal(signal.SIGTERM) is before


@pytest.mark.parametrize(
    ("resource", "message"),
    [
        ({"path": 1, "sha256": _ZERO_SHA}, "_CALCIFIER_RESOURCES entries require path and sha256"),
        ({"path": _RESOURCE, "sha256": 1}, "_CALCIFIER_RESOURCES entries require path and sha256"),
        (
            {"path": _RESOURCE, "sha256": _ZERO_SHA, "source": _ABSOLUTE},
            f"resource path must be relative, got absolute path: {_ABSOLUTE!r}",
        ),
        (
            {"path": _RESOURCE, "sha256": _ZERO_SHA, "source": ""},
            "resource path must be a non-empty relative path",
        ),
    ],
)
def test_load_recipe_rejects_each_invalid_resource_declaration(
    tmp_path: Path, resource: dict[str, Any], message: str
) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text("# skill\n")
    recipe_path = _metadata_recipe(tmp_path / "recipe.py", str(skill), [resource])

    with pytest.raises(RecipeError) as raised:
        load_recipe(recipe_path)

    assert str(raised.value) == message


_RESOURCE_READER = """from calcifier import recipe, resource


@recipe
def read_all() -> list[str]:
    return [
        resource("references/deep/first.txt").read_text(),
        resource("references/deep/second.txt").read_text(),
    ]
"""


def test_packaged_resources_are_snapshotted_into_nested_directories(tmp_path: Path) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text("# skill\n")
    deep = tmp_path / "references" / "deep"
    deep.mkdir(parents=True)
    (deep / "first.txt").write_bytes(b"first\n")
    (deep / "second.txt").write_bytes(b"second\n")
    resources = [
        {"path": "references/deep/first.txt", "sha256": _sha(b"first\n")},
        {"path": "references/deep/second.txt", "sha256": _sha(b"second\n")},
    ]
    path = _metadata_recipe(tmp_path / "recipe.py", str(skill), resources, _RESOURCE_READER)

    result = load_recipe(path).run()

    assert result.error is None
    assert result.output == ["first\n", "second\n"]


def test_run_durations_are_elapsed_seconds_not_timestamps(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"

    @step
    def quick() -> bool:
        return True

    @recipe
    def run() -> bool:
        return quick()

    @recipe(output_root=tmp_path / "out")
    def blocked() -> bool:
        return True

    completed = run.run(trace_path=trace_path)
    refused = blocked.run(dry_run=DryRunMode.HARD)

    assert completed.error is None
    assert _elapsed(completed.duration_seconds)
    assert _elapsed(json.loads(trace_path.read_text())["duration_seconds"])
    out = (tmp_path / "out").resolve()
    assert refused.error == f"side effects blocked by hard dry-run: output root {out}"
    assert _elapsed(refused.duration_seconds)


@pytest.mark.parametrize("name", ["", 1, None])
def test_emit_rejects_blank_or_non_string_artifact_names(name: Any) -> None:
    with pytest.raises(ValueError) as raised:
        emit("value", name=name)

    assert str(raised.value) == "artifact name must be a non-empty string"
    assert emit("value", name="report") == "value"


_LANE_CHILD = """from calcifier import recipe, step, write_text


@step
def persist(value: str) -> str:
    write_text("value.txt", value)
    return value


@recipe(name="child", output_root=".")
def child_recipe(value: str) -> str:
    return persist(value)
"""

_LANE_PARENT = """from calcifier import call_recipe, parallel, recipe, step


def fan_in(prefix: str) -> list[str]:
    return parallel(
        lambda suffix: call_recipe("child.py", {{"value": prefix + suffix}}).output,
        ["-x", "-y"],
        max_workers=2,
    )


@step
def fan_out() -> list[list[str]]:
    return parallel(fan_in, ["a", "b"], max_workers=2)


@recipe(output_root={root!r})
def parent() -> list[list[str]]:
    return fan_out()
"""


def test_nested_parallel_children_write_under_one_based_nested_lanes(tmp_path: Path) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text("# skill\n")
    root = tmp_path / "requested"
    child = _metadata_recipe(tmp_path / "child.py", str(skill), [], _LANE_CHILD)
    parent_body = _LANE_PARENT.format(root=str(root))
    parent = _metadata_recipe(tmp_path / "parent.py", str(skill), [], parent_body)
    for node in (parent, child):
        node.write_text(
            node.read_text().replace("'sha256': 'skill'", f"'sha256': {_sha(skill.read_bytes())!r}")
        )
    record_trust(parent, recorded_by="tester", dependencies=[(child, _sha(child.read_bytes()))])

    result = load_recipe(parent).run()

    assert result.error is None
    assert result.output == [["a-x", "a-y"], ["b-x", "b-y"]]
    lanes = root / ".calcifier-parallel"
    written = {
        path.relative_to(lanes).as_posix(): path.read_text() for path in lanes.rglob("value.txt")
    }
    assert written == {
        "lane-0001.0001/call-0001-child/value.txt": "a-x",
        "lane-0001.0002/call-0001-child/value.txt": "a-y",
        "lane-0002.0001/call-0001-child/value.txt": "b-x",
        "lane-0002.0002/call-0001-child/value.txt": "b-y",
    }


def test_agent_call_duration_is_elapsed_seconds(tmp_path: Path) -> None:
    @step
    def judge() -> bool:
        return agent("worker", "review", working_directory=tmp_path, readable_paths=(tmp_path,))

    @recipe(agents=("worker",), budget=Budget(2, 30))
    def run() -> bool:
        return judge()

    registry = Registry()
    registry.register(_Worker(default="true"))

    result = run.run(registry=registry)

    assert (result.error, result.output) == (None, True)
    [call] = result.records[0].agent_calls
    assert (call.role, call.attempts, call.error) == ("worker", 1, None)
    assert _elapsed(call.duration_seconds)
