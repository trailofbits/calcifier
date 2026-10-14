"""CLI provider command construction and agent configuration."""

import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest

from calcifier import (
    agent,
    agent_events,
    agent_providers,
    output_type,
    subprocesses,
)
from calcifier.agent_config import AgentConfigError, load_agent_config, override_agent_timeouts
from calcifier.agent_events import (
    AgentActivity,
    AgentResult,
    _run_cli,
    capture_agent_activity,
)
from calcifier.agent_providers import (
    ClaudeCli,
    CodexCli,
    CommandCli,
    claude_command,
    codex_command,
    host_provider_check,
)
from calcifier.filesystem_contract import AgentError, FilesystemContract
from calcifier.native_events import (
    parse_adapter_envelope,
    parse_claude_events,
    parse_codex_events,
    parse_native_events,
    provider_reported_model,
)
from calcifier.registry import Registry, configured_registry
from calcifier.subprocesses import ProcessError, ProcessOutputLimitError
from calcifier.trace import active_run, agent_call_frame, record_agent_usage
from tests.support.agents import StubProvider
from tests.support.agents import capture_run as _capture_run


@output_type
class _FakeOutput:
    ok: bool


def test_host_command_known_answers(tmp_path: Path) -> None:
    extra = tmp_path / "readable"
    extra.mkdir()
    filesystem = FilesystemContract.build(tmp_path, (extra,))
    schema = {"type": "boolean"}

    claude = claude_command("claude", model="claude-test", filesystem=filesystem, schema=schema)
    assert claude.argv == [
        "claude",
        "-p",
        "--output-format=stream-json",
        "--verbose",
        "--no-session-persistence",
        "--safe-mode",
        "--restricted",
        "--permission-prompts",
        "none",
        "--permission-mode",
        "plan",
        "--strict-mcp-config",
        "--tools",
        "Read,Glob,Grep",
        "--allowedTools",
        "Read,Glob,Grep",
        "--max-turns",
        "12",
        "--model",
        "claude-test",
        "--add-dir",
        str(extra),
        "--json-schema",
        '{"type": "boolean"}',
    ]
    assert host_provider_check("claude") == (
        ["claude", "--help"],
        claude_command("claude").preflight_features,
    )

    schema_path = tmp_path / "schema.json"
    codex = codex_command(
        "codex",
        model="gpt-test",
        reasoning_effort="high",
        filesystem=filesystem,
        schema_path=schema_path,
    )
    assert codex.argv == [
        "codex",
        "exec",
        "--color",
        "never",
        "--ephemeral",
        "--json",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--disable",
        "multi_agent",
        "--disable",
        "plugins",
        "--disable",
        "apps",
        "--disable",
        "browser_use",
        "--disable",
        "computer_use",
        "-c",
        "project_doc_max_bytes=0",
        "--model",
        "gpt-test",
        "-c",
        'model_reasoning_effort="high"',
        "--cd",
        str(tmp_path),
        "--output-schema",
        str(schema_path),
        "-",
    ]
    assert host_provider_check("codex") == (
        ["codex", "exec", "--help"],
        codex_command("codex", schema_path=Path("output-schema.json")).preflight_features,
    )


def test_host_codex_never_bypasses_approvals(monkeypatch):
    calls = _capture_run(monkeypatch)

    CodexCli().invoke("prompt", timeout=3)

    command = calls[0][0]
    assert command[:5] == ["codex", "exec", "--color", "never", "--ephemeral"]
    assert "--sandbox" in command
    assert "read-only" in command
    assert "--json" in command
    assert command[-1] == "-"
    assert "--dangerously-bypass-approvals-and-sandbox" not in command


def test_host_codex_pins_model_and_reasoning_effort(monkeypatch):
    calls = _capture_run(monkeypatch)

    CodexCli(model="gpt-5.4", reasoning_effort="xhigh").invoke("prompt", timeout=3)

    command = calls[0][0]
    assert command[command.index("--model") + 1] == "gpt-5.4"
    assert 'model_reasoning_effort="xhigh"' in command


def test_host_claude_pins_a_configured_model(monkeypatch):
    calls = _capture_run(monkeypatch)

    ClaudeCli(model="claude-opus-5-5").invoke("prompt", timeout=3)
    ClaudeCli().invoke("prompt", timeout=3)

    pinned, default = (call[0] for call in calls)
    assert pinned[pinned.index("--model") + 1] == "claude-opus-5-5"
    assert "--model" not in default


def test_a_configured_claude_model_reaches_the_provider_and_the_fingerprint(
    tmp_path: Path, monkeypatch
):
    route = {"target": {"builtin": "claude"}, "runtime": {"kind": "host"}}
    unpinned = load_agent_config(_write_config(tmp_path, {"worker": dict(route)}))
    pinned = load_agent_config(
        _write_config(tmp_path, {"worker": {**route, "model": "claude-opus-5-5"}})
    )
    calls = _capture_run(monkeypatch)

    provider = configured_registry(pinned, ("worker",)).get("worker")
    provider.invoke("prompt")

    assert provider.model == "claude-opus-5-5"
    assert calls[0][0][calls[0][0].index("--model") + 1] == "claude-opus-5-5"
    assert pinned.fingerprint() != unpinned.fingerprint()


def test_host_codex_isolates_configuration_but_preserves_authentication(tmp_path, monkeypatch):
    original = tmp_path / "codex-home"
    original.mkdir()
    (original / "auth.json").write_text('{"token":"fixture"}')
    monkeypatch.setenv("CODEX_HOME", str(original))
    observed: dict[str, object] = {}

    def fake_run(cmd, stdin, *, timeout, name, env, **kwargs):  # noqa: ANN001
        del stdin, timeout, name, kwargs
        observed["command"] = cmd
        isolated = Path(env["CODEX_HOME"])
        observed["auth"] = (isolated / "auth.json").read_text()
        return AgentResult("ok", "ok", "", 0, "codex")

    monkeypatch.setattr(agent_providers, "_run_cli", fake_run)

    CodexCli().invoke("prompt", timeout=3)

    assert observed["auth"] == '{"token":"fixture"}'
    assert "--ignore-user-config" in cast("list[str]", observed["command"])


def test_codex_receives_all_fields_required_native_schema(monkeypatch, tmp_path: Path):
    @output_type
    class DefaultedOutput:
        ok: bool
        note: str | None = None

    observed: dict[str, object] = {}

    def fake_run(cmd, stdin, **kwargs):  # noqa: ANN001
        del stdin, kwargs
        schema_path = Path(cmd[cmd.index("--output-schema") + 1])
        observed.update(json.loads(schema_path.read_text()))
        text = '{"ok":true,"note":null}'
        return AgentResult(text, text, "", 0, "worker")

    monkeypatch.setattr(agent_providers, "_run_cli", fake_run)

    result = CodexCli(role="worker", default_timeout=5).structured(
        "return a result",
        annotation=DefaultedOutput,
        filesystem=FilesystemContract.build(tmp_path, (tmp_path,)),
    )

    assert result == DefaultedOutput(ok=True, note=None)
    assert observed["required"] == ["ok", "note"]
    assert '"default"' not in json.dumps(observed)


def test_codex_receives_batch_key_tuple_as_supported_array_schema(monkeypatch, tmp_path: Path):
    @output_type
    class Disposition:
        key: tuple[int, str]

    observed: dict[str, object] = {}

    def fake_run(cmd, stdin, **kwargs):  # noqa: ANN001
        del stdin, kwargs
        schema_path = Path(cmd[cmd.index("--output-schema") + 1])
        observed.update(json.loads(schema_path.read_text()))
        text = '{"key":[1,"alpha"]}'
        return AgentResult(text, text, "", 0, "verifier")

    monkeypatch.setattr(agent_providers, "_run_cli", fake_run)

    result = CodexCli(role="verifier", default_timeout=5).structured(
        "return a disposition",
        annotation=Disposition,
        filesystem=FilesystemContract.build(tmp_path, (tmp_path,)),
    )

    key = cast("dict[str, object]", cast("dict[str, object]", observed["properties"])["key"])
    assert result == Disposition(key=(1, "alpha"))
    assert "prefixItems" not in key
    assert key["items"] == {"anyOf": [{"type": "integer"}, {"type": "string"}]}


def test_agent_stderr_is_interactive_and_not_returned(capsys):
    result = _run_cli(
        [
            sys.executable,
            "-c",
            "import sys; print('answer'); print('diagnostic', file=sys.stderr)",
        ],
        "",
        timeout=2,
        name="test",
    )

    assert result.text == "answer"
    assert result.raw_stderr == ""
    assert "diagnostic" in capsys.readouterr().err


def test_agent_error_does_not_persist_stderr(capsys):
    with pytest.raises(AgentError, match="exited 3") as excinfo:
        _run_cli(
            [
                sys.executable,
                "-c",
                "import sys; print('private', file=sys.stderr); raise SystemExit(3)",
            ],
            "",
            timeout=2,
            name="test",
        )

    assert "private" not in str(excinfo.value)
    assert "private" in capsys.readouterr().err


def test_agent_timeout_is_bounded():
    with pytest.raises(AgentError, match="timed out"):
        _run_cli(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            "",
            timeout=0.05,
            name="test",
        )


@pytest.mark.parametrize(
    ("parser", "events", "accepted"),
    [
        (parse_codex_events, [], False),
        (
            parse_codex_events,
            [{"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}}],
            False,
        ),
        (parse_codex_events, [{"type": "turn.completed"}], True),
        (parse_codex_events, [{"type": "turn.completed"}, {"type": "turn.started"}], False),
        (parse_codex_events, [{"type": "turn.completed"}, {"type": "turn.failed"}], False),
        (parse_codex_events, [{"type": "error"}], False),
        (parse_claude_events, [{"type": "assistant"}], False),
        (parse_claude_events, [{"type": "result", "subtype": "success"}], True),
        (parse_claude_events, [{"type": "result", "is_error": True}], False),
        (parse_claude_events, [{"type": "result", "subtype": "error_max_turns"}], False),
        (parse_claude_events, [{"type": "result"}], True),
        (None, [], True),
    ],
)
def test_native_completion_is_required(parser, events, accepted):
    if parser is None:
        return
    provider = "codex" if parser is parse_codex_events else "claude"
    stream = parse_native_events("\n".join(json.dumps(event) for event in events), provider)
    if accepted:
        stream.require_completion()
    else:
        with pytest.raises(AgentError, match="native"):
            stream.require_completion()


def test_incomplete_native_cli_records_usage_but_never_accepts_partial_output():
    event = json.dumps({"type": "assistant", "message": {"usage": {"input_tokens": 7}}})
    activity = AgentActivity()
    with capture_agent_activity(activity), pytest.raises(AgentError, match="no completed native"):
        _run_cli(
            [sys.executable, "-c", "import sys; print(sys.argv[1])", event],
            "",
            timeout=2,
            name="claude",
            native_format="claude",
        )
    assert activity.usage["input_tokens"] == 7
    assert activity.usage_reported_attempts == {}


def test_codex_tool_output_never_substitutes_for_final_compiler_response():
    final = '{"version":"1","module":{"kind":"Module","fields":[]}}'
    events = [
        {
            "type": "item.completed",
            "item": {"type": "command_execution", "aggregated_output": "a complete-looking recipe"},
        },
        {"type": "item.completed", "item": {"type": "agent_message", "text": final}},
        {"type": "turn.completed", "usage": {"input_tokens": 23, "output_tokens": 7}},
    ]
    text, usage = parse_codex_events("\n".join(json.dumps(event) for event in events))
    assert text == final
    assert usage["input_tokens"] == 23
    assert usage["output_tokens"] == 7


def test_timeout_retains_usage_emitted_before_process_termination(monkeypatch):
    event = json.dumps(
        {
            "type": "assistant",
            "message": {"usage": {"input_tokens": 7, "output_tokens": 3}},
        }
    )
    real_wait = subprocesses._wait_bounded

    def wait_after_usage(process, stdout, *args, **kwargs):
        # This tests usage retention after timeout, not interpreter startup
        # latency under coverage/mutation load. Keep the real timeout path.
        deadline = time.monotonic() + 10
        while not stdout.data.endswith(b"\n"):
            if process.poll() is not None or time.monotonic() >= deadline:
                pytest.fail("child did not emit its usage event")
            time.sleep(0.01)
        assert bytes(stdout.data).splitlines() == [event.encode()]
        return real_wait(process, stdout, *args, **kwargs)

    monkeypatch.setattr(subprocesses, "_wait_bounded", wait_after_usage)
    with active_run() as context:
        record_agent_usage({"input_tokens": 100})
        with (
            agent_call_frame(
                role="worker",
                target_type="builtin",
                target_name="claude",
                runtime={"kind": "host"},
            ),
            pytest.raises(AgentError, match="timed out"),
        ):
            _run_cli(
                [
                    sys.executable,
                    "-c",
                    "import sys,time; print(sys.argv[1], flush=True); time.sleep(30)",
                    event,
                ],
                "",
                timeout=0.5,
                name="claude",
                native_format="claude",
            )

        usage = context.agent_calls[0].usage
        assert usage["input_tokens"] == 7
        assert usage["output_tokens"] == 3
        assert context.agent_calls[0].usage_reported_attempts == {}
        record_agent_usage({"input_tokens": 100})
        with agent_call_frame(
            role="second",
            target_type="builtin",
            target_name="claude",
            runtime={"kind": "host"},
        ):
            record_agent_usage({"input_tokens": 7})
            record_agent_usage({"input_tokens": 5})
        assert context.agent_calls[1].usage["input_tokens"] == 12


def test_usage_recording_is_a_noop_without_an_active_call_and_accumulates():
    record_agent_usage({"input_tokens": 100})
    with active_run() as context:
        record_agent_usage({"input_tokens": 100})
        with agent_call_frame(
            role="worker",
            target_type="builtin",
            target_name="claude",
            runtime={"kind": "host"},
        ):
            record_agent_usage({"input_tokens": 7})
            record_agent_usage({"input_tokens": 5})

        assert context.agent_calls[0].usage["input_tokens"] == 12


def test_agent_trace_records_sizes_without_contents(tmp_path: Path) -> None:
    provider = StubProvider(default='{"ok": true}')
    provider.name = "worker"
    registry = Registry(configuration_hash="fixture")
    registry.register(provider)

    with active_run(registry=registry) as context:
        from calcifier import trace as trace_module

        with trace_module.step_frame("validate", _FakeOutput, {}):
            result = agent(
                "worker",
                "private candidate details",
                working_directory=tmp_path,
                readable_paths=(tmp_path,),
            )

    assert result == _FakeOutput(ok=True)
    record = context.records[0].agent_calls[0]
    assert len(record.prompt_bytes) == len(record.schema_bytes) == 1
    assert record.prompt_bytes[0] > len("private candidate details")
    assert record.schema_bytes[0] > 0
    assert "private candidate details" not in repr(record)


class _ScriptedAnswers(StubProvider):
    """Answer successive structured attempts in order, recording each prompt."""

    def __init__(self, *answers: str) -> None:
        super().__init__()
        self.answers = list(answers)
        self.prompts: list[str] = []

    def invoke(self, prompt, *, timeout=None, filesystem=None, schema=None):  # noqa: ANN001
        del timeout, filesystem, schema
        self.prompts.append(prompt)
        text = self.answers.pop(0)
        return AgentResult(text, text, "", 0, self.name)


def test_each_rejected_attempt_names_the_stage_that_failed() -> None:
    provider = _ScriptedAnswers("not json at all", "still not json")

    with pytest.raises(AgentError) as caught:
        provider.structured("return approval", annotation=dict)

    assert str(caught.value) == (
        "stub returned no valid structured response after one retry (first attempt did not "
        "match the declared output schema; second attempt did not match the declared output "
        "schema)"
    )
    assert "did not match the declared output schema: " in provider.prompts[1]


def test_schema_failures_are_named_without_quoting_the_answer() -> None:
    provider = _ScriptedAnswers('{"secret": "alpha"}', '{"secret": "beta"}')

    with pytest.raises(AgentError) as caught:
        provider.structured("return it", annotation=_FakeOutput)

    assert "alpha" not in str(caught.value) and "beta" not in str(caught.value)
    assert "alpha" in provider.prompts[1]


def test_agent_activity_is_telemetry_not_call_control():
    provider = StubProvider(default="ok")
    activity = AgentActivity(max_model_calls=1)
    with capture_agent_activity(activity):
        provider.complete("first")
        provider.complete("second")
    assert activity.attempts == 2


def test_agent_activity_wall_time_includes_machine_suspend(monkeypatch):
    activity = AgentActivity()
    wall_times = iter((1_000.0, 15_010.0))
    monkeypatch.setattr("calcifier.clock._CONTINUOUS_CLOCK", None)
    monkeypatch.setattr("calcifier.clock.time.monotonic", lambda: 100.0)
    monkeypatch.setattr("calcifier.clock.time.time", lambda: next(wall_times))

    with capture_agent_activity(activity):
        pass

    assert activity.duration_seconds == 14_010


def _write_config(tmp_path: Path, agents_config: dict[str, object]) -> Path:
    for route in agents_config.values():
        if isinstance(route, dict) and "timeout_seconds" not in route:
            cast("dict[str, Any]", route)["timeout_seconds"] = 30
    path = tmp_path / "agents.json"
    path.write_text(json.dumps({"version": 1, "agents": agents_config}))
    return path


def test_agent_config_resolves_builtin_and_command_roles(tmp_path: Path):
    path = _write_config(
        tmp_path,
        {
            "worker": {
                "target": {"builtin": "claude"},
                "runtime": {"kind": "host"},
            },
            "local-reviewer": {
                "target": {"command": ["local-agent", "--model", "qwen"]},
                "runtime": {"kind": "host"},
                "timeout_seconds": 9,
            },
        },
    )

    config = load_agent_config(path)
    registry = configured_registry(config, ("worker", "local-reviewer"))

    assert registry.get("worker").target_name == "claude"
    local = registry.get("local-reviewer")
    assert isinstance(local, CommandCli)
    assert local.command == ("local-agent", "--model", "qwen")
    assert local.default_timeout == 9
    assert registry.configuration_hash == config.fingerprint(("worker", "local-reviewer"))


def test_agent_config_pins_builtin_codex_model_and_reasoning(tmp_path: Path, monkeypatch):
    path = _write_config(
        tmp_path,
        {
            "compiler": {
                "target": {"builtin": "codex"},
                "runtime": {"kind": "host"},
                "model": "gpt-5.4",
                "reasoning_effort": "xhigh",
            }
        },
    )
    calls = _capture_run(monkeypatch)

    config = load_agent_config(path)
    route = config.routes["compiler"]
    provider = configured_registry(config, ("compiler",)).get("compiler")
    provider.invoke("prompt")

    assert route.model == "gpt-5.4"
    assert route.reasoning_effort == "xhigh"
    assert provider.model == "gpt-5.4"
    assert provider.reasoning_effort == "xhigh"
    assert calls[0][0][calls[0][0].index("--model") + 1] == "gpt-5.4"
    assert 'model_reasoning_effort="xhigh"' in calls[0][0]


def test_agent_timeout_override_is_strict(tmp_path: Path):
    config = load_agent_config(
        _write_config(
            tmp_path,
            {"worker": {"target": {"builtin": "claude"}, "runtime": {"kind": "host"}}},
        )
    )

    overridden = override_agent_timeouts(config, ["worker=7"])

    assert overridden.routes["worker"].timeout_seconds == 7
    assert config.routes["worker"].timeout_seconds == 30
    with pytest.raises(AgentConfigError, match="unknown role"):
        override_agent_timeouts(config, ["missing=1"])
    with pytest.raises(AgentConfigError, match="expects ROLE=SECONDS"):
        override_agent_timeouts(config, ["worker"])
    with pytest.raises(AgentConfigError, match="must be an integer"):
        override_agent_timeouts(config, ["worker=slow"])
    with pytest.raises(AgentConfigError, match="must be positive"):
        override_agent_timeouts(config, ["worker=0"])
    assert override_agent_timeouts(config, ["worker=1"]).routes["worker"].timeout_seconds == 1


@pytest.mark.parametrize(
    ("seconds", "accepted"), [(0, False), (0.5, False), (1.0, False), (1, True), (2, True)]
)
def test_a_configured_timeout_must_be_a_positive_integer(
    tmp_path: Path, seconds: int | float, accepted: bool
):
    route = {"target": {"builtin": "claude"}, "runtime": {"kind": "host"}}
    path = _write_config(tmp_path, {"worker": {**route, "timeout_seconds": seconds}})

    if accepted:
        assert load_agent_config(path).routes["worker"].timeout_seconds == seconds
    else:
        with pytest.raises(AgentConfigError, match="must be a positive integer"):
            load_agent_config(path)


def test_routes_describe_their_target(tmp_path: Path):
    host = {"kind": "host"}
    config = load_agent_config(
        _write_config(
            tmp_path,
            {
                "worker": {"target": {"builtin": "codex"}, "runtime": host},
                "local": {"target": {"command": ["adapter", "--flag"]}, "runtime": host},
            },
        )
    )
    builtin, command = config.routes["worker"], config.routes["local"]

    assert (builtin.target_type, builtin.target_name) == ("builtin", "codex")
    assert (command.target_type, command.target_name) == ("command", "adapter")
    assert command.as_dict()["target"] == {"command": ["adapter", "--flag"]}
    assert builtin.as_dict()["target"] == {"builtin": "codex"}


def test_filesystem_contract_is_explicit_and_adds_verified_resources(tmp_path: Path):
    cwd = tmp_path / "source"
    resources = tmp_path / "snapshot"
    cwd.mkdir()
    resources.mkdir()

    contract = FilesystemContract.build(cwd, (), resource_root=resources)

    assert contract.working_directory == cwd
    assert contract.readable_paths == (cwd, resources)
    # The adapter payload's filesystem object, field for field.
    assert contract.as_dict() == {
        "mode": "read-only",
        "working_directory": str(cwd),
        "readable_paths": [str(cwd), str(resources)],
        "writable_paths": [],
    }
    with pytest.raises(AgentError, match="does not exist"):
        FilesystemContract.build(tmp_path / "missing", ())


def test_provider_event_usage_is_normalized():
    claude_text, claude_usage = parse_claude_events(
        json.dumps(
            {
                "type": "result",
                "structured_output": {"ok": True},
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 7,
                    "cache_creation_input_tokens": 3,
                },
                "total_cost_usd": 0.01,
            }
        )
    )
    codex_text, codex_usage = parse_codex_events(
        "\n".join(
            [
                json.dumps(
                    {"type": "item.completed", "item": {"type": "agent_message", "text": "true"}}
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 11,
                            "output_tokens": 4,
                            "cached_input_tokens": 5,
                            "reasoning_output_tokens": 2,
                        },
                    }
                ),
            ]
        )
    )
    adapter_text, adapter_usage = parse_adapter_envelope(
        json.dumps(
            {
                "version": 1,
                "output": "false",
                "usage": {"input_tokens": 6, "output_tokens": 1, "cost_usd": 0.02},
            }
        )
    )

    assert claude_text == '{"ok": true}'
    assert claude_usage == {
        "input_tokens": 10,
        "output_tokens": 2,
        "cache_read_tokens": 7,
        "cache_write_tokens": 3,
        "nested_model_tokens": None,
        "cost_usd": 0.01,
    }
    assert codex_text == "true"
    assert codex_usage["cache_read_tokens"] == 5
    assert codex_usage["nested_model_tokens"] == 2
    assert adapter_text == "false"
    assert adapter_usage["cost_usd"] == 0.02


_CLAUDE_RESULT = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "ok",
    "total_cost_usd": 0.42,
    "usage": {
        "input_tokens": 4,
        "output_tokens": 30,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 10,
        "output_tokens_details": {"thinking_tokens": 12},
    },
    "modelUsage": {
        "primary-model[1m]": {
            "inputTokens": 4,
            "outputTokens": 30,
            "cacheReadInputTokens": 20,
            "cacheCreationInputTokens": 10,
            "thinkingTokens": 12,
            "costUSD": 0.4,
        },
        "auxiliary-model": {
            "inputTokens": 80,
            "outputTokens": 3,
            "cacheReadInputTokens": 0,
            "cacheCreationInputTokens": 0,
            "costUSD": 0.02,
        },
    },
}


def test_claude_usage_sums_every_model_into_the_trace_fields() -> None:
    _, usage = parse_claude_events(json.dumps(_CLAUDE_RESULT))

    # Auxiliary models are counted, thinking is the nested share of output, and
    # the cost is the provider's total.
    assert usage == {
        "input_tokens": 84,
        "output_tokens": 33,
        "cache_read_tokens": 20,
        "cache_write_tokens": 10,
        "nested_model_tokens": 12,
        "cost_usd": 0.42,
    }


def test_claude_usage_without_model_usage_keeps_the_turn_accounting() -> None:
    result = {key: value for key, value in _CLAUDE_RESULT.items() if key != "modelUsage"}
    old_style = {**_CLAUDE_RESULT, "modelUsage": {"primary-model": {"inputTokens": 4}}}

    _, usage = parse_claude_events(json.dumps(result))
    _, partial = parse_claude_events(json.dumps(old_style))

    assert usage["input_tokens"] == 4
    assert usage["nested_model_tokens"] == 12
    # A field no model reports is not overwritten with zero.
    assert (partial["input_tokens"], partial["nested_model_tokens"]) == (4, 12)
    no_details = {**result, "usage": {"input_tokens": 4, "output_tokens": 30}}
    assert parse_claude_events(json.dumps(no_details))[1]["nested_model_tokens"] is None


def test_an_unfinished_claude_stream_sums_usage_wherever_events_carry_it() -> None:
    stream = "\n".join(
        json.dumps(event)
        for event in (
            {"type": "assistant", "message": {"content": "no usage here"}},
            {"type": "assistant", "message": {"usage": {"input_tokens": 3}}},
            {"type": "stream_event", "usage": {"input_tokens": 5, "output_tokens": 2}},
        )
    )

    _, usage = parse_claude_events(stream)

    assert (usage["input_tokens"], usage["output_tokens"]) == (8, 2)
    silent = json.dumps({"type": "assistant", "message": {"content": "no usage"}})
    assert parse_claude_events(silent)[1] == {}


@pytest.mark.parametrize(
    ("parser", "stdout"),
    [
        (parse_claude_events, json.dumps(_CLAUDE_RESULT)),
        (
            parse_claude_events,
            json.dumps({"type": "assistant", "message": {"usage": {"input_tokens": 3}}}),
        ),
        (
            parse_codex_events,
            "\n".join(
                json.dumps(event)
                for event in (
                    {"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}},
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 5, "reasoning_output_tokens": 2},
                    },
                )
            ),
        ),
    ],
    ids=["claude-result", "claude-partial", "codex"],
)
def test_every_provider_reports_only_trace_schema_usage_fields(parser, stdout) -> None:  # noqa: ANN001
    schema = json.loads((Path(__file__).parents[1] / "docs/trace-schema.json").read_text())

    _, usage = parser(stdout)

    assert usage
    assert set(usage) <= set(schema["$defs"]["usage"]["properties"])


def testprovider_reported_model_is_separate_and_prefers_primary_usage() -> None:
    stream = json.dumps(
        {
            "type": "result",
            "result": "ok",
            "modelUsage": {
                "auxiliary-small": {"outputTokens": 3},
                "primary-large": {"outputTokens": 30},
            },
        }
    )

    assert provider_reported_model(stream) == "primary-large"
    assert provider_reported_model('{"type":"turn.completed","model":"exact-model"}') == (
        "exact-model"
    )
    assert provider_reported_model('{"type":"turn.completed"}') is None


def test_provider_event_and_adapter_fallback_errors():
    from calcifier.subprocesses import ProcessError

    for parser in (parse_claude_events, parse_codex_events):
        with pytest.raises(ProcessError, match="valid UTF-8 JSON"):
            parser("plain")
    assert parse_codex_events('{}\n{"item": {"type": "other"}}')[0] == ""
    with pytest.raises(AgentError, match="v1 JSON envelope"):
        parse_adapter_envelope("not-json")
    with pytest.raises(AgentError, match="protocol version 1"):
        parse_adapter_envelope('{"version": 2, "output": "x"}')
    with pytest.raises(AgentError, match="invalid output or usage"):
        parse_adapter_envelope('{"version": 1, "output": 1}')


def test_cli_runner_error_boundaries(monkeypatch):
    with pytest.raises(AgentError, match="explicit timeout"):
        _run_cli(["agent"], "", timeout=None, name="test")

    monkeypatch.setattr(
        agent_events,
        "run_process",
        lambda *args, **kwargs: (_ for _ in ()).throw(ProcessOutputLimitError("large")),
    )
    with pytest.raises(AgentError, match="output exceeded"):
        _run_cli(["agent"], "", timeout=1, name="test")
    monkeypatch.setattr(
        agent_events,
        "run_process",
        lambda *args, **kwargs: (_ for _ in ()).throw(ProcessError("start")),
    )
    with pytest.raises(AgentError, match="failed to start"):
        _run_cli(["agent"], "", timeout=1, name="test")


def test_builtin_providers_receive_native_schema_and_filesystem(monkeypatch, tmp_path: Path):
    calls = _capture_run(monkeypatch)
    extra = tmp_path / "extra"
    extra.mkdir()
    contract = FilesystemContract.build(tmp_path, (extra,))

    ClaudeCli(default_timeout=5).invoke("prompt", filesystem=contract, schema={"type": "boolean"})
    CodexCli(default_timeout=5).invoke("prompt", filesystem=contract, schema={"type": "boolean"})

    claude, codex = calls
    assert "--add-dir" in claude[0]
    assert "--json-schema" in claude[0]
    assert codex[0][codex[0].index("--cd") + 1] == str(tmp_path)
    assert "--output-schema" in codex[0]


def _fake_cli_executable(tmp_path: Path, name: str, source: str) -> Path:
    script = tmp_path / f"{name}_fake.py"
    script.write_bytes(source.encode("utf-8"))
    script.chmod(0o700)
    if sys.platform == "win32":
        executable = tmp_path / f"{name}.cmd"
        executable.write_bytes(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n'.encode())
        return executable
    return script


@pytest.mark.parametrize("provider_name", ["claude", "codex"])
def test_builtin_adapter_with_offline_fake_executable(tmp_path: Path, provider_name: str):
    executable = _fake_cli_executable(
        tmp_path,
        provider_name,
        f"""#!/usr/bin/env python3
import json
import sys

assert sys.stdin.read()
if {provider_name!r} == "claude":
    print(json.dumps({{
        "type": "result",
        "structured_output": {{"ok": True}},
        "usage": {{"input_tokens": 3, "output_tokens": 1}},
    }}))
else:
    print(json.dumps({{
        "type": "item.completed",
        "item": {{"type": "agent_message", "text": "{{\\\"ok\\\": true}}"}},
    }}))
    print(json.dumps({{
        "type": "turn.completed",
        "usage": {{"input_tokens": 4, "output_tokens": 1}},
    }}))
""",
    )
    provider = (
        ClaudeCli(str(executable), role="worker", default_timeout=5)
        if provider_name == "claude"
        else CodexCli(str(executable), role="worker", default_timeout=5)
    )

    result = provider.structured(
        "return a result",
        annotation=_FakeOutput,
        filesystem=FilesystemContract.build(tmp_path, (tmp_path,)),
    )

    assert result == _FakeOutput(ok=True)


@pytest.mark.parametrize(
    ("is_error", "subtype"), [(True, "success"), (False, "error_during_execution")]
)
def test_native_failure_signal_rejects_a_valid_answer_and_retains_usage(
    tmp_path: Path, is_error: bool, subtype: str
):
    event = {
        "type": "result",
        "is_error": is_error,
        "subtype": subtype,
        "structured_output": {"ok": True},
        "usage": {"input_tokens": 11, "output_tokens": 3},
    }
    executable = _fake_cli_executable(
        tmp_path,
        "failed",
        "#!/usr/bin/env python3\nimport sys\nassert sys.stdin.read()\n"
        f"print({json.dumps(event)!r})\n",
    )
    provider = ClaudeCli(str(executable), role="worker", default_timeout=5)
    activity = AgentActivity()
    with capture_agent_activity(activity), pytest.raises(AgentError, match="native.*failed"):
        provider.structured(
            "return a result",
            annotation=_FakeOutput,
            filesystem=FilesystemContract.build(tmp_path, (tmp_path,)),
        )

    assert activity.attempts == 1
    assert activity.usage["input_tokens"] == 11
    assert activity.usage["output_tokens"] == 3
    transports = [event for event in activity.events if event["phase"] == "native_transport"]
    assert len(transports) == 1
    assert transports[0]["status"] == "failed"


def test_custom_adapter_receives_versioned_read_only_contract(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_run(cmd, stdin, **kwargs):  # noqa: ANN001
        captured.update({"cmd": cmd, "stdin": stdin, **kwargs})
        return AgentResult("true", "true", "", 0, "worker")

    monkeypatch.setattr(agent_providers, "_run_cli", fake_run)
    contract = FilesystemContract.build(tmp_path, (tmp_path,))

    CommandCli("worker", ("adapter",), default_timeout=9).invoke(
        "prompt", filesystem=contract, schema={"type": "boolean"}
    )

    assert captured["env"]["CALCIFIER_AGENT_PROTOCOL"] == "1"
    assert captured["env"]["CALCIFIER_AGENT_CONTRACT"] == json.dumps(
        {"version": 1, "filesystem": contract.as_dict(), "schema": {"type": "boolean"}},
        sort_keys=True,
    )


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"version": 2, "agents": {}}, "version must be 1"),
        ({"version": 1, "agents": {}, "extra": True}, "unknown fields"),
        ({"version": 1, "agents": []}, "must be a JSON object"),
        (
            {
                "version": 1,
                "agents": {
                    "worker": {
                        "target": {},
                        "runtime": {"kind": "host"},
                    }
                },
            },
            "exactly one of",
        ),
        (
            {
                "version": 1,
                "agents": {
                    "worker": {
                        "target": {"builtin": "claude", "command": ["agent"]},
                        "runtime": {"kind": "host"},
                    }
                },
            },
            "exactly one of",
        ),
        (
            {
                "version": 1,
                "agents": {
                    "worker": {
                        "target": {"builtin": "claude"},
                        "runtime": {"kind": "host"},
                        "timeout_seconds": 30,
                        "remediation": 1,
                    }
                },
            },
            "remediation must be a string",
        ),
        (
            {
                "version": 1,
                "agents": {
                    "claude": {
                        "target": {"builtin": "claude"},
                        "runtime": {"kind": "host"},
                    }
                },
            },
            "reserved",
        ),
        (
            {
                "version": 1,
                "agents": {
                    "worker": {
                        "target": {"command": []},
                        "runtime": {"kind": "host"},
                    }
                },
            },
            "non-empty argv",
        ),
        (
            {
                "version": 1,
                "agents": {
                    "bad role": {
                        "target": {"builtin": "claude"},
                        "runtime": {"kind": "host"},
                    }
                },
            },
            "invalid agent role",
        ),
        (
            {
                "version": 1,
                "agents": {
                    "worker": {
                        "target": {"builtin": "other"},
                        "runtime": {"kind": "host"},
                    }
                },
            },
            "builtin must be",
        ),
        (
            {
                "version": 1,
                "agents": {
                    "worker": {
                        "target": {"command": ["agent", ""]},
                        "runtime": {"kind": "host"},
                    }
                },
            },
            "command entries",
        ),
        (
            {
                "version": 1,
                "agents": {
                    "worker": {
                        "target": {"command": ["agent"]},
                        "runtime": {"kind": "host"},
                        "timeout_seconds": 0,
                    }
                },
            },
            "positive integer",
        ),
        (
            {
                "version": 1,
                "agents": {
                    "worker": {
                        "target": {"command": ["agent"]},
                        "runtime": {"kind": "coop"},
                    }
                },
            },
            "kind must be host",
        ),
        (
            {
                "version": 1,
                "agents": {
                    "worker": {
                        "target": {"command": ["agent"]},
                        "runtime": {"kind": "host", "instance": "vm"},
                    }
                },
            },
            "accepts only kind",
        ),
        (
            {
                "version": 1,
                "agents": {
                    "worker": {
                        "target": {"command": ["adapter"]},
                        "runtime": {"kind": "host"},
                        "timeout_seconds": 30,
                        "model": "local-model",
                    }
                },
            },
            "model is only valid for builtin claude or codex",
        ),
        (
            {
                "version": 1,
                "agents": {
                    "worker": {
                        "target": {"builtin": "codex"},
                        "runtime": {"kind": "host"},
                        "timeout_seconds": 30,
                        "model": "",
                    }
                },
            },
            "model must be a non-empty model name",
        ),
        (
            {
                "version": 1,
                "agents": {
                    "worker": {
                        "target": {"builtin": "codex"},
                        "runtime": {"kind": "host"},
                        "timeout_seconds": 30,
                        "reasoning_effort": "unbounded",
                    }
                },
            },
            "reasoning_effort must be one of",
        ),
    ],
)
def test_agent_config_rejects_invalid_routes(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(AgentConfigError, match=message):
        load_agent_config(path)


def test_agent_config_rejects_unreadable_or_invalid_json(tmp_path: Path):
    with pytest.raises(AgentConfigError, match="failed to load agent config"):
        load_agent_config(tmp_path / "missing.json")

    path = tmp_path / "invalid.json"
    path.write_text("not json")
    with pytest.raises(AgentConfigError, match="failed to load agent config"):
        load_agent_config(path)


def test_agent_config_rejects_duplicate_keys(tmp_path: Path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"version":1,"agents":{"worker":{"target":{"builtin":"claude"},'
        '"runtime":{"kind":"host"}},"worker":{"target":{"builtin":"codex"},'
        '"runtime":{"kind":"host"}}}}'
    )

    with pytest.raises(AgentConfigError, match="duplicate key 'worker'"):
        load_agent_config(path)


def test_configured_registry_requires_every_custom_role(tmp_path: Path):
    config = load_agent_config(_write_config(tmp_path, {}))

    with pytest.raises(AgentConfigError, match="missing from --agent-config"):
        configured_registry(config, ("worker",))


def test_configured_registry_retains_roles_for_subrecipes(tmp_path: Path):
    config = load_agent_config(
        _write_config(
            tmp_path,
            {
                "coordinator": {
                    "target": {"command": ["coordinator-agent"]},
                    "runtime": {"kind": "host"},
                },
                "child-worker": {
                    "target": {"command": ["child-agent"]},
                    "runtime": {"kind": "host"},
                },
            },
        )
    )

    registry = configured_registry(config, ("coordinator",))

    assert registry.known() == ["child-worker", "coordinator"]


def test_configured_registry_rejects_implicit_builtin(tmp_path: Path):
    config = load_agent_config(
        _write_config(
            tmp_path,
            {
                "worker": {
                    "target": {"builtin": "claude"},
                    "runtime": {"kind": "host"},
                }
            },
        )
    )

    with pytest.raises(AgentConfigError, match="implicit builtin agent 'claude'.*not allowed"):
        configured_registry(config, ("worker", "claude"))


def test_unconfigured_registry_cannot_emit_invalid_runtime_summary():
    with pytest.raises(AgentError, match="requires a configured registry"):
        Registry().runtime_summary(())


def test_command_provider_forwards_stdin_without_a_shell(monkeypatch):
    calls = _capture_run(monkeypatch)

    provider = CommandCli("local", ("adapter", "--model", "qwen"), default_timeout=7)
    provider.invoke("packet")

    assert calls == [(["adapter", "--model", "qwen"], "packet", 7, "local")]


def test_a_template_argument_without_a_placeholder_is_rejected_before_any_call(
    tmp_path: Path,
) -> None:
    provider = _ScriptedAnswers('{"ok": true}')
    provider.name = "worker"
    registry = Registry(configuration_hash="fixture")
    registry.register(provider)

    with (
        active_run(registry=registry),
        pytest.raises(AgentError, match=r"no \{name\} placeholder.*\['exact_grouped_keys'\]"),
    ):
        agent(
            "worker",
            "Review {target}.",
            working_directory=tmp_path,
            readable_paths=(tmp_path,),
            output=_FakeOutput,
            target="alpha",
            exact_grouped_keys={},
        )

    assert provider.prompts == []


def test_agent_prompt_without_template_arguments_is_forwarded(tmp_path: Path) -> None:
    provider = _ScriptedAnswers('{"ok": true}')
    provider.name = "worker"
    registry = Registry(configuration_hash="fixture")
    registry.register(provider)

    with active_run(registry=registry):
        answer = agent(
            "worker",
            "Review all records.",
            working_directory=tmp_path,
            readable_paths=(tmp_path,),
            output=_FakeOutput,
        )

    assert answer.ok is True
    assert len(provider.prompts) == 1
    assert "Review all records." in provider.prompts[0]


def test_agent_prompt_interpolates_declared_template_arguments(tmp_path: Path) -> None:
    provider = _ScriptedAnswers('{"ok": true}')
    provider.name = "worker"
    registry = Registry(configuration_hash="fixture")
    registry.register(provider)

    with active_run(registry=registry):
        answer = agent(
            "worker",
            "Review {target}.",
            working_directory=tmp_path,
            readable_paths=(tmp_path,),
            output=_FakeOutput,
            target="alpha",
        )

    assert answer.ok is True
    assert "Review alpha." in provider.prompts[0]
    assert "{target}" not in provider.prompts[0]
