"""Exercise compilation through a bounded, offline native provider executable."""

from __future__ import annotations

import json
import sys

import pytest

from calcifier import cli
from calcifier.agent_providers import CodexCli
from calcifier.baseline import is_baseline
from calcifier.recipe_draft import draft_response
from calcifier.registry import Registry
from tests.support.agents import StubProvider
from tests.support.recipes import _GOOD_RECIPE, write_skill
from tests.support.skill_source import SKILL


@pytest.mark.parametrize("oversized", [False, True])
def test_native_calcify_distinguishes_accepted_source_from_retained_baseline(
    tmp_path, monkeypatch, oversized
):
    skill = write_skill(tmp_path / "neutral", SKILL)
    assessment = "Little standalone work is extracted; owner review is needed."
    response = json.loads(draft_response(_GOOD_RECIPE))
    response["suitability_note"] = assessment
    responses = [json.dumps(response)]
    (tmp_path / "responses.json").write_text(json.dumps(responses), encoding="utf-8")
    script = tmp_path / "native_provider.py"
    script.write_text(
        f"""#!{sys.executable}
import json, sys
from pathlib import Path

root = Path(__file__).parent
assert sys.stdin.read()
state = root / 'calls.json'
calls = json.loads(state.read_text(encoding='utf-8')) if state.exists() else []
schema_path = Path(sys.argv[sys.argv.index('--output-schema') + 1])
schema = json.loads(schema_path.read_text(encoding='utf-8'))
calls.append(schema)
state.write_text(json.dumps(calls), encoding='utf-8')
if len(calls) == 1:
    for index in range(150):
        print(json.dumps({{'type': 'tool.progress', 'padding': 'x' * 65536}}))
if {oversized!r}:
    print(json.dumps({{'type': 'usage', 'usage': {{'input_tokens': 17}}}}))
    text = 'x' * (1024 * 1024 + 1)
else:
    text = json.loads((root / 'responses.json').read_text(encoding='utf-8'))[len(calls) - 1]
print(json.dumps({{'type': 'item.completed', 'item': {{'type': 'agent_message', 'text': text}}}}))
print(json.dumps({{'type': 'turn.completed', 'usage': {{'input_tokens': 11, 'output_tokens': 7}}}}))
""",
        encoding="utf-8",
        newline="\n",
    )
    script.chmod(0o700)
    executable = script
    if sys.platform == "win32":
        executable = tmp_path / "native_provider.cmd"
        executable.write_text(
            f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8", newline=""
        )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-auth"))
    registry = Registry()
    registry.register(CodexCli(str(executable), role="compiler", default_timeout=20))
    worker = StubProvider()
    worker.name = "worker"
    registry.register(worker)
    monkeypatch.setattr(cli, "_registry_for_args", lambda *_: registry)
    trace = tmp_path / "trace.json"

    result = cli.main(["calcify", str(skill), "--require-compiled", "--compiler-trace", str(trace)])

    payload = json.loads(trace.read_text(encoding="utf-8"))
    recipe = skill.parent / ".calcifier" / "recipe.py"
    assert result == int(oversized)
    assert is_baseline(recipe.read_text(encoding="utf-8")) == oversized
    assert payload["outcome"] == ("baseline_retained" if oversized else "compiled")
    assert payload["model_calls"] == 1
    calls = json.loads((tmp_path / "calls.json").read_text(encoding="utf-8"))
    assert set(calls[0]["properties"]) == {"version", "source", "suitability_note"}
    assert calls[0]["properties"]["suitability_note"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]
    assert payload["suitability_note"] == (None if oversized else assessment)
    transports = [event for event in payload["events"] if event["phase"] == "native_transport"]
    diagnostics = json.loads(transports[0]["detail"])
    assert diagnostics["total_bytes"] > 8 * 1024 * 1024
    assert diagnostics["event_types"]["tool.progress"] == 150
    if oversized:
        assert transports[0]["status"] == "failed"
        assert payload["usage"]["input_tokens"] == 17
        assert payload["usage_reported_attempts"] == {}
        assert diagnostics["largest_message_bytes"] == 1024 * 1024 + 1
    else:
        assert payload["usage"]["input_tokens"] == 11
        assert payload["usage"]["output_tokens"] == 7
        assert payload["usage_reported_attempts"] == {"input_tokens": 1, "output_tokens": 1}
        assert cli.main(["lint", str(recipe), "--portable"]) == 0
        assert cli.main(["exec", str(recipe), "--trust"]) == 0
