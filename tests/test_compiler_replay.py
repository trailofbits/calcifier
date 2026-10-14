"""Neutral compiler recordings preserve bytes and exercise real offline checks."""

from __future__ import annotations

import json

import pytest

from calcifier.agent_events import AgentActivity
from calcifier.recipe_draft import draft_response
from scripts.compiler_replay import CompilerTape, create_packet, main, run_packet
from tests.support.compiler import ScriptedCompiler
from tests.support.recipes import _GOOD_RECIPE

pytestmark = pytest.mark.semantic_offline


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_recorded_repairs_replay_with_exact_skill_and_resource_bytes(tmp_path, newline):
    skill = tmp_path / "SKILL.md"
    skill_bytes = ("# Neutral task\nReview the input.\n").replace("\n", newline).encode("utf-8")
    skill.write_bytes(skill_bytes)
    reference = tmp_path / "references" / "note.txt"
    reference.parent.mkdir()
    resource_bytes = "café\ncomplete\n".replace("\n", newline).encode("utf-8")
    reference.write_bytes(resource_bytes)
    directory = tmp_path / "packet"
    packet = create_packet(skill, directory)
    provider = ScriptedCompiler(["malformed", draft_response(_GOOD_RECIPE)])
    tape = CompilerTape(packet, AgentActivity(), recording=(directory / "packet.json", provider))
    recorded = run_packet(directory, directory / "recorded", tape)
    assert recorded["status"] == "compiler_accepted", recorded
    assert recorded["responses_consumed"] == recorded["provider_invocations"] == 2
    assert (directory / "inputs" / "SKILL.md").read_bytes() == skill_bytes
    assert (directory / "inputs" / "references" / "note.txt").read_bytes() == resource_bytes
    recorded_packet = (directory / "packet.json").read_bytes()
    assert json.loads(recorded_packet)["calls"][0]["response"] == "malformed"
    assert main(["replay", str(directory), "--output", str(tmp_path / "replayed")]) == 0
    replayed = json.loads((tmp_path / "replayed" / "result.json").read_bytes())
    assert replayed["status"] == "compiler_accepted"
    assert replayed["responses_consumed"] == 2 and replayed["provider_invocations"] == 0
    assert replayed["request_changes"] == []
    assert (tmp_path / "replayed" / "candidate.py").read_bytes() == (
        directory / "recorded" / "candidate.py"
    ).read_bytes()
    assert (directory / "packet.json").read_bytes() == recorded_packet


@pytest.mark.parametrize("change", ["prompt", "schema", "incomplete", "phase"])
def test_replay_rejects_changed_or_incomplete_requests(tmp_path, change):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral task\nReview the input.\n")
    directory = tmp_path / "packet"
    packet = create_packet(skill, directory)
    tape = CompilerTape(
        packet,
        AgentActivity(),
        recording=(directory / "packet.json", ScriptedCompiler([draft_response(_GOOD_RECIPE)])),
    )
    assert run_packet(directory, directory / "recorded", tape)["status"] == "compiler_accepted"
    call = packet["calls"][0]
    if change == "prompt":
        call["prompt"] += "\nChanged instructions"
    elif change == "schema":
        call["schema"] = {"type": "integer"}
    elif change == "incomplete":
        call["response"] = None
    else:
        call["phase"] = "source.repair_1"
    result = run_packet(directory, tmp_path / "replayed", CompilerTape(packet, AgentActivity()))
    assert result["status"] == "replay_diverged", result
    assert result["responses_consumed"] == result["provider_invocations"] == 0
    assert not (tmp_path / "replayed" / "candidate.py").exists()
