"""A host-prepared prompt remains complete without being appended to itself."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calcifier import agent, prepare_context
from calcifier.hashing import sha256_hex
from calcifier.instruction_context import instruction_context
from calcifier.prepared_context import prepared_context_run
from calcifier.registry import Registry
from calcifier.trace import active_run
from tests.support.agents import StubProvider


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("budget", [1024, 65536])
@pytest.mark.parametrize("form", ["context", "template", "custom", "partial"])
def test_agent_delivers_full_prepared_context_once_with_originals_and_all_answers(
    tmp_path: Path, newline: str, budget: int, form: str
) -> None:
    skill = tmp_path / "SKILL.md"
    skill_bytes = ("# Neutral review" + newline + "Interpret all records." + newline).encode()
    skill.write_bytes(skill_bytes)
    reference = tmp_path / "references/guide.txt"
    reference.parent.mkdir()
    labels = [f"distinct-concern-{index}" for index in range(9)]
    reference_bytes = (newline.join(labels) + newline).encode()
    reference.write_bytes(reference_bytes)
    interface = "Keep distinct concerns at the same input location." + newline
    prior = ("complete observations café" + newline) * 100 + newline.join(labels)
    registry = Registry()
    prompts = []

    class Capture(StubProvider):
        name = "worker"

        def invoke(self, prompt, *, filesystem=None, **kwargs):
            assert filesystem is not None
            prompts.append(prompt)
            assert context.prompt in prompt
            assert prompt.count(context.prompt) == 1
            assert Path(context.directory) in filesystem.readable_paths
            assert authority is not None
            assert authority.directory in filesystem.readable_paths
            manifest = json.loads((authority.directory / "manifest.json").read_bytes())
            assert (authority.directory / manifest["skill"]["path"]).read_bytes() == skill_bytes
            assert Path(manifest["references"][0]["path"]).read_bytes() == reference_bytes
            assert interface in prompt
            assert "Do only the assigned stage" in prompt
            task = (Path(context.directory) / "task.txt").read_bytes()
            assert prior.encode() in task
            if budget == 65536:
                assert prompt.count(prior) == 1
            else:
                assert prior not in prompt
            return super().invoke(prompt, filesystem=filesystem, **kwargs)

    registry.register(Capture(default=json.dumps(labels)))
    with (
        instruction_context(
            skill_path=skill,
            skill_sha256=sha256_hex(skill_bytes),
            skill_root=tmp_path,
            resources=(("references/guide.txt", sha256_hex(reference_bytes), None),),
            resource_root=tmp_path,
            interface=interface,
        ) as authority,
        prepared_context_run(),
        active_run(registry=registry),
    ):
        context = prepare_context("Interpret all records.", prior_results=prior, byte_budget=budget)
        if form == "context":
            prompt = context.prompt
        elif form == "template":
            prompt = "{prepared}"
        elif form == "partial":
            prompt = context.prompt[:80]
        else:
            prompt = "Review every record; retain distinct concerns."
        if form == "template":
            result = agent(
                "worker", prompt, context=context, output=list[str], prepared=context.prompt
            )
        else:
            result = agent("worker", prompt, context=context, output=list[str])
        assert result == labels
        assert len(prompts) == 1
        if form == "custom":
            assert prompt in prompts[0]
        assert authority is not None
        authority.validate()
