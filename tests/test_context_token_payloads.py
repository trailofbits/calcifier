"""Lossless prompt reductions keep complete assignments and answer constraints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import pytest

from calcifier import output_type, prepare_context, word_count
from calcifier.agent_events import AgentResult
from calcifier.hashing import sha256_hex
from calcifier.instruction_context import instruction_context
from calcifier.prepared_context import prepared_context_run
from tests.support.agents import StubProvider


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("large_skill,large_task", [(False, False), (True, False), (False, True)])
def test_context_says_when_complete_text_is_embedded_and_preserves_file_fallback(
    tmp_path: Path, newline: str, large_skill: bool, large_task: bool
) -> None:
    skill = "# Neutral\nInterpret every record, including café.\n" * (100 if large_skill else 1)
    skill = skill.replace("\n", newline)
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_bytes(skill.encode("utf-8"))
    interface = "Report all distinct concerns, including shared locations." + newline
    task = ("first record" + newline) * (100 if large_task else 1) + "last record"
    prior = "complete reference" + newline + "last rule has no keyword hit"
    assignment = f"Task:\n{task}\n\nInstructions:\n\n\nPrior results:\n{prior}\n"
    with (
        instruction_context(
            skill_path=skill_path,
            skill_sha256=sha256_hex(skill_path.read_bytes()),
            skill_root=tmp_path,
            resources=(),
            resource_root=tmp_path,
            interface=interface,
        ) as authority,
        prepared_context_run(),
    ):
        assert authority is not None
        context = prepare_context(task, prior_results=prior, byte_budget=1024)
        assert len(context.prompt.encode("utf-8")) <= 1024
        assert (Path(context.directory) / "task.txt").read_bytes() == assignment.encode("utf-8")
        manifest = json.loads((authority.directory / "manifest.json").read_bytes())
        assert (authority.directory / manifest["skill"]["path"]).read_bytes() == skill.encode()
        delivered_interface = authority.directory / manifest["interface"]["path"]
        assert delivered_interface.read_bytes() == interface.encode()
        assert "Use embedded SKILL.md and interface in full" in context.prompt
        assert "otherwise read their files" in context.prompt
        assert "Read applicable references" in context.prompt
        assert "Their constraints govern this stage" in context.prompt
        assert "Use embedded assignment in full" in context.prompt
        assert "otherwise read task.txt" in context.prompt
        if large_skill:
            assert skill not in context.prompt
        else:
            assert context.prompt.count(skill) == 1
            assert context.prompt.count(interface) == 1
        if large_task:
            assert assignment not in context.prompt
        else:
            assert context.prompt.count(assignment) == 1
        # Large budgets still deliver every byte, rather than substituting a preview.
        complete = prepare_context(task, prior_results=prior, byte_budget=65536)
        assert complete.prompt.count(skill) == complete.prompt.count(assignment) == 1
        authority.validate()


@output_type
class _Observation:
    location: str
    conclusion: Annotated[str, word_count(2, 8)]
    state: Literal["confirmed", "unresolved"]


@output_type
class _Judgments:
    observations: list[_Observation]
    limitation: str | None = None


class _CaptureSchemas(StubProvider):
    def __init__(self, answers: list[str]) -> None:
        super().__init__()
        self.answers = iter(answers)
        self.calls: list[tuple[str, dict]] = []

    def invoke(self, prompt, *, schema=None, **kwargs):
        del kwargs
        assert schema is not None
        self.calls.append((prompt, schema))
        text = next(self.answers)
        return AgentResult(text, "", "", 0, self.name)


def test_compact_schema_keeps_nested_constraints_and_full_answer_after_retry() -> None:
    observations = [
        {"location": "same café", "conclusion": f"distinct concern {index}", "state": "confirmed"}
        for index in range(9)
    ]
    accepted = {"observations": observations, "limitation": None}
    rejected = {**accepted, "observations": [{**observations[0], "conclusion": "short"}]}
    provider = _CaptureSchemas([json.dumps(rejected), json.dumps(accepted)])
    result = provider.structured("Review every record.", annotation=_Judgments)

    assert isinstance(result, _Judgments)
    assert len(result.observations) == 9
    assert [o.conclusion for o in result.observations] == [o["conclusion"] for o in observations]
    assert len(provider.calls) == 2
    for prompt, native_schema in provider.calls:
        text = prompt.split("\n\nSchema:\n", 1)[1].split("\n", 1)[0]
        assert json.loads(text) == native_schema
        assert len(text.encode()) < 0.75 * len(json.dumps(native_schema, indent=2).encode())
        properties = native_schema["$defs"]["_Observation"]["properties"]
        assert properties["state"]["enum"] == ["confirmed", "unresolved"]
        assert "at most 8 words" in properties["conclusion"]["description"]
        assert native_schema["additionalProperties"] is False
        assert "limitation" in native_schema["required"]
    assert provider.calls[1][0].startswith(provider.calls[0][0])
