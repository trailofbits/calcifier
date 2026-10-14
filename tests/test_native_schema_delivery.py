"""Native schema delivery keeps validation without repeating schema text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import pytest

from calcifier import agent_providers, output_type, word_count
from calcifier.agent_events import AgentResult
from calcifier.agent_providers import CodexCli
from calcifier.filesystem_contract import FilesystemContract
from tests.support.agents import StubProvider


@output_type
class _Record:
    location: str
    conclusion: Annotated[str, word_count(2, 8)]
    state: Literal["confirmed", "unresolved"]


@output_type
class _Summary:
    records: list[_Record]
    limitation: str | None = None


class _SchemaCapture(StubProvider):
    def __init__(self, answers: list[str], native: bool) -> None:
        super().__init__()
        self.native_structured_output = native
        self.answers = iter(answers)
        self.calls: list[tuple[str, dict]] = []

    def invoke(self, prompt, *, schema=None, **kwargs):
        del kwargs
        assert schema is not None
        self.calls.append((prompt, schema))
        text = next(self.answers)
        return AgentResult(text, "", "", 0, self.name)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("open_mapping", [False, True])
@pytest.mark.parametrize("retry", [False, True])
def test_native_and_inline_schema_delivery_keep_all_constraints_and_complete_answers(
    newline: str, native: bool, open_mapping: bool, retry: bool
) -> None:
    records = [
        {"location": "same café", "conclusion": f"distinct concern {i}", "state": "confirmed"}
        for i in range(9)
    ]
    accepted = {"records": records, "limitation": None}
    rejected = {**accepted, "records": [{**records[0], "conclusion": "short"}]}

    def wire(value):
        if open_mapping:
            return json.dumps({"value": json.dumps({"group": value}), "_calcifier_json": True})
        return json.dumps(value)

    provider = _SchemaCapture(([wire(rejected)] if retry else []) + [wire(accepted)], native=native)
    task = "Review all records." + newline + "Preserve every distinct concern."
    if open_mapping:
        result = provider.structured(task, annotation=dict[str, _Summary])["group"]
    else:
        result = provider.structured(task, annotation=_Summary)
    assert isinstance(result, _Summary)
    assert len(result.records) == 9
    assert [record.conclusion for record in result.records] == [r["conclusion"] for r in records]
    assert len(provider.calls) == (2 if retry else 1)
    for prompt, native_schema in provider.calls:
        assert prompt.startswith(task)
        assert ("\n\nSchema:\n" in prompt) is not native
        assert native_schema["additionalProperties"] is False
        if open_mapping:
            description = native_schema["properties"]["value"]["description"]
            complete_schema = json.loads(description.split("schema: ", 1)[1])
            summary_schema = complete_schema["$defs"]["_Summary"]
            assert native_schema["properties"]["_calcifier_json"]["const"] is True
        else:
            complete_schema = native_schema
            summary_schema = native_schema
        properties = complete_schema["$defs"]["_Record"]["properties"]
        assert properties["state"]["enum"] == ["confirmed", "unresolved"]
        assert "at most 8 words" in properties["conclusion"]["description"]
        assert "limitation" in summary_schema["required"]
        if not native:
            inline = prompt.split("\n\nSchema:\n", 1)[1].split("\n", 1)[0]
            assert json.loads(inline) == native_schema
    if retry:
        assert provider.calls[1][0].startswith(provider.calls[0][0])
        assert provider.calls[1][1] == provider.calls[0][1]


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("open_mapping", [False, True])
@pytest.mark.parametrize("retry", [False, True])
def test_builtin_codex_delivers_complete_schema_through_cli_file_without_inline_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    newline: str,
    open_mapping: bool,
    retry: bool,
) -> None:
    records = [
        {"location": "same café", "conclusion": f"distinct concern {i}", "state": "confirmed"}
        for i in range(9)
    ]
    accepted = {"records": records, "limitation": None}
    rejected = {**accepted, "records": [{**records[0], "conclusion": "short"}]}

    def wire(value):
        if open_mapping:
            return json.dumps({"value": json.dumps({"group": value}), "_calcifier_json": True})
        return json.dumps(value)

    answers = iter(([wire(rejected)] if retry else []) + [wire(accepted)])
    calls = []
    filesystem = FilesystemContract.build(tmp_path, ())
    provider = CodexCli(role="worker", default_timeout=5)
    annotation = dict[str, _Summary] if open_mapping else _Summary
    expected_schema = provider._structured_schema(annotation)
    task = "Review every record." + newline + "Keep each distinct concern."

    def transport(command, stdin, *, filesystem, **kwargs):
        schema_path = Path(command[command.index("--output-schema") + 1])
        schema = json.loads(schema_path.read_bytes())
        assert schema == expected_schema
        assert schema["additionalProperties"] is False
        assert filesystem.working_directory == tmp_path
        assert stdin.startswith(task)
        assert "\n\nSchema:\n" not in stdin
        calls.append((stdin, schema, schema_path))
        answer = next(answers)
        return AgentResult(answer, answer, "", 0, kwargs["name"])

    monkeypatch.setattr(agent_providers, "_run_cli", transport)
    monkeypatch.setattr(agent_providers, "_seed_codex_auth", lambda _: None)
    result = provider.structured(task, annotation=annotation, filesystem=filesystem)
    if open_mapping:
        result = result["group"]
    assert isinstance(result, _Summary)
    assert [(record.location, record.conclusion, record.state) for record in result.records] == [
        (record["location"], record["conclusion"], record["state"]) for record in records
    ]
    assert result.limitation is None
    assert len(calls) == (2 if retry else 1)
    assert all(not path.exists() for _, _, path in calls)
    if retry:
        assert calls[1][0].startswith(calls[0][0])
        assert calls[1][1] == calls[0][1]
