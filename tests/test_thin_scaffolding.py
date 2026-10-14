"""Direct compilation moves concrete I/O into Python and leaves judgment with agents."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pytest

from calcifier.agent_events import AgentProvider, AgentResult
from calcifier.artifact import calcify_file, prepare_artifact
from calcifier.baseline import is_baseline
from calcifier.portable import lint_portable
from calcifier.recipe import RecipeError, load_recipe
from calcifier.registry import Registry
from tests.support.compiler import ScriptedCompiler


class InspectInstructions(AgentProvider):
    name = "worker"
    default_timeout = 30

    def __init__(self, after=None):
        self.calls = []
        self.after = after
        self.lock = threading.Lock()

    def invoke(self, prompt, *, filesystem=None, **kwargs):
        del kwargs
        assert filesystem is not None
        match = re.search(r"Authoritative recipe sources: (.+?[/\\]manifest\.json)\. Read", prompt)
        assert match is not None
        manifest_path = Path(match[1])
        assert any(manifest_path.is_relative_to(root) for root in filesystem.readable_paths)
        manifest = json.loads(manifest_path.read_bytes())
        skill = (manifest_path.parent / manifest["skill"]["path"]).read_bytes()
        interface = manifest["interface"]
        interface_text = (
            None if interface is None else (manifest_path.parent / interface["path"]).read_bytes()
        )
        references = {
            entry["source"]: Path(entry["path"]).read_bytes() for entry in manifest["references"]
        }
        assert "Their constraints govern this stage" in prompt
        assert "Do only the assigned stage" in prompt
        with self.lock:
            self.calls.append((manifest_path, skill, interface_text, references, prompt))
            if self.after is not None:
                self.after()
        return AgentResult('"judged"', "", "", 0, self.name)


_SOURCE = """from __future__ import annotations

import sys

from calcifier import Budget, agent, recipe, resource, run_command, step

@step
def prepare() -> str:
    selected = run_command([sys.executable, str(resource("scripts/select.py"))],
                           timeout_seconds=15, mutating=False)
    descriptions = resource("descriptions/options.json").read_text(encoding="utf-8")
    return selected.stdout + "\\nDescriptions:\\n" + descriptions

@step
def interpret(prepared: str) -> str:
    return agent("worker", "Interpret the selection and descriptions: {prepared}",
                 working_directory=".", readable_paths=(), output=str, prepared=prepared)

@recipe(agents=("worker",), budget=Budget(2, 60))
def workflow() -> str:
    return interpret(prepare())

if __name__ == "__main__":
    raise SystemExit(workflow.cli())
"""


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_one_compiler_call_runs_packaged_python_and_prepares_a_reasoning_task(tmp_path, newline):
    skill = tmp_path / "SKILL.md"
    original = newline.join(
        [
            "---",
            "name: neutral-selection",
            "description: Prepare local inputs and interpret a selection.",
            "allowed-tools: Bash",
            "---",
            "# Neutral selection",
            "Run scripts/select.py, load {baseDir}/descriptions/, and interpret the result.",
            "Preserve both possibilities and explain uncertainty.",
            "",
        ]
    ).encode("utf-8")
    skill.write_bytes(original)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "select.py").write_bytes(
        b'import json, secrets\nprint(json.dumps(secrets.choice(["north", "south"])))\n'
    )
    descriptions = tmp_path / "descriptions"
    descriptions.mkdir()
    reference = '{"north": "caf\u00e9", "south": "r\u00e9sum\u00e9"}\n'.encode("utf-8")
    (descriptions / "options.json").write_bytes(reference)
    compiler = ScriptedCompiler([_SOURCE], wrap_sources=True, repeat_last=True)
    worker = InspectInstructions()
    registry = Registry()
    registry.register(compiler)
    registry.register(worker)

    path = calcify_file(skill, registry)
    workflow = load_recipe(path)
    run = workflow.run(registry=registry)

    assert len(compiler.calls) == 1
    assert not is_baseline(path.read_text(encoding="utf-8"))
    assert lint_portable(path) == []
    assert run.ok, run.error
    assert run.output == "judged"
    assert len(worker.calls) == 1
    _, delivered_skill, _, references, prompt = worker.calls[0]
    assert delivered_skill == original
    assert references["descriptions/options.json"] == reference
    prepared = prompt.split("Interpret the selection and descriptions: ", 1)[1]
    selection, delivered_descriptions = prepared.split("\nDescriptions:\n")
    assert json.loads(selection) in {"north", "south"}
    assert delivered_descriptions.startswith(reference.decode("utf-8"))
    assert skill.read_bytes() == original


@pytest.mark.parametrize("parallel", [False, True])
def test_independent_reasoning_calls_keep_the_original_skill_without_inventory(tmp_path, parallel):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral review\nObtain two independent judgments of the same input.\n")
    source = _SOURCE.replace("import sys", "import sys\nfrom calcifier import parallel")
    source = source.replace("budget=Budget(2, 60)", "budget=Budget(4, 60)")
    source = source.replace("def workflow() -> str:", "def workflow() -> list[str]:")
    source = source.replace(
        "    return interpret(prepare())",
        '    return parallel(interpret, ["First independent judgment", '
        '"Second independent judgment"])'
        if parallel
        else '    return [interpret("First independent judgment"), '
        'interpret("Second independent judgment")]',
    )
    # The unused local preparation function is omitted in this separate workflow.
    start, end = source.index("@step\ndef prepare"), source.index("@step\ndef interpret")
    source = source[:start] + source[end:]
    compiler = ScriptedCompiler([source], wrap_sources=True)
    worker = InspectInstructions()
    registry = Registry()
    registry.register(compiler)
    registry.register(worker)

    path = calcify_file(skill, registry)
    run = load_recipe(path).run(registry=registry)

    assert len(compiler.calls) == 1
    assert run.ok, run.error
    assert run.output == ["judged", "judged"]
    assert len(worker.calls) == 2
    assert all(call[1] == skill.read_bytes() for call in worker.calls)


def test_direct_recipe_delivers_the_host_interface_without_obligation_bindings(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral\nInterpret the supplied task.\n")
    source = (
        _SOURCE[: _SOURCE.index("@step\ndef prepare")]
        + _SOURCE[_SOURCE.index("@step\ndef interpret") :]
    )
    source = source.replace("return interpret(prepare())", 'return interpret("Prepared task")')
    compiler = ScriptedCompiler([source], wrap_sources=True)
    worker = InspectInstructions()
    registry = Registry()
    registry.register(compiler)
    registry.register(worker)
    interface = "Keep both alternatives and describe uncertainty.\r\n"
    artifact = prepare_artifact(skill, registry, interface_requirements=interface)
    artifact.path.parent.mkdir(parents=True, exist_ok=True)
    artifact.path.write_bytes(artifact.source.encode("utf-8"))
    run = load_recipe(artifact.path).run(registry=registry)
    assert run.ok, run.error
    assert worker.calls[0][2] == interface.encode("utf-8")
    assert interface in worker.calls[0][-1]


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("count", [1, 5, 9])
def test_partial_search_results_keep_full_reference_and_all_agent_findings(
    tmp_path, monkeypatch, newline, count
):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(
        newline.join(
            [
                "# Neutral review",
                "Read references/guide.txt and interpret every rule against the input.",
                "A search hit is only an observation; also review rules with no literal trigger.",
                "Keep distinct concerns at the same location in the final report.",
                "",
            ]
        ).encode("utf-8")
    )
    labels = [f"concern-{index}" for index in range(count)]
    reference = newline.join(labels).encode("utf-8")
    (tmp_path / "references").mkdir()
    (tmp_path / "references/guide.txt").write_bytes(reference)
    source = """import json

from calcifier import (
    Budget, agent, emit, prepare_context, recipe, resource, step, write_json,
)

@step
def search() -> str:
    text = resource("references/guide.txt").read_bytes().decode("utf-8")
    return json.dumps({"reference_text": text, "search_hits": text.splitlines()[:1]})

@step
def deliver(findings: list[str]) -> list[str]:
    write_json("report.json", [{"location": "same input", "concern": item} for item in findings])
    return emit(findings, name="concerns")

@recipe(agents=("worker",), budget=Budget(2, 60), output_root="out")
def review() -> list[str]:
    context = prepare_context("Review the full input using the original guide.",
                              prior_results=search())
    findings = agent("worker", "Perform the assigned review.", output=list[str], context=context)
    return deliver(findings)

if __name__ == "__main__":
    raise SystemExit(review.cli())
"""

    class ReferenceWorker(InspectInstructions):
        def invoke(self, prompt, *, filesystem=None, **kwargs):
            super().invoke(prompt, filesystem=filesystem, **kwargs)
            assert self.calls[-1][3]["references/guide.txt"] == reference
            context = re.search(r"Prepared context directory: (.+)\n", prompt)
            assert context is not None
            task = (Path(context[1]) / "task.txt").read_text(encoding="utf-8")
            prepared = json.loads(task.split("Prior results:\n", 1)[1])
            assert prepared["reference_text"].encode("utf-8") == reference
            assert prepared["search_hits"] == labels[:1]
            # Model the reasoning result independently of the partial mechanical search.
            return AgentResult(json.dumps(labels), "", "", 0, self.name)

    compiler = ScriptedCompiler([source], wrap_sources=True)
    worker = ReferenceWorker()
    registry = Registry()
    registry.register(compiler)
    registry.register(worker)
    path = calcify_file(skill, registry)
    monkeypatch.chdir(tmp_path)
    run = load_recipe(path).run(registry=registry)

    assert run.ok, run.error
    assert run.output == labels
    assert json.loads((tmp_path / "out/report.json").read_bytes()) == [
        {"location": "same input", "concern": label} for label in labels
    ]


@pytest.mark.parametrize("changed", ["skill", "reference"])
def test_unbound_recipe_rejects_changed_originals_before_calling_an_agent(tmp_path, changed):
    skill = tmp_path / "INSTRUCTIONS.md"
    skill.write_bytes(b"# Neutral\nInterpret the complete task.\n")
    references = tmp_path / "references"
    references.mkdir()
    reference = references / "rules.md"
    reference.write_bytes(b"Preserve uncertainty.\n")
    source = (
        _SOURCE[: _SOURCE.index("@step\ndef prepare")]
        + _SOURCE[_SOURCE.index("@step\ndef interpret") :]
    )
    source = source.replace("return interpret(prepare())", 'return interpret("Prepared task")')
    compiler = ScriptedCompiler([source], wrap_sources=True)
    worker = InspectInstructions()
    registry = Registry()
    registry.register(compiler)
    registry.register(worker)
    path = calcify_file(skill, registry)
    target = skill if changed == "skill" else reference
    target.write_bytes(b"Changed original instructions.\n")
    if changed == "reference":
        with pytest.raises(RecipeError, match="packaged resource drift"):
            load_recipe(path).run(registry=registry)
    else:
        run = load_recipe(path).run(registry=registry)
        assert not run.ok
    assert worker.calls == []


@pytest.mark.parametrize("digest", [None, "", 123])
def test_unbound_recipe_rejects_missing_or_invalid_source_hash_at_load(tmp_path, digest):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Neutral\nInterpret the task.\n")
    source = "from calcifier import recipe\n@recipe\ndef workflow() -> bool:\n    return True\n"
    metadata = {"path": "SKILL.md"}
    if digest is not None:
        metadata["sha256"] = digest
    path = tmp_path / "recipe.py"
    path.write_text(source + "_CALCIFIER_SKILL = " + repr(metadata) + "\n", encoding="utf-8")
    with pytest.raises(RecipeError, match="nonempty sha256"):
        load_recipe(path)
