"""Generated recipes declare a runner environment without changing their program."""

from __future__ import annotations

import ast
import re
import sys
import tomllib

import pytest
from packaging.requirements import Requirement

from calcifier import __version__
from calcifier.agent_events import AgentActivity
from calcifier.artifact import prepare_artifact
from calcifier.calcify import calcify_source
from calcifier.metadata import attach_script_metadata
from calcifier.portable import lint_portable_source
from calcifier.registry import Registry
from calcifier.subprocesses import run_process
from tests.support.agents import StubProvider
from tests.support.compiler import ScriptedCompiler, one_draft
from tests.support.recipes import _GOOD_RECIPE

# Reference parser from the inline script metadata specification.
_BLOCK = re.compile(r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$")


def script_requirements(source: str) -> dict:
    blocks = [match for match in _BLOCK.finditer(source) if match["type"] == "script"]
    assert len(blocks) == 1
    return tomllib.loads(
        "".join(
            line[2:] if line.startswith("# ") else line[1:]
            for line in blocks[0]["content"].splitlines(keepends=True)
        )
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "python_version, minimum", [(None, "3.13"), ("3.11", "3.13"), ("3.14", "3.14")]
)
def test_requirements_replace_model_headers_preserving_program_bytes(
    newline, python_version, minimum
):
    prefix = f"#!/usr/bin/env python{newline}# coding: utf-8{newline}"
    program = f'"""Résumé."""{newline}from __future__ import annotations{newline}VALUE = 3{newline}'
    old_header = newline.join(
        ("# /// script", '# dependencies = ["unrelated-package"]', "# ///", "", "")
    )
    source = attach_script_metadata(prefix + old_header + program, python_version)

    assert source.startswith(prefix + "# /// script" + newline)
    assert source.endswith(program)
    assert attach_script_metadata(source, python_version) == source
    requirements = script_requirements(source.replace("\r\n", "\n"))
    assert requirements["requires-python"] == f">={minimum}"
    dependency = Requirement(requirements["dependencies"][0])
    assert dependency.name == "calcifier"
    assert str(dependency.specifier) == f"=={__version__}"
    assert len(requirements["dependencies"]) == 1


def test_metadata_shaped_string_data_is_preserved():
    source = 'VALUE = """\n# /// script\n# dependencies = []\n# ///\n"""\n'

    stamped = attach_script_metadata(source)

    original = ast.parse(source).body[0]
    result = ast.parse(stamped).body[0]
    assert isinstance(original, ast.Assign)
    assert isinstance(result, ast.Assign)
    assert ast.literal_eval(result.value) == ast.literal_eval(original.value)


def test_source_compilation_has_runner_requirements():
    source = calcify_source("Return whether a target was supplied.", one_draft(_GOOD_RECIPE))

    assert script_requirements(source)["dependencies"] == [f"calcifier=={__version__}"]


@pytest.mark.parametrize("optimize", [False, True])
def test_file_compilation_stamps_baselines_and_optimized_recipes(tmp_path, optimize):
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"# Target check\nReturn whether a target was supplied.\n")
    registry = Registry()
    registry.register(ScriptedCompiler([_GOOD_RECIPE], wrap_sources=True))
    worker = StubProvider()
    worker.name = "worker"
    registry.register(worker)
    activity = AgentActivity()

    artifact = prepare_artifact(skill, registry, optimize=optimize, activity=activity)

    assert script_requirements(artifact.source) == {
        "requires-python": ">=3.13",
        "dependencies": [f"calcifier=={__version__}"],
    }
    assert artifact.outcome == ("compiled" if optimize else "baseline_retained")
    assert lint_portable_source(artifact.source, artifact.path) == []
    if optimize:
        assert artifact.dry_run is not None and artifact.dry_run.status == "passed"
        artifact.path.parent.mkdir(parents=True)
        artifact.path.write_bytes(artifact.source.encode("utf-8"))
        result = run_process(
            [sys.executable, str(artifact.path), "--help"],
            timeout_seconds=20,
            max_output_bytes=10000,
        )
        assert result.returncode == 0
        assert b"--target" in result.stdout
