"""A statically accepted recipe runs once against schema-valid stub agents."""

from __future__ import annotations

import importlib
import json
import textwrap
from pathlib import Path
from typing import Annotated, Any

import pytest

import calcifier.dry_run as dry_run
from calcifier import composition
from calcifier.dry_run import DryRunResult, dry_run_recipe
from calcifier.hashing import sha256_hex
from calcifier.outputs import output_type, parse_response, strict_schema_for, synthesize, word_count
from calcifier.recipe import RecipeRun
from calcifier.run_result import RunException
from calcifier.side_effects import SideEffect, SideEffectKind
from calcifier.skill_package import ResourceEntry
from calcifier.subprocesses import ProcessError, ProcessResult
from calcifier.trace import RunHooks, StepRecord

pytestmark = pytest.mark.semantic_offline

# ``calcifier.recipe`` as an attribute is the exported decorator, not the module.
recipe_module = importlib.import_module("calcifier.recipe")


@output_type
class _Finding:
    title: str
    score: int
    ratio: float
    kind: str
    flagged: bool


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"type": "string"}, "stub"),
        ({"type": "string", "minLength": 6}, "stubss"),
        ({"type": "integer"}, 1),
        ({"type": "integer", "minimum": 5}, 5),
        ({"type": "integer", "minimum": -3}, 1),
        ({"type": "number"}, 1.0),
        ({"type": "boolean"}, False),
        ({"type": "null"}, None),
        ({"type": ["null", "integer"]}, 1),
        ({"type": ["null"]}, None),
        ({"const": "fixed"}, "fixed"),
        ({"enum": ["first", "second"]}, "first"),
        ({"anyOf": [{"type": "null"}, {"type": "integer"}]}, 1),
        ({"oneOf": [{"type": "null"}]}, None),
        ({"type": "array", "items": {"type": "integer"}}, [1]),
        ({"type": "array", "items": {"type": "integer"}, "minItems": 3}, [1, 1, 1]),
        ({"type": "array", "items": {"type": "integer"}, "minItems": 0}, [1]),
        ({"type": "array"}, [None]),
        ({"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}]}, ["stub", 1]),
        ({"type": "object"}, {}),
        ({"type": "object", "additionalProperties": {"type": "integer"}}, {"stub": 1}),
        ({"type": "object", "properties": {"a": {"type": "integer"}}}, {"a": 1}),
        ({"$defs": {"Leaf": {"type": "integer"}}, "$ref": "#/$defs/Leaf"}, 1),
    ],
)
def test_synthesized_values_satisfy_their_schema(schema: dict[str, Any], expected: Any) -> None:
    assert synthesize(schema) == expected


def test_synthesis_of_an_output_type_fills_every_field() -> None:
    value = synthesize(strict_schema_for(list[_Finding]))

    assert value == [{"title": "stub", "score": 1, "ratio": 1.0, "kind": "stub", "flagged": False}]


def test_nesting_stops_exactly_past_the_depth_limit() -> None:
    schema: dict[str, Any] = {"type": "integer"}
    for _ in range(40):
        schema = {"type": "array", "items": schema}

    value = synthesize(schema)
    depth = 0
    while isinstance(value, list):
        value = value[0]
        depth += 1

    assert (depth, value) == (32 + 1, None)


def test_numbers_keep_their_json_type() -> None:
    assert type(synthesize({"type": "integer"})) is int
    assert type(synthesize({"type": "number"})) is float


def test_self_referencing_schemas_stop_at_the_depth_limit() -> None:
    schema = {
        "$defs": {"Node": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
        "$ref": "#/$defs/Node",
    }

    value = synthesize(schema)
    depth = 0
    while isinstance(value, list):
        value = value[0]
        depth += 1

    assert value is None
    assert depth == 16


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [(Path, "workspace"), (bool, False), (int, 1), (float, 1.0), (str, "text"), (None, "text")],
)
def test_required_inputs_are_synthesized_by_annotation(
    annotation: object, expected: object, tmp_path: Path
) -> None:
    value = dry_run._input_value(annotation, tmp_path)

    if expected == "workspace":
        assert value == tmp_path
    elif expected == "text":
        assert value == "stub"
    else:
        assert value == expected
        assert type(value) is type(expected)


def test_unsupported_input_annotations_cannot_be_synthesized(tmp_path: Path) -> None:
    assert dry_run._input_value(list[object], tmp_path) is None


def _record(name: str, error: str | None = None, *children: StepRecord) -> StepRecord:
    return StepRecord(name=name, inputs={}, error=error, children=list(children))


@pytest.mark.parametrize(
    ("error", "records", "expected"),
    [
        (None, [], DryRunResult("passed")),
        (
            "IndexError: list index out of range",
            [_record("outer", "IndexError: outer", _record("inner", "IndexError: at line 4"))],
            DryRunResult("failed", "inner", "IndexError: at line 4"),
        ),
        (
            "BudgetExceeded: agent call budget exhausted",
            [_record("ok"), _record("spend", "BudgetExceeded: at line 9")],
            DryRunResult("failed", "spend", "BudgetExceeded: at line 9"),
        ),
        (
            "SomeException: boom",
            [],
            DryRunResult("failed", None, "SomeException: boom"),
        ),
        (
            "AgentError: worker returned no valid structured response after one retry",
            [_record("ask", "AgentError: x")],
            DryRunResult(
                "unverified",
                "ask",
                "AgentError: worker returned no valid structured response after one retry",
            ),
        ),
        (
            "no candidates were found",
            [_record("gate", "no candidates were found")],
            DryRunResult("unverified", "gate", "no candidates were found"),
        ),
    ],
)
def test_run_outcomes_distinguish_defects_from_stub_limits(
    error: str | None, records: list[StepRecord], expected: DryRunResult
) -> None:
    exception = (
        RunException(error.split(":", 1)[0]) if error and expected.status == "failed" else None
    )
    run = RecipeRun(name="recipe", inputs={}, error=error, exception=exception, records=records)

    assert dry_run._outcome(run) == expected


def test_file_writes_follow_execution_order_through_nested_steps() -> None:
    def effect(kind: SideEffectKind, target: str) -> SideEffect:
        return SideEffect(kind, target, idempotency_key=target)

    outer = _record("outer")
    outer.side_effects = [effect("file_write", "a.md")]
    inner = _record("inner")
    inner.side_effects = [effect("publish", "remote"), effect("file_write", "b.md")]
    outer.children = [inner]
    last = _record("last")
    last.side_effects = [effect("file_write", "a.md"), effect("file_write", "c.md")]

    assert dry_run._file_writes([outer, last]) == ("a.md", "b.md", "c.md")


def test_the_innermost_failing_step_is_reported() -> None:
    records = [
        _record("first"),
        _record("second", "E: outer", _record("clean"), _record("deeper", "E: inner")),
    ]

    assert dry_run._failing_step(records) == ("deeper", "E: inner")
    assert dry_run._failing_step([_record("only")]) == (None, None)


def test_an_error_raised_outside_the_candidate_keeps_its_plain_text(tmp_path: Path) -> None:
    try:
        int("x")
    except ValueError as exc:
        located = recipe_module._exception_location(exc, tmp_path / "candidate.py")

    assert located is None


def _fake_process(
    monkeypatch: pytest.MonkeyPatch, stdout: bytes, stderr: bytes = b""
) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: Any) -> ProcessResult:
        calls.append(list(argv))
        assert Path(argv[3]).is_file()
        assert kwargs["cwd"] == str(Path(argv[3]).parent)
        return ProcessResult(tuple(argv), 0, stdout, stderr)

    monkeypatch.setattr(dry_run, "run_process", run)
    return calls


def test_the_subprocess_outcome_is_the_last_json_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcome = {
        "status": "failed",
        "step": "persist",
        "error": "KeyError: 'x'",
        "writes": ["<workspace>/a.md"],
    }
    calls = _fake_process(monkeypatch, b"noise\n" + json.dumps(outcome).encode() + b"\n")
    recipe_path = tmp_path / "missing" / ".calcifier" / "recipe.py"

    result = dry_run_recipe("value = 1\n", recipe_path=recipe_path, skill_path=tmp_path / "S.md")

    assert result == DryRunResult("failed", "persist", "KeyError: 'x'", ("<workspace>/a.md",))
    (argv,) = calls
    assert argv[1:3] == ["-m", "calcifier.dry_run"]
    assert argv[4:] == [str(tmp_path / "S.md"), str(recipe_path)]
    # The candidate is staged elsewhere; the destination tree is never created.
    assert not Path(argv[3]).is_relative_to(tmp_path)
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize(
    ("stdout", "stderr", "detail"),
    [
        (b"", b"", "no result"),
        (b"Traceback", b"ImportError: boom", "ImportError: boom"),
        (b'{"step": "x"}', b"", "no result"),
        (b"[1]", b"", "no result"),
    ],
)
def test_a_subprocess_without_a_readable_outcome_crashed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stdout: bytes, stderr: bytes, detail: str
) -> None:
    _fake_process(monkeypatch, stdout, stderr)

    result = dry_run_recipe("value = 1\n", recipe_path=tmp_path / "r.py", skill_path=tmp_path)

    assert result == DryRunResult("failed", None, f"dry run crashed: {detail}")


def test_a_subprocess_that_does_not_finish_fails_and_is_cleaned_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail(argv: list[str], **_kwargs: Any) -> ProcessResult:
        raise ProcessError("timed out after 120s", stderr_tail=b"")

    monkeypatch.setattr(dry_run, "run_process", fail)

    result = dry_run_recipe("value = 1\n", recipe_path=tmp_path / "r.py", skill_path=tmp_path)

    assert result == DryRunResult("failed", None, "dry run did not finish: timed out after 120s")
    assert list(tmp_path.iterdir()) == []


_RECIPE = textwrap.dedent(
    """
    from pathlib import Path

    from calcifier import Budget, agent, output_type, recipe, resource, step, tool

    @output_type
    class Finding:
        title: str

    @step
    def review(working_directory: str) -> list[Finding]:
        return agent(
            "worker",
            "Review.",
            output=list[Finding],
            working_directory=working_directory,
            readable_paths=(working_directory,),
        )

    @step
    def render(findings: list[Finding], guide: str, count: int) -> str:
        return guide + findings[0].title + str(tool("Grep", pattern="x")) + str(count)

    @recipe(agents=("worker",), budget=Budget(2, 60), capabilities=("Grep",))
    def run(working_directory: str, flag: bool, count: int = 3) -> str:
        guide = resource("references/guide.md").read_text()
        return render(review(working_directory), guide, count)
    """
)


def _skill(tmp_path: Path) -> Path:
    skill = tmp_path / "skill" / "SKILL.md"
    (skill.parent / "references").mkdir(parents=True)
    (skill.parent / "references" / "guide.md").write_text("guide:", encoding="utf-8")
    skill.write_text("Review the workspace.\n", encoding="utf-8")
    return skill


def _harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str) -> tuple[Path, Path]:
    del monkeypatch
    skill = _skill(tmp_path)
    candidate = skill.parent / ".calcifier" / "candidate.py"
    candidate.parent.mkdir()
    candidate.write_text(source, encoding="utf-8")
    return candidate, skill


def test_the_harness_runs_the_recipe_with_stub_agents_tools_and_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate, skill = _harness(monkeypatch, tmp_path, _RECIPE)

    assert dry_run._run(candidate, skill, candidate) == DryRunResult("passed")
    assert composition._verify_dependency is not dry_run._skip_dependency_pin


def test_the_harness_reports_a_crash_in_its_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _RECIPE.replace("findings[0].title", "findings[2].title")
    candidate, skill = _harness(monkeypatch, tmp_path, source)
    line = next(
        (number, text.strip())
        for number, text in enumerate(source.splitlines(), start=1)
        if "findings[2].title" in text
    )

    result = dry_run._run(candidate, skill, candidate)

    # The error names the candidate line that raised, so a repair can find it.
    expected = f"IndexError: list index out of range (line {line[0]}: {line[1]})"
    assert result == DryRunResult("failed", "render", expected)


def test_the_harness_reports_a_child_recipe_crash_as_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent = textwrap.dedent(
        """
        from calcifier import call_recipe, recipe, step


        @step
        def delegate() -> bool:
            return call_recipe("child.py", {}).output


        @recipe
        def run() -> bool:
            return delegate()
        """
    )
    staged, skill = _harness(monkeypatch, tmp_path, parent)
    # The candidate is staged apart from where the recipe will live; its child sits there.
    candidate = tmp_path / "staging" / "candidate.py"
    candidate.parent.mkdir()
    staged.rename(candidate)
    recipe_path = staged.parent / "recipe.py"
    (recipe_path.parent / "child.py").write_text(
        "from calcifier import recipe\n\n\n@recipe\ndef child() -> bool:\n"
        '    return {}["missing"]\n\n\n'
        "# --- calcifier metadata (skill provenance + sha256) ---\n"
        f"_CALCIFIER_SKILL = {{'path': {str(skill)!r}, "
        f"'sha256': {sha256_hex(skill.read_bytes())!r}}}\n",
        encoding="utf-8",
    )
    line = next(n for n, text in enumerate(parent.splitlines(), 1) if "call_recipe(" in text)

    result = dry_run._run(candidate, skill, recipe_path)

    # A crash inside the child is a defect to repair, not a stub limitation to skip.
    assert result == DryRunResult(
        "failed",
        "delegate",
        "SubrecipeCrashError: subrecipe child failed: KeyError: 'missing' "
        f'(line {line}: return call_recipe("child.py", {{}}).output)',
    )


_WRITER = textwrap.dedent(
    """
    from pathlib import Path

    from calcifier import recipe, step, write_text


    @step
    def save(output_dir: str) -> Path:
        return write_text(f"{Path(output_dir).name}/entry.md", "entry")


    @recipe(output_root=".")
    def run(output_dir: str) -> Path:
        return save(output_dir)
    """
)


@pytest.mark.parametrize(
    ("annotation", "write"),
    [("str", "<workspace>/stub/entry.md"), ("Path", "<workspace>/input/entry.md")],
)
def test_a_relative_output_root_is_the_dry_run_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, annotation: str, write: str
) -> None:
    source = _WRITER.replace("output_dir: str", f"output_dir: {annotation}")
    candidate, skill = _harness(monkeypatch, tmp_path, source)
    before = Path.cwd()

    # Derive an output-root-relative destination from the separate input's name.
    assert dry_run._run(candidate, skill, candidate) == DryRunResult("passed", writes=(write,))
    assert Path.cwd() == before


@pytest.mark.parametrize(
    ("annotation", "missing"),
    [("str", Path("stub/missing.md")), ("Path", "<input>/missing.md")],
)
def test_dry_run_errors_name_the_workspace_instead_of_its_temporary_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, annotation: str, missing: str | Path
) -> None:
    source = _WRITER.replace(
        'return write_text(f"{Path(output_dir).name}/entry.md", "entry")',
        'return Path(output_dir, "missing.md").read_text()',
    ).replace("output_dir: str", f"output_dir: {annotation}")
    candidate, skill = _harness(monkeypatch, tmp_path, source)

    result = dry_run._run(candidate, skill, candidate)

    assert result.status == "failed"
    assert f"No such file or directory: {str(missing)!r}" in (result.error or "")


def test_placeholders_replace_the_longest_spelling_first() -> None:
    # On macOS the resolved workspace (/private/var/...) contains the unresolved one.
    placeholders = {"/var/w": "<workspace>", "/private/var/w": "<workspace>"}
    result = DryRunResult("failed", "step", "cannot open /private/var/w/a and /var/w/b")

    assert dry_run._with_placeholders(result, placeholders) == DryRunResult(
        "failed", "step", "cannot open <workspace>/a and <workspace>/b"
    )


def test_placeholders_cover_windows_paths_in_writes_and_exception_text() -> None:
    workspace = "C:\\Users\\runner\\work"
    missing = f"{workspace}\\missing.md"
    result = DryRunResult(
        "failed",
        "save",
        f"FileNotFoundError: [Errno 2] No such file or directory: {missing!r}",
        (missing,),
    )

    assert dry_run._with_placeholders(result, {workspace: "<workspace>"}) == DryRunResult(
        "failed",
        "save",
        "FileNotFoundError: [Errno 2] No such file or directory: '<workspace>/missing.md'",
        ("<workspace>/missing.md",),
    )


def test_the_harness_reports_a_recipe_that_cannot_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate, skill = _harness(monkeypatch, tmp_path, "from calcifier import step\n")

    result = dry_run._run(candidate, skill, candidate)

    assert result.status == "failed"
    assert (result.error or "").startswith("RecipeError: ")


def test_the_harness_skips_a_recipe_whose_inputs_it_cannot_synthesize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _RECIPE.replace("flag: bool,", "labels: list[object],")
    candidate, skill = _harness(monkeypatch, tmp_path, source)

    assert dry_run._run(candidate, skill, candidate) == DryRunResult(
        "unverified", None, "a required recipe input cannot be synthesized"
    )


def test_only_the_staged_candidate_resolves_children_from_the_destination(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "staging" / "candidate.py"
    recipe_path = tmp_path / "skill" / ".calcifier" / "recipe.py"
    child = recipe_path.with_name("child.py")
    resolve = dry_run._parent_at_destination(candidate, recipe_path)

    assert resolve(candidate) == recipe_path
    # A child calling its own children resolves them from its own file.
    assert resolve(child) == child


def test_relative_skill_and_recipe_paths_are_absolute_in_the_staged_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = _fake_process(
        monkeypatch,
        json.dumps({"status": "passed"}).encode(),
    )

    result = dry_run.dry_run_recipe(
        "value = 1\n",
        recipe_path=Path("skill/.calcifier/recipe.py"),
        skill_path=Path("skill/SKILL.md"),
    )

    assert result == DryRunResult("passed")
    assert calls[0][4:] == [
        str(tmp_path / "skill" / "SKILL.md"),
        str(tmp_path / "skill" / ".calcifier" / "recipe.py"),
    ]


def test_child_recipes_run_without_the_trust_store() -> None:
    assert dry_run._skip_dependency_pin(Path("parent.py"), Path("child.py"), "0" * 64) is None


def test_the_entry_point_prints_one_json_outcome(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    candidate, skill = _harness(monkeypatch, tmp_path, _RECIPE)

    assert dry_run.main(["calcifier.dry_run", str(candidate), str(skill), str(candidate)]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "status": "passed",
        "step": None,
        "error": None,
        "writes": [],
        "command_gap": False,
    }


def test_stub_agents_answer_unstructured_calls_with_text() -> None:
    stub = dry_run._SchemaStub("worker")

    assert stub.name == "worker"
    assert stub.invoke("prompt").text == "stub"
    assert json.loads(stub.invoke("prompt", schema={"type": "integer"}).text) == 1


def test_stub_tools_are_available_and_return_nothing() -> None:
    tool = dry_run._EmptyTool()

    assert tool.available("Grep") is True
    assert tool.invoke("Grep", pattern="x") == ""


@output_type
class _VerboseFinding:
    description: Annotated[str, word_count(3, 5)]


@output_type
class _VerboseReport:
    findings: list[_VerboseFinding]


def test_synthesized_nested_word_counts_pass_host_validation() -> None:
    value = synthesize(strict_schema_for(_VerboseReport))

    report = parse_response(json.dumps(value), _VerboseReport)
    assert len(report.findings[0].description.split()) == 3


def test_word_count_synthesis_respects_character_bounds_when_possible() -> None:
    schema = {
        "type": "string",
        "description": "At least 3 words; at most 5 words.",
        "minLength": 12,
        "maxLength": 14,
    }

    value = synthesize(schema)

    assert len(value.split()) == 3
    assert 12 <= len(value) <= 14


@pytest.mark.parametrize("with_manifest", [False, True])
def test_native_entry_point_preserves_optional_resource_manifest(
    tmp_path, monkeypatch, capsys, with_manifest
):
    candidate, skill, destination = (
        tmp_path / name for name in ("candidate.py", "SKILL.md", "recipe.py")
    )
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return DryRunResult("passed", writes=("report.md",))

    monkeypatch.setattr(dry_run, "_run", run)
    arguments = ["calcifier.dry_run", str(candidate), str(skill), str(destination)]
    entries = None
    if with_manifest:
        manifest = tmp_path / "resources.json"
        entry = {"path": "references/雪.md", "sha256": "a" * 64, "size": 17}
        manifest.write_bytes(json.dumps([entry], ensure_ascii=False).encode("utf-8"))
        arguments.append(str(manifest))
        entries = (ResourceEntry("references/雪.md", "a" * 64, 17),)
    assert dry_run.main(arguments) == 0
    assert calls == [((candidate, skill, destination, entries), {"agent_capabilities": None})]
    assert json.loads(capsys.readouterr().out) == {
        "status": "passed",
        "step": None,
        "error": None,
        "writes": ["report.md"],
        "command_gap": False,
    }


_REQUIRED_FILE = textwrap.dedent(
    """
    from pathlib import Path
    from calcifier import recipe, step, write_text

    @step
    def read_scope(target: str) -> str:
        write_text("started.txt", "started")
        return Path(target, "scope.json").read_text(encoding="utf-8")

    @recipe(output_root="out")
    def run(target: str = "target") -> str:
        return read_scope(target)
    """
)


def test_missing_default_input_is_unverified_but_real_execution_still_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate, skill = _harness(monkeypatch, tmp_path, _REQUIRED_FILE)

    result = dry_run._run(candidate, skill, candidate)

    assert result.status == "unverified"
    assert result.step == "read_scope"
    assert (result.error or "").startswith("file beneath an absent default path is unavailable")
    assert "FileNotFoundError" in (result.error or "")
    assert "scope.json" in (result.error or "")
    assert "line " in (result.error or "")
    assert result.writes == ("<workspace>/out/started.txt",)
    assert not (tmp_path / "out").exists()

    monkeypatch.chdir(tmp_path)
    run = recipe_module.load_recipe(candidate).run()
    assert not run.ok
    assert run.error_type == "FileNotFoundError"
    assert run.exception is not None
    assert run.exception.filename == str(Path("target", "scope.json"))


@pytest.mark.parametrize("with_scope", [True, False])
def test_supplied_input_root_keeps_success_and_missing_file_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, with_scope: bool
) -> None:
    candidate, skill = _harness(monkeypatch, tmp_path, _REQUIRED_FILE)
    workspace = tmp_path / "workspace"
    (workspace / "target").mkdir(parents=True)
    if with_scope:
        (workspace / "target/scope.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(workspace)

    result = dry_run._run_in_workspace(candidate, skill, workspace, RunHooks(), None)

    assert result.status == ("passed" if with_scope else "failed")
    assert (workspace / "out/started.txt").read_bytes() == b"started"


@pytest.mark.parametrize("target", [".", "..", "../outside", "", "/missing-input"])
def test_missing_files_outside_absent_relative_default_roots_still_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target: str
) -> None:
    source = _REQUIRED_FILE.replace('target: str = "target"', f"target: str = {target!r}")
    candidate, skill = _harness(monkeypatch, tmp_path, source)

    result = dry_run._run(candidate, skill, candidate)

    assert result.status == "failed"


@pytest.mark.parametrize(
    "body",
    [
        'raise FileNotFoundError("target/scope.json")',
        'raise ValueError("target/scope.json")',
        "return [][1]",
        'return Path("target-other", "scope.json").read_text()',
        'return Path(target, "..", "missing.json").read_text()',
    ],
)
def test_other_defects_and_unlocated_missing_errors_are_not_synthetic_input_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str
) -> None:
    source = _REQUIRED_FILE.replace(
        'return Path(target, "scope.json").read_text(encoding="utf-8")', body
    )
    candidate, skill = _harness(monkeypatch, tmp_path, source)

    assert dry_run._run(candidate, skill, candidate).status == "failed"


@pytest.mark.parametrize(
    "body",
    [
        'return resource("references/missing.md").read_text()',
        'return str(call_recipe("missing.py", {}).output)',
    ],
)
def test_missing_packaged_dependencies_still_fail_with_an_absent_default_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str
) -> None:
    source = _REQUIRED_FILE.replace(
        "from calcifier import recipe, step, write_text",
        "from calcifier import call_recipe, recipe, resource, step, write_text",
    ).replace('return Path(target, "scope.json").read_text(encoding="utf-8")', body)
    candidate, skill = _harness(monkeypatch, tmp_path, source)

    assert dry_run._run(candidate, skill, candidate).status == "failed"


def test_placeholder_writes_use_forward_slashes_for_native_windows_paths(monkeypatch):
    monkeypatch.setattr(dry_run, "_NATIVE_SEPARATOR", "\\")
    result = DryRunResult("failed", "step", "boom", ("C:\\tmp\\ws\\out\\started.txt",))
    replaced = dry_run._with_placeholders(result, {"C:\\tmp\\ws": "<workspace>"})
    assert replaced.writes == ("<workspace>/out/started.txt",)
