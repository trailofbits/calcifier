"""Compiler calls receive only the pinned resources needed to preserve skill contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from calcifier.agent_events import AgentActivity, AgentResult
from calcifier.calcify import CalcifyError, calcify_file
from calcifier.compiler_resources import stage_compiler_resources
from calcifier.filesystem_contract import AgentError, FilesystemContract
from calcifier.hashing import sha256_hex
from calcifier.registry import Registry
from calcifier.skill_package import (
    GENERATION_CONTEXT_FILE_SIZE_LIMIT,
    RESOURCE_FILE_SIZE_LIMIT,
    ResourceEntry,
    discover_resources,
)
from tests.support.agents import StubProvider
from tests.support.compiler import one_draft
from tests.support.recipes import _GOOD_RECIPE


@pytest.mark.parametrize("failure", [False, True])
def test_compiler_can_read_unlisted_reference_contents_in_an_isolated_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text("Read applicable guides in references/ and produce their required fields.\n")
    guide = tmp_path / "references" / "contract.md"
    guide.parent.mkdir()
    content = "Required fields: reason, uncertainty.\n" + "x" * GENERATION_CONTEXT_FILE_SIZE_LIMIT
    guide.write_text(content)
    (tmp_path / "unrelated-secret.txt").write_text("not a packaged resource")
    compiler = one_draft(_GOOD_RECIPE)
    original = compiler.invoke
    directories: list[Path] = []

    def inspect(
        prompt: str, *, filesystem: FilesystemContract | None = None, **kwargs: Any
    ) -> AgentResult:
        assert filesystem is not None
        directory = filesystem.working_directory
        directories.append(directory)
        assert filesystem.readable_paths == (directory,)
        assert directory != tmp_path
        assert (directory / "references" / "contract.md").read_text() == content
        assert sorted(
            p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
        ) == ["SKILL.md", "references/contract.md"]
        assert "Required fields: reason, uncertainty." not in prompt
        if failure:
            raise AgentError("stopped")
        return original(prompt, filesystem=filesystem, **kwargs)

    monkeypatch.setattr(compiler, "invoke", inspect)
    registry = Registry()
    registry.register(compiler)
    worker = StubProvider()
    worker.name = "worker"
    registry.register(worker)

    activity = AgentActivity()
    recipe = calcify_file(skill, registry, activity=activity)
    assert recipe.is_file()
    assert activity.max_model_calls == 5
    if failure:
        assert "Calcifier baseline:" in recipe.read_text(encoding="utf-8")
        retained = activity.events[-1]
        assert retained["phase"] == "baseline" and retained["status"] == "retained"
        assert retained["detail"] == "agent invocation failed: stopped"
    else:
        assert "Calcifier baseline:" not in recipe.read_text(encoding="utf-8")
    assert directories and all(not directory.exists() for directory in directories)
    assert guide.read_text() == content


@pytest.mark.parametrize("change", [b"other", b"longer", None])
def test_resource_drift_fails_before_a_compiler_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: bytes | None
) -> None:
    from calcifier import artifact

    skill = tmp_path / "SKILL.md"
    skill.write_text("Read references/guide.md.\n")
    guide = tmp_path / "references" / "guide.md"
    guide.parent.mkdir()
    guide.write_bytes(b"first")
    original = artifact.compile_source

    def drift(*args: Any, **kwargs: Any):
        if change is None:
            guide.unlink()
        else:
            guide.write_bytes(change)
        return original(*args, **kwargs)

    monkeypatch.setattr(artifact, "compile_source", drift)
    compiler = one_draft(_GOOD_RECIPE)
    registry = Registry()
    registry.register(compiler)
    with pytest.raises(CalcifyError, match="compiler resource"):
        calcify_file(skill, registry)
    assert compiler.calls == []
    assert not (tmp_path / ".calcifier").exists()


def test_external_alias_snapshot_and_inline_contents_agree(tmp_path: Path) -> None:
    skill = tmp_path / "package" / "skills" / "analyze" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    reference = tmp_path / "package" / "references" / "contract.md"
    reference.parent.mkdir()
    reference.write_bytes(b"Required fields: source, exception.\r\n")
    # Explicit external paths use the same discovery syntax as source skills.
    text = "Read `{baseDir}/../../references/contract.md`.\n"
    skill.write_text(text)
    entries = tuple(discover_resources(skill))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    manifest = stage_compiler_resources(snapshot, skill, text, entries)

    assert entries[0].source == "../../references/contract.md"
    assert (snapshot / "references" / "contract.md").read_bytes() == reference.read_bytes()
    assert '"references/contract.md": "Required fields: source, exception.\\r\\n"' in manifest
    assert (snapshot / "SKILL.md").read_text() == text
    assert "../../references/contract.md (sha256:" not in manifest


def test_reordered_external_alias_keeps_explicit_inline_context(tmp_path: Path) -> None:
    skill = tmp_path / "package" / "skills" / "analyze" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    reference = tmp_path / "package" / "shared" / "references" / "contract.md"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"Required fields: reason, uncertainty.\r\n")
    text = "Read `{baseDir}/../../shared/references/contract.md`.\n"
    skill.write_text(text)
    entries = tuple(discover_resources(skill))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    manifest = stage_compiler_resources(snapshot, skill, text, entries)

    assert entries[0].path == "references/shared/contract.md"
    assert entries[0].path not in text
    assert (snapshot / entries[0].path).read_bytes() == reference.read_bytes()
    expected = '"references/shared/contract.md": "Required fields: reason, uncertainty.\\r\\n"'
    assert expected in manifest


def test_skill_edit_during_compilation_cannot_be_pinned_as_the_compiler_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_bytes(b"Original instructions.\r\n")
    compiler = one_draft(_GOOD_RECIPE)
    original = compiler.invoke

    def edit_skill(prompt: str, **kwargs: Any) -> AgentResult:
        assert "Original instructions." in prompt
        skill.write_bytes(b"Changed instructions.\r\n")
        return original(prompt, **kwargs)

    monkeypatch.setattr(compiler, "invoke", edit_skill)
    registry = Registry()
    registry.register(compiler)
    with pytest.raises(CalcifyError, match="hash|drift|changed"):
        calcify_file(skill, registry)
    assert not (tmp_path / ".calcifier").exists()


@pytest.mark.parametrize(
    "aliases",
    [
        ("",),
        ("../outside",),
        ("/outside",),
        ("SKILL.md",),
        ("skill.md",),
        ("./SKILL.md",),
        ("SKILL.md/child",),
        ("C:/outside",),
        ("C:outside",),
        ("references\\outside",),
        ("references/invalid\x00name",),
        ("\\\\server\\share",),
        ("references/a", "references/a"),
        ("references/a", "references/a/b"),
        ("references/a/b", "references/a"),
    ],
)
def test_unsafe_or_colliding_aliases_are_refused(tmp_path: Path, aliases: tuple[str, ...]) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"data")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    entries = tuple(ResourceEntry(alias, sha256_hex(b"data"), 4, "source") for alias in aliases)

    with pytest.raises(CalcifyError, match="compiler resource alias"):
        stage_compiler_resources(snapshot, tmp_path / "SKILL.md", "skill", entries)


def test_snapshot_refuses_unrelated_files_in_its_destination(tmp_path: Path) -> None:
    (tmp_path / "existing").write_text("leave intact")
    with pytest.raises(CalcifyError, match="must be empty"):
        stage_compiler_resources(tmp_path, tmp_path / "SKILL.md", "skill", ())
    assert (tmp_path / "existing").read_text() == "leave intact"


def test_resource_escape_is_not_exposed(tmp_path: Path) -> None:
    root = tmp_path / "package"
    root.mkdir()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"data")
    entry = ResourceEntry("references/data", sha256_hex(b"data"), 4, "../outside")
    with pytest.raises(CalcifyError, match="escapes"):
        stage_compiler_resources(snapshot, root / "SKILL.md", "skill", (entry,))
    assert list(snapshot.iterdir()) == []


@pytest.mark.parametrize("source", [None, ""])
def test_internal_resource_symlink_cannot_expand_the_compiler_view(
    tmp_path: Path, source: str | None
) -> None:
    root = tmp_path / "package" / "skills" / "analyze"
    references = root / "references"
    references.mkdir(parents=True)
    outside = tmp_path / "package" / "outside.md"
    outside.write_bytes(b"data")
    try:
        (references / "link.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    entry = ResourceEntry("references/link.md", sha256_hex(b"data"), 4, source=source)

    with pytest.raises(CalcifyError, match="escapes"):
        stage_compiler_resources(snapshot, root / "SKILL.md", "skill", (entry,))
    assert list(snapshot.iterdir()) == []


def test_exact_resource_size_limit_remains_available(tmp_path: Path) -> None:
    source = tmp_path / "limit"
    data = b"x" * RESOURCE_FILE_SIZE_LIMIT
    source.write_bytes(data)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    entry = ResourceEntry("resources/limit.bin", sha256_hex(data), len(data), "limit")

    manifest = stage_compiler_resources(snapshot, tmp_path / "SKILL.md", "skill", (entry,))

    assert (snapshot / "resources" / "limit.bin").read_bytes() == data
    assert "resources/limit.bin (sha256:" in manifest


def test_oversize_resource_is_not_exposed(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    data = b"x" * (RESOURCE_FILE_SIZE_LIMIT + 1)
    (tmp_path / "large").write_bytes(data)
    entry = ResourceEntry("references/data", sha256_hex(data), len(data), "large")
    with pytest.raises(CalcifyError, match="changed after discovery"):
        stage_compiler_resources(snapshot, tmp_path / "SKILL.md", "skill", (entry,))
    assert list(snapshot.iterdir()) == []


def test_utf8_external_references_ignore_the_platform_default_encoding(tmp_path, monkeypatch):
    skill = tmp_path / "package" / "skills" / "analyze" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    reference = tmp_path / "package" / "references" / "contract.md"
    reference.parent.mkdir()
    reference.write_bytes(b"required content")
    skill.write_bytes("Read `{baseDir}/../../references/contract.md` for “contracts”.".encode())
    original = Path.read_text

    def windows_read(path, encoding=None, errors=None, newline=None):
        if path == skill:
            return path.read_bytes().decode(encoding or "cp1252", errors or "strict")
        return original(path, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "read_text", windows_read)
    entries = discover_resources(skill)
    assert len(entries) == 1
    assert entries[0].source == "../../references/contract.md"
    assert entries[0].sha256 == sha256_hex(reference.read_bytes())


@pytest.mark.parametrize("failure", ["missing-directory", "skill-write", "manifest-read"])
def test_snapshot_io_failures_use_normal_compiler_diagnostics(tmp_path, monkeypatch, failure):
    snapshot = tmp_path / "snapshot"
    if failure != "missing-directory":
        snapshot.mkdir()

    def fail(*args, **kwargs):
        raise OSError("unavailable filesystem")

    if failure == "skill-write":
        monkeypatch.setattr(Path, "write_text", fail)
    elif failure == "manifest-read":
        monkeypatch.setattr(Path, "read_text", fail)
    with pytest.raises(CalcifyError, match="cannot stage compiler resource snapshot") as error:
        stage_compiler_resources(snapshot, tmp_path / "SKILL.md", "skill", ())
    assert isinstance(error.value.__cause__, OSError)


def test_sibling_resources_can_share_staging_directories(tmp_path):
    resource = tmp_path / "references"
    resource.mkdir()
    for name in ["a", "b", "nested/c", "nested/d"]:
        target = resource / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(name.encode())
    skill = tmp_path / "SKILL.md"
    skill.write_text("Read references.")
    entries = tuple(discover_resources(skill))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    stage_compiler_resources(snapshot, skill, "Read references.", entries)
    for entry in entries:
        assert (snapshot / entry.path).read_bytes() == (tmp_path / entry.path).read_bytes()


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_empty_resource_snapshot_preserves_the_skill_without_inventing_references(
    tmp_path, newline
):
    skill = tmp_path / "SKILL.md"
    source = ("# Summarize notes" + newline + "Explain café clearly." + newline).encode("utf-8")
    skill.write_bytes(source)
    snapshot = tmp_path / "compiler"
    snapshot.mkdir()

    manifest = stage_compiler_resources(snapshot, skill, source.decode("utf-8"), ())

    assert manifest == ""
    assert sorted(path.name for path in snapshot.iterdir()) == ["SKILL.md"]
    assert (snapshot / "SKILL.md").read_bytes() == source
    assert skill.read_bytes() == source
