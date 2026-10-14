"""Tests for side-effect manifests."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from calcifier import recipe, record_side_effect, step
from calcifier.recipe import RecipeError
from calcifier.side_effects import DryRunMode, SideEffectBlocked
from calcifier.trace import active_run, step_frame


def test_record_side_effect_appends_to_current_step():
    with active_run() as ctx, step_frame("write", bool, {}):
        effect = record_side_effect("file_write", "/tmp/x")  # noqa: S108

    assert effect.completed is True
    assert len(ctx.records[0].side_effects) == 1
    assert ctx.records[0].side_effects[0] is effect


def test_record_side_effect_derives_idempotency_key():
    with active_run() as first, step_frame("write", bool, {}):
        a = record_side_effect("file_write", "/tmp/x", payload={"b": 2, "a": 1})  # noqa: S108
    with active_run() as second, step_frame("write", bool, {}):
        b = record_side_effect("file_write", "/tmp/x", payload={"a": 1, "b": 2})  # noqa: S108

    assert a.idempotency_key
    assert a.idempotency_key == b.idempotency_key
    first_key = first.records[0].side_effects[0].idempotency_key
    second_key = second.records[0].side_effects[0].idempotency_key
    assert first_key == second_key


def test_record_side_effect_caller_provided_key_used_as_is():
    with active_run(), step_frame("write", bool, {}):
        effect = record_side_effect("file_write", "/tmp/x", idempotency_key="user-key")  # noqa: S108

    assert effect.idempotency_key == "user-key"


def test_record_side_effect_duplicate_returns_existing(caplog):
    caplog.set_level(logging.WARNING, logger="calcifier.side_effects")
    with active_run() as ctx, step_frame("write", bool, {}):
        first = record_side_effect("file_write", "/tmp/x", idempotency_key="same")  # noqa: S108
        second = record_side_effect("publish", "slack:#general/msg-1", idempotency_key="same")

    assert second is first
    assert len(ctx.records[0].side_effects) == 1
    assert "duplicate side effect idempotency key same" in caplog.text


def test_parent_reuses_an_effect_already_recorded_in_a_nested_step():
    with active_run() as ctx, step_frame("parent", bool, {}):
        with step_frame("child", bool, {}):
            original = record_side_effect("file_write", "report.md", idempotency_key="report")
        repeated = record_side_effect("file_write", "report.md", idempotency_key="report")

    assert repeated is original
    assert ctx.records[0].side_effects == []
    assert ctx.records[0].children[0].side_effects == [original]


def test_record_side_effect_outside_step_raises():
    with pytest.raises(RecipeError, match="requires an active step"):
        record_side_effect("file_write", "/tmp/x")  # noqa: S108


def test_record_side_effect_rejects_kind_outside_trace_schema():
    with pytest.raises(RecipeError, match="unsupported side effect kind"):
        record_side_effect("recipe_execution", "/tmp/x")  # ty: ignore[invalid-argument-type]  # noqa: S108


def test_record_side_effect_dry_run_soft_returns_uncompleted():
    with active_run(dry_run=DryRunMode.SOFT) as ctx, step_frame("write", bool, {}):
        effect = record_side_effect("file_write", "/tmp/x")  # noqa: S108

    assert effect.completed is False
    assert ctx.records[0].side_effects == [effect]


def test_record_side_effect_dry_run_hard_raises():
    with (
        active_run(dry_run=DryRunMode.HARD),
        step_frame("write", bool, {}),
        pytest.raises(SideEffectBlocked),
    ):
        record_side_effect("file_write", "/tmp/x")  # noqa: S108


def test_local_file_creation_side_effect(tmp_path: Path):
    @step
    def write_file(path: Path) -> bool:
        path.write_text("ok")
        record_side_effect("file_write", str(path))
        return True

    @recipe
    def runner(path: Path) -> bool:
        return write_file(path=path)

    out = tmp_path / "report.md"
    run = runner.run({"path": out})
    effect = run.records[0].side_effects[0]
    assert out.read_text() == "ok"
    assert effect.kind == "file_write"
    assert effect.target == str(out)


def test_external_stub_mutation_side_effect():
    def mutate_stub() -> None:
        record_side_effect("external_object", target="stub:42")

    @step
    def call_stub() -> bool:
        mutate_stub()
        return True

    @recipe
    def runner() -> bool:
        return call_stub()

    run = runner.run({})
    effect = run.records[0].side_effects[0]
    assert effect.kind == "external_object"
    assert effect.target == "stub:42"
