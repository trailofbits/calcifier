"""Compiler prompt content and repair context."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

import pytest

from calcifier.calcify import CalcifyError, calcify_source
from calcifier.prompts import PORTABLE_SOURCE_PROMPT, _make_sentinel, _signature_text, build_prompt
from calcifier.recipe_draft import draft_response as _draft_payload
from tests.support.ast import assert_same_program as _assert_same_program
from tests.support.compiler import source_sequence
from tests.support.recipes import _REPAIR_RECIPE, _REPAIR_SKILL


@pytest.mark.semantic_offline
def test_generation_contract_keeps_model_locations_out_of_filesystem_authority() -> None:
    prompt = build_prompt("Inspect one candidate.")
    normalized = " ".join(prompt.split())

    assert "from trusted recipe inputs, never from agent or child-recipe output" in normalized
    assert "Model output may name locations in a report only as opaque text" in normalized


def test_make_sentinel_returns_string_not_in_skill_text():
    sentinel = _make_sentinel("a normal skill body")
    assert sentinel not in "a normal skill body"
    assert "CALCIFIER_SKILL_BOUNDARY" in sentinel


def test_make_sentinel_changes_each_call_for_same_input():
    a = _make_sentinel("body")
    b = _make_sentinel("body")
    assert a != b


def test_make_sentinel_fails_closed_after_repeated_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("calcifier.prompts.secrets.token_hex", lambda width: "a" * (width * 2))
    skill = f"<<<CALCIFIER_SKILL_BOUNDARY:{'a' * 32}>>>"

    with pytest.raises(CalcifyError, match="could not pick a unique sentinel"):
        _make_sentinel(skill)


def test_build_prompt_embeds_sentinel_around_skill():
    prompt = build_prompt("the skill body")
    # The sentinel appears twice (open and close) and the skill body once.
    assert "the skill body" in prompt
    boundary_count = prompt.count("CALCIFIER_SKILL_BOUNDARY")
    # The unique sentinel must surround the skill body at minimum (2 hits);
    # additional mentions in instruction prose are fine.
    assert boundary_count >= 2


def test_build_prompt_states_the_recipe_shape_and_effect_rules():
    prompt = " ".join(build_prompt("# report-producing skill").split())
    assert 'emit(value, name="<short-name>")' in prompt
    assert "never return only a status" in prompt
    assert 'resource("<manifest path>").read_text(encoding="utf-8")' in prompt
    assert "Plain helpers and computed constants are allowed" in prompt
    assert "use `@step` for traced phases with typed results" in prompt
    assert "Ordinary string methods are allowed" in prompt
    assert "A fully mechanical skill needs no agent calls" in prompt
    assert "Reuse supplied programs and filesystem tools" in prompt
    assert "Use ordinary bounded loops" in prompt
    assert "`fail()` after exhaustion" in prompt
    assert "`Path.replace()`" in prompt
    assert '`Path.read_text(encoding="utf-8")` or `read_bytes()` for files Python parses' in prompt
    assert "`prepare_context(files=..., readable_roots=...)` for files only agents need" in prompt
    assert "`open()` is not available" in prompt
    assert "When it calls agents" in prompt
    assert "counts every agent() call the recipe can make" in prompt
    assert "write paths are relative to that root" in prompt
    assert "record_side_effect" not in prompt
    assert "choose one the skill's work can finish within" in prompt


def test_build_prompt_lists_the_installed_recipe_api():
    prompt = build_prompt("# skill")
    api = prompt.split("## Calcifier API", 1)[1].split("## Recipe shape", 1)[0]

    assert "- `resource(rel_path: str) -> Path`" in api
    assert "- `Budget(max_agent_calls: int | None, max_wall_seconds: float)`" in api
    assert (
        "- `agent(role: str, prompt: str, /, *, working_directory: str | Path | None = None," in api
    )
    # Loader objects and exception types are not part of a recipe body's API.
    declarations = [line for line in api.splitlines() if line.startswith("- `")]
    assert not any(
        name in line for line in declarations for name in ("load_recipe", "RecipeRun", "Error")
    )


def test_packaged_manifest_files_are_recipe_resources_not_user_inputs():
    prompt = " ".join(
        build_prompt("# skill", resource_manifest="references/guide.md (sha256:example)").split()
    )

    assert "Every file in the packaged-resource manifest ships with the recipe" in prompt
    assert 'Access it only with `resource("<manifest path>")`' in prompt
    assert "never declare a recipe input for it" in prompt
    assert "resolve it against a user directory with `Path(...)` or a join" in prompt
    assert "or ask the user to supply it" in prompt
    assert "## Packaged resources references/guide.md (sha256:example)" in prompt


def test_api_signatures_render_unannotated_and_evaluated_annotations():
    def plain(value, *, limit=3):
        del value, limit

    namespace: dict[str, object] = {}
    exec("def typed(value: int) -> str:\n    return str(value)\n", namespace)  # noqa: S102

    assert _signature_text(plain) == "(value, *, limit=3)"
    assert _signature_text(cast("Callable[..., object]", namespace["typed"])) == (
        "(value: int) -> str"
    )


def test_build_prompt_gives_every_analyzing_agent_the_whole_target():
    prompt = " ".join(build_prompt("# skill").split())

    assert "an agent that analyzes a target receives the whole declared target" in prompt
    assert "it never removes work unless the skill itself sets that bound" in prompt


def test_build_prompt_starts_small_and_moves_repeatable_work_into_python():
    prompt = " ".join(build_prompt("# skill").split())
    assert (
        "Start with one agent session, then move concrete, repeatable operations into Python"
        in prompt
    )
    assert "Leave understanding, interpretation, and uncertain decisions with agents" in prompt
    assert (
        "Keep connected reasoning phases in one session, preserving their order and checks"
        in prompt
    )
    assert "available roles do not require separate calls" in prompt
    assert (
        "Add sessions for independent work, required independent review, "
        "or different filesystem access" in prompt
    )
    assert "not for each prose clause or phase" in prompt
    assert "at most 250 lines" in prompt


def test_build_prompt_preserves_only_skill_stated_answer_constraints():
    prompt = " ".join(build_prompt("# skill").split())

    assert "requested results and intermediates needed for mechanical work" in prompt
    assert "Keep reasoning and checklists internal unless requested in the answer" in prompt
    assert "never invent receipts" in prompt
    assert "Put every answer constraint stated by the skill into its output type" in prompt
    assert "word counts as `Annotated[str, word_count(minimum, maximum)]`" in prompt
    assert "prose with no stated constraint stays `str`" in prompt
    assert (
        "Host checks enforce only source-stated constraints on their subject, "
        "stage, and actual inputs" in prompt
    )
    assert "Keep qualitative answer requirements in the agent's instructions" in prompt
    assert "clear failure or a bounded retry when the skill calls for one" in prompt
    assert (
        "Reject or retry missing required results; "
        "never replace them with invented judgments or prose" in prompt
    )


@pytest.mark.parametrize("delivered", [False, True])
def test_mechanical_preparation_does_not_replace_reference_understanding(delivered):
    skill = "Read the guide, interpret every category, then validate the complete input."
    prompt = " ".join(build_prompt(skill, instruction_delivery=delivered).split())

    assert "Reading or parsing a reference does not complete its required understanding" in prompt
    assert (
        "Pass the complete read text, alongside mechanical observations, "
        "to the agent assigned that interpretation"
    ) in prompt
    assert "Search matches are observations, not an exhaustive set of semantic candidates" in prompt
    assert "Preserve the original reasoning instructions" in prompt
    assert skill in prompt


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_build_prompt_defers_connected_reasoning_to_original_source(newline: str):
    skill = newline.join(
        [
            "# Packet review",
            "Collect repeated observations about each packet.",
            "Check identity and approval against their respective packaged rules.",
            "Keep both checks even when they concern one packet and outcome.",
            "Validate both checks, then report one finding per packet.",
            "Combine all confirmed causes and their remedies in that finding.",
            "",
        ]
    )
    prompt = build_prompt(skill)
    generation_rules = " ".join(prompt.split("## Source SKILL.md", 1)[0].split())

    assert (
        "Describe the assigned scope, mechanical preparation and answer interface in agent prompts"
        in (generation_rules)
    )
    assert "leave the original reasoning workflow intact" in generation_rules
    assert "Keep connected reasoning in that one session" in generation_rules
    assert "do not invent intermediate semantic gates or ledgers" in generation_rules
    assert prompt.count(skill) == 1


def test_fixed_prompt_stays_a_generic_core():
    # Rules added one incident at a time are how the prompt drifted; keep the fixed text short.
    assert len(PORTABLE_SOURCE_PROMPT.splitlines()) <= 100


def test_build_prompt_uses_literal_for_closed_choice_inputs():
    prompt = " ".join(build_prompt("# skill").split())
    assert (
        "Annotate a recipe input accepting only a fixed set of values as `Literal[...]`, not `str`."
        in prompt
    )


def test_build_prompt_leaves_tools_unrestricted_without_allowed_tools():
    prompt = " ".join(build_prompt("# skill").split())
    assert "The skill has no allowed-tools field and does not restrict tools" in prompt
    assert "only the capabilities the recipe calls; run_command() needs Bash" in prompt


def test_scoped_preapprovals_preserve_scope_without_restricting_tools():
    prompt = " ".join(build_prompt("---\nallowed-tools: Bash(git log:*)\n---\n# skill").split())
    assert '["Bash(git log:*)"]' in prompt
    assert "unlisted tools remain available under runtime permissions" in prompt


def test_calcify_repair_request_extends_the_original_prompt_with_the_rejection():
    rejected = "def broken(—):\n    pass"
    compiler = source_sequence(rejected, _REPAIR_RECIPE)

    generated = calcify_source(_REPAIR_SKILL, compiler, agent_roles=("worker",))

    _assert_same_program(generated, _REPAIR_RECIPE)
    initial, repair = compiler.source_prompts
    assert repair.startswith(initial)
    request = repair.removeprefix(initial)
    assert "Return one complete,\ncorrected RecipeDraft for the whole module" in request
    error = request.split("Validation error (JSON string):\n", 1)[1]
    assert error.startswith('"invalid RecipeDraft: invalid or incomplete recipe draft: ')
    # Nothing parsed, so the raw response is shown as the rejected source.
    assert json.dumps(_draft_payload(rejected), ensure_ascii=False) in request


@pytest.mark.parametrize("delivered", [False, True])
def test_generation_only_promises_original_delivery_for_file_based_compilation(delivered):
    prompt = build_prompt("# Neutral", instruction_delivery=delivered)
    assert ("gives every agent the original SKILL.md" in prompt) is delivered
    assert ("do not receive the original SKILL.md automatically" in prompt) is not delivered
    assert ("Use complete embedded context directly" in prompt) is delivered
    assert (
        "read backing files when text is absent or a file check is required" in prompt
    ) is delivered


def test_api_reference_includes_current_helper_contracts_without_dataclass_noise():
    from calcifier import PreparedContext, prepare_context
    from calcifier.prompts import _api_reference

    api = _api_reference()
    # Real API documentation travels with the signature, not a separate incident rule list.
    context_doc = prepare_context.__doc__
    assert context_doc is not None
    assert context_doc.strip().splitlines()[0] in api
    assert "always saved in full" in api
    assert f"`PreparedContext{PreparedContext.__init__.__name__}" not in api
    assert len(api.encode("utf-8")) <= 16384
