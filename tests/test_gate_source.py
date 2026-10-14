"""Offline regressions for gate source."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from calcifier.calcify import (
    CalcifyError,
    calcify_source,
)
from calcifier.diagnostics import SkillDeclarationError
from calcifier.gate import _check_agent_call, _check_effect_contracts, verify_source
from calcifier.recipe_ast import (
    _footer_target,
    _is_verified_resource_read,
    _system_exit_cli_target,
    decorator_keyword_call,
)
from tests.support.ast import assert_same_program as _assert_same_program
from tests.support.compiler import (
    registry_emitting as _registry_emitting,
)
from tests.support.compiler import (
    source_sequence,
)
from tests.support.compiler_cases import (
    _BASH_TOOL_RECIPE,
    _COMMAND_RECIPE,
    _FOOTER,
    _TWO_ROLE_RECIPE,
    _compile_file_drafts,
    _helper_signature_recipe,
    _skill_allowing,
)
from tests.support.recipes import (
    _GOOD_RECIPE,
    _REALIZED_FILESYSTEM_RECIPE,
    _REPAIR_RECIPE,
    _REPAIR_SKILL,
)


def test_agent_call_checks_and_budget_decorator_parsing():
    bindings = {"agent": "calcifier.agent"}
    undeclared = ast.parse(
        'agent("worker", "review", working_directory=".", readable_paths=(".",))',
        mode="eval",
    ).body
    assert isinstance(undeclared, ast.Call)
    with pytest.raises(CalcifyError, match="undeclared agent role worker"):
        _check_agent_call(undeclared, None, bindings)

    one_positional = ast.parse(
        'agent("worker", working_directory=".", readable_paths=(".",))', mode="eval"
    ).body
    assert isinstance(one_positional, ast.Call)
    _check_agent_call(one_positional, ("worker",), bindings)

    mutation_prompt = ast.parse(
        'agent("worker", "write a report file", working_directory=".", readable_paths=(".",))',
        mode="eval",
    ).body
    assert isinstance(mutation_prompt, ast.Call)
    # Agents cannot write whatever the prompt says: their filesystem contract is read-only.
    _check_agent_call(mutation_prompt, ("worker",), bindings)

    dynamic_role = ast.parse(
        'agent(7, "review", working_directory=".", readable_paths=(".",))', mode="eval"
    ).body
    assert isinstance(dynamic_role, ast.Call)
    with pytest.raises(CalcifyError, match="dynamic agent role"):
        _check_agent_call(dynamic_role, ("worker",), bindings)

    def budget_value(source: str) -> tuple[int, int] | None:
        decorator = ast.parse(source, mode="eval").body
        assert isinstance(decorator, ast.Call)
        return decorator_keyword_call(
            decorator,
            "budget",
            {"recipe": "calcifier.recipe", "Budget": "calcifier.Budget"},
        )

    assert budget_value("recipe(budget=Budget(2, 180))") == (2, 180)
    assert budget_value("recipe(budget=Budget(3, max_wall_seconds=90))") == (3, 90)
    assert budget_value("recipe(budget=Budget(max_agent_calls=3, max_wall_seconds=90))") == (3, 90)
    assert budget_value("recipe()") is None
    assert budget_value("recipe(budget=Other(2, 180))") is None
    assert budget_value("recipe(budget=Budget(True, 180))") is None
    assert budget_value("recipe(budget=Budget(2, False))") is None


@pytest.mark.parametrize(
    ("annotation", "message"),
    [
        ("Missing", "unsupported step annotation `Missing`"),
        (
            "tuple[MissingB, MissingA, MissingB]",
            "unsupported step annotation names ['MissingA', 'MissingB']",
        ),
    ],
)
def test_missing_step_types_report_all_undefined_names_once(annotation, message):
    from calcifier.gate import _check_step_annotation

    function = ast.parse(f"def inspect() -> {annotation}:\n    pass\n").body[0]
    assert isinstance(function, ast.FunctionDef)
    with pytest.raises(CalcifyError) as caught:
        _check_step_annotation(function, {"tuple"})
    assert str(caught.value) == f"{message} at line 1 (no matching definition)"
    assert caught.value.line == 1


def test_ruff_shape_and_ty_findings_are_reported_together(tmp_path: Path) -> None:
    source = (
        _REPAIR_RECIPE.replace(
            "def inspect(target: str) -> bool:\n",
            "def inspect(target: str) -> bool:\n    missing\n",
        )
        .replace("return inspect(target=target)", "return inspect()")
        .split('\n\nif __name__ == "__main__":')[0]
        + "\n"
    )

    with pytest.raises(CalcifyError) as caught:
        verify_source(source, lint_path=tmp_path / "recipe.py")

    # One repair sees every static failure, not only the first tool's.
    found = [item.actual for item in caught.value.diagnostics]
    assert any(message.startswith("F821") for message in found)
    assert any(message.startswith("no CLI footer") for message in found)
    assert any(message.startswith("ty[missing-argument]") for message in found)
    assert str(caught.value).startswith("Ruff validation failed: ")
    assert "; no CLI footer at line 1; ty validation failed: " in str(caught.value)


@pytest.mark.parametrize("source", [_BASH_TOOL_RECIPE, _COMMAND_RECIPE])
@pytest.mark.parametrize("tools", ["", "Read", "Bash(git log:*)"])
def test_tool_preapprovals_do_not_restrict_declared_capabilities(source: str, tools: str) -> None:
    verify_source(source)


@pytest.mark.parametrize("source", [_COMMAND_RECIPE, _BASH_TOOL_RECIPE], ids=["command", "tool"])
def test_tool_use_without_an_allowed_tools_field_is_unrestricted(source: str) -> None:
    verify_source(source)


@pytest.mark.parametrize("skill", ["# skill\n", _skill_allowing(""), _skill_allowing("Read")])
def test_a_tool_free_recipe_compiles_whatever_the_field_says(skill: str) -> None:
    verify_source(_GOOD_RECIPE)


def test_a_malformed_allowed_tools_field_stops_before_any_model_call() -> None:
    compiler = source_sequence(_GOOD_RECIPE)

    with pytest.raises(SkillDeclarationError, match="malformed allowed-tools entry"):
        calcify_source(_skill_allowing("Bash(git:*"), compiler, agent_roles=())

    assert compiler.all_prompts == []


def test_a_recipe_may_declare_only_configured_agent_roles() -> None:
    reviewer = _REPAIR_RECIPE.replace('agents=("worker",)', 'agents=("reviewer",)').replace(
        '"worker", "review {target}"', '"reviewer", "review {target}"'
    )

    with pytest.raises(CalcifyError, match=r"agent roles \['reviewer'\] that are not configured"):
        verify_source(reviewer, agent_roles=("worker",))
    verify_source(reviewer, agent_roles=("reviewer",))


def test_calcify_source_accepts_valid_recipe():
    reg = _registry_emitting(_GOOD_RECIPE)
    out = calcify_source("# any skill", reg.get("compiler"))
    assert "@recipe" in out


def test_calcify_source_rejects_legacy_fenced_python_output():
    reg = _registry_emitting(f"```python\n{_GOOD_RECIPE}\n```")
    with pytest.raises(CalcifyError, match="invalid RecipeDraft"):
        calcify_source("# skill", reg.get("compiler"))


def test_verify_source_rejects_a_missing_footer():
    src = textwrap.dedent(
        """
        from calcifier import recipe, step

        @step
        def s(x: str) -> bool: return True

        @recipe
        def r(target: str = "x") -> bool:
            return s(x=target)
        """
    )
    with pytest.raises(CalcifyError, match="no CLI footer"):
        verify_source(src)


def test_calcify_source_rejects_bare_cli_footer():
    """A bare ``<name>.cli()`` footer (no ``raise SystemExit(...)``) is rejected
    with a specific message that points the LLM at the fix."""
    src = textwrap.dedent(
        """
        from calcifier import recipe, step

        @step
        def s(x: str) -> bool: return True

        @recipe
        def r(target: str = "x") -> bool:
            return s(x=target)

        if __name__ == "__main__":
            r.cli()
        """
    )
    reg = _registry_emitting(src)
    with pytest.raises(CalcifyError, match="bare form discards"):
        calcify_source("# skill", reg.get("compiler"))


def test_calcify_source_accepts_recipe_body_without_a_step_wrapper():
    src = (
        textwrap.dedent(
            """
            from calcifier import recipe

            @recipe
            def r(target: str = "x") -> bool:
                return True
            """
        ).strip()
        + _FOOTER
    )
    reg = _registry_emitting(src)
    assert "return True" in calcify_source("# skill", reg.get("compiler"))


def test_calcify_source_rejects_no_recipe():
    """A module that has steps and a valid footer shape but no ``@recipe`` is rejected."""
    src = textwrap.dedent(
        """
        from calcifier import step

        @step
        def s(x: str) -> bool: return True

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    ).strip()
    reg = _registry_emitting(src)
    with pytest.raises(CalcifyError, match="no @recipe"):
        calcify_source("# skill", reg.get("compiler"))


def test_verify_source_rejects_a_footer_naming_another_symbol():
    """The footer's ``<name>.cli()`` symbol must match the @recipe-decorated
    variable. Calling ``other.cli()`` where ``other`` is something else is
    rejected even if the syntax otherwise looks fine."""
    src = textwrap.dedent(
        """
        from calcifier import recipe, step

        @step
        def s(x: str) -> bool: return True

        @recipe
        def r(target: str = "x") -> bool:
            return s(x=target)

        if __name__ == "__main__":
            raise SystemExit(other.cli())
        """
    )
    with pytest.raises(CalcifyError, match="no CLI footer.*footer calls"):
        verify_source(src)


def test_calcify_source_rejects_multiple_recipes():
    src = textwrap.dedent(
        """
            from calcifier import recipe, step

            @step
            def s(x: str) -> bool: return True

            @recipe
            def a(target: str = "x") -> bool: return s(x=target)

            @recipe
            def b(target: str = "x") -> bool: return s(x=target)
            """
    ).strip() + _FOOTER.replace("r.cli()", "a.cli()")
    reg = _registry_emitting(src)
    with pytest.raises(CalcifyError, match="multiple @recipe"):
        calcify_source("# skill", reg.get("compiler"))


def test_calcify_source_rejects_invalid_unstructured_python():
    reg = _registry_emitting("def broken(:\n    pass\n" + _FOOTER)
    with pytest.raises(CalcifyError, match="invalid RecipeDraft"):
        calcify_source("# skill", reg.get("compiler"))


def test_calcify_source_rejects_forbidden_top_level_raise():
    src = "raise RuntimeError('explode')\n" + _FOOTER
    reg = _registry_emitting(src)
    with pytest.raises(CalcifyError, match="forbidden top-level statement"):
        calcify_source("# skill", reg.get("compiler"))


def test_calcify_source_rejects_step_with_unsupported_annotation():
    src = (
        textwrap.dedent(
            """
            from calcifier import recipe, step

            @step
            def bad(x: str) -> _Missing:
                return None

            @recipe
            def r(target: str = "x") -> bool:
                return True
            """
        ).strip()
        + _FOOTER
    )
    reg = _registry_emitting(src)
    with pytest.raises(CalcifyError, match="_Missing"):
        calcify_source("# skill", reg.get("compiler"))


def test_calcify_accepts_explicit_none_step_return_annotation():
    source = _GOOD_RECIPE.replace(
        "def emitted(target: str) -> bool:\n    return bool(target)",
        "def emitted(target: str) -> None:\n    return None",
    ).replace(
        'def writing_style(target: str = ".") -> bool:',
        'def writing_style(target: str = ".") -> None:',
    )

    verify_source(source)


def test_calcify_source_rejects_top_level_os_system_without_executing(tmp_path: Path):
    marker = tmp_path / "pwn_marker"
    src = textwrap.dedent(
        f"""\
        from calcifier import recipe, step
        import os

        os.system({"touch " + str(marker)!r})

        @step
        def s(x: str) -> bool: return True

        @recipe
        def r(target: str = "x") -> bool:
            return s(x=target)

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    )
    reg = _registry_emitting(src)
    with pytest.raises(CalcifyError, match="forbidden top-level statement"):
        calcify_source("# skill", reg.get("compiler"))
    assert not marker.exists()


def test_calcify_source_rejects_top_level_file_write_without_executing(tmp_path: Path):
    marker = tmp_path / "pwn_marker"
    src = textwrap.dedent(
        f"""\
        from calcifier import recipe, step
        from pathlib import Path

        Path({str(marker)!r}).write_text("pwned")

        @step
        def s(x: str) -> bool: return True

        @recipe
        def r(target: str = "x") -> bool:
            return s(x=target)

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    )
    reg = _registry_emitting(src)
    with pytest.raises(CalcifyError, match="forbidden top-level statement"):
        calcify_source("# skill", reg.get("compiler"))
    assert not marker.exists()


def test_calcify_rejects_agent_directed_writes_inside_steps(tmp_path: Path):
    marker = tmp_path / "state.json"
    source = textwrap.dedent(
        f"""
        from pathlib import Path
        from calcifier import recipe, step

        @step
        def initialize() -> bool:
            Path({str(marker)!r}).write_text("success")
            return True

        @recipe
        def run() -> bool:
            return initialize()

        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )

    with pytest.raises(
        CalcifyError,
        match="arbitrary generated write method 'write_text'.*Calcifier's declared primitives",
    ):
        calcify_source("# skill", _registry_emitting(source).get("compiler"))
    assert not marker.exists()


def test_calcify_requires_literal_output_root_for_mutations():
    source = textwrap.dedent(
        """
        from calcifier import recipe, step, write_text

        @step
        def initialize() -> bool:
            write_text("state.json", "success")
            return True

        @recipe
        def run() -> bool:
            return initialize()

        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )

    with pytest.raises(CalcifyError, match="literal output_root"):
        calcify_source("# skill", _registry_emitting(source).get("compiler"))

    portable = source.replace("@recipe\n", '@recipe(output_root="artifacts")\n')
    _assert_same_program(
        calcify_source("# skill", _registry_emitting(portable).get("compiler")), portable
    )

    module = ast.parse(source)
    recipe_node = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    with pytest.raises(CalcifyError, match="literal output_root"):
        _check_effect_contracts(module, recipe_node)


def test_calcify_source_rejection_messages_include_line_numbers():
    src = textwrap.dedent(
        """\
        from calcifier import recipe, step

        @step
        def s(x: str) -> bool:
            return True

        os.system("echo no")

        @recipe
        def r(target: str = "x") -> bool:
            return s(x=target)

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    )
    reg = _registry_emitting(src)
    with pytest.raises(CalcifyError) as excinfo:
        calcify_source("# skill", reg.get("compiler"))
    # Lines refer to the checked source the repair sees, after destination formatting.
    checked = excinfo.value.source
    assert checked is not None
    line = next(n for n, text in enumerate(checked.splitlines(), 1) if "os.system" in text)
    assert f"line {line}:" in str(excinfo.value)


def test_calcify_source_distinguishes_syntax_vs_shape_errors():
    syntax_reg = _registry_emitting("def broken(:\n    pass\n")
    with pytest.raises(CalcifyError) as syntax_exc:
        calcify_source("# skill", syntax_reg.get("compiler"))
    assert "invalid RecipeDraft" in str(syntax_exc.value)

    shape_src = textwrap.dedent(
        """\
        from calcifier import step

        @step
        def s(x: str) -> bool: return True

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    )
    shape_reg = _registry_emitting(shape_src)
    with pytest.raises(CalcifyError) as shape_exc:
        calcify_source("# skill", shape_reg.get("compiler"))
    assert "no @recipe" in str(shape_exc.value)
    assert "valid Python" not in str(shape_exc.value)


def test_calcify_rejects_unavailable_top_level_dependency():
    source = _GOOD_RECIPE.replace(
        "from calcifier import recipe, step",
        "import openpyxl\nfrom calcifier import recipe, step",
    ).replace("return bool(target)", "return bool(target and openpyxl)")

    with pytest.raises(CalcifyError, match="unavailable top-level dependencies.*openpyxl"):
        verify_source(source)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        pytest.param(
            "budget=Budget(4, 30))",
            "budget=Budget(4, 30), capabilities=())",
            id="empty-capabilities",
        ),
        pytest.param('agents=("worker", "reviewer")', 'agents=("reviewer", "worker")', id="roles"),
        pytest.param(
            "budget=Budget(4, 30)",
            "budget=Budget(max_agent_calls=4, max_wall_seconds=30)",
            id="keyword-budget",
        ),
    ],
)
def test_equivalent_recipe_declarations_are_kept_verbatim(
    tmp_path: Path, old: str, new: str
) -> None:
    # The first draft fixes the fake's plan; the second spells the same bounds differently.
    footerless = _TWO_ROLE_RECIPE.split('\n\nif __name__ == "__main__":')[0] + "\n"
    variant = _TWO_ROLE_RECIPE.replace(old, new)
    assert new in variant

    compiler, generated = _compile_file_drafts(tmp_path, footerless, variant)

    assert len(compiler.source_prompts) == 2
    assert new in generated


def test_calcify_accepts_string_replace_but_rejects_path_replace():
    string_source = _GOOD_RECIPE.replace(
        "return bool(target)", 'return bool(str(target).replace("\\t", "    "))'
    )
    assert ".replace" in calcify_source(
        "# skill", _registry_emitting(string_source).get("compiler")
    )

    path_source = _GOOD_RECIPE.replace(
        "from calcifier import recipe, step",
        "from pathlib import Path\n\nfrom calcifier import recipe, step",
    ).replace("return bool(target)", 'return bool(Path(target).replace("other"))')
    with pytest.raises(CalcifyError, match="path receiver method 'replace'"):
        calcify_source("# skill", _registry_emitting(path_source).get("compiler"))

    branch_source = _GOOD_RECIPE.replace(
        "from calcifier import recipe, step",
        "from pathlib import Path\n\nfrom calcifier import recipe, step",
    ).replace(
        "return bool(target)",
        "if target:\n        value = Path(target)\n"
        "    else:\n        value = target\n"
        '    return bool(value.replace("other"))',
    )
    # A str-or-Path receiver fails ty for str.replace before the effect check sees it.
    with pytest.raises(CalcifyError, match="replace"):
        calcify_source("# skill", _registry_emitting(branch_source).get("compiler"))


def test_calcify_rejects_bytecode_stage_syntax_warning():
    warned = _REPAIR_RECIPE.replace(
        "def inspect(target: str) -> bool:\n",
        'def inspect(target: str) -> bool:\n    assert (target, "always true")\n',
    )
    compiler = source_sequence(warned, _REPAIR_RECIPE)

    generated = calcify_source(_REPAIR_SKILL, compiler, agent_roles=("worker",))

    _assert_same_program(generated, _REPAIR_RECIPE)
    assert "agent output emitted SyntaxWarning" in compiler.source_prompts[1]
    assert "assertion is always true" in compiler.source_prompts[1]


def test_verify_source_rejects_a_tool_outside_declared_capabilities():
    src = textwrap.dedent(
        """
        from calcifier import recipe, step
        from calcifier import tool

        @step
        def s(x: str) -> bool:
            tool("Slack.postMessage", channel=x)
            return True

        @recipe(capabilities=("Read",))
        def r(target: str = "x") -> bool:
            return s(x=target)

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    )
    with pytest.raises(CalcifyError, match="undeclared tool capability Slack.postMessage"):
        verify_source(src)


def test_calcify_accepts_declared_tool_capability():
    src = textwrap.dedent(
        """
        from calcifier import recipe, step
        from calcifier import tool

        @step
        def s(x: str) -> bool:
            tool("Read", path=x)
            return True

        @recipe(capabilities=("Read",))
        def r(target: str = "x") -> bool:
            return s(x=target)

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    )
    reg = _registry_emitting(src)
    assert "tool" in calcify_source("---\nallowed-tools: Read\n---\n# skill", reg.get("compiler"))


def test_calcify_rejects_positional_tool_inputs():
    src = textwrap.dedent(
        """
        from calcifier import recipe, step
        from calcifier import tool

        @step
        def s(x: str) -> bool:
            tool("Read", {"path": x})
            return True

        @recipe(capabilities=("Read",))
        def r(target: str = "x") -> bool:
            return s(x=target)

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    )
    reg = _registry_emitting(src)
    with pytest.raises(CalcifyError, match="positional argument"):
        calcify_source("---\nallowed-tools: Read\n---\n# skill", reg.get("compiler"))


def test_verify_source_rejects_a_tool_without_declared_capabilities():
    src = textwrap.dedent(
        """
        from calcifier import recipe, step
        from calcifier import tool

        @step
        def s(x: str) -> bool:
            tool("Slack.postMessage", channel=x)
            return True

        @recipe
        def r(target: str = "x") -> bool:
            return s(x=target)

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    )
    with pytest.raises(CalcifyError, match="undeclared tool capability Slack.postMessage"):
        verify_source(src)


def test_calcify_rejects_dynamic_tool_capability():
    src = textwrap.dedent(
        """
        from calcifier import recipe, step
        from calcifier import tool

        @step
        def s(name: str) -> bool:
            tool(name)
            return True

        @recipe(capabilities=("Read",))
        def r(target: str = "Read") -> bool:
            return s(name=target)

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    )
    reg = _registry_emitting(src)
    with pytest.raises(CalcifyError, match="dynamic tool capability"):
        calcify_source("---\nallowed-tools: Read\n---\n# skill", reg.get("compiler"))


def test_calcify_accepts_literal_declared_agent_role():
    src = textwrap.dedent(
        """
        from calcifier import Budget, agent, recipe, step

        @step
        def s(x: str) -> bool:
            return agent(
                "worker", "review {x}", working_directory=".", readable_paths=(".",), x=x
            )

        @recipe(agents=("worker",), budget=Budget(2, 30))
        def r(target: str = "x") -> bool:
            return s(x=target)

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    )

    compiled = calcify_source("# skill", _registry_emitting(src).get("compiler"))
    assert "agent(" in compiled and '"worker"' in compiled


def test_calcify_accepts_agent_prompt_that_explicitly_forbids_mutation():
    source = _REPAIR_RECIPE.replace(
        '"review {target}"',
        '"Report facts about {target}; do not modify the repository."',
    )

    assert "do not modify the repository" in calcify_source(
        _REPAIR_SKILL,
        _registry_emitting(source).get("compiler"),
        agent_roles=("worker",),
    )


@pytest.mark.parametrize(
    "call, agents, match",
    [
        (
            'agent("worker", "review", working_directory=".", readable_paths=(".",))',
            "",
            "undeclared agent role worker",
        ),
        (
            'agent(role, "review", working_directory=".", readable_paths=(".",))',
            '(agents=("worker",))',
            "dynamic agent role",
        ),
    ],
)
def test_calcify_rejects_undeclared_or_dynamic_agent_roles(
    call: str,
    agents: str,
    match: str,
):
    decorator = f"@recipe{agents}" if agents else "@recipe"
    src = textwrap.dedent(
        f"""
        from calcifier import agent, recipe, step

        role = "worker"

        @step
        def s() -> bool:
            return {call}

        {decorator}
        def r() -> bool:
            return s()

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    )

    with pytest.raises(CalcifyError, match=match):
        calcify_source("# skill", _registry_emitting(src).get("compiler"))


@pytest.mark.parametrize(
    ("imports", "call"),
    [
        (
            "from calcifier import agent as invoke_agent, recipe, step",
            'invoke_agent("worker", "review", working_directory=".", readable_paths=(".",))',
        ),
        (
            "import calcifier as cf\nfrom calcifier import recipe, step",
            'cf.agent("worker", "review", working_directory=".", readable_paths=(".",))',
        ),
    ],
)
def test_calcify_rejects_undeclared_agent_throughimport_bindings(
    imports: str,
    call: str,
):
    src = (
        imports
        + "\n\n"
        + textwrap.dedent(
            f"""
        @step
        def s() -> bool:
            return {call}

        @recipe
        def r() -> bool:
            return s()

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
        )
    )

    with pytest.raises(CalcifyError, match="undeclared agent role worker"):
        calcify_source("# skill", _registry_emitting(src).get("compiler"))


def test_calcify_rejects_dynamic_calcifier_call_target():
    src = textwrap.dedent(
        """
        import calcifier
        from calcifier import recipe, step

        @step
        def s() -> bool:
            return getattr(calcifier, "agent")("worker", "review")

        @recipe
        def r() -> bool:
            return s()

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    )

    with pytest.raises(CalcifyError, match="dynamic calcifier call target"):
        calcify_source("# skill", _registry_emitting(src).get("compiler"))

    resource_read = ast.parse('resource("references/guide.md").read_text()', mode="eval").body
    assert isinstance(resource_read, ast.Call)
    assert _is_verified_resource_read(resource_read.func, {"resource": "calcifier.resource"})


def test_chained_resource_read_accepts_string_methods() -> None:
    src = textwrap.dedent(
        """
        from calcifier import recipe, resource, step

        @step
        def read_index() -> list[str]:
            return resource("references/index.md").read_text().splitlines()

        @recipe
        def review() -> list[str]:
            return read_index()

        if __name__ == "__main__":
            raise SystemExit(review.cli())
        """
    )
    verify_source(src)
    fixed = src.replace(
        'return resource("references/index.md").read_text().splitlines()',
        'text = resource("references/index.md").read_text()\n    return text.splitlines()',
    )
    verify_source(fixed)


@pytest.mark.parametrize("method", ["read_text", "read_bytes"])
def test_calcify_allows_direct_verified_resource_reads(method: str):
    returned = "str" if method == "read_text" else "bytes"
    src = textwrap.dedent(
        f"""
        from calcifier import recipe, resource, step

        @step
        def load() -> {returned}:
            return resource("references/guide.md").{method}()

        @recipe()
        def r() -> {returned}:
            return load()

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    )

    _assert_same_program(calcify_source("# skill", _registry_emitting(src).get("compiler")), src)


def test_calcify_rejects_paths_derived_outside_packaged_resource_manifest():
    source = textwrap.dedent(
        """
        from calcifier import recipe, resource, step

        @step
        def locate() -> str:
            return str(resource("references/report.md").parent / "../../scripts/export.py")

        @recipe()
        def run() -> str:
            return locate()

        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )

    with pytest.raises(CalcifyError, match="resource path escapes its manifest"):
        verify_source(source)


def test_calcify_allows_resource_lookup_from_closed_path_tuple():
    source = textwrap.dedent(
        """
        from calcifier import recipe, resource, step

        @step
        def load() -> str:
            values = []
            for path in ("references/a.json", "references/b.json"):
                values.append(resource(path).read_text())
            return "".join(values)

        @recipe()
        def run() -> str:
            return load()

        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )

    verify_source(source)


def test_calcify_checks_agent_calls_in_class_bodies():
    src = textwrap.dedent(
        """
        from calcifier import agent, recipe, step

        class Runner:
            @staticmethod
            def invoke() -> bool:
                return agent(
                    "worker", "review", working_directory=".", readable_paths=(".",)
                )

        @step
        def s() -> bool:
            return Runner.invoke()

        @recipe
        def r() -> bool:
            return s()

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    )

    with pytest.raises(CalcifyError, match="undeclared agent role worker"):
        calcify_source("# skill", _registry_emitting(src).get("compiler"))


def test_calcify_rejects_aliased_undeclared_tool_capability():
    src = textwrap.dedent(
        """
        from calcifier import recipe, step, tool as use_tool

        @step
        def s() -> bool:
            return use_tool("Write", path="out")

        @recipe(capabilities=("Read",))
        def r() -> bool:
            return s()

        if __name__ == "__main__":
            raise SystemExit(r.cli())
        """
    )

    with pytest.raises(CalcifyError, match="undeclared tool capability Write"):
        calcify_source(
            "---\nallowed-tools: Read, Write\n---\n# skill", _registry_emitting(src).get("compiler")
        )


def test_generated_calcifier_helper_call_must_bind_runtime_signature() -> None:
    with pytest.raises(
        CalcifyError,
        match=r"emit",
    ):
        verify_source(
            _helper_signature_recipe("emit()"),
        )


def test_generated_calcifier_helper_call_accepts_complete_runtime_signature() -> None:
    verify_source(
        _helper_signature_recipe("emit(target)"),
    )


@pytest.mark.parametrize(
    "call, message",
    [
        ("emit(*())", "emit"),
        ("emit(target, unknown=True)", "unknown"),
    ],
)
def test_generated_calcifier_helper_calls_with_wrong_arguments_are_rejected(
    call: str,
    message: str,
) -> None:
    with pytest.raises(CalcifyError, match=message):
        verify_source(_helper_signature_recipe(call))


@pytest.mark.parametrize(
    "statement",
    [
        "return 1",
        "raise ValueError",
        "raise Other()",
        "raise SystemExit()",
        "raise SystemExit(1, 2)",
        "raise SystemExit(code=run.cli())",
        "raise SystemExit(1)",
        "raise SystemExit(run.cli(1))",
        "raise SystemExit(run.cli(code=1))",
        "raise SystemExit(make_cli())",
        "raise SystemExit(run.other())",
    ],
)
def test_cli_footer_target_rejects_noncanonical_shapes(statement: str) -> None:
    node = ast.parse(f"def main():\n    {statement}\n").body[0]
    assert isinstance(node, ast.FunctionDef)

    assert _system_exit_cli_target(node.body[0]) is None


def test_cli_footer_target_and_guard_require_exact_shape() -> None:
    valid = ast.parse('if __name__ == "__main__":\n    raise SystemExit(run.cli())\n').body[0]
    wrong_test = ast.parse("if ready:\n    raise SystemExit(run.cli())\n").body[0]
    extra_body = ast.parse(
        'if __name__ == "__main__":\n    prepare()\n    raise SystemExit(run.cli())\n'
    ).body[0]
    with_else = ast.parse(
        'if __name__ == "__main__":\n    raise SystemExit(run.cli())\nelse:\n    pass\n'
    ).body[0]
    assert isinstance(valid, ast.If)
    assert isinstance(wrong_test, ast.If)
    assert isinstance(extra_body, ast.If)
    assert isinstance(with_else, ast.If)

    assert _footer_target(valid) == "run"
    assert _footer_target(wrong_test) is None
    assert _footer_target(extra_body) is None
    assert _footer_target(with_else) is None


def test_async_steps_must_be_awaited_from_async_callers():
    invalid = textwrap.dedent(
        """
        from calcifier import recipe, step

        @step
        async def transform(value: str) -> str:
            return value.upper()

        @recipe()
        def run(value: str = "ready") -> str:
            return transform(value)

        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )

    # ty sees the unawaited coroutine where `str` is declared; the custom check
    # covers callers whose declared types would not conflict.
    with pytest.raises(CalcifyError, match="called without await|CoroutineType"):
        verify_source(invalid)

    valid = invalid.replace(
        'def run(value: str = "ready") -> str:\n    return transform(value)',
        'async def run(value: str = "ready") -> str:\n    return await transform(value)',
    )
    verify_source(valid)


def test_the_taint_check_needs_no_plan_and_covers_commands() -> None:
    source = textwrap.dedent(
        """
        from calcifier import Budget, agent, recipe, run_command, step


        @step
        def name_commit(working_directory: str) -> str:
            return agent(
                "worker",
                "Name the commit to show.",
                working_directory=working_directory,
                readable_paths=(working_directory,),
            )


        @step
        def show(sha: str) -> str:
            return run_command(["git", "show", sha], timeout_seconds=30, mutating=False).stdout


        @recipe(agents=("worker",), budget=Budget(2, 60))
        def run(working_directory: str = ".") -> str:
            return show(name_commit(working_directory))


        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )

    with pytest.raises(CalcifyError, match="model-derived value reaches a command") as caught:
        verify_source(source)

    (diagnostic,) = caught.value.diagnostics
    assert diagnostic.code == "model_derived_values"
    assert diagnostic.node == "show"


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        (
            'call_recipe("child.py", value=target).output',
            "invalid child recipe call|ty\\[",
        ),
        (
            'call_catalog_recipe("analysis-real", {"analysis-real": "child.py"}, '
            "value=target).output",
            "invalid child recipe call|ty\\[",
        ),
    ],
)
def test_generated_child_calls_match_runtime_signature(expression: str, message: str):
    source = _GOOD_RECIPE.replace(
        "from calcifier import recipe, step",
        "from calcifier import call_catalog_recipe, call_recipe, recipe, step",
    ).replace("return bool(target)", f"return bool({expression})")

    with pytest.raises(CalcifyError, match=message):
        verify_source(source)


def test_a_child_result_may_be_kept_before_its_output_is_read():
    source = _GOOD_RECIPE.replace(
        "from calcifier import recipe, step",
        "from calcifier import call_recipe, recipe, step",
    ).replace(
        "return bool(target)",
        'result = call_recipe("child.py", {"value": target})\n    return bool(result.output)',
    )

    verify_source(source)


def test_generated_output_root_rejects_placeholders_and_double_rooting():
    placeholder = _REALIZED_FILESYSTEM_RECIPE.replace(
        'output_root="workspace"', 'output_root="artifacts/<run_id>"'
    )
    with pytest.raises(CalcifyError, match="without placeholders"):
        verify_source(placeholder)

    double_rooted = textwrap.dedent(
        """
        from calcifier import recipe, step, write_json

        @step
        def persist(output_path: str) -> bool:
            write_json(output_path + "/report.json", {})
            return True

        @recipe(output_root="artifacts")
        def run(output_path: str = "artifacts/run") -> bool:
            return persist(output_path)

        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )
    with pytest.raises(CalcifyError, match="rooted twice"):
        verify_source(double_rooted)

    target_rooted = textwrap.dedent(
        """
        from pathlib import Path
        from calcifier import ensure_directory, recipe, step

        @step
        def persist(working_directory: str, output_path: str) -> bool:
            root = Path(working_directory)
            destination = root / output_path
            ensure_directory(destination)
            return True

        @recipe(output_root=".")
        def run(working_directory: str = ".", output_path: str = "reports") -> bool:
            return persist(working_directory, output_path)

        if __name__ == "__main__":
            raise SystemExit(run.cli())
        """
    )
    with pytest.raises(CalcifyError, match="incorrectly rooted beneath working_directory"):
        verify_source(target_rooted)
