"""Scripted model transport for compiler tests."""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from calcifier.agent_events import AgentProvider, AgentResult
from calcifier.calcify import calcify_file
from calcifier.filesystem_contract import FilesystemContract
from calcifier.recipe import Recipe, load_recipe
from calcifier.recipe_draft import draft_response
from calcifier.registry import Registry

_REPAIR_HEADING = "## Rejected candidate: repair "


class ScriptedCompiler(AgentProvider):
    name = "compiler"
    default_timeout: float | None = 30

    def __init__(
        self,
        responses: Sequence[str | Callable[[str], str]],
        *,
        wrap_sources: bool = False,
        repeat_last: bool = False,
        record_usage: bool = False,
        default_timeout: float = 30,
    ) -> None:
        self.responses: list[str | Callable[[str], str]] = list(responses)
        self.wrap_sources = wrap_sources
        self.repeat_last = repeat_last
        self.record_usage = record_usage
        self.default_timeout = default_timeout
        first = responses[0] if wrap_sources and responses else ""
        self.draft = first if isinstance(first, str) else ""
        self._last: str | None = None
        self.calls: list[tuple[str, str]] = []
        self.source_prompts: list[str] = []
        self.compiler_directories: list[Path] = []

    @property
    def all_prompts(self) -> list[str]:
        return [prompt for _, prompt in self.calls]

    @property
    def source_calls(self) -> int:
        return len(self.calls)

    @property
    def roles(self) -> list[str]:
        """Agent roles declared in a single source fixture."""
        decorators = [
            decorator
            for node in ast.parse(self.draft).body
            if isinstance(node, ast.FunctionDef)
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Call) and ast.unparse(decorator.func) == "recipe"
        ]
        return [
            str(role)
            for decorator in decorators
            for keyword in decorator.keywords
            if keyword.arg == "agents"
            for role in ast.literal_eval(keyword.value)
        ]

    def prompts(self, kind: str) -> list[str]:
        return [prompt for call_kind, prompt in self.calls if call_kind == kind]

    def invoke(
        self,
        prompt: str,
        *,
        timeout: float | None = None,
        filesystem: FilesystemContract | None = None,
        schema: dict[str, Any] | None = None,
    ) -> AgentResult:
        del timeout, schema
        kind = "repair" if _REPAIR_HEADING in prompt else "source"
        self.calls.append((kind, prompt))
        self.source_prompts.append(prompt)
        if filesystem is not None:
            self.compiler_directories.append(filesystem.working_directory)
        if self.responses:
            scripted = self.responses.pop(0)
            response = scripted if isinstance(scripted, str) else scripted(prompt)
            self._last = response
        elif self.repeat_last and self._last is not None:
            response = self._last
        else:
            raise AssertionError("unexpected additional compiler request")
        text = draft_response(response) if self.wrap_sources else response
        usage = {"input_tokens": len(self.calls), "output_tokens": 1} if self.record_usage else {}
        return AgentResult(text, text, "", 0, self.name, usage=usage)


def one_draft(source: str) -> ScriptedCompiler:
    return ScriptedCompiler([source], wrap_sources=True, repeat_last=True)


def source_sequence(*sources: str, repeat_last: bool = False) -> ScriptedCompiler:
    return ScriptedCompiler(sources, wrap_sources=True, repeat_last=repeat_last, record_usage=True)


def draft_sequence(*responses: str) -> ScriptedCompiler:
    return ScriptedCompiler(responses, record_usage=True)


def registry_emitting(source: str) -> Registry:
    registry = Registry()
    registry.register(one_draft(source))
    return registry


def compile_and_load(
    skill_path: Path, registry: Registry, *, output: Path | None = None
) -> tuple[Path, Recipe]:
    path = calcify_file(skill_path, registry, output=output)
    return path, load_recipe(path)
