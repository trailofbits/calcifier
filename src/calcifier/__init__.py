"""Calcifier: typed Python recipes for LLM-assisted skills."""

from .agents import agent
from .budget import Budget, BudgetExceeded
from .composition import (
    SubrecipeError,
    SubrecipeResult,
    call_catalog_recipe,
    call_recipe,
)
from .filesystem_contract import TransientAgentError
from .mutations import (
    CommandResult,
    ensure_directory,
    run_command,
    verify_artifact,
    write_json,
    write_text,
)
from .outputs import output_type, word_count
from .prepared_context import PreparedContext, prepare_context
from .recipe import (
    Recipe,
    RecipeError,
    RecipeFailure,
    RecipeRun,
    emit,
    fail,
    load_recipe,
    parallel,
    recipe,
    resource,
    step,
)
from .side_effects import record_side_effect
from .tools import tool

__version__ = "0.2.0"

__all__ = [
    "Budget",
    "BudgetExceeded",
    "CommandResult",
    "PreparedContext",
    "Recipe",
    "RecipeError",
    "RecipeFailure",
    "RecipeRun",
    "SubrecipeError",
    "SubrecipeResult",
    "TransientAgentError",
    "__version__",
    "agent",
    "call_catalog_recipe",
    "call_recipe",
    "emit",
    "ensure_directory",
    "fail",
    "load_recipe",
    "output_type",
    "parallel",
    "prepare_context",
    "recipe",
    "record_side_effect",
    "resource",
    "run_command",
    "step",
    "tool",
    "verify_artifact",
    "word_count",
    "write_json",
    "write_text",
]
