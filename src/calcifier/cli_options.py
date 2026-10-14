"""Flags and budget overrides shared by both recipe entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from .budget import Budget
from .validation import is_positive_number


def add_agent_flags(
    parser: argparse.ArgumentParser, *, internal: bool = False, wall_budget: bool = True
) -> None:
    prefix = "_" if internal else ""
    parser.add_argument(
        "--agent-config",
        dest=f"{prefix}agent_config",
        type=Path,
        help="required version-1 semantic-role configuration",
    )
    parser.add_argument(
        "--agent-timeout",
        dest=f"{prefix}agent_timeout",
        action="append",
        default=[],
        metavar="ROLE=SECONDS",
        help="override one configured role deadline",
    )
    if wall_budget:
        parser.add_argument(
            "--max-wall-seconds",
            dest=f"{prefix}max_wall_seconds",
            type=float,
            help="limit whole-run wall time (shortens any declared deadline)",
        )


def add_tool_flags(parser: argparse.ArgumentParser, *, internal: bool = False) -> None:
    prefix = "_" if internal else ""
    parser.add_argument(
        "--tool-config",
        dest=f"{prefix}tool_config",
        type=Path,
        help="versioned JSON mapping declared capabilities to subprocess providers",
    )
    parser.add_argument(
        "--skip-tool-preflight",
        dest=f"{prefix}skip_tool_preflight",
        action="store_true",
        help="run even when a declared capability has no available provider",
    )


def add_output_flag(parser: argparse.ArgumentParser, *, internal: bool = False) -> None:
    parser.add_argument(
        "--output-dir",
        dest="_output_directory" if internal else "output_directory",
        type=Path,
        help="redirect the recipe's declared output root (relative to the current directory)",
    )


def resolve_output_directory(declared: Path | None, override: Path | None) -> Path | None:
    if override is None:
        return declared
    if declared is None:
        raise ValueError("--output-dir requires a recipe with a declared output_root")
    destination = override.resolve()
    for parent in (destination, *destination.parents):
        if parent.exists() and not parent.is_dir():
            raise ValueError(f"output directory is not a directory: {parent}")
    return destination


def shorten_budget(declared: Budget | None, override: float | None) -> Budget | None:
    if override is None:
        return declared
    if not is_positive_number(override, finite=True):
        raise ValueError("--max-wall-seconds must be positive")
    if declared is None:
        return Budget(max_agent_calls=None, max_wall_seconds=override)
    return Budget(
        max_agent_calls=declared.max_agent_calls,
        max_wall_seconds=min(declared.max_wall_seconds, override),
    )
