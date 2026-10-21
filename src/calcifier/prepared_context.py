"""Bounded prompt embedding backed by a complete task file."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from .atomic_io import atomic_write
from .exceptions import RecipeError
from .hashing import sha256_hex
from .instruction_context import current_instructions, instruction_prompt
from .outputs import output_type

_DIRECTORY: ContextVar[Path | None] = ContextVar("prepared_context_directory", default=None)


@contextmanager
def prepared_context_run() -> Generator[None]:
    """Own scratch bundles for one run, including its parallel workers."""
    with tempfile.TemporaryDirectory(prefix="calcifier-context-") as temporary:
        token = _DIRECTORY.set(Path(temporary).resolve())
        try:
            yield
        finally:
            _DIRECTORY.reset(token)


@output_type(frozen=True)
class PreparedContext:
    prompt: str
    directory: str
    byte_budget: int = 65536


def prepare_context(
    task: str,
    *,
    instructions: str = "",
    prior_results: str = "",
    byte_budget: int = 65536,
    files: Iterable[str | Path] = (),
    readable_roots: Iterable[str | Path] = (),
) -> PreparedContext:
    """Save the complete task in per-run scratch storage; the budget limits embedding only.

    The task, instructions and prior results are always saved in full as ``task.txt``
    and embedded in the prompt when they fit. Use the embedded assignment directly; read
    task.txt when it is absent or a file check is required. Include every record the agent
    must select from or refer to in ``prior_results``, for example
    ``prior_results=index.model_dump_json()``. Pass the result as
    ``agent(..., context=context)``; the bundle is readable alongside the caller's
    working directory. The runtime removes it when the recipe run ends.
    Gather target files and manifests with ``files=`` and explicit ``readable_roots=``.
    All selected paths must resolve to regular files beneath an authorized resolved root
    before any contents are read; escaping symlinks are rejected. Explicit external inputs
    and resource() snapshots remain supported by including their roots or exact file paths.
    UTF-8 content is saved completely, retaining LF/CRLF, without changing the embedding limit.
    """
    if type(byte_budget) is not int or byte_budget < 1024:
        raise ValueError("context embedding budget must be an integer of at least 1024 bytes")
    task_text = (
        f"Task:\n{task}\n\nInstructions:\n{instructions}\n\nPrior results:\n{prior_results}\n"
    )
    directory = _DIRECTORY.get()
    if directory is None:
        raise RecipeError("prepare_context() requires an active recipe run")
    task_text += _context_files(files, readable_roots)
    data = task_text.encode("utf-8")
    root = directory / sha256_hex(data)
    with tempfile.TemporaryDirectory(prefix=".bundle-", dir=directory) as temporary:
        staging = Path(temporary)
        atomic_write(staging / "task.txt", data)
        try:
            staging.replace(root)
        except OSError:
            # Existing bundles are complete: another caller published first.
            if not root.is_dir():
                raise
    prompt = (
        f"Prepared context directory: {root}\n"
        "Complete assignment: task.txt. Use embedded assignment in full; otherwise read task.txt. "
        "Never limit scope to a preview.\n"
    )
    prompt = instruction_prompt(current_instructions(), prompt, byte_budget)
    if len((prompt + task_text).encode()) <= byte_budget:
        prompt += task_text
    return PreparedContext(prompt, str(root), byte_budget)


def _context_files(files: Iterable[str | Path], readable_roots: Iterable[str | Path]) -> str:
    """Validate the complete selection before reading any file into the bundle."""
    selected = tuple(Path(path) for path in files)
    if not selected:
        return ""
    try:
        roots = tuple(Path(root).resolve(strict=True) for root in readable_roots)
        if not roots:
            raise RecipeError("prepared-context files require explicit readable_roots")
        resolved = tuple(path.resolve(strict=True) for path in selected)
        for source, target in zip(selected, resolved, strict=True):
            if not any(target.is_relative_to(root) for root in roots):
                raise RecipeError(f"prepared-context file escapes readable_roots: {source}")
            if not target.is_file():
                raise RecipeError(f"prepared-context input is not a regular file: {source}")
        records = [
            {"path": str(source), "text": target.read_bytes().decode("utf-8")}
            for source, target in zip(selected, resolved, strict=True)
        ]
    except (OSError, UnicodeError) as exc:
        raise RecipeError(f"cannot read prepared-context files: {exc}") from exc
    return "\nFiles:\n" + json.dumps(records, ensure_ascii=False) + "\n"
