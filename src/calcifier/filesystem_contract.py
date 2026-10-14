"""Filesystem access requested for agent calls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import CalcifierError


class AgentError(CalcifierError, RuntimeError):
    """Raised when an agent fails to produce a usable response."""


class TransientAgentError(AgentError):
    """A classified transport failure that recipe code may explicitly retry."""


@dataclass(frozen=True)
class FilesystemContract:
    """The complete filesystem access requested for one invocation."""

    working_directory: Path
    readable_paths: tuple[Path, ...]
    writable_paths: tuple[Path, ...] = ()

    @classmethod
    def build(
        cls,
        working_directory: str | Path,
        readable_paths: tuple[str | Path, ...],
        writable_paths: tuple[str | Path, ...] = (),
        *,
        resource_root: Path | None = None,
    ) -> FilesystemContract:
        cwd = Path(working_directory).resolve()
        if not cwd.is_dir():
            raise AgentError(f"agent working directory does not exist: {cwd}")
        writable = tuple(dict.fromkeys(Path(path).resolve() for path in writable_paths))
        for path in writable:
            if not path.is_dir():
                raise AgentError(f"agent writable path must be an existing directory: {path}")
        if writable and not any(cwd.is_relative_to(path) for path in writable):
            raise AgentError("agent working directory must be inside a writable path")
        paths = [Path(path).resolve() for path in readable_paths]
        paths.extend(writable)
        paths.append(cwd)
        if resource_root is not None:
            paths.append(resource_root.resolve())
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise AgentError(f"agent readable path does not exist: {missing[0]}")
        return cls(cwd, tuple(dict.fromkeys(paths)), writable)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "read-write" if self.writable_paths else "read-only",
            "working_directory": str(self.working_directory),
            "readable_paths": [str(path) for path in self.readable_paths],
            "writable_paths": [str(path) for path in self.writable_paths],
        }


def _filesystem_prompt(
    filesystem: FilesystemContract, prompt: str, *, host_delivery: bool = False
) -> str:
    roots = "\n".join(f"- {path}" for path in filesystem.readable_paths)
    writable = "\n".join(f"- {path}" for path in filesystem.writable_paths) or "(none)"
    policy = (
        "Filesystem contract: read-write. You may create, modify, and delete files only "
        "under the declared writable paths. Everything else is read-only. "
        ".git paths stay read-only. Agents never commit; recipes run git with run_command().\n"
        if filesystem.writable_paths
        else "Filesystem contract: read-only. Do not write, modify, or delete files.\n"
    )
    delivery = (
        "Return results in the structured answer; the host persists reports and delivery status.\n"
        if host_delivery
        else ""
    )
    return (
        f"{prompt}\n\n"
        f"{policy}"
        f"{delivery}"
        f"Working directory: {filesystem.working_directory}\n"
        f"Declared readable paths:\n{roots}\nDeclared writable paths:\n{writable}"
    )
