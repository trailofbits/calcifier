from __future__ import annotations

from calcifier import Budget, agent, recipe, step


@step
def write_notes(working_directory: str) -> str:
    return agent(
        "worker",
        "Read the merged changes in the workspace and return release notes as Markdown.",
        working_directory=working_directory,
        readable_paths=(working_directory,),
    )


@recipe(agents=("worker",), budget=Budget(2, 120))
def release_notes(working_directory: str = ".") -> str:
    return write_notes(working_directory)


if __name__ == "__main__":
    raise SystemExit(release_notes.cli())
