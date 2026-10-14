from __future__ import annotations

from calcifier import Budget, agent, output_type, recipe, step


@output_type
class Digest:
    title: str
    summary: str
    key_points: list[str]


@step
def summarize(working_directory: str) -> Digest:
    return agent(
        "worker",
        "Read every note under the workspace directory. Return a title, a one-paragraph "
        "summary, and up to five key points.",
        output=Digest,
        working_directory=working_directory,
        readable_paths=(working_directory,),
    )


@recipe(agents=("worker",), budget=Budget(2, 120))
def notes_summary(working_directory: str = ".") -> Digest:
    return summarize(working_directory)


if __name__ == "__main__":
    raise SystemExit(notes_summary.cli())
