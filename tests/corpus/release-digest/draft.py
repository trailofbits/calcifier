from __future__ import annotations

from calcifier import Budget, agent, call_recipe, output_type, recipe, step


@output_type
class Digest:
    headline: str
    digest: str


@step
def collect_notes(working_directory: str) -> str:
    return str(
        call_recipe(
            "../../release-notes/.calcifier/recipe.py", {"working_directory": working_directory}
        ).output
    )


@step
def summarize(notes: str, working_directory: str) -> Digest:
    return agent(
        "worker",
        "Write a one-line headline and a short digest of these release notes.\n{notes}",
        output=Digest,
        working_directory=working_directory,
        readable_paths=(working_directory,),
        notes=notes,
    )


@recipe(agents=("worker",), budget=Budget(4, 240))
def release_digest(working_directory: str = ".") -> Digest:
    notes = collect_notes(working_directory)
    return summarize(notes, working_directory)


if __name__ == "__main__":
    raise SystemExit(release_digest.cli())
