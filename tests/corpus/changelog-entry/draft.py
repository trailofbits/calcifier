from __future__ import annotations

from pathlib import Path

from calcifier import Budget, agent, emit, output_type, recipe, step, verify_artifact, write_text


@output_type
class Entry:
    markdown: str


@step
def draft_entry(working_directory: str) -> Entry:
    return agent(
        "worker",
        "Read the changes described in the workspace and write a changelog entry in Markdown.",
        output=Entry,
        working_directory=working_directory,
        readable_paths=(working_directory,),
    )


@step
def save_entry(entry: Entry, output_dir: str) -> Path:
    artifact = write_text(f"{output_dir}/changelog-entry.md", entry.markdown)
    verify_artifact(artifact)
    emit(artifact)
    return artifact


@recipe(agents=("worker",), budget=Budget(2, 120), output_root=".")
def changelog_entry(output_dir: str, working_directory: str = ".") -> Path:
    entry = draft_entry(working_directory)
    return save_entry(entry, output_dir)


if __name__ == "__main__":
    raise SystemExit(changelog_entry.cli())
