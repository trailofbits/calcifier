from __future__ import annotations

from calcifier import Budget, agent, output_type, recipe, step, tool


@output_type
class Match:
    path: str
    explanation: str


@step
def search(phrase: str, working_directory: str) -> str:
    return str(tool("Grep", pattern=phrase, path=working_directory))


@step
def explain(phrase: str, matches: str, working_directory: str) -> list[Match]:
    return agent(
        "worker",
        "For each file in these Grep results for {phrase}, return the path and a "
        "one-sentence explanation of why the match matters.\n{matches}",
        output=list[Match],
        working_directory=working_directory,
        readable_paths=(working_directory,),
        phrase=phrase,
        matches=matches,
    )


@recipe(agents=("worker",), budget=Budget(2, 120), capabilities=("Grep",))
def text_search(phrase: str, working_directory: str = ".") -> list[Match]:
    matches = search(phrase, working_directory)
    return explain(phrase, matches, working_directory)


if __name__ == "__main__":
    raise SystemExit(text_search.cli())
