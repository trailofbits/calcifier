"""Smoke-test fixture: a minimal recipe with one step using the stub provider."""

from __future__ import annotations

from calcifier import Budget, agent, fail, output_type, recipe, step


@output_type
class Greeting:
    text: str


@step
def greet(name: str) -> Greeting:
    return agent(
        "worker",
        "Greet {name}",
        working_directory=".",
        readable_paths=(".",),
        name=name,
    )


@recipe(version="0.1", agents=("worker",), budget=Budget(2, 30))
def trivial(greeting: str = "hello") -> bool:
    if not greeting:
        fail("greeting required")
    out = greet(name=greeting)
    return bool(out.text)


if __name__ == "__main__":
    raise SystemExit(trivial.cli())
