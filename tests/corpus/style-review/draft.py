from __future__ import annotations

from calcifier import Budget, agent, output_type, recipe, resource, step


@output_type
class Fix:
    rule: str
    excerpt: str
    rewrite: str


@step
def load_guidance() -> str:
    guide = resource("references/style-guide.md").read_text(encoding="utf-8")
    examples = resource("references/examples.md").read_text(encoding="utf-8")
    return guide + "\n\n" + examples


@step
def review(document: str, guidance: str, working_directory: str) -> list[Fix]:
    return agent(
        "worker",
        "Review {document} against every rule in this style guidance. Return each problem "
        "with the rule it breaks, the offending excerpt, and a suggested rewrite.\n{guidance}",
        output=list[Fix],
        working_directory=working_directory,
        readable_paths=(working_directory,),
        document=document,
        guidance=guidance,
    )


@recipe(agents=("worker",), budget=Budget(2, 120))
def style_review(document: str, working_directory: str = ".") -> list[Fix]:
    guidance = load_guidance()
    return review(document, guidance, working_directory)


if __name__ == "__main__":
    raise SystemExit(style_review.cli())
