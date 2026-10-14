from __future__ import annotations

from pathlib import Path

from calcifier import (
    Budget,
    agent,
    emit,
    output_type,
    recipe,
    step,
    verify_artifact,
    write_text,
)


@output_type
class Item:
    item_id: str
    description: str


@step
def describe(inventory: str, working_directory: str) -> list[Item]:
    return agent(
        "worker",
        "Describe every item in the inventory file {inventory}.",
        output=list[Item],
        working_directory=working_directory,
        readable_paths=(working_directory,),
        inventory=inventory,
    )


@step
def save_records(items: list[Item], output_dir: str) -> list[Path]:
    records = []
    for item in items:
        record = write_text(f"{output_dir}/items/{item.item_id}.md", item.description)
        verify_artifact(record)
        records.append(emit(record))
    return records


@recipe(agents=("worker",), budget=Budget(2, 120), output_root=".")
def per_item_records(inventory: str, output_dir: str, working_directory: str = ".") -> list[Path]:
    items = describe(inventory, working_directory)
    return save_records(items, output_dir)


if __name__ == "__main__":
    raise SystemExit(per_item_records.cli())
