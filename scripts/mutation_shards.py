"""Partition complete modules without omitting or repeating any mutation target."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path


def weights_for(operator: str) -> dict[str, float]:
    profile = Path(__file__).with_name("mutation_weights.json")
    return json.loads(profile.read_text(encoding="utf-8"))["weights"][operator]


def targets(
    root: Path, index: int, count: int, *, weights: Mapping[str, float] | None = None
) -> list[Path]:
    if count < 1 or not 0 <= index < count:
        raise ValueError("require count >= 1 and 0 <= index < count")
    files = sorted(root.rglob("*.py"))
    costs = weights or {}
    if any(
        isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in costs.values()
    ):
        raise ValueError("mutation weights must be finite nonnegative numbers")
    measured = [value for value in costs.values() if value > 0]
    fallback = statistics.median(measured) if measured else 1.0
    estimates = {path: costs.get(path.relative_to(root).as_posix(), fallback) for path in files}
    partitions: list[list[Path]] = [[] for _ in range(count)]
    totals = [0.0] * count
    for path in sorted(files, key=lambda path: (-estimates[path], path)):
        shard = min(range(count), key=lambda i: (totals[i], len(partitions[i]), i))
        partitions[shard].append(path)
        totals[shard] += estimates[path]
    selected = sorted(partitions[index])
    if not selected:
        raise ValueError("mutation shard has no target modules")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("src/calcifier"))
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument(
        "--operator", choices=("comparison", "arithmetic", "boolean", "boundary", "return")
    )
    args = parser.parse_args()
    weights = weights_for(args.operator) if args.operator else None
    selected = targets(args.root, args.index, args.count, weights=weights)
    print(",".join(path.as_posix() for path in selected))


if __name__ == "__main__":
    main()
