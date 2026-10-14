"""Enforce a minimum mutation kill rate for every module in pytest-gremlins JSON reports.

Usage:
    uv run pytest --gremlins ... --gremlin-report=json
    uv run python scripts/mutation_threshold.py coverage/gremlins/gremlins.json --min 90

Each report's ``results`` list holds one entry per mutant, with the mutated
``file_path`` and its ``status``. Reports from operator shards are merged,
mutants are grouped by file, and every file must reach ``--min``: a strong
module cannot hide a weak one in an aggregate. Only ``zapped`` counts as a kill.
An unvalidated ``timeout`` cannot distinguish nontermination from a slow test
selection and counts as a miss, as does a runner ``error``. ``pardoned`` mutants are
explicit human suppressions, audited with ``--gremlin-audit-pardons``, and are
excluded from the rate.
Exits 0 when every file passes, else 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_KILLS = ("zapped",)
_MISSES = ("survived", "error", "timeout")
_STATUSES = frozenset((*_KILLS, *_MISSES, "pardoned"))


def _results(data: Any, source: Path) -> list[tuple[str, str]]:
    """Return ``(file_path, status)`` for every mutant in one report."""
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise SystemExit(f"{source}: expected a top-level `results` list")
    results: list[tuple[str, str]] = []
    for item in data["results"]:
        file_path = item.get("file_path") if isinstance(item, dict) else None
        status = item.get("status") if isinstance(item, dict) else None
        if not isinstance(file_path, str) or not file_path:
            raise SystemExit(f"{source}: a result has no `file_path`: {item!r}")
        if status not in _STATUSES:
            raise SystemExit(f"{source}: unknown mutant status {status!r} for {file_path}")
        results.append((file_path, status))
    return results


def _display(file_path: str) -> str:
    # Source paths in downloaded reports may belong to another host.
    path = Path(os.path.abspath(file_path))
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return file_path


def _tally(paths: list[Path]) -> dict[str, Counter[str]]:
    per_file: dict[str, Counter[str]] = {}
    for path in paths:
        for file_path, status in _results(json.loads(path.read_text()), path):
            per_file.setdefault(_display(file_path), Counter())[status] += 1
    return per_file


def _rate(counts: Counter[str]) -> tuple[int, int]:
    kills = sum(counts[status] for status in _KILLS)
    return kills, kills + sum(counts[status] for status in _MISSES)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats_paths", nargs="+", type=Path, help="gremlins.json reports")
    parser.add_argument("--min", type=float, default=90.0, help="required kill-rate percentage")
    args = parser.parse_args(argv)

    per_file = _tally(args.stats_paths)
    print(f"reports: {len(args.stats_paths)}")
    failed: list[str] = []
    for file_path in sorted(per_file):
        kills, resolved = _rate(per_file[file_path])
        pardoned = per_file[file_path]["pardoned"]
        if resolved == 0:
            print(f"   n/a  0/0  {file_path} ({pardoned} pardoned)")
            continue
        rate = kills * 100.0 / resolved
        print(
            f"{rate:6.2f}%  {kills}/{resolved}  {file_path} "
            f"({pardoned} pardoned, {per_file[file_path]['timeout']} timeout, "
            f"{per_file[file_path]['error']} error)"
        )
        if rate + 1e-9 < args.min:
            failed.append(f"{file_path} {rate:.2f}%")
    rates = [_rate(counts) for counts in per_file.values()]
    total_kills = sum(kills for kills, _ in rates)
    total = sum(resolved for _, resolved in rates)
    if total == 0:
        print("FAIL: no resolved mutants", file=sys.stderr)
        return 1
    print(
        f"overall: {total_kills * 100.0 / total:.2f}% of {total} (per-file threshold {args.min}%)"
    )
    if failed:
        print(f"FAIL: below {args.min}%: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
