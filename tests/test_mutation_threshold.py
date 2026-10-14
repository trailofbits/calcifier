"""Tests for the per-module pytest-gremlins kill-rate gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# scripts/ is not a package, so add it to sys.path explicitly for the import below.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import mutation_threshold  # ty: ignore[unresolved-import]  # noqa: E402


def _report(tmp_path: Path, name: str = "gremlins.json", **files: dict[str, int]) -> Path:
    """Write a report with ``count`` results of each status for each named file."""
    results = [
        {"file_path": str(tmp_path / f"{file}.py"), "status": status}
        for file, counts in files.items()
        for status, count in counts.items()
        for _ in range(count)
    ]
    path = tmp_path / name
    path.write_text(json.dumps({"summary": {}, "files": {}, "results": results}))
    return path


def test_every_file_at_the_threshold_passes(tmp_path, capsys):
    report = _report(tmp_path, a={"zapped": 9, "survived": 1}, b={"zapped": 10})

    assert mutation_threshold.main([str(report), "--min", "90"]) == 0
    out = capsys.readouterr().out
    assert " 90.00%  9/10  " in out
    assert "PASS" in out


def test_a_strong_file_cannot_hide_a_weak_one(tmp_path, capsys):
    report = _report(tmp_path, weak={"zapped": 8, "survived": 2}, strong={"zapped": 90})

    assert mutation_threshold.main([str(report), "--min", "90"]) == 1
    captured = capsys.readouterr()
    assert "overall: 98.00% of 100" in captured.out
    assert "weak.py 80.00%" in captured.err
    assert "strong.py" not in captured.err


def test_operator_shards_merge_per_file(tmp_path, capsys):
    first = _report(tmp_path, "first.json", a={"zapped": 5, "survived": 1})
    second = _report(tmp_path, "second.json", a={"zapped": 4})

    assert mutation_threshold.main([str(first), str(second), "--min", "90"]) == 0
    assert " 90.00%  9/10  " in capsys.readouterr().out


def test_timeouts_and_errors_miss_and_pardons_are_excluded(tmp_path, capsys):
    report = _report(
        tmp_path, a={"zapped": 7, "timeout": 2, "error": 1, "survived": 0, "pardoned": 5}
    )

    assert mutation_threshold.main([str(report), "--min", "90"]) == 1
    output = capsys.readouterr().out
    assert " 70.00%  7/10  " in output
    assert "2 timeout, 1 error" in output
    assert mutation_threshold.main([str(report), "--min", "70"]) == 0


def test_a_file_with_only_pardoned_mutants_is_listed_but_not_rated(tmp_path, capsys):
    report = _report(tmp_path, a={"zapped": 1}, b={"pardoned": 2})

    assert mutation_threshold.main([str(report), "--min", "90"]) == 0
    assert "0/0" in capsys.readouterr().out


def test_no_resolved_mutants_fails(tmp_path, capsys):
    report = _report(tmp_path, a={"pardoned": 3})

    assert mutation_threshold.main([str(report), "--min", "90"]) == 1
    assert "no resolved mutants" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"summary": {"zapped": 1}}, "top-level `results` list"),
        ({"results": [{"status": "zapped"}]}, "no `file_path`"),
        ({"results": [{"file_path": "a.py", "status": "skipped"}]}, "unknown mutant status"),
        ({"results": ["a.py"]}, "no `file_path`"),
    ],
)
def test_malformed_reports_are_refused(tmp_path, data, message):
    path = tmp_path / "gremlins.json"
    path.write_text(json.dumps(data))

    with pytest.raises(SystemExit, match=message):
        mutation_threshold.main([str(path), "--min", "90"])


@pytest.mark.parametrize(
    "reported_path",
    ["/home/runner/work/project/src/module.py", r"Z:\foreign-runner\project\src\module.py"],
    ids=["posix", "windows"],
)
def test_reported_source_paths_do_not_trigger_filesystem_lookups(
    tmp_path, monkeypatch, capsys, reported_path
):
    report = tmp_path / "gremlins.json"
    report.write_bytes(
        json.dumps({"results": [{"file_path": reported_path, "status": "zapped"}]}).encode()
    )

    def reject_resolution(*args, **kwargs):
        raise AssertionError("A reported source path must not be resolved on this host")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "resolve", reject_resolution)
        assert mutation_threshold.main([str(report)]) == 0
    assert reported_path in capsys.readouterr().out


def test_relative_and_absolute_local_paths_share_one_file_score(tmp_path, monkeypatch, capsys):
    first = _report(tmp_path, "first.json", a={"zapped": 5, "survived": 1})
    second = tmp_path / "second.json"
    second.write_bytes(
        json.dumps({"results": [{"file_path": "a.py", "status": "zapped"}] * 4}).encode()
    )
    monkeypatch.chdir(tmp_path)

    assert mutation_threshold.main([str(first), str(second)]) == 0
    assert " 90.00%  9/10  a.py" in capsys.readouterr().out
    assert not (tmp_path / "a.py").exists()
