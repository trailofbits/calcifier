"""The mutation gate: every module is mutated and reported against the threshold on its own."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_OPERATORS = {"comparison", "arithmetic", "boolean", "boundary", "return"}


def test_ci_mutates_the_whole_package_in_operator_shards() -> None:
    from scripts.mutation_shards import targets, weights_for

    workflow = (_ROOT / ".github/workflows/mutation.yml").read_text()
    partitions = re.findall(r"\{operator: (\w+), index: (\d+), count: (\d+)\}", workflow)

    assert {operator for operator, _, _ in partitions} == _OPERATORS
    root = _ROOT / "src/calcifier"
    all_modules = set(root.rglob("*.py"))
    for operator in _OPERATORS:
        modules = [
            path
            for name, index, count in partitions
            if name == operator
            for path in targets(root, int(index), int(count), weights=weights_for(operator))
        ]
        assert len(modules) == len(all_modules)
        assert set(modules) == all_modules
    assert "GREMLIN_OPERATORS: ${{ matrix.operator }}" in workflow
    assert (
        'scripts/mutation_shards.py --operator "$GREMLIN_OPERATORS" '
        '--index "$SHARD_INDEX" --count "$SHARD_COUNT"'
    ) in workflow
    assert "export MUTATION_RUN_TARGETS" in workflow
    assert "make mutate\n" in workflow
    assert "test -s coverage/gremlins/gremlins.json" in workflow
    assert "--gremlin-targets" not in workflow
    assert workflow.count("name: gremlins-report-${{ matrix.operator }}-${{ matrix.index }}") == 1
    # Every shard must succeed before the per-module rate is reported.
    assert "SHARDS: ${{ needs.gremlins.result }}" in workflow
    assert 'test "$SHARDS" = success' in workflow
    assert 'scripts/mutation_threshold.py "${reports[@]}" --min 90' in workflow
    # Report-only until v0.1.0 ships.
    report = workflow.split("- name: Report the per-module kill rate", 1)[1]
    assert report.lstrip().startswith("continue-on-error: true")


def test_ci_mutation_cache_is_exact_tree_only() -> None:
    workflow = (_ROOT / ".github/workflows/mutation.yml").read_text()

    assert workflow.count("uses: actions/cache@") == 1
    assert "restore-keys:" not in workflow
    for hashed_input in (
        "uv.lock",
        "pyproject.toml",
        "Makefile",
        "src/**",
        "tests/**",
        "plugins/**",
        "scripts/mutation_shards.py",
        "scripts/mutation_weights.json",
        ".github/workflows/mutation.yml",
    ):
        assert workflow.count(f"'{hashed_input}'") == 1
    assert "${{ matrix.index }}of${{ matrix.count }}" in workflow


@pytest.mark.parametrize("index,count", [(-1, 4), (4, 4), (0, 0)])
def test_invalid_mutation_partition_is_rejected(tmp_path, index, count):
    from scripts.mutation_shards import targets

    with pytest.raises(ValueError, match="require count"):
        targets(tmp_path, index, count)


def test_module_partitions_are_disjoint_complete_and_nonempty(tmp_path):
    from scripts.mutation_shards import targets

    files = [tmp_path / f"module_{index}.py" for index in range(11)]
    for path in files:
        path.touch()
    partitions = [targets(tmp_path, index, 4) for index in range(4)]
    assert [len(partition) for partition in partitions] == [3, 3, 3, 2]
    assert sorted(path for partition in partitions for path in partition) == sorted(files)
    with pytest.raises(ValueError, match="no target modules"):
        targets(tmp_path, 11, 12)


def test_measured_cost_balances_heavy_modules_instead_of_alphabetical_positions(tmp_path):
    from scripts.mutation_shards import targets

    files = [tmp_path / f"module_{index:02}.py" for index in range(16)]
    for path in reversed(files):
        path.touch()
    weights = {path.name: 100.0 if index % 4 == 0 else 1.0 for index, path in enumerate(files)}
    partitions = [targets(tmp_path, index, 4, weights=weights) for index in range(4)]

    assert sorted(path for partition in partitions for path in partition) == files
    assert all(sum(weights[path.name] == 100 for path in part) == 1 for part in partitions)
    assert [sum(weights[path.name] for path in part) for part in partitions] == [103] * 4
    assert partitions == [
        targets(tmp_path, index, 4, weights=dict(reversed(list(weights.items()))))
        for index in range(4)
    ]


def test_unmeasured_and_zero_cost_modules_stay_in_mutation_scope(tmp_path):
    from scripts.mutation_shards import targets

    files = [tmp_path / name for name in ("measured.py", "new.py", "zero.py")]
    for path in files:
        path.touch()
    weights = {"measured.py": 100.0, "zero.py": 0.0}
    partitions = [targets(tmp_path, index, 2, weights=weights) for index in range(2)]

    assert sorted(path for partition in partitions for path in partition) == files
    assert partitions == [[files[0], files[2]], [files[1]]]


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan"), True])
def test_invalid_mutation_weights_fail_before_selecting_targets(tmp_path, value):
    from scripts.mutation_shards import targets

    (tmp_path / "module.py").touch()
    with pytest.raises(ValueError, match="finite nonnegative"):
        targets(tmp_path, 0, 1, weights={"module.py": value})


def test_makefile_mutates_the_package_and_every_changed_module() -> None:
    import tomllib

    makefile = (_ROOT / "Makefile").read_text()

    assert "MUTATION_TARGETS := src/calcifier\n" in makefile
    assert "MUTATION_RUN_TARGETS ?= $(MUTATION_TARGETS)" in makefile
    assert "CHANGED_MUTATION_PATHS := $(filter src/calcifier/%.py," in makefile
    assert "git diff --name-only --diff-filter=ACMR $(MUTATION_BASE)" in makefile
    assert "git ls-files --others --exclude-standard" in makefile
    assert "--gremlin-targets=$(CHANGED_MUTATION_TARGETS)" in makefile
    assert (
        makefile.count("scripts/mutation_threshold.py coverage/gremlins/gremlins.json --min 90")
        == 2
    )
    config = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    assert "targets" not in config["tool"]["pytest-gremlins"]
    from fnmatch import fnmatchcase

    includes = config["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    assert any(fnmatchcase("/scripts/mutation_weights.json", pattern) for pattern in includes)


def test_coverage_does_not_exclude_variadic_tuple_annotations() -> None:
    import tomllib

    root = Path(__file__).parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    patterns = config["tool"]["coverage"]["report"]["exclude_lines"]
    for source in (
        "def required_children() -> tuple[str, ...]:",
        "    ) -> tuple[tuple[str, ...], ...]:",
        '    description = "summary..."',
    ):
        assert not any(re.search(pattern, source) for pattern in patterns)
    assert any(re.search(pattern, "    ... # protocol stub") for pattern in patterns)


def test_parametrized_test_ids_cannot_corrupt_mutation_coverage_contexts(
    request: pytest.FixtureRequest,
) -> None:
    # Gremlins encodes contexts as nodeid|phase and strips " [PLUGIN]" suffixes.
    # Source snippets in default pytest IDs must not become either delimiter.
    for item in request.session.items:
        assert "|" not in item.nodeid, item.nodeid
        assert " [" not in item.nodeid, item.nodeid
        assert "\n" not in item.nodeid, item.nodeid


def test_parameter_ids_preserve_distinctions_and_simple_readable_values() -> None:
    from tests.conftest import pytest_make_parametrize_id

    assert pytest_make_parametrize_id(1) is None
    assert pytest_make_parametrize_id("alpha-beta") is None
    ids = [pytest_make_parametrize_id(value) for value in ("a | b", "a [b]", "a\nb", "")]
    assert len(set(ids)) == 4
    assert all(value is not None and re.fullmatch(r"[A-Za-z0-9_.-]+", value) for value in ids)
    assert pytest_make_parametrize_id("a | b") == ids[0]
