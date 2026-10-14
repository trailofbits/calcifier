"""Mutation controls and nested interpreters must execute the same real package."""

from __future__ import annotations

import ast
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import pytest_gremlins.plugin as gremlins

from tests.conftest import pytest_configure
from tests.mutation_sources import _bootstrap

pytestmark = pytest.mark.semantic_offline


def test_spawn_import_keeps_overlay_paths_without_relaunching_pytest(tmp_path, monkeypatch):
    bootstrap = tmp_path / "bootstrap.py"
    bootstrap.write_bytes(_bootstrap().encode())
    root = tmp_path / "modules"
    root.mkdir()
    (tmp_path / "workspace.json").write_bytes(
        json.dumps(
            {
                "roots": [str(root)],
                "workspace": None,
                "original": str(tmp_path),
            }
        ).encode()
    )
    monkeypatch.setattr(sys, "path", sys.path[:])
    monkeypatch.setattr(sys, "argv", [str(bootstrap), "test_worker.py"])
    monkeypatch.setenv("PYTHONPATH", "")

    def unexpected_pytest(*args):
        pytest.fail("spawn import must not relaunch pytest")

    monkeypatch.setattr(pytest, "main", unexpected_pytest)
    runpy.run_path(str(bootstrap), run_name="__mp_main__")
    assert sys.path[0] == str(root)
    assert os.environ["PYTHONPATH"] == str(root)
    assert sys.argv == [str(bootstrap), "test_worker.py"]


def test_bootstrap_executes_a_plain_spawn_worker_without_recursive_collection(tmp_path):
    # This is an unmodified package and ordinary pytest, with no active mutation.
    bootstrap = tmp_path / "bootstrap.py"
    bootstrap.write_bytes(_bootstrap().encode())
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "subject.py").write_bytes(
        b"from pathlib import Path\ndef origin():\n    return str(Path(__file__).parent)\n"
    )
    (tmp_path / "workspace.json").write_bytes(
        json.dumps(
            {
                "roots": [str(modules)],
                "workspace": None,
                "original": str(tmp_path),
            }
        ).encode()
    )
    selected = tmp_path / "test_worker.py"
    selected.write_bytes(b"""import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from subject import origin

def test_worker():
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        assert pool.submit(origin).result(timeout=30) == str(Path(__file__).parent / "modules")
""")
    result = subprocess.run(
        [sys.executable, str(bootstrap), "-q", "--noconftest", "-o", "addopts=", str(selected)],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "ACTIVE_GREMLIN": ""},
        capture_output=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


@pytest.mark.parametrize("package_target", [False, True])
def test_instrumented_package_keeps_metadata_resources_and_native_mutations(
    tmp_path, monkeypatch, request, package_target
):
    monkeypatch.setattr(request.config.option, "gremlins", True, raising=False)
    pytest_configure(request.config)
    package = tmp_path / "src" / "samplepkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    target = package / ("__init__.py" if package_target else "worker.py")
    target.write_text('def value():\n    return "stable"\n', encoding="utf-8")
    module = "samplepkg" if package_target else "samplepkg.worker"
    (package / "__main__.py").write_text(
        f'from {module} import value\nprint(value(), end="")\n', encoding="utf-8"
    )
    (package / "reference.txt").write_text("required context", encoding="utf-8")
    (tmp_path / "child").mkdir()
    source = ast.parse(
        'def value():\n    return "changed" if __gremlin_active__ == "fault" else "stable"\n'
    )
    instrumented = gremlins._write_instrumented_sources({str(target): source}, tmp_path)
    test = tmp_path / "test_native.py"
    test.write_text(
        "import subprocess, sys\nfrom pathlib import Path\n"
        f"import {module} as subject\n"
        "def test_metadata():\n"
        "    path = Path(subject.__file__)\n"
        "    assert path.is_file()\n"
        "    assert subject.__spec__.origin == str(path)\n"
        '    assert path.with_name("reference.txt").read_text() == "required context"\n'
        "def test_native():\n"
        '    answer = subprocess.check_output([sys.executable, "-m", "samplepkg"],\n'
        '        cwd="child", text=True)\n'
        '    assert answer == "stable"\n',
        encoding="utf-8",
    )
    try:
        for active, status in [("", 0), ("fault", 1)]:
            result = _run(instrumented, tmp_path, active, "test_native.py")
            assert result.returncode == status, result.stdout + result.stderr
            if active:
                assert "changed" in result.stdout
                assert "Mutation selection control:" in result.stdout
                assert "2 passed" in result.stdout
    finally:
        gremlins._cleanup_instrumented_dir(instrumented)


def test_failing_instrumented_control_is_an_error_not_a_kill(tmp_path, monkeypatch, request):
    monkeypatch.setattr(request.config.option, "gremlins", True, raising=False)
    pytest_configure(request.config)
    source = ast.parse('value = 2 if __gremlin_active__ == "fault" else 1\n')
    instrumented = gremlins._write_instrumented_sources(
        {str(tmp_path / "alpha.py"): source}, tmp_path
    )
    (tmp_path / "test_alpha.py").write_text(
        "import alpha\ndef test_bad_control():\n    assert alpha.value == 0\n",
        encoding="utf-8",
    )
    try:
        result = _run(instrumented, tmp_path, "fault", "test_alpha.py")
        assert result.returncode == 3, result.stdout + result.stderr
        assert "Mutation selection control:" in result.stdout
        assert "test_bad_control" in result.stderr[:2000]
        assert "assert 1 == 0" in result.stderr[:2000]
        assert not list(instrumented.glob("control-*.passed"))
    finally:
        gremlins._cleanup_instrumented_dir(instrumented)


@pytest.mark.parametrize("import_fault", [False, True])
def test_checkout_keeps_imports_tests_source_pins_and_git_metadata_together(
    tmp_path, monkeypatch, request, import_fault
):
    monkeypatch.setattr(request.config.option, "gremlins", True, raising=False)
    pytest_configure(request.config)
    package = tmp_path / "src" / "samplepkg"
    package.mkdir(parents=True)
    target = package / "__init__.py"
    original = 'value = "stable"\n'
    target.write_text(original, encoding="utf-8")
    (tmp_path / "support.py").write_text(
        "from pathlib import Path\nROOT = Path(__file__).parent\n", encoding="utf-8"
    )
    test = tmp_path / "test_checkout.py"
    test.write_text(
        "import subprocess, sys\nfrom pathlib import Path\n"
        "from support import ROOT\n"
        "assert Path(__file__).is_relative_to(ROOT)\n"
        "import samplepkg\n"
        "def test_checkout():\n"
        "    assert Path(samplepkg.__file__).is_relative_to(ROOT)\n"
        "    assert Path(__file__).is_relative_to(ROOT)\n"
        "    assert Path.cwd() == ROOT\n"
        '    assert (ROOT / "note.txt").read_text() == "untracked"\n'
        '    assert not (ROOT / "cache").exists()\n'
        '    assert subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()\n'
        '    value = subprocess.check_output([sys.executable, "-c",\n'
        '        "import samplepkg; print(samplepkg.value)"], text=True)\n'
        '    assert value.strip() == "stable"\n',
        encoding="utf-8",
    )
    (tmp_path / ".gitignore").write_text("cache/\n", encoding="utf-8")
    for args in [
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            f"core.hooksPath={tmp_path / 'no-hooks'}",
            "commit",
            "-qm",
            "fixture",
        ],
    ]:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True)
    (tmp_path / "note.txt").write_text("untracked", encoding="utf-8")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "ignored").touch()
    source = ast.parse(
        'if __gremlin_active__ == "fault":\n    raise ValueError("mutant import fault")\n'
        'value = "stable"\n'
        if import_fault
        else 'value = "changed" if __gremlin_active__ == "fault" else "stable"\n'
    )
    instrumented = gremlins._write_instrumented_sources({str(target): source}, tmp_path)
    try:
        for active, status in [("", 0), ("fault", 1)]:
            result = _run(instrumented, tmp_path, active, str(test) + "::test_checkout")
            assert result.returncode == status, result.stdout + result.stderr
            if active and import_fault:
                assert "Mutation import probe (active):" in result.stdout
        assert target.read_text(encoding="utf-8") == original
    finally:
        gremlins._cleanup_instrumented_dir(instrumented)


def _run(directory: Path, cwd: Path, active: str, selector: str):
    return subprocess.run(
        [sys.executable, str(directory / "gremlin_lightweight_runner.py"), selector],
        cwd=cwd,
        env={
            **os.environ,
            "PYTEST_GREMLINS_SOURCES_FILE": str(directory / "sources.json"),
            "ACTIVE_GREMLIN": active,
        },
        capture_output=True,
        text=True,
        # Initial collection, selector control, and two bounded import probes.
        timeout=180,
        check=False,
    )
