"""Run instrumented packages from real files, including in native child processes."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest_gremlins.plugin as gremlins
from pytest_gremlins.instrumentation.origin_finder import normalize_origin

_WRITE_SOURCES = gremlins._write_instrumented_sources
_BUILD_TEST_COMMAND = gremlins._build_test_command


def write_sources(
    instrumented_asts: dict[str, ast.Module],
    rootdir: Path,
    duplicates: dict[str, list[str]] | None = None,
) -> Path:
    """Keep upstream mutation metadata while replacing its process-local import hook."""
    directory: Path = _WRITE_SOURCES(instrumented_asts, rootdir, duplicates).resolve()
    sources: dict[str, str] = {
        origin: entry["source"]
        for origin, entry in json.loads(
            (directory / "sources.json").read_text(encoding="utf-8")
        ).items()
    }
    if (rootdir / ".git").exists():
        workspace = _copy_checkout(rootdir, directory / "workspace")
        for original in instrumented_asts:
            source = rootdir / original
            target = workspace / source.relative_to(rootdir)
            target.write_text(
                sources[normalize_origin(str(source))], encoding="utf-8", newline="\n"
            )
        roots = [str(workspace / "src"), str(workspace)]
    else:
        workspace = None
        roots = [str(_copy_packages(instrumented_asts, sources, rootdir, directory))]
    (directory / "workspace.json").write_text(
        json.dumps(
            {
                "roots": roots,
                "workspace": str(workspace) if workspace else None,
                "original": str(rootdir),
            }
        ),
        encoding="utf-8",
    )
    (directory / "gremlin_bootstrap.py").write_text(_bootstrap(), encoding="utf-8")
    (directory / "gremlin_lightweight_runner.py").write_text(
        gremlins._get_lightweight_runner_script(), encoding="utf-8"
    )
    return directory


def build_test_command(
    instrumented_dir: Path | None,
    preserved_addopts: str = "",
    *,
    xdist_loaded: bool = False,
) -> list[str]:
    if instrumented_dir is not None:
        # Our runner owns pytest options and confirms failures against passing controls.
        return [sys.executable, str(instrumented_dir / "gremlin_lightweight_runner.py")]
    return _BUILD_TEST_COMMAND(instrumented_dir, preserved_addopts, xdist_loaded=xdist_loaded)


def _copy_checkout(rootdir: Path, workspace: Path) -> Path:
    """Keep source fingerprints, scripts and tests in the same disposable checkout."""
    raw = subprocess.check_output(
        ["git", "-C", str(rootdir), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    )
    workspace.mkdir()
    for name in sorted(set(raw.split(b"\0")) - {b""}):
        relative = Path(os.fsdecode(name))
        source = rootdir / relative
        if source.is_file():
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    common = subprocess.check_output(
        ["git", "-C", str(rootdir), "rev-parse", "--git-common-dir"], encoding="utf-8"
    ).strip()
    objects = (rootdir / common / "objects").resolve()
    (workspace / ".git/objects/info/alternates").write_text(
        str(objects) + "\n", encoding="utf-8", newline="\n"
    )
    head = subprocess.check_output(
        ["git", "-C", str(rootdir), "rev-parse", "HEAD"], encoding="utf-8"
    )
    (workspace / ".git/HEAD").write_text(head, encoding="utf-8", newline="\n")
    return workspace


def _copy_packages(
    instrumented_asts: dict[str, ast.Module],
    sources: dict[str, str],
    rootdir: Path,
    directory: Path,
) -> Path:
    overlay = directory / "modules"
    overlay.mkdir()
    if (rootdir / "docs").is_dir():
        # Preserve source-checkout resource fallbacks relative to package files.
        shutil.copytree(rootdir / "docs", directory / "docs")
    copied: set[str] = set()
    for original in instrumented_asts:
        source = rootdir / original
        relative = source.relative_to(rootdir)
        if relative.parts[0] == "src":
            relative = relative.relative_to("src")
        if len(relative.parts) > 1:
            top = source.parents[len(relative.parts) - 2]
            if relative.parts[0] not in copied:
                shutil.copytree(
                    top, overlay / relative.parts[0], ignore=shutil.ignore_patterns("__pycache__")
                )
                copied.add(relative.parts[0])
        target = overlay / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(sources[normalize_origin(str(source))], encoding="utf-8", newline="\n")
    return overlay


def _bootstrap() -> str:
    return '''"""Physical mutation overlay; inherited by native Python subprocesses."""
import os
import json
import sys
from pathlib import Path

config = json.loads(Path(__file__).with_name("workspace.json").read_text(encoding="utf-8"))
if config["workspace"] is not None:
    original, workspace = Path(config["original"]), Path(config["workspace"])
    for index, argument in enumerate(sys.argv[1:], 1):
        path, separator, node = argument.partition("::")
        if Path(path).is_absolute() and Path(path).is_relative_to(original):
            sys.argv[index] = str(workspace / Path(path).relative_to(original)) + separator + node
    os.chdir(workspace)
sys.path[:0] = config["roots"]
os.environ["PYTHONPATH"] = os.pathsep.join(
    [*config["roots"], *filter(None, [os.environ.get("PYTHONPATH", "")])]
)
if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main(sys.argv[1:]))
'''
