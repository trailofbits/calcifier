"""Run instrumented packages from real files, including in native child processes."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest_gremlins.plugin as gremlins

_WRITE_SOURCES = gremlins._write_instrumented_sources


def write_sources(instrumented_asts: dict[str, ast.Module], rootdir: Path) -> Path:
    """Keep upstream mutation metadata while replacing its process-local import hook."""
    directory: Path = _WRITE_SOURCES(instrumented_asts, rootdir).resolve()
    sources: dict[str, str] = json.loads((directory / "sources.json").read_text())
    if (rootdir / ".git").exists():
        workspace = _copy_checkout(rootdir, directory / "workspace")
        for original in instrumented_asts:
            name = gremlins._path_to_module_name(Path(original), rootdir)
            target = workspace / Path(original).relative_to(rootdir)
            target.write_text(sources[name], encoding="utf-8", newline="\n")
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
    return directory


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
        source = Path(original)
        name = gremlins._path_to_module_name(source, rootdir)
        is_package = source.name == "__init__.py"
        parts = name.removesuffix(".__init__").split(".")
        if len(parts) > 1 or is_package:
            top = source.parents[len(parts) - 2 + int(is_package)]
            if parts[0] not in copied:
                shutil.copytree(
                    top, overlay / parts[0], ignore=shutil.ignore_patterns("__pycache__")
                )
                copied.add(parts[0])
        target = overlay.joinpath(*parts)
        target = target / "__init__.py" if is_package else target.with_suffix(".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(sources[name], encoding="utf-8", newline="\n")
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
