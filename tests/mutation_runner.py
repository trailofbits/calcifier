"""Real pytest mutation runner, including controlled import-failure probes."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

_FLAGS = ["-x", "-q", "--tb=short", "-p", "no:gremlins", "-o", "addopts="]
# Windows CI starts several nested interpreters concurrently. This bounds probe
# startup and execution, not the behavior of Calcifier's subprocess deadlines.
_PROBE_TIMEOUT_SECONDS = 60
_PROBE = """import importlib.util
import json
import os
import sys

def test_selected_modules_import():
    for index, path in enumerate(json.loads(os.environ["CALCIFIER_MUTATION_PROBE_FILES"])):
        name = f"_mutation_import_probe_{index}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
"""


def _unique_mutant_module(sources: dict[str, str], active: str) -> str | None:
    matches = [
        name
        for name, source in sources.items()
        if any(
            isinstance(node, ast.Constant) and node.value == active
            for node in ast.walk(ast.parse(source))
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _probe_result(report: Path, *, failed: bool) -> bool:
    try:
        # Local XML written by the isolated pytest probe, not external input.
        root = ET.fromstring(report.read_text(encoding="utf-8"))  # noqa: S314
        cases = list(root.iter("testcase"))
        return (
            len(cases) == 1
            and cases[0].get("name") == "test_selected_modules_import"
            and not list(root.iter("error"))
            and len(list(root.iter("failure"))) == int(failed)
            and not list(root.iter("skipped"))
        )
    except (OSError, ET.ParseError):
        return False


def _selectors_collect(bootstrap: Path, selectors: list[str]) -> bool:
    try:
        collection = subprocess.run(
            [sys.executable, str(bootstrap), *_FLAGS, "--collect-only", *selectors],
            env={**os.environ, "ACTIVE_GREMLIN": "", "PYTHONIOENCODING": "utf-8"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"Mutation selector control could not finish: {exc}", file=sys.stderr, flush=True)
        return False
    print(f"Mutation selector control:\n{collection.stdout}{collection.stderr}", flush=True)
    return collection.returncode == 0 and "::" in collection.stdout


def _import_failure_is_tested(bootstrap: Path, selectors: list[str]) -> bool:
    active = os.environ.get("ACTIVE_GREMLIN", "")
    try:
        sources = json.loads(
            Path(os.environ["PYTEST_GREMLINS_SOURCES_FILE"]).read_text(encoding="utf-8")
        )
        if not isinstance(sources, dict) or any(not isinstance(v, str) for v in sources.values()):
            return False
        if not active or _unique_mutant_module(sources, active) is None:
            return False
        files = sorted({str(Path(selector.split("::", 1)[0]).resolve()) for selector in selectors})
        if not files or any(not Path(path).is_file() for path in files):
            return False
    except (KeyError, OSError, ValueError, TypeError, SyntaxError):
        return False
    if not _selectors_collect(bootstrap, selectors):
        return False
    return _run_import_probes(bootstrap, files, active)


def _run_import_probes(bootstrap: Path, files: list[str], active: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="calcifier-mutation-import-") as directory:
        root = Path(directory)
        probe = root / "test_import_probe.py"
        probe.write_text(_PROBE, encoding="utf-8", newline="\n")
        # Bootstrap changes cwd to the instrumented checkout. Keep this separate
        # one-test probe rooted here rather than discovering checkout/ancestor
        # pytest settings (or crossing Windows drives during collection).
        config = root / "pytest.ini"
        config.write_bytes(b"[pytest]\n")
        env = {
            **os.environ,
            "CALCIFIER_MUTATION_PROBE_FILES": json.dumps(files),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_ADDOPTS": "",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": os.pathsep.join([str(Path.cwd()), os.environ.get("PYTHONPATH", "")]),
        }
        for label, mutant, expected_code in [("control", "", 0), ("active", active, 1)]:
            report = root / f"{label}.xml"
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(bootstrap),
                        *_FLAGS,
                        "--noconftest",
                        "-c",
                        str(config),
                        f"--confcutdir={root}",
                        f"--junitxml={report}",
                        str(probe),
                    ],
                    cwd=root,
                    env={**env, "ACTIVE_GREMLIN": mutant},
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=_PROBE_TIMEOUT_SECONDS,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                print(
                    f"Mutation import probe ({label}) could not finish: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                return False
            print(f"Mutation import probe ({label}):\n{result.stdout}{result.stderr}", flush=True)
            if result.returncode != expected_code or not _probe_result(report, failed=bool(mutant)):
                return False
    return True


def _control_passes(bootstrap: Path, selectors: list[str]) -> bool:
    """A failing mutant counts only when the same instrumented selection passes unchanged."""
    key = hashlib.sha256(json.dumps(selectors).encode()).hexdigest()
    marker = bootstrap.parent / f"control-{key}.passed"
    if marker.is_file():
        return True
    try:
        control = subprocess.run(
            [sys.executable, str(bootstrap), *_FLAGS, *selectors],
            env={**os.environ, "ACTIVE_GREMLIN": "", "PYTHONIOENCODING": "utf-8"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        traceback.print_exc()
        return False
    print(f"Mutation selection control:\n{control.stdout}{control.stderr}", flush=True)
    if control.returncode != 0:
        # The parallel runner retains only the first 2,000 stderr characters.
        # Keep the actual control failure visible without changing its status.
        print(
            f"Mutation selection control failed (exit {control.returncode}).\n"
            f"stdout tail:\n{control.stdout[-1200:]}\n"
            f"stderr tail:\n{control.stderr[-400:]}",
            file=sys.stderr,
            flush=True,
        )
        return False
    # The directory is unique to this immutable instrumentation run. Duplicate
    # parallel controls are harmless; only successful completed controls are cached.
    marker.touch()
    return True


def main() -> int:
    try:
        bootstrap = Path(sys.argv[1]).resolve()
        selectors = sys.argv[2:]
        sys.argv = [str(bootstrap), *_FLAGS, *selectors]
        runpy.run_path(str(bootstrap), run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 3
        # Bootstrap may map absolute selectors into its disposable checkout.
        # Controls and import probes must use the paths that actually ran.
        selectors = sys.argv[1 + len(_FLAGS) :]
    except Exception:
        traceback.print_exc()
        return 3
    else:
        code = 0
    try:
        if code == 1 and os.environ.get("ACTIVE_GREMLIN"):
            return 1 if _control_passes(bootstrap, selectors) else 3
        # SystemExit during collection is pytest's INTERNAL_ERROR (3), unlike
        # ordinary import exceptions (2). Neither is a kill unless the same
        # isolated import probe fails as a test and its unchanged control passes.
        if code in {2, 3, 4} and _import_failure_is_tested(bootstrap, selectors):
            return 1
    except Exception:
        traceback.print_exc()
        return 3
    return code


if __name__ == "__main__":
    raise SystemExit(main())
