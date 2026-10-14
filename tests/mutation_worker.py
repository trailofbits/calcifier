"""Keep the active mutation visible even if a CI worker shuts down mid-test."""

from __future__ import annotations

import os
import tempfile

from pytest_gremlins.parallel.pool import WorkerResult, _run_gremlin_test


def run_logged_test(
    gremlin_id: str,
    test_command: list[str],
    rootdir: str,
    env_vars: dict[str, str],
    timeout: int,
) -> WorkerResult:
    # This is the pool's picklable worker entry point, before the test subprocess
    # captures stdout/stderr. Flush starts as well as results so interrupted runs
    # identify their outstanding mutations without needing the final report.
    prefix = f"Mutation worker {os.getpid()}: {gremlin_id}"
    print(f"{prefix} started", flush=True)
    try:
        # Pytest retains session directories, including after failed selections.
        # Keep each attempt's fixtures, probes and native child scratch files in
        # one owned directory, and remove them when its result is collected.
        with tempfile.TemporaryDirectory(prefix="cmw-", ignore_cleanup_errors=True) as scratch:
            attempt_env = {**env_vars, "TMPDIR": scratch, "TMP": scratch, "TEMP": scratch}
            result = _run_gremlin_test(gremlin_id, test_command, rootdir, attempt_env, timeout)
    except BaseException as exc:
        print(f"{prefix} raised {type(exc).__name__}: {exc}", flush=True)
        raise
    print(f"{prefix} finished {result.status.value}", flush=True)
    return result
