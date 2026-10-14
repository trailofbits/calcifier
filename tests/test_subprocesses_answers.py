"""Bounded-subprocess known answers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from calcifier.subprocesses import ProcessOutputLimitError, run_process

# -- bounded subprocesses -------------------------------------------------------


# The child arms its SIGTERM handler before it overflows stdout, so termination
# can never race interpreter startup. The handler's in-process delay fits well
# inside the two-second grace period but not inside an immediate SIGKILL.
_GRACEFUL_CHILD = """\
import signal, sys, time
from pathlib import Path


def cleanup(signum, frame):
    time.sleep(0.3)
    Path(sys.argv[1]).write_text("cleaned")
    sys.exit(0)


signal.signal(signal.SIGTERM, cleanup)
sys.stdout.buffer.write(b"x" * 2048)
sys.stdout.flush()
while True:
    time.sleep(0.05)
"""


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups and SIGTERM handlers")
def test_terminated_process_group_receives_its_termination_grace_period(tmp_path: Path) -> None:
    marker = tmp_path / "cleaned"

    with pytest.raises(ProcessOutputLimitError) as raised:
        run_process(
            [sys.executable, "-c", _GRACEFUL_CHILD, str(marker)],
            timeout_seconds=60,
            max_output_bytes=1024,
        )

    assert str(raised.value) == "process stdout exceeded 1024 bytes"
    assert raised.value.stdout == b"x" * 1024
    assert marker.read_text() == "cleaned"
