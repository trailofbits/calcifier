"""Durable, atomic replacement of one file."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write(
    path: Path, data: bytes, *, mode: int | None = None, overwrite: bool = True
) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=path.parent) as handle:
            temporary = Path(handle.name)
            if mode is not None:
                os.chmod(temporary, mode)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
            temporary = None
        else:
            # Publish without replacing even a concurrently created destination.
            # Both paths are on the same filesystem, including on Windows.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
