"""Read JSON configuration with caller-owned schema and duplicate-key policy."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def load_json_config(
    path: str | Path,
    *,
    object_pairs_hook: Callable[[list[tuple[str, Any]]], dict[str, Any]] | None = None,
) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=object_pairs_hook)
