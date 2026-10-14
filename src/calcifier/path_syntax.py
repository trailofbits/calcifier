"""Cross-platform path syntax; callers enforce their own containment policy."""

from __future__ import annotations

from dataclasses import dataclass
from ntpath import isreserved
from pathlib import Path, PurePosixPath, PureWindowsPath


@dataclass(frozen=True)
class PathSyntax:
    native: Path
    posix: PurePosixPath
    windows: PureWindowsPath
    has_backslash: bool

    @property
    def has_windows_reserved_name(self) -> bool:
        """Identify device names, alternate streams, and other nonportable components."""
        return isreserved(str(self.windows))

    @classmethod
    def inspect(cls, raw: str | Path, *, normalize_backslashes: bool = False) -> PathSyntax:
        text = str(raw)
        return cls(
            native=Path(raw),
            posix=PurePosixPath(text.replace("\\", "/") if normalize_backslashes else text),
            windows=PureWindowsPath(text),
            has_backslash="\\" in text,
        )
