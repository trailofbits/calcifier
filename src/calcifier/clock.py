"""Suspend-aware elapsed-time measurement.

``time.monotonic()`` is intentionally immune to wall-clock changes, but on
some platforms it stops while the machine is suspended. Process deadlines
need elapsed real time instead: prefer a boot/continuous clock and otherwise
combine monotonic and wall time defensively.
"""

from __future__ import annotations

import ctypes
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass


def _clock_boottime() -> Callable[[], float] | None:
    clock_id = getattr(time, "CLOCK_BOOTTIME", None)
    if clock_id is None or not hasattr(time, "clock_gettime"):
        return None
    return lambda: time.clock_gettime(clock_id)


class _MachTimebaseInfo(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


def _mach_continuous_clock() -> Callable[[], float] | None:
    if sys.platform != "darwin":
        return None
    try:
        library = ctypes.CDLL("/usr/lib/libSystem.dylib")
        continuous = library.mach_continuous_time
        timebase = library.mach_timebase_info
        continuous.argtypes = []
        continuous.restype = ctypes.c_uint64
        timebase.argtypes = [ctypes.POINTER(_MachTimebaseInfo)]
        timebase.restype = ctypes.c_int
        info = _MachTimebaseInfo()
        if timebase(ctypes.byref(info)) != 0 or info.denom == 0:
            return None
        seconds_per_tick = info.numer / info.denom / 1_000_000_000
    except (AttributeError, OSError):
        return None
    return lambda: float(continuous()) * seconds_per_tick


_CONTINUOUS_CLOCK = _clock_boottime() or _mach_continuous_clock()


@dataclass(frozen=True)
class SuspendAwareTimer:
    """Measure elapsed real time without trusting a single clock source."""

    monotonic_started: float
    wall_started: float
    continuous_started: float | None

    @classmethod
    def start(cls) -> SuspendAwareTimer:
        continuous = _CONTINUOUS_CLOCK
        return cls(
            monotonic_started=time.monotonic(),
            wall_started=time.time(),
            continuous_started=continuous() if continuous is not None else None,
        )

    def elapsed(self) -> float:
        """Return non-negative elapsed time, including machine suspension."""
        monotonic_elapsed = max(0.0, time.monotonic() - self.monotonic_started)
        continuous = _CONTINUOUS_CLOCK
        if continuous is not None and self.continuous_started is not None:
            return max(
                monotonic_elapsed,
                max(0.0, continuous() - self.continuous_started),
            )
        wall_elapsed = max(0.0, time.time() - self.wall_started)
        return max(monotonic_elapsed, wall_elapsed)
