"""Tests for suspend-aware elapsed-time measurement."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from calcifier import clock
from calcifier.clock import SuspendAwareTimer


def test_boot_clock_uses_the_available_suspend_aware_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    native_clock = Mock(return_value=123.5)
    monkeypatch.setattr(
        clock, "time", SimpleNamespace(CLOCK_BOOTTIME=7, clock_gettime=native_clock)
    )

    continuous = clock._clock_boottime()

    assert continuous is not None
    assert continuous() == 123.5
    native_clock.assert_called_once_with(7)


@pytest.mark.parametrize("attributes", [{}, {"CLOCK_BOOTTIME": 7}, {"clock_gettime": Mock()}])
def test_boot_clock_falls_back_when_the_platform_lacks_the_api(monkeypatch, attributes) -> None:
    monkeypatch.setattr(clock, "time", SimpleNamespace(**attributes))
    assert clock._clock_boottime() is None


@pytest.mark.parametrize("denominator", [0, 1])
def test_mach_clock_handles_zero_and_unit_timebase_denominators(monkeypatch, denominator) -> None:
    def fill_timebase(info):
        info._obj.numer = 1
        info._obj.denom = denominator
        return 0

    native_clock = Mock(return_value=2_000_000_000)
    library = SimpleNamespace(
        mach_continuous_time=native_clock,
        mach_timebase_info=Mock(side_effect=fill_timebase),
    )
    monkeypatch.setattr(clock, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(clock.ctypes, "CDLL", lambda _path: library)

    continuous = clock._mach_continuous_clock()

    if denominator == 0:
        assert continuous is None
        native_clock.assert_not_called()
    else:
        assert continuous is not None
        assert continuous() == pytest.approx(2.0)


def test_mach_clock_converts_non_unit_timebase_ticks_to_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fill_timebase(info):
        info._obj.numer = 125
        info._obj.denom = 3
        return 0

    library = SimpleNamespace(
        mach_continuous_time=Mock(side_effect=[24_000_000, 120_000_000]),
        mach_timebase_info=Mock(side_effect=fill_timebase),
    )
    monkeypatch.setattr(clock, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(clock.ctypes, "CDLL", lambda _path: library)

    continuous = clock._mach_continuous_clock()

    assert continuous is not None
    assert continuous() == pytest.approx(1.0)
    assert continuous() == pytest.approx(5.0)


def test_timer_falls_back_to_wall_time_during_suspend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("calcifier.clock._CONTINUOUS_CLOCK", None)
    monkeypatch.setattr("calcifier.clock.time.monotonic", lambda: 100.0)
    monkeypatch.setattr("calcifier.clock.time.time", lambda: 460.0)
    timer = SuspendAwareTimer(
        monotonic_started=100.0,
        wall_started=40.0,
        continuous_started=None,
    )

    assert timer.elapsed() == 420


def test_timer_fallback_resists_wall_clock_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("calcifier.clock._CONTINUOUS_CLOCK", None)
    monkeypatch.setattr("calcifier.clock.time.monotonic", lambda: 125.0)
    monkeypatch.setattr("calcifier.clock.time.time", lambda: 900.0)
    timer = SuspendAwareTimer(
        monotonic_started=100.0,
        wall_started=1_000.0,
        continuous_started=None,
    )

    assert timer.elapsed() == 25


def test_timer_prefers_larger_continuous_elapsed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("calcifier.clock._CONTINUOUS_CLOCK", lambda: 250.0)
    monkeypatch.setattr("calcifier.clock.time.monotonic", lambda: 105.0)
    monkeypatch.setattr("calcifier.clock.time.time", lambda: 9_999.0)
    timer = SuspendAwareTimer(
        monotonic_started=100.0,
        wall_started=1_000.0,
        continuous_started=200.0,
    )

    assert timer.elapsed() == 50


def test_timer_falls_back_if_continuous_clock_becomes_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("calcifier.clock._CONTINUOUS_CLOCK", None)
    monkeypatch.setattr("calcifier.clock.time.monotonic", lambda: 101.0)
    monkeypatch.setattr("calcifier.clock.time.time", lambda: 1_020.0)
    timer = SuspendAwareTimer(
        monotonic_started=100.0,
        wall_started=1_000.0,
        continuous_started=500.0,
    )

    assert timer.elapsed() == 20


def test_timer_clamps_rolled_back_clocks_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("calcifier.clock._CONTINUOUS_CLOCK", None)
    monkeypatch.setattr("calcifier.clock.time.monotonic", lambda: 90.0)
    monkeypatch.setattr("calcifier.clock.time.time", lambda: 900.0)
    timer = SuspendAwareTimer(
        monotonic_started=100.0,
        wall_started=1_000.0,
        continuous_started=None,
    )

    assert timer.elapsed() == 0
