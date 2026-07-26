"""Tests for `RealTimeSimulator` -- the wall-clock-paced backend.

`RealTimeSimulator` inherits every physical/value behaviour of `Simulator` but
paces `advance` to a wall clock: it sleeps out the real time a step represents,
then settles the virtual clock to the tick real time has reached. These tests
drive the whole path on an *injected fake clock* (so no real seconds are spent):
sleeping advances the fake clock, so a full run reproduces the deterministic
virtual makespan while actually exercising the sleep/monotonic path, and the
overshoot rule (adopt the tick real time reached, never less than `until`) is
pinned directly.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ofplang.run.simulator import RealTimeSimulator, Simulator, realtime_backend_factory

FIXTURES = Path(__file__).parent / "fixtures"
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")
SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")


class FakeClock:
    """A monotonic clock where `sleep(d)` is the only thing that makes time pass
    (plus explicit `jump`s, standing in for real time consumed elsewhere -- e.g. a
    slow solve between advances). No real seconds are spent."""

    def __init__(self, overshoot: float = 0.0) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []
        # Real `time.sleep(d)` returns after *at least* d; `overshoot` models that
        # granularity by advancing the clock by `d + overshoot` on every sleep while
        # still recording the requested `d`.
        self.overshoot = overshoot

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        assert seconds > 0  # advance only sleeps a positive remaining
        self.sleeps.append(seconds)
        self.t += seconds + self.overshoot

    def jump(self, seconds: float) -> None:
        self.t += seconds


# -- unit: advance pacing / overshoot ------------------------------------------


def test_advance_sleeps_out_the_step_and_reaches_until():
    clock = FakeClock()
    sim = RealTimeSimulator(
        SIMPLE_ENV, seconds_per_tick=1.0, monotonic=clock.monotonic, sleep=clock.sleep
    )
    # First advance pins the epoch at the current clock; reaching tick 3 sleeps 3s.
    assert sim.advance(3) == 3
    assert sim.now == 3
    assert clock.sleeps == [3.0]
    # A further step sleeps only the delta.
    assert sim.advance(5) == 5
    assert clock.sleeps == [3.0, 2.0]


def test_advance_overshoots_when_real_time_already_passed():
    clock = FakeClock()
    sim = RealTimeSimulator(
        SIMPLE_ENV, seconds_per_tick=1.0, monotonic=clock.monotonic, sleep=clock.sleep
    )
    assert sim.advance(2) == 2  # sleeps 2s -> t == 2
    # Real time is consumed outside advance (a slow replan): the clock jumps past the
    # next target, so advance must not sleep and must adopt the reached tick.
    clock.jump(5.0)  # t == 7
    assert sim.advance(3) == 7  # target 3 already overshot; adopt tick 7
    assert sim.now == 7
    assert clock.sleeps == [2.0]  # no sleep on the overshooting step


def test_advance_adopts_sleep_granularity_overshoot():
    # A real sleep returns *late*: it overshoots the requested remaining. The reached
    # tick must follow the wall clock past `until` (never lag it), and the clock must
    # never run backward -- the same overshoot rule as a slow solve, but arising from
    # sleep granularity itself.
    clock = FakeClock(overshoot=1.5)  # every sleep lands 1.5 ticks late
    sim = RealTimeSimulator(
        SIMPLE_ENV, seconds_per_tick=1.0, monotonic=clock.monotonic, sleep=clock.sleep
    )
    # advance(2): sleep 2s but land at t == 3.5 -> adopt reached tick 3 (> until).
    reached = sim.advance(2)
    assert reached == 3
    assert reached >= 2  # never lags the target
    assert sim.now == 3
    # A follow-on step is already past its target from the prior overshoot, so it does
    # not sleep and simply re-adopts the current tick (monotonic, never backward).
    assert sim.advance(3) == 3
    assert clock.sleeps == [2.0]  # only the first step slept


def test_speed_scales_wall_time():
    clock = FakeClock()
    sim = RealTimeSimulator(
        SIMPLE_ENV, seconds_per_tick=1.0, speed=2.0,
        monotonic=clock.monotonic, sleep=clock.sleep,
    )
    # speed 2 -> half a real second per tick; reaching tick 4 sleeps 2s.
    assert sim.advance(4) == 4
    assert clock.sleeps == [2.0]


def test_non_positive_pace_rejected():
    with pytest.raises(ValueError):
        RealTimeSimulator(SIMPLE_ENV, seconds_per_tick=0.0)
    with pytest.raises(ValueError):
        RealTimeSimulator(SIMPLE_ENV, speed=0.0)


# -- integration: drive a full run through the runner --------------------------


def test_realtime_backend_drives_run_and_matches_default():
    pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")
    from ofplang.run.runner import RollingRunner

    default_status = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0).run()

    clock = FakeClock()
    factory = realtime_backend_factory(
        seconds_per_tick=1.0, monotonic=clock.monotonic, sleep=clock.sleep
    )
    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, backend_factory=factory, random_seed=0)
    status = runner.run()

    # Same deterministic outcome as the instant simulator: it is still a Simulator,
    # only paced.
    assert isinstance(runner.sim, RealTimeSimulator)
    assert status["now"] == default_status["now"]
    assert (
        {a["status"] for a in status["activities"]}
        == {a["status"] for a in default_status["activities"]}
    )
    # The wall-clock path really ran: total real time slept equals the makespan
    # (one tick == one second here), never overshooting on a hardware-free run where
    # sleep is the only source of elapsed time.
    assert sum(clock.sleeps) == pytest.approx(float(status["now"]))


def test_realtime_simulator_is_a_simulator():
    # It is a drop-in Simulator subclass, so anything typed against Simulator accepts
    # it and its non-timing behaviour is inherited unchanged.
    sim = RealTimeSimulator(SIMPLE_ENV)
    assert isinstance(sim, Simulator)


def test_default_wiring_really_sleeps():
    """Smoke test the *real* clock path: with no injected clock, `advance` drives the
    default `time.monotonic` / `time.sleep`. This is the one test that spends real
    time -- kept tiny (a few ms) with a very small tick. It asserts only a *lower*
    bound on elapsed (real time really passed) and never an upper bound, which would
    be flaky under CI/timer-granularity jitter."""
    spt = 0.005  # 5 ms per tick
    sim = RealTimeSimulator(SIMPLE_ENV, seconds_per_tick=spt)
    start = time.monotonic()
    reached = sim.advance(3)  # ~15 ms of real sleeping
    elapsed = time.monotonic() - start

    assert reached >= 3  # reached the target (or beyond, on overshoot)
    assert sim.now == reached
    # It genuinely waited: a generous lower bound (half the nominal 3 ticks) so a slow,
    # coarse-timer machine that oversleeps still passes -- only under-sleeping fails,
    # which `time.sleep` does not do.
    assert elapsed >= 3 * spt * 0.5
