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

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        assert seconds > 0  # advance only sleeps a positive remaining
        self.sleeps.append(seconds)
        self.t += seconds

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
