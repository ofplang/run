"""A wall-clock-paced simulator: the `Simulator` with real time added back in.

`VirtualTimeSimulator.advance(until)` settles its virtual clock to `until` *instantly*
-- it is deterministic and as fast as the CPU allows, which is exactly what a test or
a scheduler-in-the-loop replan wants. A real backend, by contrast, cannot skip time:
between now and `until` real seconds must actually pass before the machine has made
its progress. `RealTimeSimulator` bridges the two by keeping all of the simulator's
virtual completion logic but *pacing* `advance` to a wall clock -- it sleeps out the
real time a step represents, then delegates to the shared `_settle` engine to apply
whatever completed.

That makes it the general, hardware-free stand-in for a real backend: it exercises
the runner's wall-clock path end to end (the runner adopts `advance`'s return as
`now`, so solve latency and sleep overshoot flow through exactly as they would with
real hardware) without needing any device. It is also what a demo or an operator
rehearsal runs on -- a plan that unfolds in real (or scaled) time.

Time model (the `Backend` contract, see `..backend`):

* One environment time tick (§4.1) maps to `seconds_per_tick` real seconds, divided
  by `speed` (so `speed=2.0` runs twice as fast, `speed=60.0` turns tick-minutes into
  real seconds). The product must be > 0.
* A wall-clock *epoch* is pinned on the first `advance`, mapping the clock's then-value
  to `monotonic()` now. Thereafter tick `k` is due at `epoch + k * eff_seconds`.
* `advance(until)` sleeps until the wall clock reaches `until`'s due time, then adopts
  the tick the wall clock has *actually* reached -- which may exceed `until` if a slow
  solve already consumed that time -- and settles the virtual clock (applying every
  completion up to it) there. The reached tick is returned, never less than `until`.

`monotonic` and `sleep` are injectable so a test can drive the whole path on a fake
clock without spending real seconds; production uses `time.monotonic` / `time.sleep`.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from .core import Simulator


class RealTimeSimulator(Simulator):
    """A `Simulator` whose `advance` is paced to a (real or fake) wall clock.

    Every physical and value behaviour is inherited unchanged; only the *timing* of
    `advance` differs -- it blocks until real time has caught up before settling the
    virtual clock. Suitable as an injected `Backend` (see `realtime_backend_factory`)
    to drive the runner on real time without any hardware.
    """

    def __init__(
        self,
        environment,
        *,
        device_model=None,
        seconds_per_tick: float = 1.0,
        speed: float = 1.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        super().__init__(environment, device_model=device_model)
        if not (seconds_per_tick > 0 and speed > 0):
            raise ValueError(
                f"seconds_per_tick and speed must both be > 0, got "
                f"{seconds_per_tick} and {speed}"
            )
        self._seconds_per_tick = seconds_per_tick / speed
        self._monotonic = monotonic
        self._sleep = sleep
        # Wall-clock instant that the virtual clock's current value maps to; pinned
        # lazily on the first advance so construction-to-first-advance idle time does
        # not count against the schedule.
        self._epoch: float | None = None

    def advance(self, until: int) -> int:
        """Block until the wall clock reaches `until`'s due time, then settle the
        virtual clock to the tick real time has actually reached (>= `until`) and
        apply every completion up to it. Returns that reached tick.

        Overshoot is intended, not error: if the caller (the runner, between plans)
        already spent more real time than one step, the extra ticks are adopted and
        their completions applied -- the same way a real backend would have kept
        running while the scheduler thought.
        """
        if self._epoch is None:
            # Pin the epoch so the *current* clock value is "now" in wall time.
            self._epoch = self._monotonic() - self.now * self._seconds_per_tick

        due = self._epoch + until * self._seconds_per_tick
        remaining = due - self._monotonic()
        if remaining > 0:
            self._sleep(remaining)

        elapsed = self._monotonic() - self._epoch
        reached = int(elapsed / self._seconds_per_tick)
        # Never settle before `until` (sleep granularity / clock jitter) and never go
        # backward past the base clock's monotonic invariant.
        reached = max(reached, until, self.now)
        return self._settle(reached)


def realtime_backend_factory(
    *,
    seconds_per_tick: float = 1.0,
    speed: float = 1.0,
    device_model=None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[dict], RealTimeSimulator]:
    """Build a `backend_factory(environment) -> RealTimeSimulator` for the runner.

    Pass the result as `RollingRunner(..., backend_factory=realtime_backend_factory(...))`
    to run a plan paced to a wall clock instead of instant virtual time. See
    `RealTimeSimulator` for the timing parameters; `monotonic`/`sleep` are injectable
    for tests.
    """

    def factory(environment: dict) -> RealTimeSimulator:
        return RealTimeSimulator(
            environment,
            device_model=device_model,
            seconds_per_tick=seconds_per_tick,
            speed=speed,
            monotonic=monotonic,
            sleep=sleep,
        )

    return factory
