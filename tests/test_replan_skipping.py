"""Replanning happens when the answer can differ, not on every tick (D41).

Every tick still advances the clock and polls the backend, so what a run observes
-- and therefore the status document it produces -- is exactly what it was. What
drops is the number of CP-SAT solves: with `poll_interval=1` the runner used to
solve once per time unit, i.e. as many times as the makespan is long, even while
nothing had happened. It now keeps the plan from the last replan until an
operation finishes, a machine goes down, or a pending activity comes due.

The behavioural half of that claim is guarded by the rest of the suite (every
existing timing assertion must still hold); these tests pin the part that is
otherwise invisible -- how often the scheduler is called -- plus the one timing
consequence that a naive "replan when work comes due" rule would get wrong.
The scheduler is a required dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import RollingRunner  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")

# simple: source (2) -> transport (1) -> target (2); makespan 5.


def _timeline(status) -> list[tuple]:
    """Each activity as (what ran, start, end) -- the whole observable schedule."""
    return [(a.get("process") or a["kind"], a["start"], a["end"]) for a in status["activities"]]


def _slow_env(tmp_path, factor: int = 10) -> str:
    """The `simple` environment with every duration multiplied, so the makespan is
    long relative to the number of activities -- the shape that made per-tick
    replanning expensive (a real protocol has minutes-long steps and a
    seconds-granularity clock)."""
    env = yaml.safe_load(Path(SIMPLE_ENV).read_text(encoding="utf-8"))
    for process in env["processes"].values():
        for mode in process["modes"]:
            mode["duration"] *= factor
    for transport in env.get("transports", []):
        transport["duration"] *= factor
    path = tmp_path / "slow.env.yaml"
    path.write_text(yaml.safe_dump(env, sort_keys=False), encoding="utf-8")
    return str(path)


def test_a_long_makespan_replans_per_event_not_per_tick(tmp_path):
    runner = RollingRunner(SIMPLE_WF, _slow_env(tmp_path), random_seed=0, poll_interval=1)
    status = runner.run()

    # Unchanged: the clock is still polled every unit, and the plan is the same.
    assert status["now"] == 50
    assert runner.ticks == 51
    assert _timeline(status) == [
        ("source", 0, 20),
        ("transport", 20, 30),
        ("target", 30, 50),
    ]
    # Four solves for three activities: the initial plan, then one per completion
    # observed (the last of which also confirms there is nothing left to do).
    assert runner.replans == 4


def test_the_run_is_identical_to_event_boundary_mode(tmp_path):
    """A poll interval of 1 lands exactly on the integer event times of this
    environment, so with no duration variance the two poll modes must produce the
    same activities -- they now also take a similar number of solves to do it."""
    slow_env = _slow_env(tmp_path)
    polled = RollingRunner(SIMPLE_WF, slow_env, random_seed=0, poll_interval=1)
    exact = RollingRunner(SIMPLE_WF, slow_env, random_seed=0, poll_interval=None)

    polled_status, exact_status = polled.run(), exact.run()
    assert polled_status["activities"] == exact_status["activities"]
    assert polled.replans == exact.replans


def test_an_early_completion_is_replanned_at_once():
    """The timing consequence a due-time-only rule would get wrong.

    `source` finishes in 1 instead of its planned 2. The poll at t=1 sees it, and
    the replan that follows moves the rest of the run up: transport 1-2, target 2-4,
    makespan 4. A runner that only replanned when work came due would hold the
    original plan until t=2 and finish at 5 instead.
    """

    def source_finishes_early(activity, planned):
        return 1 if activity.get("process") == "source" else planned

    runner = RollingRunner(
        SIMPLE_WF,
        SIMPLE_ENV,
        random_seed=0,
        poll_interval=1,
        running_task_margin=1,
        duration_model=source_finishes_early,
    )
    status = runner.run()

    assert status["now"] == 4
    assert _timeline(status) == [
        ("source", 0, 1),
        ("transport", 1, 2),
        ("target", 2, 4),
    ]


def test_needs_replan_asks_only_about_what_the_scheduler_reads():
    """The decision itself, without running anything.

    White-box because the truth table is what the saving rests on, and a run only
    shows its consequences. The gated-activity arm is not set up here -- it needs a
    composite whose `requires` is unchecked -- and is covered by the gate tests
    (`test_composite_requires_gate.py`), which pin the deferral behaviour itself.
    """
    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0, poll_interval=1)
    pending = [{"kind": "processing", "node": ["SampleTarget"], "start": 10, "end": 12}]

    # No plan yet: the first tick always replans.
    assert runner._needs_replan(set()) is True

    # A plan exists and nothing has happened: the same question would get the same
    # answer, so it is not asked again.
    runner._observed_change = False
    runner._last_pending = pending
    assert runner._needs_replan(set()) is False

    # An operation was observed to finish or fail -> the history has changed.
    runner._observed_change = True
    assert runner._needs_replan(set()) is True
    runner._observed_change = False

    # A machine went down -> the environment being scheduled against has changed.
    assert runner._needs_replan({"station_1"}) is True

    # `now` reached the moment the pending activity is due -> it must be dispatched,
    # and dispatch always follows a fresh plan.
    runner.now = 10
    assert runner._needs_replan(set()) is True

    # Nothing left to dispatch and nothing running: ask once more, so the loop's
    # completion test reads a current plan rather than a stale one.
    runner._last_pending = []
    assert runner._needs_replan(set()) is True
