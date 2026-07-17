"""Tests for duration variance (milestone 2b-2b-2).

Variance is injected externally through a duration model the runner consults when
dispatching: `fn(activity, planned_duration) -> actual_duration` (D13/D23). The
backend runs the actual duration; the runner reports the planned expected end
while an op runs and the poll-observed time once it completes -- it never assumes
it knows the actual finish. A deterministic model makes every outcome exact.

Variance is only coherent under fixed-interval polling and needs a positive
running-task margin (so a successor of an overrunning op is not dispatched onto a
still-busy resource); the runner validates both. The scheduler is a required
dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import RollingRunner, RunnerError  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")

# simple: source (2) -> transport (1) -> target (2).


def _transport_overruns_to(actual):
    """A deterministic model: the transport takes `actual`, everything else its
    planned duration."""

    def model(activity, planned):
        return actual if activity["kind"] == "transport" else planned

    return model


def test_overrun_defers_successor_and_completes():
    # The transport overruns (planned 1 -> actual 5, true finish at t=7). With
    # poll interval 2 and margin 2, target is not dispatched until the transport is
    # observed complete (at the t=8 poll), so nothing lands on a busy resource.
    runner = RollingRunner(
        SIMPLE_WF, SIMPLE_ENV, random_seed=0, poll_interval=2, running_task_margin=2,
        duration_model=_transport_overruns_to(5),
    )
    status = runner.run()

    assert all(a["status"] == "completed" for a in status["activities"])
    target = next(a for a in status["activities"] if a.get("process") == "target")
    assert target["start"] == 8  # deferred until the transport was observed done
    assert status["now"] == 10


def test_reported_end_is_poll_estimate_of_true_finish():
    # The transport's true finish (2 + 5 = 7, from the backend history) is only
    # seen at the t=8 poll, so the runner reports 8 -- an upper bound.
    runner = RollingRunner(
        SIMPLE_WF, SIMPLE_ENV, random_seed=0, poll_interval=2, running_task_margin=2,
        duration_model=_transport_overruns_to(5),
    )
    status = runner.run()
    transport = next(a for a in status["activities"] if a["kind"] == "transport")
    true_end = next(e.time for e in runner.sim._history() if e.kind == "transport")
    assert true_end == 7  # actual finish
    assert transport["end"] == 8  # poll-based estimate
    assert transport["end"] >= true_end


def test_variance_is_deterministic():
    # A deterministic model gives a reproducible run.
    def run():
        return RollingRunner(
            SIMPLE_WF, SIMPLE_ENV, random_seed=0, poll_interval=2, running_task_margin=2,
            duration_model=_transport_overruns_to(4),
        ).run()

    assert run()["now"] == run()["now"]


def test_variance_requires_poll_interval():
    # Variance with event-boundary advance (poll_interval=None) is incoherent -- an
    # off-plan finish can't be observed -- so it is rejected.
    with pytest.raises(RunnerError):
        RollingRunner(
            SIMPLE_WF, SIMPLE_ENV, poll_interval=None, running_task_margin=2,
            duration_model=_transport_overruns_to(5),
        )


def test_variance_requires_positive_margin():
    with pytest.raises(RunnerError):
        RollingRunner(
            SIMPLE_WF, SIMPLE_ENV, poll_interval=2, running_task_margin=0,
            duration_model=_transport_overruns_to(5),
        )


def test_no_model_is_unaffected():
    # Without a model the run is exactly the no-variance fixed-interval case.
    status = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0, poll_interval=2).run()
    assert status["now"] == 6


REROUTE_ENV = str(FIXTURES / "reroute.env.yaml")


def test_variance_composes_with_reroute():
    # Variance and re-routing compose: station_1 goes down (target re-routes to
    # station_2) while every transport overruns (planned + 3). The run still
    # completes on the re-routed device.
    def model(activity, planned):
        return planned + 3 if activity["kind"] == "transport" else planned

    runner = RollingRunner(
        SIMPLE_WF, REROUTE_ENV, random_seed=0, poll_interval=2, running_task_margin=2, duration_model=model
    )
    runner.sim.schedule_device_down(3, "station_1")
    status = runner.run()

    assert all(a["status"] == "completed" for a in status["activities"])
    assert status["now"] == 16
    target = next(a for a in status["activities"] if a.get("process") == "target")
    assert target["input_spots"]["target_in"] == "station_2.core"
