"""Tests for device-less Pure-Data processes (dev-notes design.md D12).

A Pure-Data-only process occupies no device and no spot; it exists only to take
time and to impose ordering through its Pure Data arcs. The runner must still
plan and execute it, and honour the precedence it participates in -- even though
nothing is transported to or from it. The fixture
(tests/fixtures/pure_data.*) chains

    sample -> Measure (Object) -> Analyze (device-less Pure Data, duration 0)
                     \\                         |
                      +----(object)----> Finish <-- score (Pure Data)

so Analyze must run after Measure (Pure Data in from its producer) and Finish
after Analyze (Pure Data out to its consumer).

`replay` is schedule-independent (it reproduces a frozen plan exactly), so it
asserts exact times. The rolling-horizon runner re-solves each tick, and a
zero-duration slack activity's exact placement is solver-chosen, so those tests
assert the robust invariants instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ofplang.run.runner import Runner, load_document

FIXTURES = Path(__file__).parent / "fixtures"
PD_WF = str(FIXTURES / "pure_data.workflow.yaml")
PD_ENV = str(FIXTURES / "pure_data.env.yaml")
PD_DOC = FIXTURES / "pure_data.document.yaml"
PD_PLAN = FIXTURES / "pure_data.plan.yaml"


def _by_node(status: dict, node: str) -> dict:
    """The single activity whose workflow node path is [node]."""
    (act,) = [a for a in status["activities"] if a.get("node") == [node]]
    return act


# -- replay (schedule-independent, exact plan reproduction) --------------------

def test_replay_reproduces_pure_data_plan():
    # Replaying the frozen plan reproduces every activity at its planned times,
    # including the instantaneous device-less Analyze.
    plan = load_document(PD_PLAN)
    status = Runner(plan, PD_ENV).run()
    assert all(a["status"] == "completed" for a in status["activities"])
    assert status["now"] == 11  # the plan makespan

    measure = _by_node(status, "Measure")
    analyze = _by_node(status, "Analyze")
    finish = _by_node(status, "Finish")

    # Analyze is a device-less, zero-duration processing activity: it occupies no
    # device and no spot, and reproduces instantaneously (end == start).
    assert analyze["kind"] == "processing" and analyze["mode"] == "mean_v1"
    assert analyze["start"] == analyze["end"] == 7
    assert "devices" not in analyze
    assert "input_spots" not in analyze and "output_spots" not in analyze

    # Both precedence directions hold: Analyze after its Object producer Measure,
    # and Finish after Analyze (a Pure Data dependency, no transport between them).
    assert analyze["start"] >= measure["end"] == 6
    assert finish["start"] >= analyze["end"]
    assert (finish["start"], finish["end"]) == (8, 11)


def test_replay_pure_data_leaves_nothing_resting():
    # Measure transforms the plate in place, it is transported to printer, and
    # Finish consumes it -- every spot is empty at the end.
    plan = load_document(PD_PLAN)
    runner = Runner(plan, PD_ENV)
    runner.run()
    assert runner.sim.spot_state() == {}


# -- rolling horizon (re-solves each tick) -------------------------------------

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import RollingRunner  # noqa: E402


def _interface():
    return load_document(PD_DOC)["interface"]


def _assert_pure_data_run(status: dict) -> None:
    """The invariants a pure-data run must satisfy regardless of poll mode or the
    solver's placement of the zero-duration Analyze (which has slack)."""
    assert all(a["status"] == "completed" for a in status["activities"])
    assert status["now"] == 11  # makespan is unaffected by Analyze's slack

    measure = _by_node(status, "Measure")
    analyze = _by_node(status, "Analyze")
    finish = _by_node(status, "Finish")

    # Analyze ran as a device-less pure-data processing activity.
    assert analyze["kind"] == "processing" and analyze["mode"] == "mean_v1"
    assert "devices" not in analyze
    assert "input_spots" not in analyze and "output_spots" not in analyze

    # The physical steps keep their deterministic times.
    assert (measure["start"], measure["end"]) == (2, 6)
    assert (finish["start"], finish["end"]) == (8, 11)

    # The Pure Data precedence holds: Analyze cannot start before Measure, which
    # produced the reading it consumes, has finished.
    assert analyze["start"] >= measure["end"]
    # It also sits within the window bounded by the downstream Object step.
    assert analyze["start"] <= finish["start"]


def test_rolling_pure_data_event_boundary():
    # Event-boundary advance (exact, deterministic times).
    runner = RollingRunner(PD_WF, PD_ENV, _interface(), poll_interval=None, random_seed=0)
    _assert_pure_data_run(runner.run())


def test_rolling_pure_data_fixed_interval():
    # Default fixed-interval polling (the standard mode).
    runner = RollingRunner(PD_WF, PD_ENV, _interface(), random_seed=0)
    _assert_pure_data_run(runner.run())


def test_rolling_pure_data_round_trips_through_replans():
    # The run drives through several replan cycles, each feeding the committed
    # pure-data activity back to the scheduler.
    runner = RollingRunner(PD_WF, PD_ENV, _interface(), poll_interval=None, random_seed=0)
    runner.run()
    assert runner.ticks > 1
