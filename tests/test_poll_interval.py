"""Tests for fixed-interval polling + completion-time estimation (milestone 2b-2b-1).

With `poll_interval=None` (the default) the runner advances to plan event
boundaries: deterministic and exact. With `poll_interval=D` it polls every D
units, learning of completions only at a poll and recording their time as that
poll (an upper bound on the true finish, D22). Durations are deterministic here
(no variance), so every outcome is an exact function of D and can be asserted.
The scheduler is a required dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.cli import EXIT_OK, main  # noqa: E402
from ofplang.run.runner import RollingRunner, load_document  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")
REROUTE_ENV = str(FIXTURES / "reroute.env.yaml")

# simple: source (2) -> transport (1) -> target (2); event-boundary makespan 5.


def test_event_boundary_default_is_exact():
    status = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0).run()
    assert status["now"] == 5
    transport = next(a for a in status["activities"] if a["kind"] == "transport")
    assert transport["end"] == 3  # exact


def test_fine_poll_matches_exact():
    # A poll interval of 1 lines up with the integer event times, so there is no
    # drift: the run is identical to the event-boundary case.
    status = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0, poll_interval=1).run()
    assert status["now"] == 5


def test_coarse_poll_drifts_deterministically():
    # With a poll interval of 2, the transport (true finish at t=3) is only seen at
    # the poll at t=4, which pushes the target one interval later -> makespan 6.
    status = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0, poll_interval=2).run()
    assert status["now"] == 6
    transport = next(a for a in status["activities"] if a["kind"] == "transport")
    assert transport["end"] == 4  # the poll upper bound, not the true 3


def test_estimated_end_is_an_upper_bound_of_the_true_finish():
    # The reported completion is an upper bound: the backend's own event history
    # shows the transport truly finished at t=3, but a poll-only observer (interval
    # 2) records 4. This is exactly the estimation the fixed-interval mode makes.
    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0, poll_interval=2)
    status = runner.run()
    transport = next(a for a in status["activities"] if a["kind"] == "transport")
    true_end = next(e.time for e in runner.sim._history() if e.kind == "transport")
    assert true_end == 3  # the real finish, from the backend history
    assert transport["end"] == 4  # the runner's poll-based estimate
    assert transport["end"] >= true_end  # never earlier than the truth


def test_reroute_still_works_under_fixed_interval():
    # Re-routing (2b-2a) composes with fixed-interval polling: station_1 goes down,
    # the runner re-routes target to station_2, and the run still completes.
    runner = RollingRunner(SIMPLE_WF, REROUTE_ENV, random_seed=0, poll_interval=2)
    runner.sim.schedule_device_down(3, "station_1")
    status = runner.run()
    assert all(a["status"] == "completed" for a in status["activities"])
    assert status["now"] == 10
    target = next(a for a in status["activities"] if a.get("process") == "target")
    assert target["input_spots"]["target_in"] == "station_2.core"


def test_cli_poll_interval(tmp_path):
    out = tmp_path / "status.yaml"
    code = main(
        ["run", SIMPLE_WF, "--env", SIMPLE_ENV, "--seed", "0", "--poll-interval", "2", "-o", str(out)]
    )
    assert code == EXIT_OK
    assert load_document(out)["now"] == 6
