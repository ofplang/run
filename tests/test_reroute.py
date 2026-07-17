"""Tests for re-routing in the rolling-horizon runner (milestone 2b-2a).

A device goes down mid-run; the runner discovers it (polling the backend),
reduces the environment it schedules against (dropping the down device's process
modes, spec §7 / D21), and the scheduler re-routes the pending work through a
relay and a re-transport. Timing stays deterministic (event-boundary advance), so
the resulting makespan is exact. The scheduler is a required dependency; these
tests skip if it is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.cli import EXIT_FAILED, EXIT_OK, main  # noqa: E402
from ofplang.run.runner import RollingRunner, RunnerError, load_document  # noqa: E402
from ofplang.run.simulator import DeviceDown, Simulator  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")
REROUTE_ENV = str(FIXTURES / "reroute.env.yaml")


# -- simulator device up/down (unit) --------------------------------------

def test_processing_on_down_device_is_rejected():
    env = load_document(Path(SIMPLE_ENV))
    sim = Simulator(env)
    sim.schedule_device_down(0, "station_1")
    sim.advance(0)  # apply the fault
    assert sim.down_devices() == ["station_1"]
    # target runs on station_1, which is down -> rejected.
    sim.place("station_1.core")
    with pytest.raises(DeviceDown):
        sim.dispatch_processing("target", "0")


def test_transport_from_down_device_is_allowed():
    # A down device still holds material and can be transported from (D21).
    env = load_document(Path(REROUTE_ENV))
    sim = Simulator(env)
    sim.place("station_1.core")
    sim.schedule_device_down(0, "station_1")
    sim.advance(0)
    uid = sim.dispatch_transport("transport", "station_1.core", "station_2.core")
    sim.advance(4)
    assert sim.state(uid) == {"status": "completed"}
    assert sim.spot_state("station_2.core") is not None


def test_running_op_unaffected_by_down():
    # Taking a device down does not fail an operation already running on it (D21).
    env = load_document(Path(SIMPLE_ENV))
    sim = Simulator(env)
    sim.place("station_1.core")  # feed target's input
    uid = sim.dispatch_processing("target", "0")  # runs on station_1 [0, 2]
    sim.schedule_device_down(1, "station_1")  # down while it runs
    sim.advance(2)
    assert sim.state(uid) == {"status": "completed"}
    assert "station_1" in sim.down_devices()


# -- reroute end to end ----------------------------------------------------

def test_reroute_when_device_goes_down():
    # station_1 goes down at t=3, just as the sample has been delivered there and
    # before target starts. The run re-routes target to station_2.
    runner = RollingRunner(SIMPLE_WF, REROUTE_ENV, random_seed=0)
    runner.sim.schedule_device_down(3, "station_1")
    status = runner.run()

    assert all(a["status"] == "completed" for a in status["activities"])
    # source(0-2) + deliver to station_1(2-3) + re-transport to station_2(3-7)
    # + target on station_2(7-9).
    assert status["now"] == 9
    # target ran on station_2 (re-routed), not station_1.
    target = next(a for a in status["activities"] if a.get("process") == "target")
    assert target["input_spots"]["target_in"] == "station_2.core"
    # The re-transport leg station_1.core -> station_2.core is present.
    assert any(
        a["kind"] == "transport" and a["from_spot"] == "station_1.core" and a["to_spot"] == "station_2.core"
        for a in status["activities"]
    )


def test_no_reroute_when_nothing_goes_down():
    # Without a fault, the run stays on the cheap route (target on station_1).
    runner = RollingRunner(SIMPLE_WF, REROUTE_ENV, random_seed=0)
    status = runner.run()
    assert status["now"] == 5
    target = next(a for a in status["activities"] if a.get("process") == "target")
    assert target["input_spots"]["target_in"] == "station_1.core"


def test_reroute_with_no_alternative_fails():
    # simple.env has target only on station_1; taking it down after delivery leaves
    # the process with no capability, so the replan is infeasible.
    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0)
    runner.sim.schedule_device_down(3, "station_1")
    with pytest.raises(RunnerError):
        runner.run()


# -- CLI -------------------------------------------------------------------

def test_cli_run_with_down(tmp_path):
    out = tmp_path / "status.yaml"
    code = main(
        ["run", SIMPLE_WF, "--env", REROUTE_ENV, "--seed", "0", "--down", "station_1@3", "-o", str(out)]
    )
    assert code == EXIT_OK
    status = load_document(out)
    assert status["now"] == 9


def test_cli_invalid_down_spec_is_usage_error(capsys):
    code = main(["run", SIMPLE_WF, "--env", REROUTE_ENV, "--down", "bogus"])
    assert code == 2
    assert "invalid --down" in capsys.readouterr().err
