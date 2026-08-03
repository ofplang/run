"""Tests for re-routing in the rolling-horizon runner (milestone 2b-2a).

A device goes down mid-run; the runner discovers it (polling the backend) and
reduces the environment it schedules against: the down device's process modes are
dropped *and* every transport touching one of its spots is removed (an offline
device is unreachable, not merely un-processable, spec §7 / D21). The scheduler
then routes around it -- or, if pending material is stranded on the down device
with no route off, the replan is infeasible and the run stops. Timing stays
deterministic (event-boundary advance), so makespans are exact. The scheduler is a
required dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import DownScope, RollingRunner, RunnerError, load_document  # noqa: E402
from ofplang.run.simulator import (  # noqa: E402
    DeviceDown,
    UnknownReference,
    VirtualTimeSimulator,
)

FIXTURES = Path(__file__).parent / "fixtures"
SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")
REROUTE_ENV = str(FIXTURES / "reroute.env.yaml")


# -- simulator device up/down (unit) --------------------------------------

def test_processing_on_down_device_is_rejected():
    env = load_document(Path(SIMPLE_ENV))
    sim = VirtualTimeSimulator(env)
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
    sim = VirtualTimeSimulator(env)
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
    sim = VirtualTimeSimulator(env)
    sim.place("station_1.core")  # feed target's input
    uid = sim.dispatch_processing("target", "0")  # runs on station_1 [0, 2]
    sim.schedule_device_down(1, "station_1")  # down while it runs
    sim.advance(2)
    assert sim.state(uid) == {"status": "completed"}
    assert "station_1" in sim.down_devices()


def test_device_up_clears_down_and_allows_processing():
    # A device that comes back up leaves the down-set and can run processes again.
    env = load_document(Path(SIMPLE_ENV))
    sim = VirtualTimeSimulator(env)
    sim.schedule_device_down(2, "station_1")
    sim.schedule_device_up(5, "station_1")
    sim.place("station_1.core")

    sim.advance(3)
    assert sim.down_devices() == ["station_1"]
    with pytest.raises(DeviceDown):
        sim.dispatch_processing("target", "0")

    sim.advance(6)  # crosses the up at t=5
    assert sim.down_devices() == []
    uid = sim.dispatch_processing("target", "0")  # station_1 usable again
    sim.advance(8)
    assert sim.state(uid) == {"status": "completed"}


# -- environment reduction (unit) -----------------------------------------

def test_reduce_environment_by_scope():
    # A down device is made unschedulable along the axes its `down_scope` selects.
    from ofplang.run.runner.rolling import _normalize_mode_ids, _reduce_environment

    env = _normalize_mode_ids(load_document(Path(REROUTE_ENV)))

    def target_modes(reduced):
        return [m["devices"] for m in reduced["processes"]["target"]["modes"]]

    def routes(reduced):
        return {(t["from"], t["to"]) for t in reduced["transports"]}

    # BOTH (default): station_1's mode AND every transport touching station_1.core go;
    # only the direct station_0 -> station_2 route remains.
    both = _reduce_environment(env, {"station_1"}, DownScope.BOTH)
    assert ["station_1"] not in target_modes(both)
    assert routes(both) == {("station_0.core", "station_2.core")}

    # PROCESSING: drop only the mode; transports are kept (material can be moved off).
    proc = _reduce_environment(env, {"station_1"}, DownScope.PROCESSING)
    assert ["station_1"] not in target_modes(proc)
    assert len(proc["transports"]) == 3

    # TRANSPORT: keep the mode; drop only the transports touching station_1.core.
    trans = _reduce_environment(env, {"station_1"}, DownScope.TRANSPORT)
    assert ["station_1"] in target_modes(trans)
    assert routes(trans) == {("station_0.core", "station_2.core")}

    # Device/spot definitions are always kept (an isolated spot, never routed to).
    assert any(d["id"] == "station_1" for d in both["devices"])
    # Recovery is by reconstruction from the full env: nothing down -> everything back.
    full = _reduce_environment(env, set(), DownScope.BOTH)
    assert len(full["transports"]) == 3
    assert len(full["processes"]["target"]["modes"]) == 2


# -- reroute end to end ----------------------------------------------------

def test_reroute_off_down_device_with_processing_scope():
    # PROCESSING scope (the classic re-route): station_1 goes down at t=3, just after
    # the sample was delivered there. Only its modes are dropped -- transports are
    # kept -- so the runner re-transports the material off station_1.core to station_2
    # and runs target there.
    runner = RollingRunner(SIMPLE_WF, REROUTE_ENV, random_seed=0, down_scope=DownScope.PROCESSING)
    runner.sim.schedule_device_down(3, "station_1")
    status = runner.run()

    assert all(a["status"] == "completed" for a in status["activities"])
    # source(0-2) + deliver to station_1(2-3) + re-transport to station_2(3-7)
    # + target on station_2(7-9).
    assert status["now"] == 9
    target = next(a for a in status["activities"] if a.get("process") == "target")
    assert target["input_spots"]["target_in"] == "station_2.core"
    # The re-transport leg station_1.core -> station_2.core is present (kept by PROCESSING).
    assert any(
        a["kind"] == "transport"
        and a["from_spot"] == "station_1.core"
        and a["to_spot"] == "station_2.core"
        for a in status["activities"]
    )


def test_material_stranded_on_down_device_fails():
    # Default BOTH scope: station_1 goes down at t=3, after the sample has been
    # delivered to it. Because transports to/from a down device's spots are dropped
    # (an offline device is unreachable), the material on station_1.core cannot be
    # moved off to reach the station_2 target mode: the replan is infeasible and the
    # run stops (GAP 1 kept terminal for the first version). Contrast
    # test_reroute_off_down_device_with_processing_scope, which keeps the transports.
    runner = RollingRunner(SIMPLE_WF, REROUTE_ENV, random_seed=0)
    runner.sim.schedule_device_down(3, "station_1")
    with pytest.raises(RunnerError):
        runner.run()


def test_reroute_avoids_down_device_from_start():
    # station_1 is down from the start and stays down. Its target mode and every
    # transport touching station_1.core are dropped, so the run routes around it
    # entirely: target on station_2, reached via the direct station_0 -> station_2 link.
    runner = RollingRunner(SIMPLE_WF, REROUTE_ENV, random_seed=0)
    runner.sim.schedule_device_down(0, "station_1")
    status = runner.run()

    assert all(a["status"] == "completed" for a in status["activities"])
    # source(0-2) + station_0 -> station_2 (2-12) + target on station_2 (12-14).
    assert status["now"] == 14
    target = next(a for a in status["activities"] if a.get("process") == "target")
    assert target["input_spots"]["target_in"] == "station_2.core"
    # Nothing was ever routed through the down device's spot.
    assert not any(
        a["kind"] == "transport"
        and "station_1.core" in (a.get("from_spot"), a.get("to_spot"))
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


def test_device_coming_back_up_restores_routing():
    # station_1 is down from the start (the initial plan avoids it, routing target
    # to station_2), but comes back up at t=1 -- before anything commits to
    # station_2. The next replan then routes target back to the cheap station_1.
    runner = RollingRunner(SIMPLE_WF, REROUTE_ENV, random_seed=0)
    runner.sim.schedule_device_down(0, "station_1")
    runner.sim.schedule_device_up(1, "station_1")
    status = runner.run()

    assert all(a["status"] == "completed" for a in status["activities"])
    assert status["now"] == 5  # the cheap station_1 route, as if nothing happened
    target = next(a for a in status["activities"] if a.get("process") == "target")
    assert target["input_spots"]["target_in"] == "station_1.core"


# -- a down transporter (D39) ----------------------------------------------
#
# A transporter is the other kind of machine a run depends on, and it goes down the
# same way. What differs is that carrying material is all a transporter does: there
# is no axis to select, so its transports are dropped whatever the `down_scope`, and
# the simulator's dispatch rules are deliberately left alone (it is the *planning*
# that changes). `reroute_transporter.env.yaml` serves one route with two
# transporters, arm_a in 1 and arm_b in 4, so the makespan says which one carried it.

TRANSPORTER_ENV = str(FIXTURES / "reroute_transporter.env.yaml")


def test_reduce_environment_drops_a_down_transporters_transports():
    # In every scope: the axes are about a down *device*, and a transporter has none.
    from ofplang.run.runner.rolling import _normalize_mode_ids, _reduce_environment

    env = _normalize_mode_ids(load_document(Path(TRANSPORTER_ENV)))

    def carriers(reduced):
        return {t["transporter"] for t in reduced["transports"]}

    for scope in (DownScope.BOTH, DownScope.PROCESSING, DownScope.TRANSPORT):
        reduced = _reduce_environment(env, {"arm_a"}, scope)
        assert carriers(reduced) == {"arm_b"}, scope
        # The transporter definition itself stays, as a device's does.
        assert any(t["id"] == "arm_a" for t in reduced["transporters"]), scope
        # A transporter has no modes, so nothing about processing changes.
        assert len(reduced["processes"]["target"]["modes"]) == 1, scope

    # Nothing down -> nothing dropped (recovery is by reconstruction).
    assert carriers(_reduce_environment(env, set(), DownScope.BOTH)) == {"arm_a", "arm_b"}


def test_a_down_transporter_still_serves_a_dispatched_transport():
    # The oracle is deliberately unchanged (D39): what a down machine changes is what
    # gets planned, not what the backend permits. Keeping it that way means a plan that
    # somehow still routes through a down transporter fails as that operation (the
    # device command raises), not as an exception escaping the run.
    env = load_document(Path(TRANSPORTER_ENV))
    sim = VirtualTimeSimulator(env)
    sim.place("station_0.core")
    sim.schedule_device_down(0, "arm_a")
    sim.advance(0)
    assert sim.down_devices() == ["arm_a"]
    uid = sim.dispatch_transport("arm_a", "station_0.core", "station_1.core")
    sim.advance(1)
    assert sim.state(uid) == {"status": "completed"}


def test_fault_registration_accepts_a_transporter_but_not_an_unknown_id():
    env = load_document(Path(TRANSPORTER_ENV))
    sim = VirtualTimeSimulator(env)
    sim.schedule_device_down(0, "arm_b")  # a transporter id is accepted (D39)
    sim.schedule_device_down(0, "station_1")  # so is a device id, as before
    sim.advance(0)
    assert sim.down_devices() == ["arm_b", "station_1"]
    with pytest.raises(UnknownReference):
        sim.schedule_device_down(0, "no_such_machine")


def test_reroute_onto_another_transporter():
    # arm_a is down from the start, so its entry for the route is dropped and the
    # scheduler carries the same move with arm_b: source(0-2) + move(2-6) + target(6-8).
    runner = RollingRunner(SIMPLE_WF, TRANSPORTER_ENV, random_seed=0)
    runner.sim.schedule_device_down(0, "arm_a")
    status = runner.run()

    assert all(a["status"] == "completed" for a in status["activities"])
    assert status["now"] == 8  # 5 would mean arm_a was still used
    move = next(a for a in status["activities"] if a["kind"] == "transport")
    assert (move["from_spot"], move["to_spot"]) == ("station_0.core", "station_1.core")


def test_down_transporter_with_no_alternative_fails():
    # Both transporters down leaves the route uncarried -- and `transports` empty --
    # so the replan is infeasible and the run stops (as for a device, D21).
    runner = RollingRunner(SIMPLE_WF, TRANSPORTER_ENV, random_seed=0)
    runner.sim.schedule_device_down(0, "arm_a")
    runner.sim.schedule_device_down(0, "arm_b")
    with pytest.raises(RunnerError):
        runner.run()


def test_transporter_coming_back_up_restores_the_cheap_route():
    # arm_a is down at the start (the initial plan reaches for arm_b) but returns at
    # t=1, before anything commits to the slow carry: the next replan uses arm_a again.
    runner = RollingRunner(SIMPLE_WF, TRANSPORTER_ENV, random_seed=0)
    runner.sim.schedule_device_down(0, "arm_a")
    runner.sim.schedule_device_up(1, "arm_a")
    status = runner.run()

    assert all(a["status"] == "completed" for a in status["activities"])
    assert status["now"] == 5  # the cheap carry, as if nothing happened


def test_nothing_down_uses_the_cheap_transporter():
    runner = RollingRunner(SIMPLE_WF, TRANSPORTER_ENV, random_seed=0)
    status = runner.run()
    assert status["now"] == 5
