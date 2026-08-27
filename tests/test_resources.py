"""Tests for device-local consumables in the rolling-horizon runner (S-D1).

The scheduler owns the consumable model (`ofplang-schedule` >= 0.2.0): a device
declares what it can hold, a mode declares what it draws, and the *execution
document* says what each stock held when the run began. The runner's part is to
supply that last piece and to keep supplying it, unchanged, on every replan -- the
level at `now` is never stated, it is replayed from those starting levels plus the
`consumption` each fixed activity echoes (SPEC §4.7.2).

What is pinned here:

  - the door: `boundary.inventories.levels` reaches the scheduler as the §6.10
    section of the status, on the first replan and on every one after it;
  - the execution: a refill is dispatched like any other activity, holding the
    device it fills and the replenisher that fills it, and moving no material;
  - the off switch (`ignore_resources`, §4.7.3) and the warning it raises;
  - that the result boundary does **not** echo the starting levels back.

The scheduler is a required dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import RollingRunner, RunnerError, load_document  # noqa: E402
from ofplang.run.runner.rolling import DownScope, _reduce_environment  # noqa: E402
from ofplang.run.simulator import SimulatorError  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
WF = str(FIXTURES / "simple.workflow.yaml")
CONSUMABLE_ENV = str(FIXTURES / "consumable.env.yaml")
REFILL_ENV = str(FIXTURES / "consumable_refill.env.yaml")


def _boundary(level: int) -> dict:
    return {"boundary": {"inventories": {"levels": {"station_1": {"reagent": level}}}}}


def _processing(status: dict) -> list[dict]:
    return [a for a in status["activities"] if a.get("kind") == "processing"]


# -- the door ----------------------------------------------------------------


def test_stocked_run_completes_and_status_carries_the_starting_levels():
    """A run whose stock covers its consumption completes, and the status it renders
    is a §6.10 document: the levels it started with, not the ones it ends on."""
    runner = RollingRunner(WF, CONSUMABLE_ENV, boundary=_boundary(2), random_seed=0)
    status = runner.run()
    assert status["inventories"] == {"levels": {"station_1": {"reagent": 2}}}
    assert not runner.failed


def test_the_levels_survive_every_replan_not_just_the_first():
    """The section is re-sent unchanged on each replan. If it were dropped after the
    first, the second would fail `missing_inventories` -- so a multi-replan run
    completing is the assertion, and the replan count is what makes it one."""
    runner = RollingRunner(WF, CONSUMABLE_ENV, boundary=_boundary(2), random_seed=0)
    runner.run()
    assert runner.replans > 1


def test_consumption_travels_with_each_fixed_activity():
    """The scheduler replays levels from the history, so a completed activity has to
    carry what it drew: a replan may withdraw the very mode it used, and a fixed
    activity is never re-read against the current environment (§7)."""
    status = RollingRunner(WF, CONSUMABLE_ENV, boundary=_boundary(2), random_seed=0).run()
    drawn = [a.get("consumption") for a in _processing(status)]
    assert {"station_1.reagent": 1} in drawn


def test_too_little_stock_is_infeasible():
    """Not enough to run the workflow is a planning failure, not a runtime one: the
    run never starts."""
    with pytest.raises(RunnerError, match="infeasible"):
        RollingRunner(WF, CONSUMABLE_ENV, boundary=_boundary(0), random_seed=0).run()


def test_a_consuming_environment_without_inventories_is_refused_by_the_scheduler():
    """The runner does not invent an empty `levels`. "Every stock starts empty" and
    "the run does not say" are different claims, and only the scheduler knows whether
    any invoked mode consumes -- so the omission surfaces as its `missing_inventories`
    rather than as a silently under-stocked run."""
    with pytest.raises(RunnerError, match="missing_inventories"):
        RollingRunner(WF, CONSUMABLE_ENV, random_seed=0).run()


def test_an_environment_that_does_not_consume_needs_no_inventories():
    """Declaring a stock nothing draws on demands nothing of the boundary."""
    env = load_document(CONSUMABLE_ENV)
    del env["processes"]["target"]["modes"][0]["consumption"]
    status = RollingRunner(WF, env, random_seed=0).run()
    assert "inventories" not in status


# -- refills, executed ------------------------------------------------------


def test_a_refillable_environment_runs_and_carries_out_the_refill():
    """An empty stock is topped up rather than ending the run: the scheduler places a
    refill, the runner dispatches it, and it appears in the history as a completed
    activity like any other."""
    runner = RollingRunner(WF, REFILL_ENV, boundary=_boundary(0), random_seed=0)
    status = runner.run()
    assert not runner.failed
    refills = [a for a in status["activities"] if a.get("kind") == "replenishment"]
    assert len(refills) == 1
    assert refills[0]["status"] == "completed"
    assert refills[0]["device"] == "station_1" and refills[0]["replenisher"] == "dispenser"
    assert refills[0]["amounts"] == {"reagent": 4}  # a refill fills to capacity


def test_the_refill_holds_both_machines_while_it_works():
    """The scheduler plans refills exclusive on both machines, so the simulator has to
    keep them that way -- a backend that let them overlap would be running a schedule
    nobody proved. `dispatch_replenishment` refuses a double booking."""
    from ofplang.run.simulator import VirtualTimeSimulator

    sim = VirtualTimeSimulator(load_document(REFILL_ENV))
    sim.dispatch_replenishment("dispenser", "station_1", {"reagent": 4}, duration=2)
    with pytest.raises(SimulatorError):  # the reader is held
        sim.dispatch_processing("target", "m0", duration=2)
    with pytest.raises(SimulatorError):  # and so is the dispenser
        sim.dispatch_replenishment("dispenser", "station_1", {"reagent": 4}, duration=2)


def test_both_machines_are_released_when_the_refill_completes():
    from ofplang.run.simulator import VirtualTimeSimulator

    sim = VirtualTimeSimulator(load_document(REFILL_ENV))
    uuid = sim.dispatch_replenishment("dispenser", "station_1", {"reagent": 4}, duration=2)
    sim.advance(2)
    assert sim.state(uuid)["status"] == "completed"
    sim.dispatch_replenishment("dispenser", "station_1", {"reagent": 4}, duration=2)


def test_a_refill_leaves_the_spots_alone():
    """The scheduler holds the *device* for a refill, not its spots, so material
    resting on a stage does not stop the stage's device being topped up -- and a
    refill moves nothing."""
    from ofplang.run.simulator import VirtualTimeSimulator

    sim = VirtualTimeSimulator(load_document(REFILL_ENV))
    sim.place("station_1.core", "plate-1")
    uuid = sim.dispatch_replenishment("dispenser", "station_1", {"reagent": 4}, duration=2)
    sim.advance(2)
    assert sim.state(uuid)["status"] == "completed"
    assert sim.spot_state("station_1.core") == "plate-1"


def test_a_refill_a_replenisher_cannot_perform_is_refused():
    from ofplang.run.simulator import VirtualTimeSimulator

    sim = VirtualTimeSimulator(load_document(REFILL_ENV))
    with pytest.raises(SimulatorError):  # no (dispenser, station_0) entry
        sim.dispatch_replenishment("dispenser", "station_0")
    with pytest.raises(SimulatorError):
        sim.dispatch_replenishment("no_such_machine", "station_1", duration=2)


def test_a_refill_table_over_modes_that_draw_nothing_plans_no_refill():
    """Declaring a way to refill a stock nothing consumes constrains nothing."""
    env = load_document(REFILL_ENV)
    del env["processes"]["target"]["modes"][0]["consumption"]
    status = RollingRunner(WF, env, random_seed=0).run()
    assert not [a for a in status["activities"] if a.get("kind") == "replenishment"]


def test_ignoring_resources_plans_no_refill_either():
    """With the model off there is nothing to run out of, so nothing to top up."""
    runner = RollingRunner(WF, REFILL_ENV, random_seed=0, ignore_resources=True)
    status = runner.run()
    assert not runner.failed
    assert not [a for a in status["activities"] if a.get("kind") == "replenishment"]


# -- the off switch ----------------------------------------------------------


def test_ignoring_resources_runs_a_consuming_environment_with_no_inventories():
    """§4.7.3: the declarations are shape-checked and nothing is applied, so the
    boundary need not say what the stocks started with."""
    runner = RollingRunner(WF, CONSUMABLE_ENV, random_seed=0, ignore_resources=True)
    status = runner.run()
    assert "inventories" not in status
    assert [a.get("consumption") for a in _processing(status)] == [None, None]


def test_ignoring_resources_lifts_the_stock_limit():
    """Off is always a relaxation: what was infeasible on the real stock now runs."""
    RollingRunner(
        WF, CONSUMABLE_ENV, boundary=_boundary(0), random_seed=0, ignore_resources=True
    ).run()


def test_switching_the_model_off_says_so():
    """A model that silently does nothing is indistinguishable from one that agrees."""
    runner = RollingRunner(WF, CONSUMABLE_ENV, random_seed=0, ignore_resources=True)
    runner.run()
    assert [d.code for d in runner.scheduler_warnings] == ["resources_ignored"]


# -- the result boundary -----------------------------------------------------


def test_the_result_boundary_does_not_echo_the_starting_levels():
    """A result boundary is written to be fed back (labcode round-trips Object ids
    through exactly that), and a second run replays no history -- so echoing the stock
    this run started with would hand the next one stock this one already spent."""
    runner = RollingRunner(WF, CONSUMABLE_ENV, boundary=_boundary(2), random_seed=0)
    runner.run()
    assert "inventories" not in runner.result_boundary["boundary"]


# -- reduction ---------------------------------------------------------------


def test_a_down_replenisher_cannot_refill():
    """A replenisher that is down performs no refill, whatever the scope -- the same
    rule a down transporter gets, and for the same reason: refilling is all it does."""
    env = load_document(REFILL_ENV)
    for scope in DownScope:
        reduced = _reduce_environment(env, {"dispenser"}, scope)
        assert reduced["replenishments"] == []
        assert reduced["replenishers"] == [{"id": "dispenser"}]  # the machine still exists


def test_a_down_device_cannot_be_refilled_when_it_cannot_be_reached():
    """Putting stock into a device is material movement, so a refill of it is
    withdrawn on the same axis as the transports that would reach it."""
    env = load_document(REFILL_ENV)
    for scope in (DownScope.BOTH, DownScope.TRANSPORT):
        assert _reduce_environment(env, {"station_1"}, scope)["replenishments"] == []
    kept = _reduce_environment(env, {"station_1"}, DownScope.PROCESSING)["replenishments"]
    assert kept == env["replenishments"]


def test_reduction_leaves_the_caller_environment_alone():
    env = load_document(REFILL_ENV)
    before = copy.deepcopy(env)
    _reduce_environment(env, {"dispenser"})
    assert env == before


# -- the replay path ---------------------------------------------------------


def test_replay_reproduces_a_plan_that_carries_a_refill():
    """`ofp-run replay` claims to reproduce a plan the scheduler produced, and the
    scheduler produces plans with refills. Replay drives the virtual-time simulator,
    so only the timing and the occupancy are at stake -- but a plan it cannot run at
    all is a hole in that claim, and it was one."""
    from ofplang.schedule import schedule

    from ofplang.run.runner import Runner

    report = schedule(WF, REFILL_ENV, document_path={
        "inventories": {"levels": {"station_1": {"reagent": 0}}}, "activities": []
    })
    assert report.plan is not None, [d.code for d in report.diagnostics]
    assert [a for a in report.plan["activities"] if a["kind"] == "replenishment"]

    status = Runner(report.plan, load_document(REFILL_ENV)).run()
    replayed = [a for a in status["activities"] if a.get("kind") == "replenishment"]
    assert len(replayed) == 1
    assert all(a["status"] == "completed" for a in status["activities"])
    # The replay reproduces the plan's own timing.
    planned = next(a for a in report.plan["activities"] if a["kind"] == "replenishment")
    assert (replayed[0]["start"], replayed[0]["end"]) == (planned["start"], planned["end"])


# -- availability -----------------------------------------------------------


def test_a_down_replenisher_is_scheduled_around():
    """A replenisher can go down like any other machine, and the reduction drops the
    refills it would have performed. With no other way to top the stock up the replan
    is infeasible -- which is the honest answer, and the first time this path has been
    reachable at all (nothing used to report a replenisher down)."""
    from ofplang.run.simulator import DeviceDown, VirtualTimeSimulator

    def factory(environment):
        sim = VirtualTimeSimulator(environment)
        sim.schedule_device_down(0, "dispenser")
        return sim

    runner = RollingRunner(
        WF, REFILL_ENV, boundary=_boundary(0), backend_factory=factory, random_seed=0
    )
    with pytest.raises(RunnerError, match="infeasible"):
        runner.run()
    assert DeviceDown is not None  # the injection API this test drives


def test_a_replenisher_is_a_machine_the_simulator_knows():
    """Fault injection used to accept only devices and transporters, so a down
    replenisher could not be expressed."""
    from ofplang.run.simulator import UnknownReference, VirtualTimeSimulator

    sim = VirtualTimeSimulator(load_document(REFILL_ENV))
    sim.schedule_device_down(1, "dispenser")  # must not raise
    with pytest.raises(UnknownReference):
        sim.schedule_device_down(1, "no_such_machine")


# -- variance ---------------------------------------------------------------


def test_variance_cannot_crush_a_refill_to_nothing():
    """A refill's duration is positive (§5.7), so the variance floor is 1 as it is for
    a processing -- only a transport may be zero, a same-spot hop being a real no-op.
    A zero-length refill would be a visit that held two machines for no time."""
    runner = RollingRunner(
        WF,
        REFILL_ENV,
        boundary=_boundary(0),
        random_seed=0,
        poll_interval=1,
        running_task_margin=1,
        duration_model=lambda activity, planned: 0,
    )
    status = runner.run()
    for activity in status["activities"]:
        if activity.get("kind") == "replenishment":
            assert activity["end"] > activity["start"]
