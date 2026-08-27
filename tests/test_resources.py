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
  - the refusal: an environment whose refills the runner cannot execute is rejected
    before any work is dispatched, rather than crashing on the first refill;
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


# -- refills the runner cannot execute ---------------------------------------


def test_a_refillable_environment_is_refused_before_anything_runs():
    """The scheduler would answer this environment with `kind: replenishment`, which
    has no dispatch here. Refusing at construction is the honest form of that; the
    alternative is scheduling happily and dying on the first refill."""
    with pytest.raises(RunnerError, match="replenishments"):
        RollingRunner(WF, REFILL_ENV, boundary=_boundary(0), random_seed=0)


def test_the_refusal_names_both_ways_out():
    with pytest.raises(RunnerError) as excinfo:
        RollingRunner(WF, REFILL_ENV, boundary=_boundary(0), random_seed=0)
    message = str(excinfo.value)
    assert "replenishments" in message and "resources ignored" in message


def test_refills_are_only_refused_when_something_consumes():
    """A refill table over modes that draw nothing constrains nothing, so the
    scheduler proposes no refill and there is nothing to refuse."""
    env = load_document(REFILL_ENV)
    del env["processes"]["target"]["modes"][0]["consumption"]
    RollingRunner(WF, env, random_seed=0).run()  # must not raise


def test_ignoring_resources_makes_a_refillable_environment_runnable():
    """With the model off no refill is planned, so the refusal does not apply."""
    runner = RollingRunner(WF, REFILL_ENV, random_seed=0, ignore_resources=True)
    runner.run()
    assert not runner.failed


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
