"""The solver's budget on a replan (schedule design.md D67, stage 3), and on the first plan.

`replan_max_time_seconds` is handed to every plan after the first. The first plan has a
budget of its own, `first_plan_max_time_seconds` (a minute by default): past it the first
plan is the best schedule found or the constructed one (schedule design.md D68), and only
where there is neither is it solved again with no budget. Within the budget a replan
the solver does not finish gets the schedule constructed from the plan it replaces --
after an overrun, a branch decided on the arm not assumed, a loop going on -- so a run
whose solver keeps running out of time still runs to the end.
"""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import (  # noqa: E402
    JobRequest,
    RollingRunner,
    rolling,  # noqa: E402
)
from tests.test_branch_measured import ENV as CUPS  # noqa: E402
from tests.test_branch_measured import MEASURED  # noqa: E402
from tests.test_loops import ENV as RINSE_ENV  # noqa: E402
from tests.test_loops import PLATE, reading, rinse  # noqa: E402
from tests.test_regions import POLISH_THEN_INSPECT, _cup, _overrun_first_rinse  # noqa: E402


def _budgets(monkeypatch, give_up: bool = False) -> list:
    """Record the budget every plan is asked with; with `give_up`, the solver finds
    nothing within it, so a replan given one is whatever was constructed."""
    from ofplang.schedule.scheduler import api
    from ofplang.schedule.scheduler.result import Solution

    asked = []
    real = rolling.replan

    def replan(*args, **kwargs):
        budget = kwargs.get("max_time_seconds")
        asked.append(budget)
        if not (give_up and budget is not None):
            return real(*args, **kwargs)
        solver = api._solver
        api._solver = lambda: lambda *a, **k: Solution("unknown", None, (), ())
        try:
            return real(*args, **kwargs)
        finally:
            api._solver = solver

    monkeypatch.setattr(rolling, "replan", replan)
    return asked


def _two_loops(budget, by: int = 4, **kwargs):
    workflow = rinse()
    jobs = [JobRequest("a", copy.deepcopy(workflow), copy.deepcopy(PLATE)),
            JobRequest("b", copy.deepcopy(workflow), {"boundary": {
                "inputs": {"plate": {"spot": "rack.in2"}},
                "outputs": {"plate": {"spot": "rack.out2"}}}})]
    return RollingRunner(
        jobs, copy.deepcopy(RINSE_ENV), poll_interval=1, running_task_margin=1,
        random_seed=0, device_model=reading(1), duration_model=_overrun_first_rinse(by),
        replan_max_time_seconds=budget, **kwargs,
    )


def test_by_default_the_first_plan_gets_a_minute_and_no_replan_a_budget(monkeypatch):
    asked = _budgets(monkeypatch)
    runner = _two_loops(None)
    runner.run()
    assert not runner.failed, runner.failure
    assert asked[0] == 60.0
    assert len(asked) > 1 and set(asked[1:]) == {None}


def test_the_replan_budget_is_asked_on_every_replan_and_not_the_first_plan(monkeypatch):
    asked = _budgets(monkeypatch)
    runner = _two_loops(5.0)
    runner.run()
    assert not runner.failed, runner.failure
    assert asked[0] == 60.0
    assert len(asked) > 1 and set(asked[1:]) == {5.0}


def test_zero_lifts_the_first_plans_budget(monkeypatch):
    asked = _budgets(monkeypatch)
    runner = _two_loops(None, first_plan_max_time_seconds=0)
    runner.run()
    assert not runner.failed, runner.failure
    assert asked[0] is None


def test_a_loop_going_on_runs_to_the_end_with_every_replan_constructed(monkeypatch):
    # Every replan's solver gives up: the plans are the previous ones laid out again,
    # with the next invocation spliced in when a loop goes on.
    _budgets(monkeypatch, give_up=True)
    runner = _two_loops(1.0)
    status = runner.run()
    assert not runner.failed, runner.failure
    assert sorted(e["expansion"]["loops"][0]["count"] for e in status["jobs"]) == [1, 2]
    assert "plan_constructed_after_solver" in [w.code for w in runner.scheduler_warnings]


def test_a_branch_off_the_arm_assumed_runs_to_the_end_with_every_replan_constructed(
    monkeypatch,
):
    _budgets(monkeypatch, give_up=True)
    jobs = [_cup("a", MEASURED, "a", "a"), _cup("b", POLISH_THEN_INSPECT, "b", "b")]
    runner = RollingRunner(jobs, copy.deepcopy(CUPS), poll_interval=None, random_seed=0,
                           replan_max_time_seconds=1.0)
    status = runner.run()
    assert not runner.failed, runner.failure
    arms = next(e for e in status["jobs"] if e["id"] == "a")["expansion"]["arms"]
    assert arms == [{"node": ["H"], "arm": "else"}]
    assert "plan_constructed_after_solver" in [w.code for w in runner.scheduler_warnings]


def test_a_first_plan_the_solver_does_not_finish_is_the_constructed_one(monkeypatch):
    # The solver gives up within every budget, the first plan's included: the run
    # starts on the constructed plan (schedule design.md D68, A2) and runs to the end.
    asked = _budgets(monkeypatch, give_up=True)
    runner = _two_loops(1.0)
    runner.run()
    assert not runner.failed, runner.failure
    assert asked[0] == 60.0 and None not in asked  # the first plan was not asked again
    assert "plan_constructed_after_solver" in [w.code for w in runner.scheduler_warnings]


def test_with_nothing_constructed_the_first_plan_is_solved_to_the_end(monkeypatch):
    # Within its budget the solver finds nothing and nothing can be built: the first
    # plan is asked again with no budget, so a run that would have started still does.
    from ofplang.schedule.scheduler import greedy

    asked = _budgets(monkeypatch, give_up=True)
    monkeypatch.setattr(greedy, "construct", lambda *_a, **_k: None)
    runner = _two_loops(None)
    runner.run()
    assert not runner.failed, runner.failure
    assert asked[:2] == [60.0, None]
