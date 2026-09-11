"""Taking a job out of a running plan: `RollingRunner.withdraw` (SPEC §6.11).

The roster is the set of jobs something of which is still in the laboratory --
unfinished work, or material nobody has collected -- so an entry is removed when
neither is true. Whether the room is actually clear is not something the runner can
see, so leaving is asked for: calling `withdraw` *is* the statement that what the job
bound has been collected.

🔴 **What makes it more than dropping an entry is the arithmetic.** This runner states
what the run *started* with and has no way to work out a level -- replaying the history
is the scheduler's job (§4.7.2) -- so a job leaving with its history would hand the
stock back everything that job drew. The scheduler carries the levels forward to `now`
and says so (`inventories.at`, §6.10); the test that matters is that the runner adopts
that rather than going on quoting the opening levels.

The other half is the spots. A **bound** final output's is freed -- the caller named
it, so leaving says they collected it there -- and the backend is told, or the next
delivery to that spot is refused for material nobody has. An **unbound** one's is not:
the schedule chose where that material came to rest and the caller was never told, so
it becomes an occupancy (§6.12).

The scheduler is a required dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import (  # noqa: E402
    JobRequest,
    RollingRunner,
    RunnerError,
    load_document,
)

EXAMPLES = Path(__file__).parent.parent / "examples"
FIXTURES = Path(__file__).parent / "fixtures"
REFILL_WF = EXAMPLES / "shared_refill.workflow.yaml"
REFILL_ENV = str(EXAMPLES / "shared_refill.env.yaml")
OPENING = {"levels": {"reader": {"reagent": 2}}}

# loader -> heater -> two racks, so two jobs with boundary material can each deliver
# somewhere of their own and a withdrawal's effect on a spot is visible.
BAY_ENV = {
    "time": {"unit": "second"},
    "devices": [
        {"id": "loader", "spots": ["stage"]},
        {"id": "heater", "spots": ["stage"]},
        {"id": "output", "spots": ["rack_a", "rack_b"]},
    ],
    "transporters": [{"id": "arm"}],
    "transports": [
        {"transporter": "arm", "from": "loader.stage", "to": "heater.stage", "duration": 2},
        {"transporter": "arm", "from": "heater.stage", "to": "output.rack_a", "duration": 2},
        {"transporter": "arm", "from": "heater.stage", "to": "output.rack_b", "duration": 2},
    ],
    "processes": {
        "heat": {
            "modes": [
                {
                    "devices": ["heater"],
                    "duration": 10,
                    "input_spots": {"plate": "heater.stage"},
                    "output_spots": {"out": "heater.stage"},
                }
            ]
        },
    },
}


def _refill_run(*ids):
    workflow = load_document(REFILL_WF)
    return RollingRunner(
        [JobRequest(id=job_id, workflow=copy.deepcopy(workflow)) for job_id in ids],
        REFILL_ENV,
        poll_interval=None,
        random_seed=0,
        inventories=copy.deepcopy(OPENING),
    )


def _bay_run(tmp_path, job1_spot):
    """Two jobs with real boundary material; `job1_spot` binds job1's output or not."""
    env = tmp_path / "bay.env.yaml"
    env.write_text(yaml.safe_dump(BAY_ENV, sort_keys=False), encoding="utf-8")
    workflow = load_document(FIXTURES / "interface_load.workflow.yaml")

    def boundary(spot):
        return {
            "boundary": {
                "inputs": {"sample": {"spot": "loader.stage"}},
                "outputs": {"result": {"spot": spot} if spot else {}},
            }
        }

    return RollingRunner(
        [
            JobRequest(id="job1", workflow=copy.deepcopy(workflow),
                       boundary=boundary(job1_spot)),
            JobRequest(id="job2", workflow=copy.deepcopy(workflow), release=30,
                       boundary=boundary("output.rack_b")),
        ],
        str(env),
        poll_interval=None,
        random_seed=0,
    )


def _withdraw_when_done(runner, job_id):
    """Run, calling `withdraw(job_id)` the tick its own work is all finished.

    Driven from the clock rather than by counting ticks: the moment a job has nothing
    running and nothing left to dispatch is the moment a person watching the bench
    would collect its plate, which is what this is standing in for. Returns the time
    the call was made and a snapshot of the backend's spots just before it.
    """
    original = runner.sim.advance
    marks: list[int] = []
    before: dict = {}

    def advance(until):
        reached = original(until)
        job = runner._by_id.get(job_id)
        if (
            not marks
            and job is not None
            and not any(runner._job_of(r.activity) is job for r in runner.log.running())
            and not any(runner._job_of(a) is job for a in runner._undispatched())
            and any(r.activity.get("job") == job_id for r in runner.log.records())
        ):
            before.update(runner.sim.spot_state())
            runner.withdraw(job_id)
            marks.append(reached)
        return reached

    runner.sim.advance = advance  # type: ignore[method-assign]
    return runner.run(), (marks[0] if marks else None), before


# -- what it refuses ---------------------------------------------------------


def test_withdraw_refuses_a_job_this_run_does_not_plan():
    runner = _refill_run("job1", "job2")
    with pytest.raises(RunnerError, match="this run plans"):
        runner.withdraw("nobody")


def test_withdraw_refuses_the_last_job_of_the_run():
    """A plan of no jobs is not a plan (`withdrawal_empties_roster`, §6.11). Refused
    here because the scheduler refusing it would fail the replan, and a failed replan
    stops every job in a run of named jobs."""
    runner = _refill_run("job1", "job2")
    runner.withdraw("job1")
    with pytest.raises(RunnerError, match="last job of the run"):
        runner.withdraw("job2")


def test_withdraw_refuses_a_single_workflow():
    """There is no roster to leave: the one workflow is the whole run."""
    runner = RollingRunner(
        str(FIXTURES / "simple.workflow.yaml"),
        str(FIXTURES / "simple.env.yaml"),
        random_seed=0,
    )
    with pytest.raises(RunnerError, match="needs a run of named jobs"):
        runner.withdraw("")


def test_a_job_that_has_not_started_may_leave():
    """🔴 Deliberate, and the same rule the scheduler applies (SPEC §6.11): nothing of
    the job is in the laboratory, so there is nothing to collect. Its work is simply
    never planned."""
    runner = _refill_run("job1", "job2")
    runner.withdraw("job1")  # before any replan at all
    status = runner.run()
    assert [entry["id"] for entry in status["jobs"]] == ["job2"]
    assert not any(a.get("job") == "job1" for a in status["activities"])
    assert not runner.failed


# -- the arithmetic ----------------------------------------------------------


def test_a_withdrawal_carries_the_stock_levels_forward():
    """🔴 The reason R2 exists. Without adopting the baseline the scheduler carried
    forward, the next replan replays a history the departed job is no longer in and
    the stock is handed back everything that job drew."""
    runner = _refill_run("job1", "job2")
    status, when, _before = _withdraw_when_done(runner, "job1")
    assert when is not None and not runner.failed

    carried = status["inventories"]
    assert carried["at"] == when
    # 2 the run opened with, plus the 4 one refill put in, less the 2 job1 drew.
    assert carried["levels"] == {"reader": {"reagent": 4}}
    assert runner.inventories == carried

    # And the opening levels are gone from the document, which is the whole point:
    # replaying them against a history without job1 is what would give the reagent
    # back.
    assert carried["levels"] != OPENING["levels"]


def test_a_withdrawal_costs_the_run_no_extra_refill():
    """The arithmetic is right, not merely different: the run does exactly the work it
    would have done without the withdrawal."""
    plain = _refill_run("job1", "job2").run()
    runner = _refill_run("job1", "job2")
    withdrawn, _when, _before = _withdraw_when_done(runner, "job1")

    def refills(status):
        return len([a for a in status["activities"] if a["kind"] == "replenishment"])

    assert refills(withdrawn) == refills(plain)
    assert withdrawn["now"] == plain["now"]


# -- the spots ---------------------------------------------------------------


def test_a_bound_output_is_collected_and_the_backend_is_told(tmp_path):
    """The caller named the spot, so leaving says they took it from there. The backend
    has to be told, or the next delivery to that spot is refused for material nobody
    has -- which is exactly what freeing a spot is *for*."""
    runner = _bay_run(tmp_path, "output.rack_a")
    status, when, before = _withdraw_when_done(runner, "job1")
    assert when is not None and not runner.failed
    assert before.get("output.rack_a") is not None  # it was there
    assert "occupied" not in status  # and it is not declared held
    assert runner.sim.spot_state("output.rack_a") is None  # backend agrees


def test_an_unbound_output_becomes_an_occupancy(tmp_path):
    """Nobody named that spot -- the schedule chose it -- so the caller cannot have
    collected what they were never told the location of (§6.12). The backend keeps
    holding it, because it really is still there."""
    runner = _bay_run(tmp_path, None)
    status, when, before = _withdraw_when_done(runner, "job1")
    assert when is not None and not runner.failed

    resting = next(iter(before))  # the one spot job1's plate had reached
    assert {entry["spot"] for entry in status["occupied"]} == {resting}
    assert runner.sim.spot_state(resting) is not None


def test_the_occupancy_is_dated_when_the_plate_was_left_there(tmp_path):
    """Not when the withdrawal noticed. The scheduler dates a frozen hold `now`; this
    runner knows the delivery's own end, and §6.12 leaves the date free to record what
    actually happened -- so the runner's answer is the one that survives."""
    runner = _bay_run(tmp_path, None)
    status, when, _before = _withdraw_when_done(runner, "job1")
    assert status["occupied"][0]["since"] < when, (status["occupied"], when)
