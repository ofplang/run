"""Taking a job into a running plan: `RollingRunner.admit` (SPEC §6.11).

The mirror of `withdraw`. A job is everything derived from one workflow, and that
derivation reads nothing but its arguments -- not the runner, the clock, or the other
jobs -- so admitting one is appending to a list and letting the tick find it there.
Two things then make it an *arrival* rather than an entry that merely exists: its
material appears on its spots (`_place_released`), and the next status hands the
scheduler a roster with one more entry in it.

🔴 **What makes it more than an append is the release and the promise.** The scheduler
defaults an arriving job to `now`, but it recognises an arrival as a job *absent from
the roster the document carries* -- and by the time the runner replans, the newcomer is
in the roster it builds from its own job list. So the runner sets the release itself,
and the test that matters is that a job admitted at t is released at t rather than at
0, free to have started before it existed.

The other half is priority: an arrival owes the jobs already being planned their
promises (§6.11), so the incumbent must finish exactly when it would have finished
alone. That is the guarantee the joint plan exists for, and the arrival is the case
that can silently lose it.

The scheduler is a required dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.runner import (  # noqa: E402
    JobRequest,
    RollingRunner,
    RunnerError,
    load_document,
)

FIXTURES = Path(__file__).parent / "fixtures"
LOAD_WF = FIXTURES / "interface_load.workflow.yaml"
PREFLIGHT_WF = str(FIXTURES / "preflight.workflow.yaml")
PREFLIGHT_ENV = str(FIXTURES / "preflight.env.yaml")

# A loading bay per job, one heater between them, a rack per job: each job has an entry
# spot of its own (so an arrival is not refused for a spot the incumbent is still
# holding) while both compete for the single heater (so an arrival *could* disturb the
# incumbent, which is what the priority test is about).
BAY_ENV = {
    "time": {"unit": "second"},
    "devices": [
        {"id": "loader", "spots": ["bay_a", "bay_b"]},
        {"id": "heater", "spots": ["stage"]},
        {"id": "output", "spots": ["rack_a", "rack_b"]},
    ],
    "transporters": [{"id": "arm"}],
    "transports": [
        {"transporter": "arm", "from": "loader.bay_a", "to": "heater.stage", "duration": 2},
        {"transporter": "arm", "from": "loader.bay_b", "to": "heater.stage", "duration": 2},
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


def _bay_request(job_id: str, bay: str, release: int = 0) -> JobRequest:
    """One `interface_load` job loading from `loader.bay_<bay>` and delivering to
    `output.rack_<bay>`."""
    return JobRequest(
        id=job_id,
        workflow=copy.deepcopy(load_document(LOAD_WF)),
        boundary={
            "boundary": {
                "inputs": {"sample": {"spot": f"loader.bay_{bay}"}},
                "outputs": {"result": {"spot": f"output.rack_{bay}"}},
            }
        },
        release=release,
    )


def _bay_run(*specs, **kwargs) -> RollingRunner:
    """A run of the jobs named by `(id, bay)` pairs, against `BAY_ENV`."""
    return RollingRunner(
        [_bay_request(job_id, bay) for job_id, bay in specs],
        copy.deepcopy(BAY_ENV),
        poll_interval=None,
        random_seed=0,
        **kwargs,
    )


def _preflight_run(*specs) -> RollingRunner:
    """A run of the device-less `preflight` workflow, one job per `(id, limit)` pair.

    `limit` is a run-phase boundary input with `requires: limit >= 0`, so a negative
    one violates a precondition that is checked before anything is dispatched (D37).
    """
    return RollingRunner(
        [
            JobRequest(
                id=job_id,
                workflow=load_document(PREFLIGHT_WF),
                boundary={
                    "boundary": {
                        "inputs": {"seed": {"view": 10}, "limit": {"view": limit}}
                    }
                },
            )
            for job_id, limit in specs
        ],
        PREFLIGHT_ENV,
        poll_interval=None,
        random_seed=0,
    )


def _admit_at(runner, request, after: int):
    """Run, calling `admit(request)` at the first tick whose clock has reached `after`.

    Called from the clock rather than by counting ticks, and *before* the advance, so
    `runner.now` is the runner's current time -- the same value `admit` reads to derive
    the release. That is what a caller doing this between ticks would see.

    Returns the final status, the time the call was made, and a snapshot of the
    backend's spots just before it.
    """
    original = runner.sim.advance
    marks: list[int] = []
    before: dict = {}

    def advance(until):
        if not marks and runner.now >= after:
            marks.append(runner.now)
            before.update(runner.sim.spot_state())
            runner.admit(request)
        return original(until)

    runner.sim.advance = advance  # type: ignore[method-assign]
    return runner.run(), (marks[0] if marks else None), before


def _entry(status, job_id) -> dict:
    """One job's roster entry of a status (§6.11)."""
    return next(e for e in status["jobs"] if e["id"] == job_id)


def _activities(status, job_id) -> list[dict]:
    return [a for a in status["activities"] if a.get("job") == job_id]


# -- what it refuses ---------------------------------------------------------


def test_admit_refuses_a_single_workflow():
    """There is no roster to join: the one workflow is the whole run, and it carries no
    job identity for an arrival to be told apart by."""
    runner = RollingRunner(
        str(FIXTURES / "simple.workflow.yaml"),
        str(FIXTURES / "simple.env.yaml"),
        random_seed=0,
    )
    with pytest.raises(RunnerError, match="needs a run of named jobs"):
        runner.admit(_bay_request("late", "b"))


def test_admit_refuses_a_job_with_no_id():
    runner = _bay_run(("job1", "a"))
    with pytest.raises(RunnerError, match="needs an id"):
        runner.admit(_bay_request("", "b"))


def test_admit_refuses_an_id_the_run_already_plans():
    runner = _bay_run(("job1", "a"))
    with pytest.raises(RunnerError, match="already plans it"):
        runner.admit(_bay_request("job1", "b"))


def test_admit_refuses_an_id_that_has_already_left():
    """🔴 `_history` filters a departed job's records out of every status *by id*, so a
    newcomer reusing one would have its own history dropped along with them -- silently,
    and only from the scheduler's view of the world."""
    runner = _bay_run(("job1", "a"), ("job2", "b"))
    runner.withdraw("job1")  # before any replan: nothing of it is in the laboratory
    runner.run()
    assert not runner.failed
    with pytest.raises(RunnerError, match="already left this run"):
        runner.admit(_bay_request("job1", "a"))


def test_admit_refuses_a_run_whose_every_job_has_stopped():
    """Nothing more is dispatched, so an arriving job would sit in the roster and never
    run. `limit = -1` violates the entry precondition at run start (D37)."""
    runner = _preflight_run(("job1", -1))
    runner.run()
    assert runner.failed
    with pytest.raises(RunnerError, match="every job of this run has stopped"):
        runner.admit(_bay_request("late", "b"))


def test_admit_refuses_a_boundary_that_states_starting_inventories():
    """The stock is the laboratory's, and the levels are the ones the run *began* with
    (§6.10). By the time a job arrives they have been drawn on and refilled, so a
    newcomer restating them would be describing a moment that has passed."""
    runner = _bay_run(("job1", "a"))
    request = _bay_request("job2", "b")
    request.boundary["boundary"]["inventories"] = {"levels": {"reader": {"reagent": 6}}}
    with pytest.raises(RunnerError, match="not a job's to state"):
        runner.admit(request)


def test_admit_refuses_an_entry_spot_the_run_is_already_holding():
    """The one placement conflict the runner can see for itself: it never asks the
    backend what is on a spot, so what it can check is what it accounts for -- the
    spots it declares occupied (§6.12)."""
    runner = _bay_run(("job1", "a"), occupied=[{"spot": "loader.bay_b"}])
    with pytest.raises(RunnerError, match="already holding something on"):
        runner.admit(_bay_request("job2", "b"))


# -- arriving ----------------------------------------------------------------


def test_an_admitted_job_runs_to_completion():
    """End to end: it joins the roster, its material appears, it is planned against the
    incumbent's machines and it delivers."""
    runner = _bay_run(("job1", "a"))
    status, when, _before = _admit_at(runner, _bay_request("job2", "b"), after=1)
    assert when is not None and not runner.failed

    # It joined at the end of the roster, which is the end of the priority order.
    assert [e["id"] for e in status["jobs"]] == ["job1", "job2"]
    assert _activities(status, "job2")
    assert all(a["status"] == "completed" for a in status["activities"])
    late = next(job for job in runner.jobs if job.id == "job2")
    assert "result" in late.outputs


def test_a_job_admitted_before_the_run_is_simply_one_of_its_jobs():
    """No special case anywhere: admitting before the loop starts is the same thing as
    having listed the job, which is what makes the seam trustworthy mid-run."""
    listed = _bay_run(("job1", "a"), ("job2", "b")).run()
    runner = _bay_run(("job1", "a"))
    runner.admit(_bay_request("job2", "b"))
    admitted = runner.run()

    assert not runner.failed
    assert admitted["now"] == listed["now"]
    assert [e["id"] for e in admitted["jobs"]] == [e["id"] for e in listed["jobs"]]


def test_an_arriving_job_is_released_when_it_arrives():
    """🔴 The reason `admit` sets the release at all. A job admitted with none stated
    would otherwise be released at 0 -- free, as far as the document says, to have
    started before it existed -- because the scheduler's own default fires only for a
    job absent from the roster, and this one is in it by the time we replan."""
    runner = _bay_run(("job1", "a"))
    status, when, _before = _admit_at(runner, _bay_request("job2", "b"), after=1)
    assert when is not None and when > 0

    assert _entry(status, "job2")["release"] == when
    assert all(a["start"] >= when for a in _activities(status, "job2"))


def test_a_release_stated_in_the_past_is_raised_to_now():
    """A job that did not exist cannot have been released. Raised rather than refused:
    a caller naming a time already gone means "as soon as possible", which is what
    `now` is."""
    runner = _bay_run(("job1", "a"))
    status, when, _before = _admit_at(runner, _bay_request("job2", "b", release=1), after=2)
    assert when is not None and when > 1

    assert _entry(status, "job2")["release"] == when


def test_a_release_stated_in_the_future_is_honoured():
    """Arriving and being released are two different moments: a job can be handed over
    now and be unable to start for an hour."""
    # A fixed time, far enough out to be past the incumbent's own work: the run has to
    # stay alive waiting for it rather than finishing when the incumbent does.
    runner = _bay_run(("job1", "a"))
    status, when, _before = _admit_at(runner, _bay_request("job2", "b", release=40), after=1)
    assert when is not None and when < 40 and not runner.failed

    assert _entry(status, "job2")["release"] == 40
    assert all(a["start"] >= 40 for a in _activities(status, "job2"))


def test_its_material_appears_when_it_arrives_and_not_before():
    """Entry material is *there*, given, from the job's release (§6.8) -- so the spot is
    empty right up to the moment it arrives, and the scheduler is never planning around
    a place it believes free while it is full."""
    runner = _bay_run(("job1", "a"))
    status, when, before = _admit_at(runner, _bay_request("job2", "b"), after=1)
    assert when is not None

    assert before.get("loader.bay_b") is None      # nothing there when it was admitted
    assert before.get("loader.bay_a") is None      # and the incumbent's was long gone
    # The plate was collected from the bay by job2's boundary load, which the plan
    # cannot start before the release.
    load = [a for a in _activities(status, "job2") if a["kind"] == "transport"]
    assert load and min(a["start"] for a in load) >= when


# -- priority ----------------------------------------------------------------


def test_an_incumbent_is_not_disturbed_by_an_arrival():
    """🔴 The guarantee planning jobs together exists for (§6.11): an earlier job is not
    disturbed by a later one. The arrival competes for the same single heater, so a
    scheduler free to reshuffle would happily delay the incumbent -- and its promise
    (`bound`), taken back from each plan, is what stops it.

    Read through the incumbent's actual completion rather than through the `bound` its
    final roster entry carries: that entry goes on being re-derived on every replan
    after the job has finished, which is not what this test is about.
    """
    alone = _bay_run(("job1", "a"))
    alone_status = alone.run()

    together = _bay_run(("job1", "a"))
    status, when, _before = _admit_at(together, _bay_request("job2", "b"), after=1)
    assert when is not None and not together.failed

    def completion(doc):
        return max(a["end"] for a in _activities(doc, "job1"))

    assert completion(status) == completion(alone_status)
    # And the arrival really did have work to fit in around it.
    assert status["now"] > alone_status["now"]
    assert _activities(status, "job2")


def test_admit_refuses_a_job_whose_own_precondition_is_violated():
    """An arriving job is held to the checks run start makes, at the moment it is
    admitted rather than at a run start it missed. A violation **refuses the call**:
    the caller hears why and can offer the job again with a boundary that holds.

    🔴 Refusing is also the only answer that leaves the run intact. A job stopped
    before it has any history has no terminal activity to tell the scheduler it
    stopped, so its work would stay pending in every later plan while this runner
    declined to dispatch it -- and the run would never finish.
    """
    runner = _preflight_run(("job1", 3))
    bad = JobRequest(
        id="job2",
        workflow=load_document(PREFLIGHT_WF),
        boundary={"boundary": {"inputs": {"seed": {"view": 10}, "limit": {"view": -1}}}},
    )
    with pytest.raises(RunnerError, match="does not satisfy its own preconditions"):
        runner.admit(bad)

    # The run is as it was: the candidate never joined, and nothing is recorded
    # against a run that nothing has happened to.
    assert [job.id for job in runner.jobs] == ["job1"]
    assert not runner.failed and runner.failure is None
    status = runner.run()
    assert not runner.failed
    assert runner.jobs[0].outputs == {"ok": 14}  # Slow.d = 10 + 1, ok = 11 + 3
    assert not _activities(status, "job2")
