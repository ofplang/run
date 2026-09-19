"""A backend that refuses a dispatch or a placement (`BackendRefused`).

There are two ways a backend can say something did not work, and until now only one
of them was survivable. An operation that fails *once running* is reported through
`state(handle)`, and the runner records it `failed` and stops its job. An operation
the backend **refuses to start** was an exception, and it escaped `run()` -- taking
with it the status of every other job in the laboratory, work that had completed
perfectly well. That is the opposite of what planning jobs together is for.

So a refusal now lands where the other one lands: the activity `failed`, the job
stopped, the run's status returned. What is pinned here:

  - the isolation: the refused job stops, the others finish, and there is a status;
  - 🔴 what the refusal stranded is **derived**, not declared -- a failed transport
    claims *both* ends, because nobody can say which one still holds the Object;
  - a backend of its own making says it the same way (`BackendRefused` is the
    contract, not `SimulatorError`);
  - 🔴 and a *broken plan* still propagates: an unknown reference is a fault in the
    program, and filing one as a laboratory mishap would hide it in a document;
  - placement, which has no activity to fail, stops its job at run start and
    mid-run alike -- and neither hangs the run.

The scheduler is a required dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.schedule import derived_holds  # noqa: E402

from ofplang.run.backend import BackendRefused  # noqa: E402
from ofplang.run.runner import JobRequest, RollingRunner, load_document  # noqa: E402
from ofplang.run.simulator import UnknownReference, VirtualTimeSimulator  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = Path(__file__).parent.parent / "examples"
OVEN_WF = EXAMPLES / "shared_refill.workflow.yaml"
OVEN_ENV = str(EXAMPLES / "stopped_job.env.yaml")
# The same workflow against the laboratory that *does* declare a stock, so the
# scheduler plans the one refill the pair of them needs (see shared_refill.run.yaml).
REFILL_ENV = str(EXAMPLES / "shared_refill.env.yaml")
LOAD_WF = FIXTURES / "interface_load.workflow.yaml"

# A loading bay and a rack per job, one heater between them: each job's entry material
# has a spot of its own, which is what lets one job's placement be refused while the
# other's succeeds.
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


def _oven_run(*ids: str, loaded: str | None = None, **kwargs):
    """Plates through a two-tray oven, with `loaded` already holding something nobody
    declared -- material a person left there and told the run nothing about.

    That is the honest shape of this failure: the document cannot be wrong about a
    spot it never mentions, so the disagreement can only surface as the backend
    refusing the move the plan makes onto it.
    """
    workflow = load_document(OVEN_WF)
    runner = RollingRunner(
        [JobRequest(id=job_id, workflow=workflow) for job_id in ids],
        OVEN_ENV,
        poll_interval=None,
        random_seed=0,
        **kwargs,
    )
    if loaded is not None:
        runner.sim.place(loaded)
    return runner.run(), runner


def _bay_request(job_id: str, bay: str, release: int = 0) -> JobRequest:
    """One `interface_load` job loading from `loader.bay_<bay>`, delivering to
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
    return RollingRunner(
        [_bay_request(job_id, bay, release) for job_id, bay, release in specs],
        copy.deepcopy(BAY_ENV),
        poll_interval=None,
        random_seed=0,
        # Placement stops a job without any operation having run, which is the shape
        # that used to leave a run with nothing to do and nothing to wait for. A low
        # cap turns that into a failure here rather than a hang.
        max_ticks=200,
        **kwargs,
    )


def _of(status: dict, job: str) -> list[dict]:
    return [a for a in status["activities"] if a.get("job") == job]


class _OwnRefusal(BackendRefused):
    """What a backend of someone else's making raises: it declares `BackendRefused`
    and nothing else, which is the whole of the contract."""


def _refusing(exc: Exception, on: str = "dispatch_transport", to_spot: str | None = None):
    """A `backend_factory` whose backend refuses the first `on` call with `exc`.

    Deliberately not a `SimulatorError`: what the runner acts on is what the
    exception *means*, declared by the class it derives from, not which backend
    raised it.

    `to_spot` narrows a transport refusal to one destination. Which move is refused
    decides how much of the laboratory the refusal strands -- a failed move holds
    both of its ends -- so a test about isolation has to refuse one that leaves the
    other jobs somewhere to go, exactly as a real mishap on one tray would.
    """

    class _Refusing(VirtualTimeSimulator):
        def __init__(self, environment):
            super().__init__(environment)
            self._refused = False

        def _maybe_refuse(self):
            if not self._refused:
                self._refused = True
                raise exc

        def dispatch_transport(self, *args, **kwargs):
            destination = args[2] if len(args) > 2 else kwargs.get("to_spot")
            if on == "dispatch_transport" and to_spot in (None, destination):
                self._maybe_refuse()
            return super().dispatch_transport(*args, **kwargs)

        def dispatch_replenishment(self, *args, **kwargs):
            if on == "dispatch_replenishment":
                self._maybe_refuse()
            return super().dispatch_replenishment(*args, **kwargs)

        def place(self, spot):
            if on == "place":
                self._maybe_refuse()
            return super().place(spot)

    return _Refusing


# -- the isolation ------------------------------------------------------------


def test_a_refused_dispatch_no_longer_takes_the_whole_run_down():
    """🔴 The point of the change. Before it, `SpotConflict` came out of `run()` and
    the two jobs that had done nothing wrong lost their status along with it."""
    status, runner = _oven_run("job1", "job2", "job3", loaded="oven.tray_1")
    assert runner.failed  # the run says something went wrong ...
    assert [job.id for job in runner.jobs if job.stopped] == ["job1"]
    # ... and the other two ran their whole workflow, which is what there is to lose.
    for job in ("job2", "job3"):
        assert {a["status"] for a in _of(status, job)} == {"completed"}


def test_the_refusal_is_recorded_as_the_activity_failing():
    """The translation: a physical fact ("that tray is full") in the planning layer's
    only word for it. Zero-length, because nothing ran."""
    status, runner = _oven_run("job1", "job2", "job3", loaded="oven.tray_1")
    failed = [a for a in status["activities"] if a["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["kind"] == "transport" and failed[0]["to_spot"] == "oven.tray_1"
    assert failed[0]["start"] == failed[0]["end"]
    # The work behind it is abandoned, not silently dropped.
    assert [a["status"] for a in _of(status, "job1")][-1] == "cancelled"

    reason = runner.jobs[0].failure
    assert reason is not None and reason.kind == "backend_refused_dispatch"
    # The detail carries the backend's own words, which name the spot.
    assert "SpotConflict" in reason.detail and "oven.tray_1" in reason.detail
    # Only that job has a reason; the run keeps the first of the run, which is its.
    assert [job.failure for job in runner.jobs[1:]] == [None, None]
    assert runner.failure is reason


def test_a_refused_transport_holds_both_ends():
    """🔴 Nothing declares this. A failed transport applied no material effect, so the
    Object is at one of its two ends and nothing says which -- the scheduler derives
    that from the document on every solve (§6.12), and over-claiming costs a slower
    plan while under-claiming puts a plate where a plate already is.

    The destination is held for a second reason the document cannot see: someone
    else's material is on it. That it comes out held anyway is why the run survives --
    no later job is planned onto the tray.
    """
    status, _runner = _oven_run("job1", "job2", "job3", loaded="oven.tray_1")
    failed = [a for a in status["activities"] if a["status"] == "failed"][0]
    assert "occupied" not in status  # said by nobody
    assert {entry["spot"] for entry in derived_holds(status)} == {
        failed["from_spot"],
        failed["to_spot"],
    }


def test_the_stop_policy_still_stops_everything():
    status, runner = _oven_run(
        "job1", "job2", "job3", loaded="oven.tray_1", on_job_failure="stop"
    )
    assert all(job.stopped for job in runner.jobs)
    assert "completed" not in {a["status"] for a in _of(status, "job3")}


def test_a_single_workflow_run_reports_it_as_any_other_failure():
    """A run of one workflow is a run of one job: it stops, and it still returns the
    status it always returned for a failure -- where before it raised."""
    status, runner = _oven_run("only", loaded="oven.tray_1")
    assert runner.failed and runner.jobs[0].stopped
    assert [a["status"] for a in status["activities"]].count("failed") == 1


# -- the contract, not the simulator ------------------------------------------


def test_a_backend_of_its_own_making_says_it_the_same_way():
    """`BackendRefused` is the contract. A backend that raises it -- and nothing of
    the simulator's taxonomy -- is understood, which is the whole point of putting the
    class in `..backend` rather than catching the simulator's own errors."""
    runner = RollingRunner(
        [
            JobRequest(id=job_id, workflow=load_document(OVEN_WF))
            for job_id in ("job1", "job2", "job3")
        ],
        OVEN_ENV,
        poll_interval=None,
        random_seed=0,
        backend_factory=_refusing(
            _OwnRefusal("the arm will not take it"), to_spot="oven.tray_1"
        ),
    )
    status = runner.run()
    assert runner.failed
    assert [job.id for job in runner.jobs if job.stopped] == ["job1"]
    reason = runner.jobs[0].failure
    assert reason is not None and reason.kind == "backend_refused_dispatch"
    assert "_OwnRefusal" in reason.detail and "will not take it" in reason.detail
    # The others are untouched, and the status is there to read.
    for job in ("job2", "job3"):
        assert {a["status"] for a in _of(status, job)} == {"completed"}


def test_a_broken_plan_still_propagates():
    """🔴 The other half of the split. An unknown reference says the *plan* or the
    runner is wrong, not the laboratory -- translating it into a failed activity would
    file a bug as a mishap and leave it sitting quietly in a status document."""
    runner = RollingRunner(
        [JobRequest(id=job_id, workflow=load_document(OVEN_WF)) for job_id in ("a", "b")],
        OVEN_ENV,
        poll_interval=None,
        random_seed=0,
        backend_factory=_refusing(UnknownReference("no such transporter")),
    )
    with pytest.raises(UnknownReference):
        runner.run()


# -- placement, which has no activity to fail ---------------------------------


def test_a_refused_placement_at_run_start_stops_that_job_alone():
    """Entry material with nowhere to go. No activity is marked failed because none
    ran -- the same shape as a job whose whole-workflow `requires` is violated, which
    also stops a job at run start rather than raising."""
    runner = _bay_run(("job1", "a", 0), ("job2", "b", 0))
    runner.sim.place("loader.bay_b")  # something is on the bay job2 loads from
    status = runner.run()

    assert runner.failed
    assert [job.id for job in runner.jobs if job.stopped] == ["job2"]
    reason = runner.jobs[1].failure
    assert reason is not None and reason.kind == "backend_refused_placement"
    assert "loader.bay_b" in reason.detail
    # Nothing of job2's ran, and job1 is untouched by any of it.
    assert "completed" not in {a["status"] for a in _of(status, "job2")}
    assert {a["status"] for a in _of(status, "job1")} == {"completed"}


def test_a_refused_placement_mid_run_stops_that_job_alone():
    """A job released later has its material appear later, so the refusal happens with
    the run well under way -- and by then the plan does carry that job's work, which is
    what has to end up `cancelled` for the scheduler to learn it stopped (§6.2)."""
    runner = _bay_run(("job1", "a", 0), ("job2", "b", 40))
    runner.sim.place("loader.bay_b")
    status = runner.run()

    assert runner.failed and runner.now >= 40  # it got there, rather than failing early
    assert [job.id for job in runner.jobs if job.stopped] == ["job2"]
    assert runner.jobs[1].failure.kind == "backend_refused_placement"
    assert {a["status"] for a in _of(status, "job1")} == {"completed"}
    assert {a["status"] for a in _of(status, "job2")} <= {"cancelled"}


# -- where the isolation ends -------------------------------------------------


def test_a_refusal_the_others_cannot_be_planned_around_stops_the_run():
    """🔴 The limit of isolating one job, and not a new one: what a stopped job is
    holding is declared so nothing is sent onto it, and that is exactly what can leave
    the rest nowhere to go. Two jobs sharing a two-tray oven, with the refusal on the
    first move: both ends of it are held, the remaining job cannot be planned around
    them, and a replan nothing can be planned from stops every job (it cannot be
    attributed to one).

    The oven has **one** tray, so that is a property of the laboratory. With two, the
    other job takes the tray that is free, and whether the run corners itself comes
    down to which of two interchangeable bays each job was handed -- the scheduler's
    choice to make, and not one a test may read.

    What matters is that this still ends in a *status* rather than an exception -- the
    run says what happened to everyone, which is what was lost before.
    """
    runner = RollingRunner(
        [JobRequest(id=job_id, workflow=load_document(OVEN_WF)) for job_id in ("a", "b")],
        str(FIXTURES / "one_tray_oven.env.yaml"),
        poll_interval=None,
        random_seed=0,
        backend_factory=_refusing(_OwnRefusal("the arm will not take it")),
    )
    status = runner.run()
    assert all(job.stopped for job in runner.jobs)
    # The run's reason is the refusal (first failure wins), and the second job stopped
    # with none of its own: nothing of *its* went wrong.
    assert runner.failure.kind == "backend_refused_dispatch"
    assert runner.jobs[1].failure is None
    assert [a["status"] for a in _of(status, "b")] == ["cancelled"] * 5


def test_a_refused_refill_stops_the_run():
    """A refill belongs to no job (`_job_of` answers None), so its refusal cannot be
    attributed and the stock it was topping up is still short -- the rule a refill's
    failure has always followed.

    It is also the one failure whose subject has no node and no spots: a refill is a
    visit, and naming it by who filled what is what makes the reason readable (it used
    to render as `None -> None`).
    """
    runner = RollingRunner(
        [
            JobRequest(id=job_id, workflow=load_document(OVEN_WF))
            for job_id in ("morning", "afternoon")
        ],
        REFILL_ENV,
        poll_interval=None,
        random_seed=0,
        inventories={"levels": {"reader": {"reagent": 2}}},
        backend_factory=_refusing(
            _OwnRefusal("no reagent in the cart"), on="dispatch_replenishment"
        ),
    )
    status = runner.run()
    assert all(job.stopped for job in runner.jobs)
    assert runner.failure.kind == "backend_refused_dispatch"
    assert runner.failure.subject == "dispenser -> reader"
    refills = [a for a in status["activities"] if a["kind"] == "replenishment"]
    assert [a["status"] for a in refills] == ["failed"]
