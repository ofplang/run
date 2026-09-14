"""The document a rolling run carries from one replan to the next (design.md D48/D49).

The runner used to *rebuild* what it handed the scheduler: the committed history
written out as a fresh status, with everything the scheduler had decided and the
document could not re-derive -- each job's promise, its release, the moment the stock
levels are stated for, what a stopped job was holding -- caught on the way past and
copied back in. Each of those was a round trip, and each was dropped at least once.

It carries the plan now. What is tested here is the seam: exactly what the runner adds
to a plan before handing it back, and that it adds nothing else.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

import ofplang.run.runner.rolling as rolling  # noqa: E402
from ofplang.run.runner import (  # noqa: E402
    JobRequest,
    RollingRunner,
    RunnerError,
    load_document,
)
from ofplang.run.runner.echo import Echo  # noqa: E402
from ofplang.run.runner.provenance import Committed  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = Path(__file__).parent.parent / "examples"
SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")
# Three stations, and the simple workflow needs two of them -- so the third is a
# spot a run can be told to keep frozen without that costing it anything.
ROOMY_ENV = str(FIXTURES / "reroute.env.yaml")
SPARE_SPOT = "station_2.core"
OVEN_ENV = str(EXAMPLES / "stopped_job.env.yaml")
SHARED_WF = EXAMPLES / "shared_refill.workflow.yaml"

ARC = {"from": {"node": ["A"], "port": "out"}, "to": {"node": ["B"], "port": "in"}}

# Everything else in a document the runner hands over has to be the previous plan,
# character for character. `now` moves, history is stamped onto the activities, and the
# objective is restated from the roster (§4.8); the rest a plan writes about itself.
_MAY_DIFFER = {"now", "activities", "objective", "outcome", "time", "meta"}


def _watch(runner):
    """Run, recording `(document handed over, plan answered with)` for every replan."""
    original = rolling.replan
    seen: list[tuple[dict, dict]] = []

    def watching(workflows, environment, status_document, **kwargs):
        report = original(workflows, environment, status_document, **kwargs)
        if report.ok:
            seen.append((copy.deepcopy(status_document), copy.deepcopy(report.plan)))
        return report

    rolling.replan = watching
    try:
        status = runner.run()
    finally:
        rolling.replan = original
    return status, seen


def _admitting_run(workflow, at=4):
    """A run of one job that admits a second once the clock reaches `at`."""
    runner = RollingRunner(
        [JobRequest(id="job1", workflow=workflow)],
        OVEN_ENV,
        poll_interval=None,
        random_seed=0,
    )
    original = runner.sim.advance
    admitted: list[int] = []

    def advance(until):
        reached = original(until)
        if not admitted and runner.now >= at:
            admitted.append(runner.now)
            runner.admit(JobRequest(id="job2", workflow=workflow))
        return reached

    runner.sim.advance = advance
    return runner, admitted


# -- the seam: what the runner is allowed to add ------------------------------


def test_every_document_is_the_previous_plan_plus_history():
    """🔴 The invariant the whole design rests on, measured end to end.

    Each tick's document is checked against the plan that came back the tick before:
    every section the runner is not entitled to write must be identical, and every
    activity must be the plan's own entry with at most a status and times added.

    This is the check that would have caught the promises, the releases and the stock
    baseline being dropped, each of which was found by its symptom instead.
    """
    workflow = load_document(SHARED_WF)
    runner = RollingRunner(
        [JobRequest(id=job_id, workflow=workflow) for job_id in ("job1", "job2")],
        OVEN_ENV,
        poll_interval=None,
        random_seed=0,
    )
    _status, seen = _watch(runner)
    assert len(seen) > 3, "not enough replans to be measuring anything"

    for (_given, plan), (next_given, _next_plan) in zip(seen, seen[1:], strict=False):
        for section in set(plan) | set(next_given):
            if section in _MAY_DIFFER:
                continue
            assert next_given.get(section) == plan.get(section), section
        # The activities are the plan's, in the plan's order, with only the history
        # written onto them.
        assert len(next_given["activities"]) == len(plan["activities"])
        for was, is_now in zip(plan["activities"], next_given["activities"], strict=True):
            keys = ("status", "start", "end")
            assert {k: v for k, v in is_now.items() if k not in keys} == {
                k: v for k, v in was.items() if k not in keys
            }


def test_the_objective_follows_the_roster_rather_than_the_opening_count():
    """The one planning input the runner restates. With nothing declared the objective
    is a function of how many jobs there are (§4.8), and the roster is the runner's --
    so a run that starts with one job and gains a second is optimised for two from then
    on, instead of keeping the one-job objective it opened with."""
    runner, admitted = _admitting_run(load_document(SHARED_WF))
    status, seen = _watch(runner)
    assert admitted, "the second job never arrived"

    # Read off the plans: what a document *declares* is the full default for its
    # roster, and what a plan reports is the part of it this instance can tell two
    # schedules apart by (§4.8) -- here, no refill is possible, so the count drops out.
    kinds = [plan["objective"]["kind"] for _, plan in seen]
    assert kinds[0] == "makespan"  # one job: nothing for a sum of completions to weigh
    assert kinds[-1] == ["completion_time_sum", "makespan"]  # two: which finishes when
    assert status["objective"]["kind"] == ["completion_time_sum", "makespan"]


def test_an_arriving_job_is_released_when_it_arrived():
    """Not at the next replan. The scheduler's default for an arrival recognises a job
    the roster does not name -- but the runner has to name it, since a job's `interface`
    is read from its entry, so the release is the runner's to state. Measured: leaving
    it out released a job admitted at 23 at 24."""
    runner, admitted = _admitting_run(load_document(SHARED_WF))
    status = runner.run()
    entries = {entry["id"]: entry for entry in status["jobs"]}
    assert entries["job2"]["release"] == admitted[0]
    assert entries["job1"]["release"] == 0


# -- stamping -----------------------------------------------------------------


def _leg(start, end, source, destination, seq=None):
    activity = {"kind": "transport", "from_spot": source, "to_spot": destination,
                "arc": ARC}
    if seq is not None:
        activity["seq"] = seq
    return Committed(activity=activity, kind="transport", status="completed",
                     start=start, end=end)


def test_a_leg_committed_before_its_arc_grew_still_finds_its_entry():
    """🔴 A leg dispatched while its arc had only one carries no position -- that is how
    a single-hop arc is written (§6.6) -- and if the arc later grows, the plan gives that
    leg the 0 it held all along. Measured on a re-route: the record says `seq: None` and
    the next plan says `seq: 0`. Read as two identities, the leg would look as if it had
    never run, and the scheduler would be asked to plan it a second time."""
    echo = Echo()
    echo.adopt({
        "now": 0,
        "activities": [
            {"kind": "transport", "status": "pending", "from_spot": "a.core",
             "to_spot": "b.core", "seq": 0, "arc": ARC},
            {"kind": "relay", "seq": 1, "spot": "b.core", "arc": ARC},
        ],
    })
    echo.stamp([_leg(2, 3, "a.core", "b.core")], 3, set())  # no `seq` on the record
    stamped = echo.document["activities"][0]
    assert stamped["status"] == "completed"
    assert (stamped["start"], stamped["end"]) == (2, 3)


def test_a_relay_is_never_stamped():
    """The runner does not dispatch one -- a relay is a scheduling junction, not an
    operation -- so it has no record of one, and the junctions are in the plan it
    carries. A stopped job's relay is not cancelled either: it is not work."""
    echo = Echo()
    relay = {"kind": "relay", "job": "job1", "seq": 1, "spot": "b.core", "arc": ARC}
    echo.adopt({"now": 0, "activities": [relay]})
    echo.stamp([], 9, {"job1"})
    assert echo.document["activities"][0] == relay


def test_a_stopped_jobs_pending_work_is_cancelled_where_it_stands():
    """🔴 This is how the scheduler learns a job stopped (§6.2) -- and a job can stop with
    nothing of its own having failed. Without it that job's work stays pending in every
    later plan, and because what a stopped job holds is derived from its being stopped,
    its plate stops being visible too."""
    echo = Echo()
    echo.adopt({
        "now": 4,
        "activities": [
            {"kind": "processing", "job": "job1", "node": ["Bake"], "start": 6, "end": 9},
            {"kind": "processing", "job": "job2", "node": ["Bake"], "start": 6, "end": 9},
        ],
    })
    echo.stamp([], 4, {"job1"})
    done = echo.document["activities"]
    assert done[0]["status"] == "cancelled"
    assert done[0]["start"] == done[0]["end"] == 4
    assert "status" not in done[1]  # job2 is still going; its work is still to come


def test_a_single_workflows_abandoned_work_is_cancelled_too():
    """Its one job is named by the empty string and its activities carry no `job` at
    all, so the two spellings of "the only job there is" have to read alike."""
    echo = Echo()
    echo.adopt({
        "now": 0,
        "activities": [{"kind": "processing", "node": ["Target"], "start": 2, "end": 4}],
    })
    echo.stamp([], 5, {""})
    assert echo.document["activities"][0]["status"] == "cancelled"


# -- freeing a spot -----------------------------------------------------------


def test_a_declared_freeze_can_be_freed_mid_run_and_the_backend_is_told():
    """The mirror of the `occupied` a run opens with, and the only way a freeze ends
    short of the job holding it leaving the plan."""
    runner = RollingRunner(SIMPLE_WF, ROOMY_ENV, random_seed=0,
                           occupied=[{"spot": SPARE_SPOT}])
    original = runner.sim.advance
    seen_before: list[tuple] = []

    def advance(until):
        reached = original(until)
        if not seen_before and runner.now >= 1:
            # Held in the document and in the world, both, before the call.
            seen_before.append((
                [entry["spot"] for entry in runner.occupied],
                runner.sim.spot_state(SPARE_SPOT) is not None,
            ))
            runner.free_spot(SPARE_SPOT)
        return reached

    runner.sim.advance = advance
    status, seen = _watch(runner)

    assert seen_before == [([SPARE_SPOT], True)]
    assert runner.occupied == []
    # The world is told, not asked: a spot left holding something the plan believes
    # free is how a later delivery comes to be refused for a place that is full.
    assert runner.sim.spot_state(SPARE_SPOT) is None
    assert not runner.failed
    assert "occupied" not in status
    # Stated in the documents before the call and in none after it.
    stated = [bool(given.get("occupied")) for given, _ in seen]
    assert stated[0] is True and stated[-1] is False


def test_freeing_a_spot_nothing_states_is_refused():
    """🔴 Including the one a stopped job is holding. That hold follows from the job's own
    history, which the document carries, so the scheduler derives it afresh on every
    solve: there is no entry to remove, and removing one would not stop the next solve
    deriving it again. What ends it is the job leaving the plan."""
    from ofplang.schedule import derived_holds

    workflow = load_document(SHARED_WF)
    runner = RollingRunner(
        [JobRequest(id=job_id, workflow=workflow) for job_id in ("job1", "job2")],
        OVEN_ENV, poll_interval=None, random_seed=0,
    )
    runner.sim.schedule_process_failure("assay", "tray_1")
    status = runner.run()

    held = {entry["spot"] for entry in derived_holds(status)}
    assert "oven.tray_1" in held  # derived, and so not stated
    with pytest.raises(RunnerError, match="states no freeze"):
        runner.free_spot("oven.tray_1")


def test_a_run_start_freeze_is_carried_through_every_plan():
    """It cannot be derived from anything -- a person put it there -- so it is stated,
    and every plan echoes it back."""
    runner = RollingRunner(SIMPLE_WF, ROOMY_ENV, random_seed=0,
                           occupied=[{"spot": SPARE_SPOT}])
    _status, seen = _watch(runner)
    assert seen
    for given, plan in seen:
        assert given["occupied"] == [{"spot": SPARE_SPOT, "since": 0}]
        assert plan["occupied"] == [{"spot": SPARE_SPOT, "since": 0}]
