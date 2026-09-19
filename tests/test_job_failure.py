"""Tests for isolating one job's failure from the rest of the run (SPEC §6.11 / §6.2).

A run of one workflow stops when anything fails, and always has. A run of a
*laboratory* should not: a cracked plate in one job is not a reason to abandon the
other two. What is pinned here:

  - the isolation itself: the failing job stops, the others finish;
  - 🔴 what the stopped job left behind is declared (`occupied`, §6.12) -- without
    which the scheduler believes the spot free and plans another job's material
    straight onto it;
  - ownership: a spot this job merely *used* earlier, and another job's material now
    sits on, is not claimed as this job's residue;
  - 🔴 and that the residue is **derived**, not asked: the runner replays what its own
    operations did rather than reading the backend's occupancy, and the two agree;
  - `--on-job-failure stop` stops everything, and stops it *consistently* -- no job
    is left looking finished while its work was cancelled;
  - and that a single workflow behaves exactly as it always did.

The scheduler is a required dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.schedule import derived_holds  # noqa: E402

from ofplang.run.cli import EXIT_FAILED, main  # noqa: E402
from ofplang.run.runner import (  # noqa: E402
    JobRequest,
    RollingRunner,
    RunnerError,
    load_document,
)

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = Path(__file__).parent.parent / "examples"
WF = EXAMPLES / "shared_refill.workflow.yaml"
OVEN_ENV = str(EXAMPLES / "stopped_job.env.yaml")
SIMPLE_WF = str(FIXTURES / "simple.workflow.yaml")
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")


def _oven_run(*ids: str, fail_tray: str | None = "tray_1", **kwargs):
    """Three plates through a two-tray oven, every assay on `fail_tray` failing.

    Self-limiting: the first job to reach that tray fails there and its plate stays,
    so the tray is declared occupied and no later job is ever sent to it.
    """
    workflow = load_document(WF)
    runner = RollingRunner(
        [JobRequest(id=job_id, workflow=workflow) for job_id in ids],
        OVEN_ENV,
        poll_interval=None,
        random_seed=0,
        **kwargs,
    )
    if fail_tray is not None:
        runner.sim.schedule_process_failure("assay", fail_tray)
    return runner.run(), runner


def _of(status: dict, job: str) -> list[dict]:
    return [a for a in status["activities"] if a.get("job") == job]


def _spots(activity: dict) -> set[str]:
    """Every spot an activity touches: a processing's inputs and outputs, a transport's
    two ends. What it could be holding, in other words."""
    spots = set((activity.get("input_spots") or {}).values())
    spots |= set((activity.get("output_spots") or {}).values())
    spots |= {activity[k] for k in ("from_spot", "to_spot") if activity.get(k)}
    return spots


# -- the isolation -----------------------------------------------------------


def test_one_jobs_failure_does_not_stop_the_others():
    status, runner = _oven_run("job1", "job2", "job3")
    assert runner.failed  # something failed, and the run says so
    stopped = [job.id for job in runner.jobs if job.stopped]
    assert stopped == ["job1"]
    # The other two ran their whole workflow, nothing cancelled.
    for job in ("job2", "job3"):
        assert {a["status"] for a in _of(status, job)} == {"completed"}
    # And the stopped one's remaining work is reported, not silently dropped.
    assert [a["status"] for a in _of(status, "job1")] == [
        "completed", "completed", "failed", "cancelled", "cancelled",
    ]


def test_the_stopped_job_carries_the_reason_and_the_others_do_not():
    _status, runner = _oven_run("job1", "job2", "job3")
    reasons = {job.id: job.failure for job in runner.jobs}
    assert reasons["job1"] is not None
    assert reasons["job1"].kind == "activity_failed"
    assert reasons["job2"] is None and reasons["job3"] is None
    # The run keeps the first failure of the run, which here is that job's.
    assert runner.failure is reasons["job1"]


def test_a_stopped_job_is_left_out_of_the_result_boundary():
    _status, runner = _oven_run("job1", "job2", "job3")
    assert set(runner.result_boundary["jobs"]) == {"job2", "job3"}


def test_a_stopped_job_promises_nothing():
    """A bound it can never meet is withdrawn from the roster, not restated."""
    status, _runner = _oven_run("job1", "job2", "job3")
    entries = {entry["id"]: entry for entry in status["jobs"]}
    assert "bound" not in entries["job1"]
    assert entries["job2"]["bound"] is not None


# -- what it left behind ------------------------------------------------------


def test_the_stopped_jobs_material_is_held_though_nothing_declares_it():
    """🔴 Not bookkeeping: the scheduler models occupancy through activity intervals,
    and the failed assay's interval has ended, so unless that tray is held the next
    job's plate is carried onto the cracked one.

    It is **derived, not declared**. Every input to the derivation is in the document
    -- which activity failed, what it touched, which jobs a terminal status stopped --
    so the scheduler works it out on every solve and the document says nothing. Stating
    it as well would say the same hold twice, which is refused
    (`occupied_already_derived`), and would leave the run's own report unable to be
    planned from again.
    """
    status, _runner = _oven_run("job1", "job2", "job3")
    failed = [a for a in status["activities"] if a["status"] == "failed"]
    assert len(failed) == 1
    # Nothing declared -- and the tray is held all the same.
    assert "occupied" not in status
    assert derived_holds(status) == [
        # Dated when the plate was actually left there, not when we noticed. No `job`:
        # the section says a spot is held, not by whom (§6.12).
        {"spot": "oven.tray_1", "since": failed[0]["end"]}
    ]
    # ... and nothing was ever planned onto it after the failure.
    later = [
        a for a in status["activities"]
        if a.get("status") == "completed" and a["start"] >= 23
    ]
    assert later and all("tray_1" not in str(a.get("to_spot") or "") for a in later)


def test_the_residue_is_dated_when_the_plate_was_left_there():
    """The truthful moment, not the moment we noticed. A plan holds the spot from
    `max(since, now)` whatever this says (schedule SPEC §6.12), so the date is free to
    record what actually happened -- and it is what a withdrawal writes down, by which
    time the history that would have said so is gone."""
    status, runner = _oven_run("job1", "job2", "job3")
    entry = derived_holds(status)[0]
    failed = [a for a in status["activities"] if a["status"] == "failed"]
    assert entry["since"] == failed[0]["end"]
    assert entry["since"] < status["now"] == runner.now  # long before the run ended


def test_a_spot_another_job_now_holds_is_not_claimed_as_this_jobs_residue():
    """🔴 Ownership, not acquaintance. Two jobs of one workflow use the same bench slot
    one after the other; claiming every spot a job ever touched takes the plate the
    next job has just made -- which was measured to make that job unplannable and take
    the whole run down with it."""
    status, runner = _oven_run("job1", "job2", "job3")
    held = {entry["spot"] for entry in derived_holds(status)}
    assert "bench.slot_a" not in held  # used by job2 and then job3, held by neither now
    # The run survived, which is the symptom the wrong rule produced.
    assert not [job for job in runner.jobs if job.id != "job1" and job.stopped]


def test_a_spot_the_job_itself_emptied_is_not_claimed():
    """The case ownership alone gets wrong. A job owns a spot it *emptied* -- it carried
    its own plate away, or collected its own entry material -- and nothing has touched it
    since, so it is still the last toucher. Only replaying what its own operations did
    says the spot is free; without that every job would claim its bench slot for the rest
    of the run."""
    status, _runner = _oven_run("job1", "job2", "job3")
    held = {entry["spot"] for entry in derived_holds(status)}
    moved = [
        a for a in status["activities"]
        if a.get("job") == "job1" and a["kind"] == "transport" and a["status"] == "completed"
    ]
    assert moved, "the fixture no longer has job1 move its own plate"
    # It left where it departed from, and is at where it arrived.
    assert all(a["from_spot"] not in held for a in moved)
    assert any(a["to_spot"] in held for a in moved)


# -- the policy ---------------------------------------------------------------


def test_stop_abandons_every_job_and_says_so():
    status, runner = _oven_run("job1", "job2", "job3", on_job_failure="stop")
    assert all(job.stopped for job in runner.jobs)
    # 🔴 Consistently: no job is left looking finished while its work was cancelled,
    # so none is checked for delivery or echoed into the result boundary.
    assert runner.result_boundary == {"jobs": {}}
    assert {a["status"] for a in _of(status, "job3")} == {"completed", "cancelled"}
    # Every job's material is held, not just the one that failed -- read as the spots,
    # since neither the section nor the derivation says whose each one is (§6.12).
    assert len(derived_holds(status)) == 3


def test_stop_ends_sooner_than_continue():
    stopped, _ = _oven_run("job1", "job2", "job3", on_job_failure="stop")
    carried_on, _ = _oven_run("job1", "job2", "job3")
    assert stopped["now"] < carried_on["now"]


def test_an_unknown_policy_is_refused():
    with pytest.raises(RunnerError, match="on_job_failure"):
        RollingRunner(SIMPLE_WF, SIMPLE_ENV, on_job_failure="carry on")


# -- the single-workflow run is untouched -------------------------------------


def test_a_single_workflow_still_stops_whole_and_declares_nothing():
    """One workflow is one job, so it stops the run either way -- and its failure ends
    the run, so there is nothing left to plan around and no `occupied` to write."""
    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0)
    runner.sim.schedule_process_failure("target", "m0")
    status = runner.run()
    assert runner.failed and runner._stopping
    assert "occupied" not in status and "jobs" not in status
    assert [a["status"] for a in status["activities"]] == [
        "completed", "completed", "failed",
    ]


@pytest.mark.parametrize("policy", ["continue", "stop"])
def test_the_policy_makes_no_difference_to_a_single_workflow(policy):
    runner = RollingRunner(SIMPLE_WF, SIMPLE_ENV, random_seed=0, on_job_failure=policy)
    runner.sim.schedule_process_failure("target", "m0")
    status = runner.run()
    assert runner.failed
    assert [a["status"] for a in status["activities"]] == [
        "completed", "completed", "failed",
    ]


# -- the CLI ------------------------------------------------------------------


def test_cli_names_the_job_that_stopped(tmp_path, capsys):
    """Several jobs commonly run the same workflow, so a reason with no job on it
    sends the reader looking in the wrong place."""
    import yaml

    contract_wf = str(FIXTURES / "contract.workflow.yaml")
    run_doc = tmp_path / "run.yaml"
    run_doc.write_text(
        yaml.safe_dump(
            {
                "jobs": [
                    # `requires: inputs.raw.view >= 0` -- the second job breaks it, so
                    # it stops at run start while the first runs to completion.
                    {"id": "good", "workflow": contract_wf,
                     "boundary": {"boundary": {"inputs": {"raw": {"view": 72}}}}},
                    {"id": "bad", "workflow": contract_wf,
                     "boundary": {"boundary": {"inputs": {"raw": {"view": -5}}}}},
                ]
            }
        ),
        encoding="utf-8",
    )
    code = main(
        ["run", "--jobs", str(run_doc), "--env", str(FIXTURES / "contract.env.yaml"),
         "-o", str(tmp_path / "s.yaml")]
    )
    assert code == EXIT_FAILED
    err = capsys.readouterr().err
    assert "ofp-run: job 'bad' failed: contract_requires" in err
    assert "job 'good'" not in err
    # And `good` really did run: a stopped job at run start must not take the run
    # with it, which is the whole point of naming them separately.
    status = yaml.safe_load((tmp_path / "s.yaml").read_text(encoding="utf-8"))
    done = [a for a in status["activities"] if a.get("job") == "good"]
    assert done and all(a["status"] == "completed" for a in done)
    # `bad`'s work is reported rather than silently dropped: the invocation whose
    # precondition it broke `failed`, and what would have followed it `cancelled`.
    abandoned = [a["status"] for a in status["activities"] if a.get("job") == "bad"]
    assert abandoned == ["failed", "cancelled"]
    # The subject names the job too, for the same reason the report line does.
    assert "bad:Score" in err


def test_cli_reports_the_single_workflow_failure_as_it_always_did(capsys):
    code = main(["run", SIMPLE_WF, "--env", str(FIXTURES / "simple_no_target.env.yaml")])
    assert code == EXIT_FAILED
    err = capsys.readouterr().err
    assert "ofp-run: execution failed" in err
    assert "job '" not in err  # no job prefix: this run has no named jobs


# -- a spot a running activity holds is not residue ---------------------------


def test_a_spot_a_running_activity_holds_is_not_derived_as_residue():
    """🔴 §6.12 is for what the plan "does not otherwise account for", and a running
    activity accounts for its spots perfectly well -- the model holds them over its
    interval. Holding them a second time describes the same material twice, and the two
    descriptions overlap: measured to make the replan infeasible and stop every other
    job in the run, the exact opposite of what isolating a failure is for.

    Here one job has two branches: A is still baking on tray_2 when B's transport into
    tray_1 fails. Checked on every document the run hands the scheduler, because that is
    where getting it wrong would land.

    The laboratory pins both halves of that sentence rather than leaving them to the
    scheduler: B's plate fits only tray_1, and A's is the longer bake, so it is under
    way by the time B moves. See the fixtures for why that had to be nailed down.
    """
    import ofplang.run.runner.rolling as rolling

    workflow = load_document(FIXTURES / "two_branch.workflow.yaml")
    runner = RollingRunner(
        [JobRequest(id=job_id, workflow=workflow) for job_id in ("job1", "job2")],
        str(FIXTURES / "two_branch_narrow.env.yaml"),
        poll_interval=None,
        random_seed=0,
    )
    for source in ("bench.slot_a", "bench.slot_b"):
        runner.sim.schedule_transport_failure("arm", source, "oven.tray_1")

    original = rolling.replan
    saw_a_running_spot = False

    def watching(workflows, environment, status_document, **kwargs):
        nonlocal saw_a_running_spot
        running = {
            spot
            for activity in status_document["activities"]
            if activity.get("status") == "running"
            for spot in _spots(activity)
        }
        saw_a_running_spot = saw_a_running_spot or bool(running)
        held = {entry["spot"] for entry in derived_holds(status_document)}
        assert not (held & running), (held, running)
        return original(workflows, environment, status_document, **kwargs)

    rolling.replan = watching
    try:
        status = runner.run()
    finally:
        rolling.replan = original
    # The check is only worth anything if something really was running at the time.
    assert saw_a_running_spot
    # And the spot is held once that bake has finished, not before.
    assert "oven.tray_2" in {entry["spot"] for entry in derived_holds(status)}


def test_a_job_that_stops_with_work_still_running_lets_the_others_finish():
    """The end-to-end shape of the same thing: job1's transport fails while its other
    branch is still baking. Neither the residue nor the plan may put job2's work
    behind something that never happened -- job2 runs to completion."""
    workflow = load_document(FIXTURES / "two_branch.workflow.yaml")
    runner = RollingRunner(
        [JobRequest(id=job_id, workflow=workflow) for job_id in ("job1", "job2")],
        str(FIXTURES / "two_branch.env.yaml"),
        poll_interval=None,
        random_seed=0,
    )
    for source in ("bench.slot_a", "bench.slot_b"):
        runner.sim.schedule_transport_failure("arm", source, "oven.tray_1")
    status = runner.run()

    assert [job.id for job in runner.jobs if job.stopped] == ["job1"]
    assert {a["status"] for a in _of(status, "job2")} == {"completed"}
    # job1 really did have work in flight when it stopped: its bake finished after
    # the transport failed, which is the case that used to take the run down.
    failed = [a for a in _of(status, "job1") if a["status"] == "failed"]
    later = [a for a in _of(status, "job1") if a["status"] == "completed"]
    assert failed and max(a["end"] for a in later) > failed[0]["end"]


# -- derived, not asked ------------------------------------------------------


def test_what_is_derived_to_be_held_is_what_the_backend_is_holding():
    """🔴 The derivation is of the world, and this pins that it describes the real one.

    Where material is used to be settled by asking the backend which spots were
    occupied -- a query the runner is not entitled to make (D15) and a real backend
    could not answer, since a laboratory keeps no spot ledger. It is now replayed from
    what the operations did, in the scheduler, from the document alone.

    The two must agree, and neither is derived from the other: the derivation applies
    the rules SPEC gives (§5.4, §5.5), the simulator applies them to its own
    bookkeeping, and they match because both follow the specification. Extracting a
    shared implementation would couple them, which is the coupling being removed -- so
    the agreement is measured here instead.
    """
    status, runner = _oven_run("job1", "job2", "job3")
    held = {entry["spot"] for entry in derived_holds(status)}
    assert held, "nothing was derived; the fixture no longer strands anything"
    for spot in sorted(held):
        assert runner.sim.spot_state(spot) is not None, f"derived held, backend says free: {spot}"
    # And the other way, for the spots this run's own history is the only account of:
    # a spot the backend holds and nothing derives is one a *running* job is using, or
    # one a job that is still going will come back to.
    stopped = {job.id for job in runner.jobs if job.stopped}
    touched = {
        spot
        for activity in status["activities"]
        if activity.get("job") in stopped
        for spot in _spots(activity)
    }
    for spot in sorted(touched - held):
        assert runner.sim.spot_state(spot) is None, f"derived free, backend holds: {spot}"
