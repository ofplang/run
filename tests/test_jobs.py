"""Tests for running several jobs together (SPEC §6.11).

A run used to be a run of a *workflow*; these pin what it means for one to be a run
of a *laboratory* with several workflows in it. What is checked here:

  - the roster goes out and comes back: each job's `release` is handed to the
    scheduler, and the `bound` it promised is carried into the next replan (the
    status is rebuilt from the commit log every tick, so a promise the runner did
    not hold would vanish on the second one);
  - identity is per job: two jobs of one workflow render the same node paths, and
    committing one job's activity must not mark the other's committed;
  - the laboratory's own state -- its stocks and the spots it is already holding --
    is stated once for the run, not per job;
  - the run document (`--jobs`) is read strictly, and a typo in it is met with the
    typo.

The scheduler is a required dependency; these tests skip if it is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("ofplang.schedule", reason="ofplang-schedule not installed")

from ofplang.run.cli import EXIT_OK, EXIT_USAGE, main  # noqa: E402
from ofplang.run.runner import (  # noqa: E402
    JobRequest,
    RollingRunner,
    RunnerError,
    load_document,
    parse_run_document,
)

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = Path(__file__).parent.parent / "examples"
SIMPLE_WF = FIXTURES / "simple.workflow.yaml"
SIMPLE_ENV = str(FIXTURES / "simple.env.yaml")
REFILL_WF = EXAMPLES / "shared_refill.workflow.yaml"
REFILL_ENV = str(EXAMPLES / "shared_refill.env.yaml")
REFILL_RUN = str(EXAMPLES / "shared_refill.run.yaml")


def _requests(*ids: str, workflow=SIMPLE_WF, **kwargs) -> list[JobRequest]:
    doc = load_document(workflow)
    return [JobRequest(id=job_id, workflow=doc, **kwargs) for job_id in ids]


def _of(status: dict, job: str) -> list[dict]:
    return [a for a in status["activities"] if a.get("job") == job]


# -- several jobs actually run -----------------------------------------------


def test_two_jobs_of_one_workflow_both_run_to_completion():
    runner = RollingRunner(_requests("job1", "job2"), SIMPLE_ENV, random_seed=0)
    status = runner.run()
    assert not runner.failed
    assert all(a["status"] == "completed" for a in status["activities"])
    # Each job ran its whole workflow: source, transport, target.
    for job in ("job1", "job2"):
        assert [a["kind"] for a in _of(status, job)].count("processing") == 2
        assert any(a["kind"] == "transport" for a in _of(status, job))


def test_one_jobs_work_is_not_committed_against_the_other():
    """🔴 The two jobs render identical node paths, so if the job were not part of a
    committed activity's identity, committing one would mark the other committed --
    and the second job's work would never be dispatched at all."""
    runner = RollingRunner(_requests("job1", "job2"), SIMPLE_ENV, random_seed=0)
    status = runner.run()
    nodes = {(a.get("job"), tuple(a["node"])) for a in status["activities"]
             if a["kind"] == "processing"}
    assert nodes == {
        ("job1", ("SampleSource",)), ("job1", ("SampleTarget",)),
        ("job2", ("SampleSource",)), ("job2", ("SampleTarget",)),
    }


def test_two_jobs_cannot_share_an_id():
    with pytest.raises(RunnerError, match="distinct"):
        RollingRunner(_requests("job1", "job1"), SIMPLE_ENV, random_seed=0)


def test_a_run_of_named_jobs_carries_no_top_level_interface():
    """`interface` is the single-workflow form (§6.8). A joint run has one per job,
    in the roster, and the two are mutually exclusive."""
    runner = RollingRunner(_requests("job1", "job2"), SIMPLE_ENV, random_seed=0)
    status = runner.run()
    assert "interface" not in status
    assert [entry["id"] for entry in status["jobs"]] == ["job1", "job2"]


def test_the_result_boundary_of_a_joint_run_is_keyed_by_job():
    runner = RollingRunner(_requests("job1", "job2"), SIMPLE_ENV, random_seed=0)
    runner.run()
    assert set(runner.result_boundary["jobs"]) == {"job1", "job2"}


def test_a_single_unnamed_workflow_still_gets_the_document_it_always_did():
    """The single-workflow route is untouched: a top-level `interface`, no roster,
    and a result boundary that is the boundary document rather than a mapping of
    them."""
    runner = RollingRunner(str(SIMPLE_WF), SIMPLE_ENV, random_seed=0)
    status = runner.run()
    assert "jobs" not in status
    assert "jobs" not in runner.result_boundary
    assert all("job" not in a for a in status["activities"])


# -- the roster: what goes out and what comes back ---------------------------


def test_the_promise_the_scheduler_made_each_job_survives_the_replans():
    """🔴 The status is rebuilt from the commit log every tick, so a `bound` the
    runner did not hold on to would be gone by the second one -- and every job would
    look like a new arrival with no promise. Then the guarantee that an earlier job
    is not disturbed by a later one would hold inside one solve and nowhere else."""
    runner = RollingRunner(_requests("job1", "job2"), SIMPLE_ENV, random_seed=0)
    status = runner.run()
    assert runner.ticks > 1  # several replans, so the promise really round-tripped
    bounds = {entry["id"]: entry.get("bound") for entry in status["jobs"]}
    assert bounds["job1"] is not None and bounds["job2"] is not None
    # Each job finished by the time it was promised.
    for job, bound in bounds.items():
        assert max(a["end"] for a in _of(status, job)) <= bound


def test_a_release_keeps_a_job_from_starting_early():
    doc = load_document(SIMPLE_WF)
    runner = RollingRunner(
        [
            JobRequest(id="now", workflow=doc),
            JobRequest(id="later", workflow=doc, release=6),
        ],
        SIMPLE_ENV,
        random_seed=0,
    )
    status = runner.run()
    assert min(a["start"] for a in _of(status, "later")) >= 6
    assert min(a["start"] for a in _of(status, "now")) == 0
    assert [e.get("release") for e in status["jobs"]] == [None, 6]


# -- the laboratory's own state ----------------------------------------------


def test_two_jobs_together_need_the_refill_neither_needs_alone():
    """The whole point of planning jobs together: the stock belongs to the device
    (§4.7), so the pair draws on one reader. And the refill belongs to no job."""
    levels = {"levels": {"reader": {"reagent": 2}}}

    alone = RollingRunner(
        _requests("morning", workflow=REFILL_WF), REFILL_ENV,
        random_seed=0, inventories=levels,
    ).run()
    assert not [a for a in alone["activities"] if a["kind"] == "replenishment"]

    together = RollingRunner(
        _requests("morning", "afternoon", workflow=REFILL_WF), REFILL_ENV,
        random_seed=0, inventories=levels,
    ).run()
    refills = [a for a in together["activities"] if a["kind"] == "replenishment"]
    assert len(refills) == 1
    assert "job" not in refills[0]  # it serves both, so it belongs to neither
    assert all(a["status"] == "completed" for a in together["activities"])


def test_the_starting_levels_are_carried_into_the_status():
    levels = {"levels": {"reader": {"reagent": 2}}}
    status = RollingRunner(
        _requests("morning", "afternoon", workflow=REFILL_WF), REFILL_ENV,
        random_seed=0, inventories=levels,
    ).run()
    assert status["inventories"] == levels


def test_a_spot_the_laboratory_already_holds_is_planned_around():
    """A spot named in `occupied` (§6.12) is held by something this run does not
    account for, so nothing may be planned onto it -- and the runner holds it in the
    backend too, so a plan that ignored it would fail loudly rather than quietly
    succeed against a world it disagrees with."""
    levels = {"levels": {"reader": {"reagent": 6}}}
    status = RollingRunner(
        _requests("morning", "afternoon", workflow=REFILL_WF), REFILL_ENV,
        random_seed=0, inventories=levels,
        occupied=[{"spot": "bench.slot_a"}],
    ).run()
    made = [a for a in status["activities"] if a.get("process") == "make_plate"]
    assert made and all(a["mode"] == "slot_b" for a in made)
    # `since` defaults to 0: what a run's opening state can mean is "from the start".
    assert status["occupied"] == [{"since": 0, "spot": "bench.slot_a"}]


def test_jobs_disagreeing_about_the_starting_stock_is_an_error():
    doc = load_document(REFILL_WF)
    boundary = lambda n: {"boundary": {"inventories": {"levels": {"reader": {"reagent": n}}}}}  # noqa: E731
    with pytest.raises(RunnerError, match="conflicting starting inventories"):
        RollingRunner(
            [
                JobRequest(id="a", workflow=doc, boundary=boundary(2)),
                JobRequest(id="b", workflow=doc, boundary=boundary(4)),
            ],
            REFILL_ENV,
            random_seed=0,
        )


def test_a_run_level_boundary_has_no_job_to_belong_to():
    with pytest.raises(RunnerError, match="boundary per job"):
        RollingRunner(_requests("job1"), SIMPLE_ENV, boundary={"boundary": {}})


# -- the run document --------------------------------------------------------


def test_the_run_document_resolves_paths_against_its_own_directory():
    doc = load_document(REFILL_RUN)
    parsed = parse_run_document(doc, EXAMPLES)
    assert [job.id for job in parsed.jobs] == ["morning", "afternoon"]
    assert parsed.jobs[0].workflow["entry"] == "main"
    assert parsed.inventories == {"levels": {"reader": {"reagent": 2}}}


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        ({}, "`jobs` must be a non-empty list"),
        ({"jobs": []}, "`jobs` must be a non-empty list"),
        ({"jobs": [{"workflow": "w.yaml"}]}, "`id` is required"),
        ({"jobs": [{"id": "a", "workflow": 7}]}, "must be a path or a mapping"),
        (
            {"jobs": [{"id": "a", "workflow": {}, "realease": 3}]},
            r"unknown key\(s\) \['realease'\]",
        ),
        (
            {"jobs": [{"id": "a", "workflow": {}, "release": -1}]},
            "`release` must be a non-negative integer",
        ),
        ({"jobs": [{"id": "a", "workflow": {}}], "stocks": {}}, r"unknown key\(s\)"),
        (
            {"jobs": [{"id": "a", "workflow": {}}], "inventories": []},
            "`inventories` must be a mapping",
        ),
    ],
)
def test_a_typo_in_the_run_document_is_met_with_the_typo(doc, message):
    with pytest.raises(RunnerError, match=message):
        parse_run_document(doc)


# -- the CLI -----------------------------------------------------------------


def test_cli_runs_a_run_document(tmp_path, capsys):
    out = tmp_path / "status.yaml"
    code = main(["run", "--jobs", REFILL_RUN, "--env", REFILL_ENV, "-o", str(out)])
    assert code == EXIT_OK
    status = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert [entry["id"] for entry in status["jobs"]] == ["morning", "afternoon"]
    assert sum(1 for a in status["activities"] if a["kind"] == "replenishment") == 1


def test_cli_wants_either_a_workflow_or_a_run_document(capsys):
    assert main(["run", "--env", REFILL_ENV]) == EXIT_USAGE
    assert main(["run", str(SIMPLE_WF), "--jobs", REFILL_RUN, "--env", SIMPLE_ENV]) == (
        EXIT_USAGE
    )
    assert "not both" in capsys.readouterr().err


def test_cli_refuses_a_run_level_boundary_with_jobs(capsys):
    code = main(
        ["run", "--jobs", REFILL_RUN, "--env", REFILL_ENV, "--boundary", REFILL_RUN]
    )
    assert code == EXIT_USAGE
    assert "each job carries its own" in capsys.readouterr().err


def test_cli_names_the_job_whose_workflow_is_rejected(tmp_path, capsys):
    bad = tmp_path / "bad.workflow.yaml"
    bad.write_text("spec_version: '0.0'\nentry: nowhere\n", encoding="utf-8")
    run_doc = tmp_path / "run.yaml"
    run_doc.write_text(
        yaml.safe_dump({"jobs": [{"id": "second", "workflow": str(bad)}]}),
        encoding="utf-8",
    )
    assert main(["run", "--jobs", str(run_doc), "--env", SIMPLE_ENV]) == EXIT_USAGE
    assert "job 'second'" in capsys.readouterr().err
