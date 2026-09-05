"""Thin wrapper over `ofplang.schedule.schedule()` for the rolling-horizon runner.

The scheduler is called in-process (D20) and takes its inputs as documents, so the
workflow, the environment and the status this runner just rendered are handed over as
they are -- no temporary files, and nothing re-serialized or re-parsed on the way. It
reads them and never writes to them (`ofplang-schedule` >= 0.1.6 states that), so the
same environment dict is passed every replan rather than copied.

`ofplang.schedule` is imported lazily so the plan-replay path (`replay`, milestone 2a)
keeps working even when the scheduler is not installed.
"""

from __future__ import annotations

from .runner import RunnerError


def replan(
    workflow,
    environment,
    status_document: dict,
    *,
    running_task_margin: int = 0,
    random_seed: int | None = None,
    max_time_seconds: float | None = None,
    environment_source: str | None = None,
    ignore_resources: bool = False,
):
    """Run the scheduler on `status_document` and return its `ScheduleReport`.

    `workflow` is one workflow document, or -- for a run of named jobs -- a list of
    `(job id, workflow document)` pairs, which are planned together (§6.11).

    `workflow` and `environment` are documents (or paths); the environment is normally
    the runner's normalized dict, reduced when machines are down (D21). Since that dict
    is not the file it came from, `environment_source` names the file for the plan's
    `meta.environment` provenance -- normalization and reduction happen in memory, and
    the file is still where the environment came from.

    `ignore_resources` switches the consumable model off (SPEC §4.7.3): the environment's
    resource declarations are still shape-checked but nothing is applied, so a lab that
    declares stocks can be run without the document stating what it started with. Off is
    always a relaxation, so no schedule is lost by it.

    Raises `RunnerError` with guidance if `ofplang.schedule` is not importable.
    """
    try:
        from ofplang.schedule.scheduler.api import JobInput
        from ofplang.schedule.scheduler.api import schedule as _schedule
        from ofplang.schedule.scheduler.api import schedule_jobs as _schedule_jobs
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise RunnerError(
            "ofplang.schedule is required for rolling-horizon `run`; install the "
            "sibling repo (e.g. `pip install -e ../ofplang-schedule`)"
        ) from exc

    # A run of named jobs is planned jointly (SPEC §6.11): they compete for the
    # laboratory's machines and draw on its stocks, which is the whole reason to plan
    # them together rather than one after another. A single unnamed workflow keeps the
    # entry point it always had, so its plan is what it always was.
    if isinstance(workflow, list):
        return _schedule_jobs(
            [JobInput(job_id, doc) for job_id, doc in workflow],
            environment,
            document_path=status_document,
            running_task_margin=running_task_margin,
            random_seed=random_seed,
            max_time_seconds=max_time_seconds,
            environment_source=environment_source,
            ignore_resources=ignore_resources,
        )

    return _schedule(
        workflow,
        environment,
        document_path=status_document,
        running_task_margin=running_task_margin,
        random_seed=random_seed,
        max_time_seconds=max_time_seconds,
        environment_source=environment_source,
        ignore_resources=ignore_resources,
    )
