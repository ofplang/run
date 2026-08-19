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
):
    """Run the scheduler on `status_document` and return its `ScheduleReport`.

    `workflow` and `environment` are documents (or paths); the environment is normally
    the runner's normalized dict, reduced when machines are down (D21). Since that dict
    is not the file it came from, `environment_source` names the file for the plan's
    `meta.environment` provenance -- normalization and reduction happen in memory, and
    the file is still where the environment came from.

    Raises `RunnerError` with guidance if `ofplang.schedule` is not importable.
    """
    try:
        from ofplang.schedule.scheduler.api import schedule as _schedule
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise RunnerError(
            "ofplang.schedule is required for rolling-horizon `run`; install the "
            "sibling repo (e.g. `pip install -e ../ofplang-schedule`)"
        ) from exc

    return _schedule(
        workflow,
        environment,
        document_path=status_document,
        running_task_margin=running_task_margin,
        random_seed=random_seed,
        max_time_seconds=max_time_seconds,
        environment_source=environment_source,
    )
