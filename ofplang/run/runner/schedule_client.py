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


def _scheduler():
    """The scheduler package, or the error that says how to get it."""
    try:
        import ofplang.schedule as mod
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise RunnerError(
            "ofplang.schedule is required for rolling-horizon `run`; install the "
            "sibling repo (e.g. `pip install -e ../ofplang-schedule`)"
        ) from exc
    return mod


def default_objective(job_count: int) -> tuple[str, ...]:
    """What a run of `job_count` jobs is optimised for when nobody says otherwise.

    Asked rather than reproduced. The default is the scheduler's rule (SPEC §4.8) and
    depends on nothing but the count -- but it is a *rule*, and a second copy of it
    here would drift the day it changed, in the quiet way a differing objective drifts:
    the same plan, optimised for something else.

    The runner has to ask because it is the one that states it. With nothing declared
    the objective is a function of the roster, and the roster is the runner's; a
    document that echoed a defaulted objective would keep the count it was opened with
    after a job arrived or left (`echo.Echo.declare_objective`).
    """
    from ofplang.schedule.core.objective import default

    _scheduler()  # the install check, and the error that explains it
    return default(job_count)


def derived_holds(document: dict) -> list[dict]:
    """The spots `document` implies are held, beyond the ones it states (§6.12).

    What a stopped job's own history leaves behind is derived by the scheduler on every
    solve rather than declared, so this is how the runner asks what it will conclude --
    which it needs before admitting a job, since entry material cannot be placed on a
    spot something is already sitting on and a replan that fails stops every job in the
    run.
    """
    return _scheduler().derived_holds(document)


def replan(
    workflow,
    environment,
    status_document: dict,
    *,
    withdraw=(),
    carry_levels_to_now: bool = False,
    running_task_margin: int = 0,
    random_seed: int | None = None,
    max_time_seconds: float | None = None,
    environment_source: str | None = None,
    ignore_resources: bool = False,
    max_transport_legs: int = 1,
):
    """Run the scheduler on `status_document` and return its `ScheduleReport`.

    `workflow` is one workflow document, or -- for a run of named jobs -- a list of
    `(job id, workflow document)` pairs, which are planned together (§6.11).

    `workflow` and `environment` are documents (or paths); the environment is normally
    the runner's normalized dict, reduced when machines are down (D21). Since that dict
    is not the file it came from, `environment_source` names the file for the plan's
    `meta.environment` provenance -- normalization and reduction happen in memory, and
    the file is still where the environment came from.

    `withdraw` names jobs **leaving** the plan (SPEC §6.11): their roster entry and
    their history go, and the stock levels are carried forward to `now` so that what
    their work drew is not given back (`inventories.at`, §6.10). They must still be in
    `status_document` -- the scheduler reads a departing job's draws from the
    `consumption` echoes there -- and no workflow is passed for them.

    `carry_levels_to_now` restates `inventories` as of `now` instead of echoing the
    moment it was given (SPEC §6.10). The scheduler never moves that moment by itself:
    working the levels out is the half it can do and the caller cannot, and deciding
    whether the history before the moment may be let go of is the half only the caller
    can. A withdrawal whose job drew on a stock after that moment needs it, or leaving
    would hand those draws back.

    `ignore_resources` switches the consumable model off (SPEC §4.7.3): the environment's
    resource declarations are still shape-checked but nothing is applied, so a lab that
    declares stocks can be run without the document stating what it started with. Off is
    always a relaxation, so no schedule is lost by it.

    `max_transport_legs` is how many transport activities one Object-bearing arc may be
    carried in (SPEC §6.4.1), joined by relays. One -- the single hop this has always
    planned -- unless the caller raises it, which is what a device the transporter
    reaches at one position only, or a plate that has to cross a hand-off station,
    needs. Raising it can only find routes a lower setting reported unreachable.

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
            withdraw=tuple(withdraw),
            carry_levels_to_now=carry_levels_to_now,
            running_task_margin=running_task_margin,
            random_seed=random_seed,
            max_time_seconds=max_time_seconds,
            environment_source=environment_source,
            ignore_resources=ignore_resources,
            max_transport_legs=max_transport_legs,
        )

    # A single workflow has no roster to leave, so `withdraw` cannot apply; the
    # runner refuses the call before it reaches here. The levels flag is not about
    # leaving, though, so it is passed either way.
    return _schedule(
        workflow,
        environment,
        document_path=status_document,
        carry_levels_to_now=carry_levels_to_now,
        running_task_margin=running_task_margin,
        random_seed=random_seed,
        max_time_seconds=max_time_seconds,
        environment_source=environment_source,
        ignore_resources=ignore_resources,
        max_transport_legs=max_transport_legs,
    )
