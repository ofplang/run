"""Thin wrapper over `ofplang.schedule.schedule()` for the rolling-horizon runner.

The scheduler is called in-process (D20). Its API takes file paths, so the
runner writes the current execution status to a temporary file and passes its
path. `ofplang.schedule` is imported lazily so the plan-replay path (`replay`,
milestone 2a) keeps working even when the scheduler is not installed.
"""

from __future__ import annotations

import os
import tempfile

import yaml

from .runner import RunnerError


def replan(
    workflow_path,
    environment_path,
    status_document: dict,
    *,
    running_task_margin: int = 0,
    random_seed: int | None = None,
    max_time_seconds: float | None = None,
):
    """Run the scheduler on `status_document` and return its `ScheduleReport`.

    The status is written to a temp file (the scheduler reads a document path) and
    removed afterward. Raises `RunnerError` with guidance if `ofplang.schedule` is
    not importable.
    """
    try:
        from ofplang.schedule.scheduler.api import schedule as _schedule
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise RunnerError(
            "ofplang.schedule is required for rolling-horizon `run`; install the "
            "sibling repo (e.g. `pip install -e ../ofplang-schedule`)"
        ) from exc

    # Serialize the status to a temp file, schedule against it, then clean up.
    fd, path = tempfile.mkstemp(suffix=".status.yaml")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(status_document, f, sort_keys=False, allow_unicode=True)
        return _schedule(
            workflow_path,
            environment_path,
            document_path=path,
            running_task_margin=running_task_margin,
            random_seed=random_seed,
            max_time_seconds=max_time_seconds,
        )
    finally:
        os.unlink(path)
