"""Committed-activity bookkeeping for the rolling-horizon runner (D10/D20).

The backend knows only physical operations; the runner holds the mapping back to
workflow activities (D10). During a rolling-horizon run the runner is the source
of truth for what has physically happened: it records each activity it dispatches
as a `Committed` entry and, each tick, replays those entries as the execution
status it feeds back to the scheduler. Only committed history is stable across
replans (D9); pending work is re-read from each fresh plan.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Committed:
    """One activity the runner has started (and possibly finished).

    `activity` is the original §6 activity dict from the plan it was dispatched
    from -- it carries the provenance (`node`, or `arc` + `seq`) and the assignment
    echo (`process`/`mode`, or `from_spot`/`to_spot`/`transporter`) the scheduler
    pins on a replan. `uuid` is the backend operation id, or `None` for a
    bookkeeping activity (a same-spot transport) that has no physical operation.
    """

    activity: dict
    kind: str  # "processing" | "transport"
    status: str  # "running" | "completed"
    start: int
    end: int  # actual finish once completed; expected finish while running
    uuid: str | None = None


class CommitLog:
    """The committed history: an ordered list of `Committed`, indexed by backend
    operation id for polling."""

    def __init__(self) -> None:
        self._records: list[Committed] = []
        self._by_uuid: dict[str, Committed] = {}

    def add(self, record: Committed) -> None:
        self._records.append(record)
        if record.uuid is not None:
            self._by_uuid[record.uuid] = record

    def by_uuid(self, uuid: str) -> Committed | None:
        return self._by_uuid.get(uuid)

    def records(self) -> list[Committed]:
        return list(self._records)

    def running(self) -> list[Committed]:
        return [r for r in self._records if r.status == "running"]
