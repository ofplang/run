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

# The identity of one logical connection (spec §6.6): the two endpoints an arc
# joins, as a hashable pair. A transport leg and the relays between the legs of
# that same arc all carry the identical `arc` mapping, so this is what groups a
# multi-leg move back together -- and it is the arc half of the provenance key
# the rolling runner matches a committed activity against a pending one by.
ArcKey = tuple[tuple[tuple[str, ...], object], tuple[tuple[str, ...], object]]


def arc_key(arc: dict | None) -> ArcKey:
    """`arc` as a hashable `(from, to)` pair of `(node path, port)` endpoints.

    A boundary arc keys distinctly from any interior one without a special case:
    its outside endpoint simply carries an empty node path (§6.4, §6.8).
    """

    def endpoint(e):
        e = e or {}
        return (tuple(e.get("node") or ()), e.get("port"))

    arc = arc or {}
    return (endpoint(arc.get("from")), endpoint(arc.get("to")))


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
    status: str  # "running" | "completed" | "failed"
    start: int
    end: int  # actual finish once completed / failed; expected finish while running
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
