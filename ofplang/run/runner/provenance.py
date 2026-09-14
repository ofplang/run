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


def activity_key(activity: dict):
    """A stable identity for an activity across replans: its workflow provenance
    (a processing's `node` path, a transport's `arc` endpoints + chain position).

    Pending identities are regenerated on every replan, but provenance is not, so
    this is what lines a plan's entry up against the committed record that started
    it -- and, once the runner carries the plan itself from one replan to the next,
    what says which entry a piece of history belongs to.

    🔴 The job is part of the identity, not decoration. Two jobs of one workflow
    render the same `node` and the same arc (§6.11), so without it committing one
    job's activity would mark the other's committed too.

    🔴 A transport's `seq` is read with `None` meaning 0. A leg dispatched while its
    arc still had only one carries no position -- that is the form a single-hop arc
    is written in (§6.6) -- and if the arc later grows a second leg, the plan gives
    the first one the 0 it held all along. Measured: a re-route commits its first leg
    with `seq: None` and the next plan calls the same leg `seq: 0`. Treating them as
    two identities would leave that leg looking as if it had never run.

    A kind this function does not know is **refused**, not given a key. The tempting
    alternative -- let anything that is not a processing fall through to the transport
    shape -- is silent and wrong: an activity with no `arc` and no `seq` yields the
    *same* key for every such activity, so two of them would collapse into one.
    """
    kind = activity.get("kind")
    job = activity.get("job", "")
    if kind == "processing":
        return ("processing", job, tuple(activity.get("node") or ()))
    if kind == "transport":
        seq = activity.get("seq")
        return ("transport", job, *arc_key(activity.get("arc")), 0 if seq is None else int(seq))
    if kind == "replenishment":
        # A refill has no workflow provenance -- it exists because the solver put it
        # there, not because the workflow asked for it -- so its `id` is the identity.
        # That is stable exactly where it has to be: the scheduler numbers new
        # candidates around the ids the document already uses, so a refill that has
        # *started* keeps its id across replans, while a pending one may be renumbered
        # and does not need to survive.
        return ("replenishment", activity.get("id"))
    raise UnknownActivityKind(kind)


class UnknownActivityKind(Exception):
    """An activity whose kind carries no provenance this runner can identify.

    Raised rather than guessed: a kind that cannot be identified cannot be dispatched
    either, so failing loudly costs a run nothing it was going to complete. The
    runner translates it into its own error.
    """

    def __init__(self, kind) -> None:
        super().__init__(f"cannot identify an activity of kind {kind!r} across replans")
        self.kind = kind


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
