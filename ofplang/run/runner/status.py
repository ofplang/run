"""Build the execution status document (spec §6/§7) the runner feeds to the
scheduler each tick.

The status is the runner's committed history rendered as a §6 document: `now`,
the `interface` boundary constraint (§6.8) and the `inventories` starting stock
(§6.10) carried through unchanged, and one activity entry per committed record
marked `completed` / `running`. A completed
record's `end` is its observed finish; a running record's `end` is the planned
expected finish (the runner does not know the actual until it observes
completion). Pending work is omitted; the scheduler re-derives it from the
workflow.

Where an arc's Object was carried in more than one leg -- a re-route off a machine
that went down (§7), or a move too far for one hop (§6.4.1) -- the junction between
two consecutive legs is a **relay**, and the relays are *derived* here rather than
recorded. The runner never dispatches one (a relay is an instantaneous scheduling
junction, not a physical operation), so nothing about it reaches the commit log;
but the two legs that surround it do, and they say everything a relay states. The
scheduler regenerates relays from the committed legs the same way (§7), so the
document round-trips either way -- deriving them is what makes the status describe
the move the way the plan did.
"""

from __future__ import annotations

from .provenance import Committed, arc_key


def build_status(
    records: list[Committed],
    now: int,
    interface: dict | None = None,
    time_section: dict | None = None,
    cancelled: list[dict] | None = None,
    inventories: dict | None = None,
    jobs: list[dict] | None = None,
    occupied: list[dict] | None = None,
) -> dict:
    """Assemble a §6 execution status from committed records at time `now`.

    Each record's original activity dict is copied (preserving its provenance and
    assignment echo) and stamped with the record's status and actual times.

    `cancelled` (D25) are plan activities that never ran because the run stopped on
    a failure. Each is stamped `cancelled` with a zero-length interval at `now` (it
    reached a terminal, non-running state without executing) and appended after the
    committed history.

    `inventories` (§6.10) is what the run *started* with, so -- exactly like
    `interface` -- it is the same on every tick rather than something recomputed
    here. The level at `now` is not stated anywhere: the scheduler replays it from
    these levels and the `consumption` each fixed activity echoes (§4.7.2), which is
    why a completed activity's original dict is copied whole below.

    The failure *reason* (D36) is deliberately NOT put in this document: the status
    must stay a valid §6 execution document (it can be validated and fed back to the
    scheduler), so the reason is exposed out of band via `RollingRunner.failure` and
    the CLI's stderr instead.
    """
    committed = []
    for rec in records:
        entry = dict(rec.activity)  # keep node / arc / seq / process / mode / spots
        entry["status"] = rec.status
        entry["start"] = rec.start
        entry["end"] = rec.end
        committed.append(entry)
    # The junctions between the legs of a multi-leg move, put back (§6.4.1). Only the
    # committed history takes part: a relay is the record of an arrival that actually
    # happened, so cancelled work below has none.
    activities = _with_relays(committed)
    for act in cancelled or []:
        entry = dict(act)  # keep its provenance / assignment echo from the last plan
        entry["status"] = "cancelled"
        entry["start"] = now
        entry["end"] = now
        activities.append(entry)

    # Readable top-level order: time, now, interface, inventories, activities --
    # the order a rendered plan uses, so a status and a plan read the same way.
    doc: dict = {}
    if time_section:
        doc["time"] = time_section
    doc["now"] = now
    # The roster of jobs this run covers (§6.11), each entry carrying what the
    # scheduler needs to recognise it again -- including the completion it promised,
    # which is only ever true because the runner held on to it.
    if jobs:
        doc["jobs"] = jobs
    # `interface` is the single-workflow form; a joint run carries one per job in the
    # roster above, and the two are mutually exclusive (§6.11).
    if interface:
        doc["interface"] = interface
    if inventories:
        doc["inventories"] = inventories
    # Spots held by something this run does not otherwise account for (§6.12): what a
    # stopped job left behind, and whatever the laboratory was already holding.
    if occupied:
        doc["occupied"] = occupied
    doc["activities"] = activities
    return doc


def _with_relays(committed: list[dict]) -> list[dict]:
    """`committed` with a relay (§6.4.1) inserted at every multi-leg junction.

    An arc's Object may be carried in more than one leg: the scheduler re-routes a
    delivery off a machine that went down (§7), or an arc whose endpoints are
    further apart than a single hop is carried through a waypoint (§6.4.1).
    Consecutive legs of one arc meet at a **relay** -- the spot the Object occupies
    after one leg delivers it and before the next picks it up, instantaneous by
    definition.

    The runner never dispatches a relay: it is a scheduling junction, not a physical
    operation, so nothing about it reaches the commit log and there is no record to
    echo. It is reconstructed here from the pair of legs around it -- exactly how the
    scheduler regenerates it on a replan (§7), which is why the document round-trips
    whether or not it is stated. Two things fall out of reconstructing rather than
    echoing: the junction is timed by the *observed* arrival (the delivering leg's
    actual finish) rather than by whatever the plan predicted for it, and a chain of
    any length needs no special case -- one relay per consecutive pair.

    `seq` is not guessed. Positions along an arc are handed out in order (§6.6), so
    the relay between the legs at positions n and n+2 sits at n+1: the gap the legs
    themselves leave. A first leg dispatched while the arc still had only one is
    recorded with no `seq` (a single-leg arc omits it), so where a chain later grew
    around it, it is stamped with the position it held all along.
    """
    # Group the legs by the connection they serve. The job is part of that identity:
    # two jobs of one workflow render the very same arc (§6.11), and pairing one
    # job's leg with another's would invent a junction between two separate Objects.
    chains: dict[tuple, list[tuple[int, dict]]] = {}
    for index, entry in enumerate(committed):
        if entry.get("kind") == "transport":
            key = (entry.get("job", ""), arc_key(entry.get("arc")))
            chains.setdefault(key, []).append((index, entry))

    relays_after: dict[int, list[dict]] = {}
    for chain in chains.values():
        # One leg is the whole move: no junction, and its bare `seq` stays bare --
        # that is the form an arc one hop wide has always been written in (§6.6).
        if len(chain) < 2:
            continue
        chain.sort(key=lambda pair: (pair[1]["start"], pair[1]["end"], _sort_seq(pair[1])))
        chain[0][1].setdefault("seq", 0)

        for (prev_index, prev), (_, nxt) in zip(chain, chain[1:], strict=False):
            # Only a delivery that has *finished* makes a junction. While the leg is
            # still running its `end` is the planned finish, and stating a completed
            # relay at a time that has not arrived yet would contradict `now` (§7) --
            # harmlessly, since the next leg cannot have started either.
            if prev.get("status") != "completed":
                continue
            # The two legs meet at a shared spot (§6.4.1). §7 already guarantees a
            # committed chain is continuous, so this only refuses to invent a
            # junction between legs that do not actually connect.
            spot = prev.get("to_spot")
            if not spot or spot != nxt.get("from_spot"):
                continue
            # A departing leg that goes nowhere is the folded stay-put case: the
            # Object is consumed where the previous leg delivered it, and §6.4.1 says
            # that relay and its no-op leg are left out. The scheduler folds them
            # before the runner ever sees them; this keeps the rule true here too.
            if nxt.get("from_spot") == nxt.get("to_spot"):
                continue
            # Positions come from the legs themselves. If a leg somehow carries none,
            # the position of the junction after it is not known -- and a wrong
            # ordinal would collide with a real one (§6.6), so say nothing instead.
            position = prev.get("seq")
            if position is None:
                continue
            relay: dict = {"kind": "relay"}
            if prev.get("job") is not None:
                relay["job"] = prev["job"]
            relay["status"] = "completed"
            relay["start"] = relay["end"] = prev["end"]
            relay["seq"] = int(position) + 1
            relay["spot"] = spot
            relay["arc"] = prev.get("arc")
            relays_after.setdefault(prev_index, []).append(relay)

    if not relays_after:
        return committed
    # Each relay follows the leg that delivered to it, so an arc reads in chain order
    # the way a rendered plan does.
    out: list[dict] = []
    for index, entry in enumerate(committed):
        out.append(entry)
        out.extend(relays_after.get(index, ()))
    return out


def _sort_seq(activity: dict) -> int:
    """A leg's chain position for ordering. A leg recorded before its arc had any
    other carries no `seq` and is the first one (§6.6), which is what -1 says here."""
    seq = activity.get("seq")
    return -1 if seq is None else int(seq)
