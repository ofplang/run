"""Build the execution status document (spec §6/§7) the runner feeds to the
scheduler each tick.

The status is the runner's committed history rendered as a §6 document: `now`,
the `interface` boundary constraint (§6.8) carried through unchanged, and one
activity entry per committed record marked `completed` / `running` at its actual
times. Relays are **not** emitted -- the scheduler regenerates them from the
committed transport legs (§7). Pending work is omitted; the scheduler re-derives
it from the workflow.
"""

from __future__ import annotations

from .provenance import Committed


def build_status(
    records: list[Committed],
    now: int,
    interface: dict | None = None,
    time_section: dict | None = None,
) -> dict:
    """Assemble a §6 execution status from committed records at time `now`.

    Each record's original activity dict is copied (preserving its provenance and
    assignment echo) and stamped with the record's status and actual times.
    """
    activities = []
    for rec in records:
        entry = dict(rec.activity)  # keep node / arc / seq / process / mode / spots
        entry["status"] = rec.status
        entry["start"] = rec.start
        entry["end"] = rec.end
        activities.append(entry)

    # Readable top-level order: time, now, interface, activities.
    doc: dict = {}
    if time_section:
        doc["time"] = time_section
    doc["now"] = now
    if interface:
        doc["interface"] = interface
    doc["activities"] = activities
    return doc
