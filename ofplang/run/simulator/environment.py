"""The physical environment the simulator executes against (spec §5).

The simulator is a *physical-only* backend (dev-notes design.md D10): it knows
devices, spots, transporters, replenishers, transport and refill durations, and
per-process capabilities, but nothing about workflows or plans. This module loads the execution
environment definition (§5) into an immutable model the simulator resolves
dispatches against (D14: the simulator reads the environment itself).

Only the physical layer is modelled here; workflow provenance lives in the
runner. This loader trusts a well-formed environment (`ofplang.schedule` owns
validation, spec §9.1) and extracts just what the simulator needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass(frozen=True)
class Mode:
    """One way to run a process (spec §5.5).

    A mode occupies its `devices` and the spots bound to its Object-bearing input
    and output ports for its `duration`. A Pure-Data-only mode names no device and
    no spot -- it just takes time (D12).
    """

    id: str
    devices: tuple[str, ...]
    duration: int
    # Object-bearing port name -> qualified spot "<device>.<spot>" (§8.2).
    input_spots: dict[str, str]
    output_spots: dict[str, str]


@dataclass(frozen=True)
class Process:
    """An execution capability: a process definition name and its modes,
    keyed by mode id (spec §5.5)."""

    name: str
    modes: dict[str, Mode]


@dataclass(frozen=True)
class Environment:
    """The physical execution environment (spec §5), as the simulator sees it."""

    time_unit: str
    # device id -> its spot names (local, unqualified).
    devices: dict[str, tuple[str, ...]]
    # every qualified spot id "<device>.<spot>" defined in the environment.
    spots: frozenset[str]
    transporters: frozenset[str]
    # (transporter, from_spot, to_spot) -> duration; a missing key means that
    # transporter cannot make that move (§5.4). Same-spot moves are omitted here
    # and treated as duration 0 on lookup.
    transports: dict[tuple[str, str, str], int]
    processes: dict[str, Process]
    # Replenishers (§5.6) and the (replenisher, device) -> duration table (§5.7).
    # A third kind of machine, sharing the one id space with devices and
    # transporters: a refill holds the device it fills *and* the replenisher that
    # fills it, so the simulator has to know they exist to keep them exclusive.
    # Defaulted so an environment that refills nothing builds unchanged.
    replenishers: frozenset[str] = frozenset()
    replenishments: dict[tuple[str, str], int] = field(default_factory=dict)

    def refill_duration(self, replenisher: str, device: str) -> int | None:
        """How long `replenisher` takes to refill `device`, or None if it cannot
        (no table entry, §5.7). A missing pair means "cannot refill it" rather than
        "instantly", the same reading `transport_duration` gives a missing route."""
        return self.replenishments.get((replenisher, device))


def environment_from_dict(raw: dict) -> Environment:
    """Build an `Environment` from an already-parsed §5 mapping.

    Kept separate from file loading so tests can construct environments inline.
    """

    # time.unit -- carried through for reference; the simulator works in the
    # integer ticks of this unit (§4.1).
    time_unit = (raw.get("time") or {}).get("unit", "")

    # Devices and their spots. The globally unique spot id is the qualified form
    # "<device>.<spot>" (§8.2); build the set of all of them for existence checks.
    devices: dict[str, tuple[str, ...]] = {}
    spots: set[str] = set()
    for entry in raw.get("devices") or []:
        did = entry["id"]
        dspots = tuple(entry.get("spots") or [])
        devices[did] = dspots
        for s in dspots:
            spots.add(f"{did}.{s}")

    # Transporters and the transport-duration table keyed by (transporter, from, to).
    transporters = {t["id"] for t in raw.get("transporters") or []}
    transports: dict[tuple[str, str, str], int] = {}
    for t in raw.get("transports") or []:
        transports[(t["transporter"], t["from"], t["to"])] = int(t["duration"])

    # Replenishers and the refill-duration table keyed by (replenisher, device),
    # exactly as transporters and transports are (§5.6, §5.7).
    replenishers = {r["id"] for r in raw.get("replenishers") or []}
    replenishments: dict[tuple[str, str], int] = {}
    for r in raw.get("replenishments") or []:
        replenishments[(r["replenisher"], r["device"])] = int(r["duration"])

    # Processes -> modes. A mode with no explicit `id` is assigned the decimal
    # string of its position ("0", "1", ...), matching how a plan records the
    # selected mode (§5.5).
    processes: dict[str, Process] = {}
    for name, pdef in (raw.get("processes") or {}).items():
        modes: dict[str, Mode] = {}
        for i, mdef in enumerate(pdef.get("modes") or []):
            raw_id = mdef.get("id")
            mid = str(raw_id) if raw_id is not None else str(i)
            modes[mid] = Mode(
                id=mid,
                devices=tuple(mdef.get("devices") or []),
                duration=int(mdef["duration"]),
                input_spots=dict(mdef.get("input_spots") or {}),
                output_spots=dict(mdef.get("output_spots") or {}),
            )
        processes[name] = Process(name=name, modes=modes)

    return Environment(
        time_unit=time_unit,
        devices=devices,
        spots=frozenset(spots),
        transporters=frozenset(transporters),
        transports=transports,
        processes=processes,
        replenishers=frozenset(replenishers),
        replenishments=replenishments,
    )


def load_environment(path) -> Environment:
    """Load an execution environment definition (§5) from a YAML file."""

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return environment_from_dict(raw)


def device_of(qualified_spot: str) -> str:
    """Return the device id of a qualified spot "<device>.<spot>" (§8.2).

    The qualified form has exactly one `.`, and neither part may contain one, so
    the prefix before the first `.` is unambiguously the device.
    """

    return qualified_spot.partition(".")[0]
