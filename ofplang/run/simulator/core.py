"""The simulated execution backend (dev-notes design.md D10-D18).

`Simulator` stands in for real hardware so the runner can be driven end to end
without a lab. It knows only the physical world (D10): devices, spots,
transporters, and opaque timed operations. It does not know workflows or plans --
the runner keeps that provenance.

Contract (D14/D15), summarised:

* Construct from an environment (§5); the simulator reads it itself.
* `dispatch_processing` / `dispatch_transport` register an operation that runs
  over ``[now, now + duration]`` and return an opaque id. `dispatch_relay` is
  rejected (a relay is not a physical operation, D14).
* Object identity is not tracked, but each occupied spot carries an opaque
  string id (D15), sim-generated unless supplied to `place`.
* `observe` / `state` report only an operation's ``status`` (running / completed),
  never times -- faithful to a real backend's blind polling (D18). Exact event
  times are available only via `_advance`, for tests and debugging.
* `advance(until)` moves the virtual clock forward to ``until``, always reaching
  it (no early return on an event); the runner decides how far to advance (D11).

The simulator is a *validating oracle* (D16): every dispatch checks its physical
preconditions (inputs present, outputs / destinations free, resources idle) and
raises if a runner drives an inconsistent plan. A valid plan never trips these.

Scope: this first cut models correct behaviour only -- no duration variance and
no device up/down (D13/D17). Variance is injected externally by passing a
`duration` to a dispatch; faults come in a later milestone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .environment import Environment, device_of, environment_from_dict, load_environment
from .errors import (
    ClockError,
    MissingObject,
    RelayNotSupported,
    ResourceBusy,
    SpotConflict,
    UnknownReference,
)


@dataclass(frozen=True)
class Event:
    """A completion event, returned by `_advance` for tests / debugging (D18).

    Marks that operation `uuid` (of kind `kind`) finished at virtual time `time`.
    """

    time: int
    uuid: str
    kind: str


@dataclass
class _Op:
    """A dispatched operation the simulator is tracking.

    Holds the resources it occupies and the spot effects to apply when it
    completes. `completed` operations stay in the registry so `state(uuid)` keeps
    working after the fact.
    """

    uuid: str
    kind: str  # "processing" | "transport"
    start: int
    end: int
    seq: int  # dispatch order, for a deterministic tie-break on equal end times
    devices: tuple[str, ...]
    transporter: str | None
    # processing: the spots its input / output ports bind (D15 object effects).
    input_spots: tuple[str, ...]
    output_spots: tuple[str, ...]
    # transport: the physical hop.
    from_spot: str | None
    to_spot: str | None
    status: str = "running"  # "running" | "completed"


class Simulator:
    """A physical-only, discrete-event execution backend (see module docstring)."""

    def __init__(self, environment):
        # Accept a ready `Environment`, a §5 mapping, or a path to a §5 YAML file
        # (D14: the simulator may read the environment itself).
        if isinstance(environment, Environment):
            self._env = environment
        elif isinstance(environment, dict):
            self._env = environment_from_dict(environment)
        else:
            self._env = load_environment(environment)

        # Virtual clock, in the integer ticks of the environment's time unit (§4.1).
        self._clock = 0

        # Spot occupancy: qualified spot -> opaque object id. A spot is free iff
        # it is absent. Occupation persists (material rests, §4.4) until a
        # consuming operation completes.
        self._spot_holds: dict[str, str] = {}

        # Resources currently being accessed by a running operation. Unlike spots,
        # these are released the moment the operation completes (§4.4: idle
        # material in a spot does not occupy its device).
        self._busy_devices: set[str] = set()
        self._busy_transporters: set[str] = set()

        # Operation registry (running and completed), keyed by id.
        self._ops: dict[str, _Op] = {}

        # Monotonic counters for opaque, unstable ids (D15). Deterministic so
        # tests are reproducible; nothing may read meaning from the values.
        self._next_op = 0
        self._next_obj = 0

    # -- ids ---------------------------------------------------------------

    def _new_op_id(self) -> str:
        uid = f"op-{self._next_op}"
        self._next_op += 1
        return uid

    def _new_obj_id(self) -> str:
        oid = f"obj-{self._next_obj}"
        self._next_obj += 1
        return oid

    # -- clock -------------------------------------------------------------

    @property
    def now(self) -> int:
        """The current virtual time."""
        return self._clock

    # -- initial placement (D15) ------------------------------------------

    def place(self, spot: str, obj_id: str | None = None) -> str:
        """Put material on a spot (e.g. seed the interface inputs before a run).

        `obj_id` is optional; when omitted the simulator generates an opaque id.
        Returns the id now held. Errors if the spot is unknown or already occupied.
        """
        if spot not in self._env.spots:
            raise UnknownReference(f"unknown spot: {spot}")
        if spot in self._spot_holds:
            raise SpotConflict(f"spot already occupied: {spot}")
        self._spot_holds[spot] = obj_id if obj_id is not None else self._new_obj_id()
        return self._spot_holds[spot]

    def remove(self, spot: str) -> str:
        """Take material off a spot and return its id. Errors if unknown or empty."""
        if spot not in self._env.spots:
            raise UnknownReference(f"unknown spot: {spot}")
        if spot not in self._spot_holds:
            raise MissingObject(f"spot is empty: {spot}")
        return self._spot_holds.pop(spot)

    # -- dispatch (D14/D15/D16) -------------------------------------------

    def dispatch_processing(self, process: str, mode, duration: int | None = None) -> str:
        """Dispatch a processing operation, resolving its physical detail from the
        environment via (`process`, `mode`) (D14). Runs over ``[now, now + duration]``;
        `duration` defaults to the mode's own (override it to inject variance, D13).
        Returns the operation id.
        """
        # Resolve the capability. Workflow provenance (the node) is not needed here
        # (D14) -- the environment mode alone gives devices, spots, and duration.
        proc = self._env.processes.get(process)
        if proc is None:
            raise UnknownReference(f"unknown process: {process}")
        m = proc.modes.get(str(mode))
        if m is None:
            raise UnknownReference(f"unknown mode {mode!r} for process {process!r}")

        dur = m.duration if duration is None else int(duration)
        if dur < 0:
            raise ValueError(f"duration must be non-negative, got {dur}")

        in_spots = tuple(m.input_spots.values())
        out_spots = tuple(m.output_spots.values())
        inputs = set(in_spots)

        # Preconditions (D16, the validating oracle). Devices idle; every input
        # spot holds material; every output spot is free -- unless it is also one
        # of this operation's own input spots (an in-place transform, §5.5).
        self._require_devices_free(m.devices)
        for s in in_spots:
            if s not in self._spot_holds:
                raise MissingObject(f"input spot is empty: {s}")
        for s in out_spots:
            if s not in inputs and s in self._spot_holds:
                raise SpotConflict(f"output spot already occupied: {s}")

        # Commit: occupy the devices for the run (spots change only on completion).
        for d in m.devices:
            self._busy_devices.add(d)
        return self._register(
            kind="processing",
            duration=dur,
            devices=m.devices,
            transporter=None,
            input_spots=in_spots,
            output_spots=out_spots,
            from_spot=None,
            to_spot=None,
        )

    def dispatch_transport(
        self,
        transporter: str | None,
        from_spot: str,
        to_spot: str,
        duration: int | None = None,
    ) -> str:
        """Dispatch a transport operation moving material `from_spot` -> `to_spot`
        (D14). `duration` defaults to the environment's transport table; a same-spot
        move is a duration-0 no-op whose `transporter` may be `None` (§5.4 / §6.4).
        Occupies the source device, the destination device, and the transporter
        over the move (§4.5). Returns the operation id.
        """
        if from_spot not in self._env.spots:
            raise UnknownReference(f"unknown spot: {from_spot}")
        if to_spot not in self._env.spots:
            raise UnknownReference(f"unknown spot: {to_spot}")

        same_spot = from_spot == to_spot

        # Resolve the transporter and duration. A real move needs a transporter and
        # a route in the table; a same-spot move is a physical no-op (duration 0,
        # transporter optional).
        if same_spot:
            if transporter is not None and transporter not in self._env.transporters:
                raise UnknownReference(f"unknown transporter: {transporter}")
            dur = 0 if duration is None else int(duration)
        else:
            if transporter is None:
                raise ValueError("a non-same-spot transport requires a transporter")
            if transporter not in self._env.transporters:
                raise UnknownReference(f"unknown transporter: {transporter}")
            if duration is None:
                dur = self._env.transports.get((transporter, from_spot, to_spot))
                if dur is None:
                    raise UnknownReference(
                        f"no route for {transporter}: {from_spot} -> {to_spot}"
                    )
            else:
                dur = int(duration)
        if dur < 0:
            raise ValueError(f"duration must be non-negative, got {dur}")

        # The move occupies the source and destination devices (deduplicated when
        # they coincide) plus the transporter (§4.5).
        devices = tuple(dict.fromkeys([device_of(from_spot), device_of(to_spot)]))

        # Preconditions (D16). Devices and transporter idle; the source holds
        # material; the destination is free (a same-spot move keeps its own spot).
        self._require_devices_free(devices)
        if transporter is not None and transporter in self._busy_transporters:
            raise ResourceBusy(f"transporter busy: {transporter}")
        if from_spot not in self._spot_holds:
            raise MissingObject(f"source spot is empty: {from_spot}")
        if not same_spot and to_spot in self._spot_holds:
            raise SpotConflict(f"destination spot already occupied: {to_spot}")

        # Commit: occupy devices and the transporter for the move.
        for d in devices:
            self._busy_devices.add(d)
        if transporter is not None:
            self._busy_transporters.add(transporter)
        return self._register(
            kind="transport",
            duration=dur,
            devices=devices,
            transporter=transporter,
            input_spots=(),
            output_spots=(),
            from_spot=from_spot,
            to_spot=to_spot,
        )

    def dispatch_relay(self, *args, **kwargs):
        """Reject a relay dispatch: a relay is a scheduling junction, not a
        physical operation (D14). The runner keeps relays as bookkeeping."""
        raise RelayNotSupported("relay has no physical operation to dispatch")

    def _require_devices_free(self, devices) -> None:
        """Raise `ResourceBusy` if any of `devices` is currently being accessed."""
        for d in devices:
            if d in self._busy_devices:
                raise ResourceBusy(f"device busy: {d}")

    def _register(
        self,
        *,
        kind,
        duration,
        devices,
        transporter,
        input_spots,
        output_spots,
        from_spot,
        to_spot,
    ) -> str:
        """Record a running operation over ``[now, now + duration]`` and return its
        id. Dispatch is now-start only (D15)."""
        op = _Op(
            uuid=self._new_op_id(),
            kind=kind,
            start=self._clock,
            end=self._clock + duration,
            seq=len(self._ops),
            devices=tuple(devices),
            transporter=transporter,
            input_spots=tuple(input_spots),
            output_spots=tuple(output_spots),
            from_spot=from_spot,
            to_spot=to_spot,
        )
        self._ops[op.uuid] = op
        return op.uuid

    # -- observation (D15/D18) --------------------------------------------

    def observe(self) -> dict[str, dict]:
        """Return every operation's status, keyed by id: ``{uuid: {"status": ...}}``.

        Status only -- no times (D18), faithful to a real backend's blind polling.
        The dict shape leaves room to add fields (times, echoes) later.
        """
        return {u: {"status": op.status} for u, op in self._ops.items()}

    def state(self, uuid: str) -> dict:
        """Return one operation's state dict (``{"status": ...}``). Errors if unknown."""
        op = self._ops.get(uuid)
        if op is None:
            raise UnknownReference(f"unknown operation: {uuid}")
        return {"status": op.status}

    def spot_state(self, spot: str | None = None):
        """Debug / test helper (D15): inspect spot occupancy, which the runner does
        not read in normal operation. With `spot`, return its object id or `None`;
        without, return a copy of the ``{spot: obj_id}`` map of all occupied spots.
        """
        if spot is None:
            return dict(self._spot_holds)
        if spot not in self._env.spots:
            raise UnknownReference(f"unknown spot: {spot}")
        return self._spot_holds.get(spot)

    # -- time advance (D11/D15) -------------------------------------------

    def advance(self, until: int) -> None:
        """Advance the virtual clock to `until`, applying every completion on the
        way (production entry point). Always reaches `until` -- it never returns
        early on an event, mirroring a real backend where only polling reveals
        completion (D11). The events themselves are discarded here."""
        self._advance(until)

    def _advance(self, until: int) -> list[Event]:
        """Advance to `until`, returning the completion events in order (for tests /
        debugging, D18). Steps internally to each next completion time (<= `until`),
        applies it, and repeats; when no completion remains within reach, jumps the
        clock to `until` and stops."""
        if until < self._clock:
            raise ClockError(f"cannot advance to {until}, clock is at {self._clock}")

        events: list[Event] = []
        while True:
            # The next completion is the earliest end among running operations.
            running = [op for op in self._ops.values() if op.status == "running"]
            next_end = min((op.end for op in running), default=None)

            # Nothing finishes at or before `until`: settle the clock and stop.
            if next_end is None or next_end > until:
                self._clock = until
                break

            # Advance to that time and complete everything ending exactly then,
            # in dispatch order so ties are deterministic.
            self._clock = next_end
            due = sorted((op for op in running if op.end == next_end), key=lambda o: o.seq)
            for op in due:
                self._complete(op)
                events.append(Event(time=next_end, uuid=op.uuid, kind=op.kind))
        return events

    def _complete(self, op: _Op) -> None:
        """Apply an operation's completion: free its resources and move material
        per D15 (a processing consumes its inputs and produces at its outputs; a
        transport carries the object from source to destination)."""
        # Release the accessed resources (spots are handled below).
        for d in op.devices:
            self._busy_devices.discard(d)
        if op.transporter is not None:
            self._busy_transporters.discard(op.transporter)

        if op.kind == "processing":
            outputs = set(op.output_spots)
            inputs = set(op.input_spots)
            # Consume inputs that are not also outputs; an in-place port keeps its
            # spot occupied across the operation (D15).
            for s in op.input_spots:
                if s not in outputs:
                    self._spot_holds.pop(s, None)
            # Produce a fresh opaque object at each output spot. An output-only spot
            # must be free at completion (a valid plan guarantees it); the id at an
            # in-place spot is regenerated (identity is not tracked, D15).
            for s in op.output_spots:
                if s not in inputs and s in self._spot_holds:
                    raise SpotConflict(f"output spot occupied at completion: {s}")
                self._spot_holds[s] = self._new_obj_id()
        else:  # transport
            # A same-spot no-op leaves the object where it is; a real move carries
            # its id from source to destination (physical move keeps identity, D15).
            if op.from_spot != op.to_spot:
                obj = self._spot_holds.pop(op.from_spot, None)
                if obj is None:
                    raise MissingObject(f"source spot emptied mid-transport: {op.from_spot}")
                if op.to_spot in self._spot_holds:
                    raise SpotConflict(f"destination spot occupied at arrival: {op.to_spot}")
                self._spot_holds[op.to_spot] = obj

        op.status = "completed"
