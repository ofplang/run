"""The simulated execution backend (dev-notes design.md D10-D25).

`Simulator` stands in for real hardware so the runner can be driven end to end
without a lab. It knows only the physical world (D10): devices, spots,
transporters, and opaque timed operations. It does not know workflows or plans --
the runner keeps that provenance.

Contract (D14/D15/D21), summarised:

* Construct from an environment (§5); the simulator reads it itself.
* `dispatch_processing` / `dispatch_transport` register an operation that runs
  over ``[now, now + duration]`` and return an opaque id. `dispatch_relay` is
  rejected (a relay is not a physical operation, D14).
* Object identity is not tracked, but each occupied spot carries an opaque
  string id (D15), sim-generated unless supplied to `place`.
* `observe` / `state` report an operation's ``status`` (running / completed /
  failed), never times -- faithful to a real backend's blind polling (D18). Exact
  event times are available only via `_history`, for tests and debugging. A
  completed operation dispatched with a value signature (`output_schema`) also
  reveals its generated `outputs`, from the injected device model or the built-in
  `default_device_model` (the value seam, D26/D27); an operation with no signature
  stays status-only, so existing callers are unaffected.
* `schedule_process_failure` / `schedule_transport_failure` declare that a
  capability's operations fail instead of completing (D25); a failed operation
  frees its resources at its end but applies no material effect.
* `advance(until)` moves the virtual clock forward to ``until``, always reaching
  it (no early return on an event); the runner decides how far to advance (D11).
  Each `advance` accumulates its completion events into `_history`.
* `schedule_device_down` / `schedule_device_up` register timed faults; a down
  device rejects new processing but still serves transports, and operations
  already running on it are unaffected (D21). `down_devices` reports the current
  set (the runner polls it to trigger re-routing).

The simulator is a *validating oracle* (D16): every dispatch checks its physical
preconditions (inputs present, outputs / destinations free, resources idle, the
device not down) and raises if a runner drives an inconsistent plan. A valid plan
never trips these.

Scope: duration variance is injected externally by passing a `duration` to a
dispatch, not built in (D13). A down device only blocks new processing (D21).
Operation failure is injected per capability (D25): a scheduled `(process, mode)`
or `(transporter, route)` fails at its end. Data value computation remains out of
scope (D12).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .environment import Environment, device_of, environment_from_dict, load_environment
from .errors import (
    ClockError,
    DeviceDown,
    MissingObject,
    RelayNotSupported,
    ResourceBusy,
    SpotConflict,
    UnknownReference,
)


# Typed default value per built-in primitive (D27 F2), used to generate a
# schema-conformant value from a value-shape descriptor.
_PRIMITIVE_DEFAULTS = {"Bool": False, "Int": 0, "Float": 0.0, "String": ""}


def _generate_value(descriptor: dict):
    """Generate a typed default value from a neutral value-shape descriptor (D27
    F2 -- the backend's stand-in for real computation). A primitive yields its
    default, an array an empty list, a record a dict of its fields' defaults. The
    descriptor format is the runner/backend seam contract (see contracts.py); the
    backend walks it without importing the runner's type model."""
    kind = descriptor["kind"]
    if kind == "primitive":
        return _PRIMITIVE_DEFAULTS[descriptor["name"]]
    if kind == "array":
        return []
    # record: a dict of each view field's default.
    return {name: _generate_value(field) for name, field in descriptor["fields"].items()}


def default_device_model(process, mode, inputs, output_schema, definition):
    """The simulator's built-in device model (D27): used when no device model is
    injected, and a convenient base for a custom one to build on.

    It fills every output with a typed default (walking its value-shape descriptor),
    then carries each Object-bearing output declared in the process's ``objects.map``
    (``outputs.P: inputs.Q``) through from ``inputs[Q]`` -- so an identity-preserving
    object pass-through keeps its view value with no per-process code. It performs no
    genuine computation; a real / custom device model does that on top (e.g. call
    this, then overwrite the computed outputs).

    Reading the process's ``objects.map`` (and inputs) makes this default
    workflow-structure- and input-dependent -- a deliberate change from the pure,
    input-independent type defaults it replaced (D27). Only the default *model* is
    workflow-aware; the simulator's physical core still never interprets the
    definition. A custom `device_model` replaces this entirely."""
    outputs = {port: _generate_value(desc) for port, desc in output_schema.items()}
    object_map = ((definition or {}).get("objects") or {}).get("map") or {}
    for out_ref, in_ref in object_map.items():
        # `objects.map` keys/values are namespaced paths (`outputs.P` / `inputs.Q`);
        # strip the namespace to the bare port name.
        outputs[out_ref.split(".", 1)[1]] = inputs[in_ref.split(".", 1)[1]]
    return outputs


@dataclass(frozen=True)
class Event:
    """A terminal event, surfaced via `_history` for tests / debugging (D18).

    Marks that operation `uuid` (of kind `kind`) ended at virtual time `time` with
    terminal `status` -- `completed` normally, or `failed` for an injected failure.
    """

    time: int
    uuid: str
    kind: str
    status: str = "completed"


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
    # Value seam (D26/D27): the output signature this operation was dispatched with
    # -- a mapping ``{port: value-shape descriptor}`` (None = a legacy dispatch with
    # no signature; then no outputs are ever attached, so `state` / `observe` stay
    # status-only for it) -- and the outputs the backend generates at completion
    # (`{port: value}`). The backend is the value *generator* (D26 principle B): it
    # walks each descriptor to a typed default (D27 F2). `inputs` are the input
    # values passed at dispatch (D27 F4); the device model (injected, or the
    # built-in `default_device_model`) computes the outputs from them (F4b).
    output_schema: dict | None = None
    inputs: dict | None = None
    outputs: dict | None = None
    # The processing capability (process, mode), identifying it to an injected
    # device model (D27 F4b); None for a transport.
    process: str | None = None
    mode: str | None = None
    # The raw process definition (the workflow's `processes.<name>` sub-mapping:
    # kind / inputs / outputs / objects), passed through to the device model so it
    # can act on the process's declared structure (e.g. carry an object output from
    # its `objects.map` input). The runner supplies it at dispatch (D27 F4b /
    # principle A: the backend receives the process definition part per dispatch);
    # the simulator itself does not interpret it.
    definition: dict | None = None
    status: str = "running"  # "running" | "completed" | "failed"
    # Whether this operation is scheduled to fail instead of completing (D25). Set
    # at dispatch from the failing (process, mode) / (transporter, route) sets; when
    # the clock reaches its end it goes to `failed` (resources freed, no material
    # effect) rather than `completed`.
    should_fail: bool = False


class Simulator:
    """A physical-only, discrete-event execution backend (see module docstring)."""

    def __init__(self, environment, device_model=None):
        # Accept a ready `Environment`, a §5 mapping, or a path to a §5 YAML file
        # (D14: the simulator may read the environment itself).
        if isinstance(environment, Environment):
            self._env = environment
        elif isinstance(environment, dict):
            self._env = environment_from_dict(environment)
        else:
            self._env = load_environment(environment)

        # Optional device model (D27 F4b): a callback
        # `device_model(process, mode, inputs, output_schema, definition) -> {port:
        # value}` the backend calls at completion to compute a signed operation's
        # outputs from its inputs -- the pluggable stand-in for real device
        # computation (a real backend plugs a real model here). `definition` is the
        # raw process definition (its declared kind/inputs/outputs/objects), so a
        # model can act on the process structure. None means the built-in
        # `default_device_model` (typed defaults + `objects.map` object pass-through)
        # is used. Affects signed processing only.
        self._device_model = device_model

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

        # Accumulated completion events, appended by every `advance`. Production
        # code never reads this; it is the privileged debug / test channel that
        # carries times (`_history`), keeping `observe` / `state` status-only (D18).
        self._history_events: list[Event] = []

        # Devices currently down, and the timed fault schedule that drives them
        # (D21). A down device cannot run processes; transports and running ops are
        # unaffected. Faults are applied lazily as the clock reaches their time.
        self._down: set[str] = set()
        self._faults: list[dict] = []

        # Failure scenario (D25): capabilities whose operations fail instead of
        # completing. A processing failure is keyed by (process, mode) and a
        # transport failure by (transporter, from_spot, to_spot). Declared up front
        # (like device faults, independent of dispatch); every operation dispatched
        # for a failing capability fails at its end.
        self._failing_processes: set[tuple[str, str]] = set()
        self._failing_transports: set[tuple[str | None, str, str]] = set()

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

    def dispatch_processing(
        self, process: str, mode, duration: int | None = None, output_schema=None, inputs=None, definition=None
    ) -> str:
        """Dispatch a processing operation, resolving its physical detail from the
        environment via (`process`, `mode`) (D14). Runs over ``[now, now + duration]``;
        `duration` defaults to the mode's own (override it to inject variance, D13).
        Returns the operation id.

        `output_schema` is the value-seam signature (D26/D27): a mapping
        ``{port: value-shape descriptor}`` the backend uses to generate a typed value
        for each output port at completion, revealed via `state` / `observe`. When
        given (even empty), the operation carries `outputs`; when omitted (None), it
        stays status-only (backward compatible). `inputs` (``{port: value}``, D27 F4)
        are the input values the device model (injected, or `default_device_model`)
        computes the outputs from (F4b). `definition` (the raw process definition) is
        passed through to the device model unchanged; the simulator does not
        interpret it (only the device model does).
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
        input_spot_set = set(in_spots)

        # Preconditions (D16, the validating oracle). No device is down (a down
        # device cannot run processes, D21); devices idle; every input spot holds
        # material; every output spot is free -- unless it is also one of this
        # operation's own input spots (an in-place transform, §5.5).
        self._apply_faults()
        for d in m.devices:
            if d in self._down:
                raise DeviceDown(f"device is down: {d}")
        self._require_devices_free(m.devices)
        for s in in_spots:
            if s not in self._spot_holds:
                raise MissingObject(f"input spot is empty: {s}")
        for s in out_spots:
            if s not in input_spot_set and s in self._spot_holds:
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
            should_fail=(process, str(mode)) in self._failing_processes,
            output_schema=None if output_schema is None else dict(output_schema),
            inputs=None if inputs is None else dict(inputs),
            process=process,
            mode=str(mode),
            definition=definition,
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
            should_fail=(transporter, from_spot, to_spot) in self._failing_transports,
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
        should_fail=False,
        output_schema=None,
        inputs=None,
        process=None,
        mode=None,
        definition=None,
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
            should_fail=should_fail,
            output_schema=output_schema,
            inputs=inputs,
            process=process,
            mode=mode,
            definition=definition,
        )
        self._ops[op.uuid] = op
        return op.uuid

    # -- observation (D15/D18) --------------------------------------------

    def _op_view(self, op: _Op) -> dict:
        """One operation's observable state. Status-only (no times, D18), except that
        a completed operation dispatched with a value signature also reveals its
        generated `outputs` (the value seam, D26). An operation with no signature (a
        legacy dispatch) never carries `outputs`, so its view stays exactly
        ``{"status": ...}`` -- backward compatible."""
        view = {"status": op.status}
        if op.outputs is not None:
            view["outputs"] = op.outputs
        return view

    def observe(self) -> dict[str, dict]:
        """Return every operation's state, keyed by id (see `_op_view`): a
        ``{"status": ...}`` dict, plus `outputs` for a completed value-carrying op."""
        return {u: self._op_view(op) for u, op in self._ops.items()}

    def state(self, uuid: str) -> dict:
        """Return one operation's state dict (see `_op_view`). Errors if unknown."""
        op = self._ops.get(uuid)
        if op is None:
            raise UnknownReference(f"unknown operation: {uuid}")
        return self._op_view(op)

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

    # -- device faults (D21) ----------------------------------------------

    def schedule_device_down(self, time: int, device: str) -> None:
        """Register that `device` goes down at virtual time `time` -- it can no
        longer run processes from then on (transports and running ops are
        unaffected, D21). Registered via this method rather than the constructor so
        the environment (what exists) and the fault scenario (what happens) stay
        separate concerns."""
        self._register_fault(time, device, "down")

    def schedule_device_up(self, time: int, device: str) -> None:
        """Register that `device` comes back up at virtual time `time`."""
        self._register_fault(time, device, "up")

    def _register_fault(self, time: int, device: str, action: str) -> None:
        if device not in self._env.devices:
            raise UnknownReference(f"unknown device: {device}")
        self._faults.append({"time": int(time), "device": device, "action": action, "applied": False})

    # -- failure scenario (D25) -------------------------------------------

    def schedule_process_failure(self, process: str, mode) -> None:
        """Declare that every processing operation for `(process, mode)` fails
        instead of completing (D25). Like a device fault, this is a scenario set up
        front (valid any time after construction, before or after dispatch); it is
        keyed by the capability, not a specific operation id. The operation still
        runs for its duration and frees its resources at the end, but ends `failed`
        and applies no material effect."""
        proc = self._env.processes.get(process)
        if proc is None:
            raise UnknownReference(f"unknown process: {process}")
        if str(mode) not in proc.modes:
            raise UnknownReference(f"unknown mode {mode!r} for process {process!r}")
        self._failing_processes.add((process, str(mode)))

    def schedule_transport_failure(self, transporter: str | None, from_spot: str, to_spot: str) -> None:
        """Declare that every transport operation over `(transporter, from_spot,
        to_spot)` fails instead of completing (D25). The counterpart to
        `schedule_process_failure` for the transport half of the plan."""
        if transporter is not None and transporter not in self._env.transporters:
            raise UnknownReference(f"unknown transporter: {transporter}")
        if from_spot not in self._env.spots:
            raise UnknownReference(f"unknown spot: {from_spot}")
        if to_spot not in self._env.spots:
            raise UnknownReference(f"unknown spot: {to_spot}")
        self._failing_transports.add((transporter, from_spot, to_spot))

    def down_devices(self) -> list[str]:
        """The devices currently down (a sorted copy). The runner polls this each
        tick and reduces the environment it schedules against accordingly."""
        self._apply_faults()
        return sorted(self._down)

    def _apply_faults(self) -> None:
        """Apply every scheduled fault whose time has been reached, in time order.
        Idempotent (each fault applies once); called from `down_devices`, from a
        processing dispatch, and at the end of each advance."""
        for fault in sorted(self._faults, key=lambda f: f["time"]):
            if not fault["applied"] and fault["time"] <= self._clock:
                if fault["action"] == "down":
                    self._down.add(fault["device"])
                else:
                    self._down.discard(fault["device"])
                fault["applied"] = True

    # -- time advance (D11/D15) -------------------------------------------

    def advance(self, until: int) -> None:
        """Advance the virtual clock to `until`, applying every completion on the
        way (the sole clock entry point for production / the runner). Always
        reaches `until` -- it never returns early on an event, mirroring a real
        backend where only polling reveals completion (D11). The completion events
        are accumulated into the history (see `_history`) rather than returned, so
        the main loop only ever sees `advance` while tests can still inspect what
        happened."""
        self._history_events.extend(self._advance(until))

    def _history(self) -> list[Event]:
        """Debug / test channel: the completion events accumulated by every
        `advance` since construction, in order (a copy). This is where actual event
        *times* live (D18 keeps `observe` / `state` status-only); a test reads it to
        verify a run went as planned, without the main loop ever handling times."""
        return list(self._history_events)

    def _advance(self, until: int) -> list[Event]:
        """Internal clock engine: advance to `until`, returning this call's
        completion events in order. Not called directly outside `advance` (its
        events would bypass the history); use `advance` + `_history` instead. Steps
        to each next completion time (<= `until`), applies it, and repeats; when no
        completion remains within reach, jumps the clock to `until` and stops."""
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

            # Advance to that time and settle everything ending exactly then, in
            # dispatch order so ties are deterministic. An operation scheduled to
            # fail (D25) ends `failed` (no material effect) instead of `completed`.
            self._clock = next_end
            due = sorted((op for op in running if op.end == next_end), key=lambda o: o.seq)
            for op in due:
                if op.should_fail:
                    self._fail(op)
                else:
                    self._complete(op)
                events.append(Event(time=next_end, uuid=op.uuid, kind=op.kind, status=op.status))
        # Apply any device faults whose time the clock has now reached (D21).
        self._apply_faults()
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

        # Value seam (D26/D27): a completed operation dispatched with a signature
        # produces a value at each output port, via a device model -- the injected
        # one, or the built-in `default_device_model` (typed defaults + object
        # pass-through from `objects.map`) when none was injected (D27 F4b).
        if op.output_schema is not None:
            model = self._device_model if self._device_model is not None else default_device_model
            op.outputs = model(op.process, op.mode, op.inputs or {}, op.output_schema, op.definition)

        op.status = "completed"

    def _fail(self, op: _Op) -> None:
        """Apply an operation's failure (D25): free its resources but apply **no**
        material effect. Inputs are left where they are and no output is produced --
        a failed operation makes no physical progress. The spot occupancy is thus
        exactly what it was before the operation ran (idle material rests in place),
        which is coherent because the run stops on failure and nothing follows."""
        for d in op.devices:
            self._busy_devices.discard(d)
        if op.transporter is not None:
            self._busy_transporters.discard(op.transporter)
        op.status = "failed"
