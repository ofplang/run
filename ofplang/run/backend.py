"""The execution backend contract the runner drives (the `Backend` Protocol).

The runner (`ofplang.run.runner`) is written against a small, structural contract
-- the operations `RollingRunner` actually calls on whatever executes work -- not
against the concrete `Simulator`. Codifying that contract as a `typing.Protocol`
lets an alternative backend (e.g. one driving real lab hardware) be injected via
`RollingRunner(backend_factory=...)` without inheriting from anything: any object
with these methods is a `Backend`.

Three facts shape the contract, so a single Protocol covers both a simulated and a
real-hardware backend:

* **Time is the backend's.** `advance(until)` blocks until the backend's clock has
  reached `until` and returns the time actually reached. A simulator jumps its
  virtual clock there instantly (deterministic); a real backend sleeps out the
  remaining wall-clock time. The runner adopts the returned time as `now`, so the
  loop is wall-clock-driven for a real backend and unchanged for the simulator.
* **Completion is discovered, not dictated.** `dispatch_*` *starts* an operation
  and returns a handle immediately (it never blocks until the operation finishes);
  the runner learns an operation completed only by polling `state(handle)`. The
  `duration` passed to a dispatch is therefore advisory -- a simulator honours it,
  a real backend may ignore it and let the machine take however long it takes. An
  overrun is just an operation still `running` at the next poll, which the rolling
  loop absorbs via replanning and `running_task_margin`.
* 🔴 **The backend is told, not asked.** Nothing here reports the state of the
  world back. The runner says what to do and learns whether it worked; **where
  material is and what a stock holds are derived from that, never queried.** Both
  halves of that follow from the same fact: a real laboratory keeps no ledger of
  its spots or its levels, so a query has no truthful implementation and a backend
  that answered one would be answering from bookkeeping of its own. The two reads
  that are here are about the backend's own affairs rather than about the world it
  acts on -- `state(handle)` is the outcome of an operation this runner started,
  and `down_devices()` is which of the backend's machines are out of service, which
  only it can know. Where the runner's derivation and the backend's reality
  disagree, the disagreement surfaces as an operation that **fails**: the runner
  seeds declared occupancy with `place`, so work aimed at a spot that is really
  full is refused loudly rather than succeeding against a world the plan does not
  match. `place` and `clear` are that channel in both directions, and both are
  commands: they say what has happened to a spot, and nothing comes back.

Only the methods the runner actually calls live here. Simulator-specific surface
-- fault/failure injection, `observe`, `remove`, `dispatch_relay`, and `spot_state`
-- is not part of the contract. `spot_state` was, until the runner stopped asking:
it is the simulator's own occupancy ledger, useful to a test that wants to check a
derivation against it, and not something a backend can be required to have.

The contract does grow -- and shrink. Growing breaks a backend that predates the
growth, conformance being structural; shrinking cannot, since an implementation that
still has the method simply is not asked for it. `spot_state` left when the runner
stopped asking; `clear` arrived with job withdrawal, and costs the backends in this
ecosystem nothing because all of them derive from `Simulator`, which has it.
`dispatch_replenishment` arrived in 0.3.0,
and because conformance here is structural, a backend that predates it stops being
a `Backend`. That was the deliberate choice over an optional-capability probe --
a runner that can plan refills but not carry them out is a worse thing to ship than
a version bump. The replay `Runner` (deterministic plan replay) targets the
simulator directly and is not backend-injectable, so its use of `now` is not
required here.

The built-in `Simulator` declares `Backend` as an explicit base, so mypy checks it
(and its `VirtualTimeSimulator` / `RealTimeSimulator` subclasses) against this
contract statically; a third-party backend need only match structurally -- no
inheritance -- to be injectable.
"""

from __future__ import annotations

from typing import Protocol


class Backend(Protocol):
    """The minimal execution-backend surface `RollingRunner` drives.

    An implementation is built per run from the (mode-id-normalized) environment
    by a `backend_factory(environment) -> Backend`; see `RollingRunner`.
    """

    def advance(self, until: int) -> int:
        """Block until the backend's current time is at least `until`; return the
        time actually reached (>= `until`).

        A simulator settles its virtual clock to `until`, applying every completion
        along the way, and returns `until`. A real backend waits out the remaining
        real time and returns its wall-derived current time (which may exceed
        `until` if the wait overshot). Completions are revealed by `state`, never by
        this call.
        """
        ...

    def down_devices(self) -> list[str]:
        """The ids of machines currently unavailable, so the runner can schedule
        against a reduced environment (a re-route). Empty when all are up.

        A machine is a **device, a transporter or a replenisher**: an id here drops
        the device's modes / its spots' transports (per the runner's `DownScope`),
        every transport the transporter carries, or every refill the replenisher
        performs. The name is historical -- devices came first -- and the three share
        one id space, so machines with the same id cannot be told apart here (the
        scheduler refuses such an environment: `machine_id_conflict`)."""
        ...

    def place(self, spot: str, obj_id: str | None = None) -> str:
        """Put material on a spot (e.g. seed the interface inputs before a run).
        `obj_id` is optional; when omitted the backend assigns an opaque id. Returns
        the id now held."""
        ...

    def clear(self, spot: str) -> None:
        """The material on `spot` has been taken away; stop accounting for it.

        The counterpart of `place`, and a command like it -- the runner is saying what
        happened, not asking what is there. It is used where a job **leaves** the plan
        (SPEC §6.11): leaving a *bound* final output behind is the caller's assertion
        that they collected it from the spot they named, so a backend still holding it
        would refuse the next delivery to that spot for material nobody has.

        **Tolerant**: a spot that holds nothing is not an error. The runner derives
        what a job was holding rather than asking (see the third fact above), so this
        is told on the strength of a derivation, and a backend that disagrees should
        not turn a disagreement into a crash. A real backend may do nothing at all --
        the plate is already gone from the bench.
        """
        ...

    def dispatch_processing(
        self,
        process: str,
        mode: str,
        duration: int | None = None,
        output_schema=None,
        inputs=None,
        definition=None,
    ) -> str:
        """Start a processing operation and return its handle immediately (does not
        block until completion; poll `state` for that).

        `duration` is advisory (a simulator's expected runtime; a real backend may
        ignore it). `output_schema` is the value-seam signature `{port: descriptor}`
        for the typed value the backend produces per output port at completion,
        revealed via `state`. `inputs` are the assembled input view values, and
        `definition` the workflow process definition, for a value-computing backend.
        """
        ...

    def dispatch_transport(
        self,
        transporter: str | None,
        from_spot: str,
        to_spot: str,
        duration: int | None = None,
        view=None,
    ) -> str:
        """Start a transport moving material `from_spot` -> `to_spot` and return its
        handle immediately. `transporter` may be `None` for a same-spot no-op move;
        `duration` is advisory (see `dispatch_processing`).

        `view` is the view value of the Object being moved (the producing arc's
        output, resolved by the runner), passed so a backend that *runs* a transport
        (e.g. a real-hardware one) can act on what it is carrying. It is advisory and
        best-effort: `None` when the runner cannot resolve it, and the built-in
        simulator ignores it entirely (a physical move needs no view)."""
        ...

    def dispatch_replenishment(
        self,
        replenisher: str,
        device: str,
        amounts: dict | None = None,
        duration: int | None = None,
    ) -> str:
        """Start a refill of `device` by `replenisher` and return its handle
        immediately. `duration` is advisory (see `dispatch_processing`).

        A refill occupies **both** machines for its visit -- the device being filled
        and the replenisher filling it -- which is the one thing about a refill the
        scheduler relies on: it plans them exclusive, so a backend that let them
        overlap would be running a schedule nobody proved.

        `amounts` is what the visit puts in, ``{resource: amount}``, derived by the
        scheduler as a fill to capacity. It is passed for a backend that really does
        put something in; a simulator need not read it. Nothing comes back: a level is
        replayed from what the run started with plus its history (SPEC §4.7.2), never
        read off a device -- the contract's third fact, above.
        """
        ...

    def state(self, uuid: str) -> dict:
        """One operation's current state: at least `{"status": "running" |
        "completed" | "failed"}`. A completed value-carrying processing also reports
        `"outputs"` ({port: value}); a failed operation may report `"reason"`
        ((code, message)). Errors if the handle is unknown."""
        ...
