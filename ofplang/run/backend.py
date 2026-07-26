"""The execution backend contract the runner drives (the `Backend` Protocol).

The runner (`ofplang.run.runner`) is written against a small, structural contract
-- the operations `RollingRunner` actually calls on whatever executes work -- not
against the concrete `Simulator`. Codifying that contract as a `typing.Protocol`
lets an alternative backend (e.g. one driving real lab hardware) be injected via
`RollingRunner(backend_factory=...)` without inheriting from anything: any object
with these methods is a `Backend`.

Two facts shape the contract, so a single Protocol covers both a simulated and a
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

Only the methods the runner actually calls live here. Simulator-specific surface
-- fault/failure injection, `observe`, `remove`, `dispatch_relay` -- is not part
of the contract. The replay `Runner` (deterministic plan replay) targets the
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
        """The ids of devices currently unavailable, so the runner can schedule
        against a reduced environment (a re-route). Empty when all are up."""
        ...

    def place(self, spot: str, obj_id: str | None = None) -> str:
        """Put material on a spot (e.g. seed the interface inputs before a run).
        `obj_id` is optional; when omitted the backend assigns an opaque id. Returns
        the id now held."""
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
    ) -> str:
        """Start a transport moving material `from_spot` -> `to_spot` and return its
        handle immediately. `transporter` may be `None` for a same-spot no-op move;
        `duration` is advisory (see `dispatch_processing`)."""
        ...

    def state(self, uuid: str) -> dict:
        """One operation's current state: at least `{"status": "running" |
        "completed" | "failed"}`. A completed value-carrying processing also reports
        `"outputs"` ({port: value}); a failed operation may report `"reason"`
        ((code, message)). Errors if the handle is unknown."""
        ...

    def spot_state(self, spot: str | None = None):
        """Inspect spot occupancy (the runner reads this only at run end, to confirm
        a boundary output was delivered). With `spot`, return its object id or
        `None`; without, return the ``{spot: obj_id}`` map of occupied spots."""
        ...
