"""Rolling-horizon runner (spec §7; dev-notes D9/D20-D23).

Drives a workflow to completion by replanning as it goes. Each tick it renders
its committed history as an execution status (§6/§7), calls the scheduler for a
fresh plan, dispatches the pending activities that can start now, then advances
the virtual clock and polls the backend. It repeats until the scheduler returns a
plan with no pending work. Every tick feeds the committed history (completed /
running activities, `now`, the `interface` boundary) back to the scheduler, which
fixes it and re-optimises the rest -- the replan round-trip (D9). Only committed
history is stable across replans (D9); pending identities may be regenerated each
replan, so pending work is always re-read from the fresh plan.

What this layer covers, added incrementally:

* rolling-horizon core (D9/D20): the replan loop above.
* re-routing (D21): when a device goes down, the environment scheduled against is
  reduced (its process modes dropped) so the scheduler re-routes pending work.
* poll modes (D22): fixed-interval polling is the standard -- an integer
  `poll_interval` (default 1) polls every that many units and estimates each
  completion time as the observing poll. `poll_interval=None` advances to plan
  event boundaries instead (exact, deterministic), retained for tests.
* duration variance (D23): an optional `duration_model` perturbs the dispatched
  duration (the backend runs the actual, the runner reports the planned expected
  end until it observes completion). Variance requires fixed-interval polling and
  a positive running-task margin (so an overrun's successors are not dispatched
  onto a still-busy resource).
* failure stop (D25): when a poll observes an operation `failed` (an injected
  capability failure, configured on the simulator), the run stops -- it dispatches
  no more work and only waits for what is still running to finish (no abort signal
  is sent). The final status marks the failed activity `failed` and the work that
  never started `cancelled`; `self.failed` is set (the CLI maps it to exit 1).
"""

from __future__ import annotations

import copy

from ..simulator import Simulator
from .contracts import Contracts, conforms, to_descriptor
from .dataflow import from_workflow
from .loader import load_document
from .provenance import Committed, CommitLog
from .runner import RunnerError
from .schedule_client import replan
from .status import build_status
from .values import ValueStore, assemble_inputs, collect_outputs, record_outputs, seed_entry


def _normalize_mode_ids(environment: dict) -> dict:
    """Return a copy of `environment` with an explicit `id` on every process mode
    that lacks one.

    This must happen before any reduction: dropping a mode renumbers the remaining
    position-based ids, so a reduced-env plan's mode id would no longer map to the
    same physical mode in the backend's full environment. Pinning ids up front
    keeps the id -> mode mapping stable across reduction (D21). Ids are `m<i>`
    rather than the bare position, because a mode id must be a v0 identifier
    (§8.1) and so cannot start with a digit.
    """
    env = copy.deepcopy(environment)
    for process in (env.get("processes") or {}).values():
        for i, mode in enumerate(process.get("modes") or []):
            if mode.get("id") is None:
                mode["id"] = f"m{i}"
    return env


def _reduce_environment(environment: dict, down: set[str]) -> dict:
    """Return a copy of `environment` with every process mode that uses a down
    device removed, keeping the device/spot/transport definitions (spec §7, D21).

    Dropping only the modes is how a re-route is triggered: committed transports to
    a down device's spot stay valid, and a re-transport can still route through it,
    but no new processing is scheduled there.
    """
    reduced = copy.deepcopy(environment)
    for process in (reduced.get("processes") or {}).values():
        process["modes"] = [
            mode for mode in (process.get("modes") or []) if not (set(mode.get("devices") or []) & down)
        ]
    return reduced


class RollingRunner:
    """Drives workflow + environment (+ interface) to completion by replanning."""

    def __init__(
        self,
        workflow_path,
        environment_path,
        interface: dict | None = None,
        *,
        job: dict | None = None,
        device_model=None,
        running_task_margin: int = 0,
        random_seed: int | None = None,
        poll_interval: int | None = 1,
        duration_model=None,
        max_ticks: int = 100_000,
    ):
        self.workflow_path = str(workflow_path)
        self.environment_path = str(environment_path)
        # Keep the environment as a dict too: when devices go down we schedule
        # against a reduced copy of it (D21), while the backend keeps the full one.
        # Mode ids are pinned up front so they stay stable when modes are dropped.
        self._environment = _normalize_mode_ids(load_document(environment_path))
        # The backend reads the environment itself. An optional device model (D27
        # F4b) computes outputs from inputs; without one the built-in
        # `default_device_model` (type defaults + `objects.map` object carry) is
        # used. A scenario concern injected from Python, like `duration_model`.
        self.sim = Simulator(self._environment, device_model=device_model)

        # Value layer (D26). The runner owns view-value routing: `dataflow` is the
        # workflow's port-level routing view (reused from the scheduler's flattener,
        # D26-0/D26-1, so its node paths match the plan's), and `values` stores each
        # produced / seeded value keyed by (node, port). `outputs` holds the
        # whole-workflow outputs, assembled from `returns` at the end of a run. In
        # v0-lite the seam is output-only: dispatch carries the output-port signature
        # so the backend generates values; inputs are not passed (D26).
        self.dataflow = from_workflow(self.workflow_path)
        # Resolved port types (D27 F1): used to build each processing's output value
        # signature so the backend can generate typed values (F2). Precompute the
        # per-process output descriptors ({port: value-shape descriptor}).
        self.contracts = Contracts.from_workflow(self.workflow_path)
        self._output_schemas = {
            name: {port: to_descriptor(rt) for port, rt in pc.outputs.items()}
            for name, pc in self.contracts.processes.items()
        }
        # The raw process definitions (workflow `processes.<name>`), passed to the
        # device model at dispatch so it can act on a process's declared structure
        # (e.g. carry an object output from its `objects.map`). D27 F4b / principle A.
        self._process_defs = (load_document(self.workflow_path) or {}).get("processes") or {}
        # Whole-workflow input values (F4): {entry_port: view value}. Seeded at the
        # boundary at run start; a missing entry input falls back to a typed default.
        self.job = dict(job or {})
        self.values = ValueStore()
        self.outputs: dict = {}

        self.interface = interface or {}
        self.margin = running_task_margin
        self.seed = random_seed
        # Fixed-interval polling is the standard mode (D22): an integer polls every
        # that many ticks, seeing a completion only at a poll and estimating its
        # time. Default 1 (the finest interval). `poll_interval=None` selects
        # event-boundary advance instead -- deterministic and exact, retained for
        # tests.
        self.poll_interval = poll_interval
        # Optional duration variance (D23): fn(activity, planned_duration) -> actual.
        # None means every operation runs for its planned duration.
        self.duration_model = duration_model
        self.max_ticks = max_ticks

        # Variance is only coherent under fixed-interval polling (an off-plan finish
        # cannot be observed by event-boundary advance), and needs a positive
        # running-task margin so a successor of an overrunning operation is not
        # dispatched onto a still-busy resource (D23). The margin is the caller's to
        # set (ideally >= poll_interval); the runner only validates it.
        if duration_model is not None:
            if poll_interval is None:
                raise RunnerError("duration variance requires poll_interval (fixed-interval polling)")
            if running_task_margin < 1:
                raise RunnerError(
                    "duration variance requires running_task_margin >= 1 "
                    "(ideally >= poll_interval, so an overrun defers its successors)"
                )

        self.log = CommitLog()
        self.now = 0
        self.ticks = 0  # number of replan cycles (a test asserts >1: history round-trips)
        self._last_time = None  # `time` section echoed from the most recent plan

        # Failure handling (D25). When an operation is observed `failed`, the runner
        # stops: it dispatches no more work and only waits for what is still running
        # to finish (no abort signal is sent). `failed` marks the overall run as
        # failed; `_stopping` gates the loop; `_last_pending` remembers the last
        # plan's pending (non-relay) activities so they can be reported `cancelled`.
        self.failed = False
        self._stopping = False
        self._last_pending: list[dict] = []

    def run(self) -> dict:
        """Drive to completion and return the final execution status (§6/§7). Raises
        `RunnerError` if a replan produces no plan (infeasible) or the run cannot
        progress; `SimulatorError` propagates if the backend rejects a dispatch.

        On an activity failure the run stops rather than raising: it dispatches no
        more work, waits for what is still running to finish, and returns a final
        status with the failed activity `failed` and the abandoned work `cancelled`
        (D25). `self.failed` records that this happened (the CLI maps it to exit 1)."""
        # Seed the boundary inputs: the entry Objects sit on their interface spots
        # at the start of the run (§6.8), and every entry input port gets its
        # boundary view value seeded from the job (contract-checked) or a typed
        # default (D27 F4).
        for _port, spot in (self.interface.get("inputs") or {}).items():
            self.sim.place(spot)
        seed_entry(self.dataflow, self.contracts, self.values, self.job)

        while True:
            self.ticks += 1
            if self.ticks > self.max_ticks:
                raise RunnerError("exceeded max ticks (possible non-termination)")

            if not self._stopping:
                # Normal tick: replan and dispatch what can start now.
                pending = self._replan_and_dispatch()
                # The run is done when there is neither unstarted work nor anything
                # still running.
                if not pending and not self.log.running():
                    break
            else:
                # Stopping after a failure (D25): dispatch nothing more, just drain.
                # The run ends once nothing is left running -- we never abort a
                # running operation, only wait for it (the user's stop policy).
                if not self.log.running():
                    break
                pending = []

            # Advance the clock, then poll. The advance policy is the only thing that
            # differs between the two modes (D22). A poll may observe a failure and
            # flip `_stopping`.
            self.now = self._next_time(pending)
            self.sim.advance(self.now)
            self._poll()

        # Assemble the whole-workflow outputs from the produced values (D26); exposed
        # via `self.outputs` and `self.values.snapshot()` (v0-lite: a runner-internal
        # channel, not the §6/§7 document).
        self.outputs = collect_outputs(self.dataflow, self.values)

        # A stopped run reports the work that never ran as cancelled (D25).
        cancelled = self._cancelled_activities() if self._stopping else None
        return build_status(self.log.records(), self.now, self.interface, self._last_time, cancelled)

    def _replan_and_dispatch(self) -> list[dict]:
        """One normal tick: build the status from committed history, replan, and
        dispatch every pending activity that can start now. Returns the plan's
        pending (non-relay) activities (also remembered for cancellation)."""
        # Discover which devices are down and schedule against the normalized
        # environment reflecting it: the full env when nothing is down, or a reduced
        # copy (down devices' process modes dropped) that triggers a re-route (D21).
        # Always the normalized dict, so the scheduler and the backend agree on mode
        # ids. Committed history is fed back so it is fixed and the rest re-optimised.
        down = set(self.sim.down_devices())
        environment = _reduce_environment(self._environment, down) if down else self._environment
        status_doc = build_status(self.log.records(), self.now, self.interface)
        report = replan(
            self.workflow_path,
            environment,
            status_doc,
            running_task_margin=self.margin,
            random_seed=self.seed,
        )
        if not report.ok:
            raise RunnerError(self._failure_message(report))
        plan = report.plan
        self._last_time = plan.get("time")

        # Pending work is what carries no status (relays are scheduler-derived and
        # never dispatched, §7). Remembered so that, if a failure stops the run, the
        # work that never started can be reported cancelled (D25).
        pending = [
            a
            for a in plan.get("activities", [])
            if a.get("status") in (None, "pending") and a.get("kind") != "relay"
        ]
        self._last_pending = pending

        # Dispatch everything that can start now. Pending is optimised at/after
        # `now`, so these are the entries at exactly `now`; their predecessors
        # finished by now (we polled on the previous advance), so the backend's
        # preconditions hold.
        for act in pending:
            if int(act["start"]) <= self.now:
                self._commit_start(act)
        return pending

    def _next_time(self, pending: list[dict]) -> int:
        """The virtual time to advance to next. In fixed-interval mode, one poll
        interval on; in event-boundary mode, the earliest future pending start or
        running-operation finish (or `now` if there is none, letting a settle pass
        clear zero-duration work)."""
        if self.poll_interval is not None:
            return self.now + self.poll_interval
        future = [int(a["start"]) for a in pending if int(a["start"]) > self.now]
        future += [r.end for r in self.log.running() if r.end > self.now]
        return min(future) if future else self.now

    def _commit_start(self, activity: dict) -> None:
        """Start a pending activity now: dispatch it to the backend (or record a
        same-spot no-op as bookkeeping) and add it to the committed history."""
        kind = activity["kind"]
        start = self.now
        planned = int(activity["end"]) - int(activity["start"])

        # A same-spot transport is a physical no-op: no backend operation, no
        # variance, completed at once (D14/D19). It is still a committed leg, so it
        # is recorded (the scheduler pins the chain by it on the next replan).
        if kind == "transport" and activity.get("from_spot") == activity.get("to_spot"):
            self.log.add(Committed(activity, kind, "completed", start, start + planned, uuid=None))
            return

        # The backend runs the *actual* duration (the variance model perturbs the
        # plan, D23). The committed record's `end` is the *planned* expected finish:
        # the runner does not know the actual until the op is observed complete, so
        # it reports the plan and lets `_poll` overwrite `end` with the poll time.
        # A processing duration must stay positive (§5.5); a transport may be zero.
        if self.duration_model is None:
            actual = planned
        else:
            floor = 1 if kind == "processing" else 0
            actual = max(floor, int(self.duration_model(activity, planned)))
        end = start + planned

        if kind == "processing":
            # Pass the output value signature (D26/D27) so the backend generates a
            # typed value for each output port at completion, and the assembled input
            # values (F4; routed from upstream / the seeded boundary). The backend
            # records inputs but does not yet use them (F4b).
            output_schema = self._output_schemas.get(activity["process"], {})
            inputs = assemble_inputs(self.dataflow, self.contracts, self.values, activity["node"])
            uuid = self.sim.dispatch_processing(
                activity["process"], activity["mode"], duration=actual,
                output_schema=output_schema, inputs=inputs,
                definition=self._process_defs.get(activity["process"]),
            )
        elif kind == "transport":
            uuid = self.sim.dispatch_transport(
                activity.get("transporter"), activity["from_spot"], activity["to_spot"], duration=actual
            )
        else:  # pragma: no cover - schema guarantees processing/transport/relay
            raise RunnerError(f"unknown activity kind: {kind!r}")
        self.log.add(Committed(activity, kind, "running", start, end, uuid=uuid))

    def _poll(self) -> None:
        """Mark running operations the backend reports as finished (status-only, D18).

        The completion time is recorded as the current poll time `now`. In
        event-boundary mode `now` is exactly the planned end (we advanced to it); in
        fixed-interval mode it is the poll at which completion was first seen -- an
        upper bound on the true finish, the best a poll-only observer can know (D22).

        An operation observed `failed` (D25) is recorded `failed` and stops the run:
        no more work is dispatched, only the still-running operations are awaited.
        """
        for rec in self.log.running():
            if rec.uuid is None:
                continue
            observed_state = self.sim.state(rec.uuid)
            observed = observed_state["status"]
            if observed == "completed":
                rec.status = "completed"
                rec.end = self.now
                # Record the values the backend produced (D26); only value-carrying
                # processing ops report `outputs`, keyed here by their node path. Each
                # output is contract-checked against its port type (D27 F4): the F2
                # defaults always conform, but a future device model / real backend
                # (F4b) could emit a non-conformant value, caught here.
                if "outputs" in observed_state:
                    process = rec.activity["process"]
                    for port, value in observed_state["outputs"].items():
                        if not conforms(value, self.contracts.output_type(process, port)):
                            raise RunnerError(
                                f"backend output {process}.{port!r} does not conform to its declared type"
                            )
                    record_outputs(self.values, tuple(rec.activity["node"]), observed_state["outputs"])
            elif observed == "failed":
                rec.status = "failed"
                rec.end = self.now
                self.failed = True
                self._stopping = True

    def _cancelled_activities(self) -> list[dict]:
        """The last plan's pending activities that never started because the run
        stopped on a failure (D25) -- the pending set minus what got committed."""
        committed = {self._provenance_key(r.activity) for r in self.log.records()}
        return [a for a in self._last_pending if self._provenance_key(a) not in committed]

    @staticmethod
    def _provenance_key(activity: dict):
        """A stable identity for an activity across replans: its workflow provenance
        (a processing's `node` path, a transport's `arc` endpoints + `seq`). Pending
        identities are regenerated each replan, but provenance is not, so this lines
        a committed activity up against a pending one (D9)."""
        kind = activity.get("kind")
        if kind == "processing":
            return ("processing", tuple(activity.get("node") or ()))
        # transport: identify by the logical arc it serves and its chain position.
        arc = activity.get("arc") or {}

        def endpoint(e):
            e = e or {}
            return (tuple(e.get("node") or ()), e.get("port"))

        return ("transport", endpoint(arc.get("from")), endpoint(arc.get("to")), activity.get("seq"))

    @staticmethod
    def _failure_message(report) -> str:
        codes = ", ".join(str(getattr(d, "code", d)) for d in report.diagnostics)
        detail = f" ({codes})" if codes else ""
        return f"scheduler produced no plan; outcome={report.outcome}{detail}"
