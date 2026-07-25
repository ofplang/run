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
from dataclasses import dataclass

from ..simulator import Simulator
from .boundary import parse_boundary
from .contract_eval import evaluate, parse as parse_contract, referenced_ports
from .contracts import ArrayType, Contracts, conforms, to_descriptor, with_static_views
from .dataflow import from_workflow
from .loader import load_document
from .provenance import Committed, CommitLog
from .runner import RunnerError
from .schedule_client import replan
from .status import build_status
from .values import ValueStore, assemble_inputs, collect_outputs, record_outputs, seed_entry


@dataclass
class Failure:
    """Why a run stopped (D36): a machine-readable `kind` (reason code), a
    human-readable `detail`, the `subject` that failed (a node path label, `main`, or
    an activity), and the virtual time `now` at which it was detected. The runner
    records the first failure that stopped the run; the CLI prints it and the final
    status echoes it."""

    kind: str
    detail: str
    subject: str
    now: int


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
    """Drives workflow + environment (+ boundary) to completion by replanning.

    `boundary` is the run I/O document (D28): a `boundary:` mapping with per-port
    `{spot, view}` descriptors for the workflow's entry inputs and final outputs.
    The runner projects it into the scheduler `interface` (spots only), the seed
    `job` (input views), and the pinned output spots (checked at run end). None
    means no boundary (no Object placement, all entry inputs defaulted)."""

    def __init__(
        self,
        workflow_path,
        environment_path,
        boundary: dict | None = None,
        *,
        device_model=None,
        running_task_margin: int = 0,
        random_seed: int | None = None,
        poll_interval: int | None = 1,
        duration_model=None,
        contract_observer=None,
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
        # Parsed contract expressions (v0 §9 / D32), per process:
        # {process: {"requires": [(expr, ast)], "ensures": [(expr, ast)]}}. Checked for
        # each atomic process and for the top-level entry composite (Phase 1); nested
        # composite contracts are deferred. A process with no `contracts` section is
        # absent. Parsed once up front, so a malformed expression surfaces here rather
        # than mid-run (valid v0 parses cleanly -- validate has already type-checked it).
        self._contract_asts = self._parse_contracts()
        # Whether the entry process is a composite (the usual case). Its contracts are
        # the whole-workflow envelope, checked at run start / run end (D33); an atomic
        # entry is instead a single activity, checked on the activity path.
        self._entry_is_composite = (self._process_defs.get(self.contracts.entry) or {}).get("kind") == "composite"
        # Nested composite contract checks (D34): the composite invocation boundaries
        # (keyed by node path) and the per-invocation sets of already-checked contracts,
        # so each `requires` / `ensures` fires once, when its values become available.
        self._composites = self.dataflow.composites
        self._checked_requires: set = set()
        self._checked_ensures: set = set()
        # The run boundary (D28): the single run-facing I/O document. Parsed against
        # the resolved contracts into the pieces the run needs -- the §6.8 `interface`
        # (spots only) handed to the scheduler, the `job` ({entry_port: view value})
        # seeded at run start, and the Object outputs pinned to a delivery spot
        # (checked at run end, P3). View values never reach the scheduler: the
        # interface projection is value-free, so an unpinned output can never become a
        # scheduling constraint on a replan. Structural boundary errors (an unknown
        # port, a missing / stray spot) surface here, up front; a supplied view value's
        # conformance is checked when it is seeded.
        self.boundary = parse_boundary(boundary, self.contracts)
        self.interface = self.boundary.interface
        # Whole-workflow input values (F4): {entry_port: view value}. Seeded at the
        # boundary at run start; a missing entry input falls back to a typed default.
        self.job = self.boundary.job
        self.values = ValueStore()
        self.outputs: dict = {}
        # The result boundary (D28): the same-schema document echoing the produced
        # output views, assembled at the end of a run (the CLI writes it to
        # `--boundary-out`). Empty until `run()` completes.
        self.result_boundary: dict = {}
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
        # Observability (D36): `failure` is the first reason that stopped the run (a
        # `Failure`, or None), reported by the CLI and echoed into the final status.
        # `contract_observer`, if given, is called for every contract check (held or
        # violated) with a `{subject, process, section, expr, held, now}` record -- an
        # optional trace hook; None (the default) means no per-check reporting.
        self.failure: Failure | None = None
        self._contract_observer = contract_observer

    @staticmethod
    def _fmt_node(node) -> str:
        """A node path rendered as a readable label for reasons / traces; the empty
        path (the workflow boundary) is `main`."""
        return "/".join(node) if node else "main"

    @staticmethod
    def _activity_subject(activity: dict) -> str:
        """A readable subject label for a failed activity (D36): a processing's node
        path, or a transport's `from_spot -> to_spot`."""
        node = activity.get("node")
        if node is not None:
            return "/".join(node) if node else "main"
        return f"{activity.get('from_spot')} -> {activity.get('to_spot')}"

    def _record_failure(self, kind: str, detail: str, subject: str) -> None:
        """Record why the run stopped (D36), first failure wins (later ones are the
        cascade of the first). Does not itself stop the run -- the caller sets
        `failed` / `_stopping`."""
        if self.failure is None:
            self.failure = Failure(kind=kind, detail=detail, subject=subject, now=self.now)

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

        # Whole-workflow precondition contracts (v0 §9 `requires` on the entry composite,
        # D32 Phase 1): checked once the boundary inputs are seeded, before any work is
        # dispatched. A violation stops the run before it starts (graceful, D25): no
        # activity runs, `self.failed`/`_stopping` are set, and the loop below breaks
        # immediately (nothing is running), so the final status is emptily terminal.
        entry = self.contracts.entry
        if self._entry_is_composite and self._contract_asts.get(entry, {}).get("requires"):
            if self._violated_contract(entry, "requires", self._main_contract_inputs(), {}, "main") is not None:
                self.failed = True
                self._stopping = True

        # Atomic preconditions that are knowable at run start -- `requires` referencing
        # only run/graph-phase inputs (D37) -- are checked now, before any dispatch, so a
        # violation stops the run before the (possibly late-dispatched) process runs.
        self._preflight_atomic_requires()

        # Nested composite contracts whose values are already available at run start
        # (inputs bound only to the boundary / literals) are checked now, before any
        # dispatch -- so a `requires` violation still precedes the composite's body (D34).
        self._check_ready_composites()

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
            # A poll may have recorded the last value a nested composite's contract was
            # waiting on -- check any that just became ready (D34).
            self._check_ready_composites()

        # Assemble the whole-workflow outputs from the produced values (D26); exposed
        # via `self.outputs` and `self.values.snapshot()` (v0-lite: a runner-internal
        # channel, not the §6/§7 document).
        self.outputs = collect_outputs(self.dataflow, self.values)

        # On a run that completed, verify each pinned Object output actually reached
        # its declared delivery spot (P3, D28). The §6.8 interface_out node holds the
        # spot to the makespan, so a completed run must leave it occupied; an empty
        # spot means the boundary delivery did not happen -- an inconsistency, raised.
        # Skipped on a failed / stopped run (delivery legitimately may not have run).
        if not self.failed:
            self._check_output_spots()
            # Whole-workflow postcondition contracts (v0 §9 `ensures` on the entry
            # composite, D32 Phase 1): checked once the outputs are assembled, over the
            # boundary inputs and produced outputs. A violation is a runtime contract
            # violation (v0 §9.3): set `self.failed` (exit 1). The activities stay
            # `completed` -- the failure is at the whole-workflow boundary, not any one
            # activity. Only checked on an otherwise-successful run (the guard above).
            entry = self.contracts.entry
            if self._entry_is_composite and self._contract_asts.get(entry, {}).get("ensures"):
                if self._violated_contract(entry, "ensures", self._main_contract_inputs(), self.outputs, "main") is not None:
                    self.failed = True
        # Echo the produced output views back into a result document of the same
        # boundary schema (D28), for `--boundary-out`.
        self.result_boundary = self.boundary.result(self.outputs)

        # A stopped run reports the work that never ran as cancelled (D25). The failure
        # reason (D36) is NOT put in the status -- it stays a valid §6 document -- but is
        # exposed via `self.failure` (and printed by the CLI).
        cancelled = self._cancelled_activities() if self._stopping else None
        return build_status(self.log.records(), self.now, self.interface, self._last_time, cancelled)

    def _check_output_spots(self) -> None:
        """Verify every pinned Object output landed on its declared delivery spot
        (P3, D28). The runner does not read spot state in normal operation (D15);
        this is the one end-of-run sanity read. Raises `RunnerError` on a spot the
        boundary delivery left empty."""
        for port, spot in self.boundary.output_spots.items():
            if self.sim.spot_state(spot) is None:
                raise RunnerError(
                    f"boundary output {port!r} did not reach its declared spot {spot!r}"
                )

    def _parse_contracts(self) -> dict:
        """Parse every process's `contracts` (v0 §9) into ASTs, keyed by process and
        section. All process kinds are parsed here; where each is *checked* is decided
        at run time by the process's role -- an atomic process on the activity path
        (D32), the entry composite at the run boundary (D33), a nested composite when
        its values become available (D34). A process with no `contracts` produces no
        entry.

        For an atomic process, `requires` is split by phase (D37): an expression
        referencing only run/graph-phase inputs is knowable at run start and goes to
        `requires_preflight` (checked before dispatch); one reading any data-phase input
        stays in `requires` (checked at dispatch). Composite `requires` is not split
        (the entry composite is already a run-boundary check, D33)."""
        result: dict = {}
        for name, pdef in self._process_defs.items():
            pdef = pdef or {}
            contracts = pdef.get("contracts") or {}
            if not contracts:
                continue
            is_atomic = pdef.get("kind") == "atomic"
            parsed: dict = {}
            for section in ("requires", "ensures"):
                exprs = [
                    (item["expr"], parse_contract(item["expr"]))
                    for item in (contracts.get(section) or [])
                    if item and item.get("expr") is not None
                ]
                if not exprs:
                    continue
                if section == "requires" and is_atomic:
                    # `requires` references only inputs (v0 §9.1); an expression is
                    # preflightable iff every input it reads is non-data phase (v0 §6),
                    # hence knowable at run start.
                    preflight = [
                        (expr, ast)
                        for expr, ast in exprs
                        if all(self.contracts.input_phase(name, port) != "data" for _s, port in referenced_ports(ast))
                    ]
                    runtime = [pair for pair in exprs if pair not in preflight]
                    if preflight:
                        parsed["requires_preflight"] = preflight
                    if runtime:
                        parsed["requires"] = runtime
                else:
                    parsed[section] = exprs
            if parsed:
                result[name] = parsed
        return result

    def _preflight_atomic_requires(self) -> None:
        """Run-start preflight (D37): check each atomic invocation's phase-hoisted
        preconditions (`requires_preflight` -- those over run/graph-phase inputs alone)
        before any work is dispatched, over the run-start-available values. A violation
        stops the run before it starts (like an atomic `requires`, but caught up front,
        so no dependent work runs). Skipped once the run is already stopping."""
        if self._stopping:
            return
        for node, process in self.dataflow.process_of.items():
            # Only the preflight candidates whose referenced inputs are *actually*
            # fixed at run start for this node (boundary / literal / unconnected) are
            # checked here; any candidate reading a producer-fed input is deferred to
            # dispatch (checked once the producer has run), so we never evaluate a
            # requires against a not-yet-produced input's typed default.
            checkable, _deferred = self._split_preflight(node, process)
            if not checkable:
                continue
            inputs = assemble_inputs(self.dataflow, self.contracts, self.values, node)
            if self._violated_exprs(process, "requires_preflight", checkable, inputs, {}, self._fmt_node(node)) is not None:
                self.failed = True
                self._stopping = True
                return

    def _main_contract_inputs(self) -> dict:
        """The entry composite's input view values, for its own whole-workflow contract
        checks (v0 §9 on `main`, D32 Phase 1): each declared entry input read from the
        boundary-seeded value store. Every entry input is seeded at run start
        (`seed_entry`), so all are present."""
        entry = self.contracts.entry
        return {port: self.values.get((), port) for port in self.contracts.processes[entry].inputs}

    def _contract_resolver(self, process: str, inputs: dict, outputs: dict):
        """Build the `resolve(scope, port, fields)` callback `contract_eval` needs:
        map a `.view` reference to this invocation's actual view value. A bare `.view`
        is the value itself (a primitive scalar or a view record); `.view.length` on
        an Array is its element count; `.view.<field>` on a nominal is that view
        field (v0 §9.1)."""
        def resolve(scope: str, port: str, fields: tuple):
            value = (inputs if scope == "inputs" else outputs)[port]
            if not fields:
                return value
            rtype = (
                self.contracts.input_type(process, port)
                if scope == "inputs"
                else self.contracts.output_type(process, port)
            )
            if isinstance(rtype, ArrayType):
                return len(value)  # the only Array view field is `length`
            return value[fields[0]]  # nominal view record field

        return resolve

    def _input_available_at_start(self, node, port: str) -> bool:
        """Whether input `port` of `node` has a value fixed at run start.

        True when the port is fed by the boundary (a seeded entry input), bound to a
        static literal, or unconnected (a typed default) -- all fixed before any work
        runs. False when a producing node feeds it: that value is not known until the
        producer completes, so a `requires` over it cannot be preflighted (D37 assumed
        run-phase inputs are always boundary/literal; a legal run->run producer output
        breaks that assumption). `input_source` uses `()` for the boundary node."""
        source = self.dataflow.input_source.get((tuple(node), port))
        if source is not None:
            return source[0] == ()
        return True  # a static literal or an unconnected input: fixed at run start

    def _split_preflight(self, node, process: str):
        """Partition a process's preflight-candidate `requires` at `node` into those
        actually checkable at run start (every referenced input fixed at run start)
        and those deferred to dispatch (a referenced input is producer-fed). The split
        is a static property of the dataflow, so it is the same at preflight and at
        dispatch -- guaranteeing each expression is checked exactly once."""
        candidates = self._contract_asts.get(process, {}).get("requires_preflight") or []
        checkable, deferred = [], []
        for pair in candidates:
            _expr, ast = pair
            if all(self._input_available_at_start(node, port) for _s, port in referenced_ports(ast)):
                checkable.append(pair)
            else:
                deferred.append(pair)
        return checkable, deferred

    def _violated_contract(self, process: str, section: str, inputs: dict, outputs: dict, subject: str):
        """Evaluate `process`'s stored `section` (requires / ensures) contracts for
        `subject`; see `_violated_exprs`."""
        return self._violated_exprs(
            process, section, self._contract_asts.get(process, {}).get(section) or [], inputs, outputs, subject
        )

    def _violated_exprs(self, process: str, section: str, exprs, inputs: dict, outputs: dict, subject: str):
        """Evaluate an explicit list of `(expr, ast)` contracts for `subject` and return
        the first violated expression, or None if all hold.

        Every expression is evaluated (so the optional `contract_observer` sees each
        one, held or violated, D36; v0 §9.2 permits evaluating all at runtime), and the
        first violation is recorded as the run's failure reason under `section`. A
        contract that evaluates false -- or whose runtime evaluation errors (v0 §9.2) --
        is a runtime contract violation (v0 §9.3)."""
        if not exprs:
            return None
        resolve = self._contract_resolver(process, inputs, outputs)
        first_violation = None
        for expr, ast in exprs:
            try:
                held = bool(evaluate(ast, resolve))
            except Exception:
                held = False  # a runtime evaluation error counts as a violation (v0 §9.2)
            if self._contract_observer is not None:
                self._contract_observer(
                    {"subject": subject, "process": process, "section": section,
                     "expr": expr, "held": held, "now": self.now}
                )
            if not held and first_violation is None:
                first_violation = expr
        if first_violation is not None:
            self._record_failure(f"contract_{section}", f"{subject}: {first_violation}", subject)
        return first_violation

    def _composite_ready(self, mapping: dict) -> bool:
        """Whether every value-store key in `mapping` (a composite's inputs or outputs,
        port -> (node, port)) has been produced / seeded. Literal-bound ports are not
        in `mapping`, so they never gate readiness (their value is always available)."""
        return all(self.values.has(node, port) for (node, port) in mapping.values())

    def _composite_values(self, mapping: dict, literals: dict) -> dict:
        """A composite's port -> view value map: each routed port read from the value
        store, plus each literal-bound port's constant."""
        values = {cport: self.values.get(node, port) for cport, (node, port) in mapping.items()}
        values.update(literals)
        return values

    def _check_ready_composites(self) -> None:
        """Check each nested composite invocation's contracts (v0 §9 / D34) as soon as its
        values are available: `requires` once all its inputs are present, `ensures`
        once all its inputs and outputs are. Each invocation is checked once (tracked in
        `_checked_requires` / `_checked_ensures`). A violation stops the run gracefully
        (D25) at the composite boundary -- no single activity is marked failed, like the
        entry composite (D33). Skipped once the run is already stopping."""
        if self._stopping:
            return
        for path, b in self._composites.items():
            asts = self._contract_asts.get(b.process)
            if not asts:
                continue  # this composite declares no contracts
            # `requires`: over the composite's inputs, checked before its body's
            # input-dependent activities can run (they wait on the same values).
            if "requires" in asts and path not in self._checked_requires and self._composite_ready(b.inputs):
                self._checked_requires.add(path)
                inputs = self._composite_values(b.inputs, b.input_literals)
                if self._violated_contract(b.process, "requires", inputs, {}, self._fmt_node(path)) is not None:
                    self.failed = True
                    self._stopping = True
                    return
            # `ensures`: over the composite's inputs and outputs, checked once its
            # outputs exist (before any downstream consumer of them runs).
            if (
                "ensures" in asts
                and path not in self._checked_ensures
                and self._composite_ready(b.inputs)
                and self._composite_ready(b.outputs)
            ):
                self._checked_ensures.add(path)
                inputs = self._composite_values(b.inputs, b.input_literals)
                outputs = self._composite_values(b.outputs, b.output_literals)
                if self._violated_contract(b.process, "ensures", inputs, outputs, self._fmt_node(path)) is not None:
                    self.failed = True
                    self._stopping = True
                    return

    def _requires_gate_open(self, node) -> bool:
        """Whether every ancestor composite of `node` that declares a `requires` has
        had it checked (v0 §9). A composite's precondition gates its whole body: no
        body activity may run until the composite's `requires` is verified. Because the
        composite is flattened, `node` (a tuple path) lies inside composite `c` exactly
        when `c` is a proper prefix of it.

        The gate closes the D34 gap where a body activity independent of the
        composite's inputs (e.g. an `objects.create`, or a literal-only binding) had no
        dataflow dependency on those inputs, so it could dispatch at run start -- before
        the composite's `requires` (checked once its producer-fed inputs arrive) was
        ever evaluated. A violation sets `_stopping`, which halts all dispatch, so the
        gate only needs "checked": a checked-and-held composite lets its body run, a
        checked-and-violated one stops the run before this is reached again.

        No deadlock: a composite's inputs come from outside it (its producers are never
        gated by it), and an unbound input is never tracked as pending readiness, so the
        `requires` always becomes checkable and the gate always eventually opens."""
        node = tuple(node)
        for path, boundary in self._composites.items():
            if len(path) < len(node) and node[: len(path)] == path:
                if "requires" in (self._contract_asts.get(boundary.process) or {}) and path not in self._checked_requires:
                    return False
        return True

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
            # A `requires` violation in `_commit_start` sets `_stopping` (D32); stop
            # dispatching the rest of this tick's pending work at once (D25).
            if self._stopping:
                break
            if int(act["start"]) <= self.now:
                # Gate a body activity on its composite's `requires` being checked
                # (v0 §9 / D34 gap): an input-independent body node must not run before
                # its composite's precondition is verified. Deferred (not failed): a
                # later tick redispatches it once the `requires` is checked. Only
                # process invocations are gated; a transport is ordered by the Object it
                # moves, which a gated producer has not yet created.
                if act.get("kind") == "processing" and not self._requires_gate_open(act["node"]):
                    continue
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
            # Precondition contracts (v0 §9 `requires`, D32): checked before the op runs,
            # over its assembled inputs. A violation must prevent the op from running,
            # so the activity is recorded `failed` and never dispatched, stopping the
            # run gracefully (D25) -- like an observed activity failure, but caught up
            # front. `requires` may reference only inputs (v0 §9.1), so outputs is empty.
            # Besides the data-phase `requires` (D32), also check any preflight
            # candidate deferred from run start because it reads a producer-fed input
            # (now available) -- so it is verified here rather than skipped (D37 gap).
            proc = activity["process"]
            _checkable, deferred = self._split_preflight(activity["node"], proc)
            requires = (self._contract_asts.get(proc, {}).get("requires") or []) + deferred
            violated = self._violated_exprs(
                proc, "requires", requires, inputs, {}, self._fmt_node(tuple(activity["node"]))
            )
            if violated is not None:
                self.log.add(Committed(activity, kind, "failed", start, start, uuid=None))
                self.failed = True
                self._stopping = True
                return
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
                outputs = observed_state.get("outputs")
                if outputs is not None:
                    process = rec.activity["process"]
                    normalized: dict = {}
                    for port, value in outputs.items():
                        resolved = self.contracts.output_type(process, port)
                        if not conforms(value, resolved):
                            raise RunnerError(
                                f"backend output {process}.{port!r} does not conform to its declared type"
                            )
                        # Project type-level static view values onto the produced value
                        # (D35): the stored / downstream / contract-visible value carries
                        # the static constant even if the backend emitted a default (or a
                        # differing value, option A).
                        normalized[port] = with_static_views(value, resolved)
                    record_outputs(self.values, tuple(rec.activity["node"]), normalized)
                    outputs = normalized
                # Postcondition contracts (v0 §9 `ensures`, D32): checked once the outputs
                # exist, over this invocation's assembled inputs and produced outputs. A
                # violation is a runtime contract violation (v0 §9.3): mark the (physically
                # completed) activity `failed` and stop the run gracefully (D25). Only
                # processing activities carry a `process` (a transport leg does not).
                process = rec.activity.get("process")
                if process is not None and self._contract_asts.get(process, {}).get("ensures"):
                    inputs = assemble_inputs(self.dataflow, self.contracts, self.values, rec.activity["node"])
                    subject = self._fmt_node(tuple(rec.activity["node"]))
                    if self._violated_contract(process, "ensures", inputs, outputs or {}, subject) is not None:
                        rec.status = "failed"
                        self.failed = True
                        self._stopping = True
            elif observed == "failed":
                rec.status = "failed"
                rec.end = self.now
                # Record why (D36): a model-driven failure carries a (code, message)
                # reason (e.g. a script error, v0 §22.2); an injected D25 failure has none,
                # so it is reported generically against the activity's subject.
                subject = self._activity_subject(rec.activity)
                reason = observed_state.get("reason")
                if reason is not None:
                    self._record_failure(reason[0], reason[1], subject)
                else:
                    self._record_failure("activity_failed", f"activity {subject} failed", subject)
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
