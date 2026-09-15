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
* re-routing (D21/D39): when a machine goes down, the environment scheduled against
  is reduced so the scheduler re-routes pending work. For a device, what is dropped
  is its `down_scope` (see `DownScope`): by default both its process modes and its
  spots' transports (fully unreachable, safe for real hardware), or modes only
  (`PROCESSING`, so material can still be moved off it -- the classic re-route). A
  down transporter has no scope -- its transports are dropped either way.
* poll modes (D22): fixed-interval polling is the standard -- an integer
  `poll_interval` (default 1) polls every that many units and estimates each
  completion time as the observing poll. `poll_interval=None` advances to plan
  event boundaries instead (exact, deterministic), retained for tests.
* replanning only when the answer can differ (D41): every tick polls, but a tick
  that changed nothing the scheduler reads -- no operation finished, no machine
  went down, nothing came due -- keeps the plan from the last replan instead of
  computing the same one again. The poll cadence, and so every observed time, is
  untouched; only the number of CP-SAT solves drops (from one per time unit to
  roughly one per activity event). See `_needs_replan`.
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
import inspect
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from ..backend import Backend, BackendRefused
from ..simulator import VirtualTimeSimulator
from .contract_eval import evaluate, referenced_ports
from .contracts import ArrayType, conforms, with_static_views
from .echo import Echo
from .failure import Failure
from .job import Job, JobRequest, build_job
from .loader import load_document
from .observation import ObservationRecorder
from .provenance import CommitLog, Committed, UnknownActivityKind, activity_key
from .runner import RunnerError
from .schedule_client import default_objective, derived_holds, replan
from .values import (
    assemble_inputs,
    collect_outputs,
    record_outputs,
    seed_entry,
    unproduced_inputs,
)


def _accepts(func, keyword: str) -> bool:
    """Whether `func` (a backend's ``dispatch_*``) accepts `keyword` -- a parameter of
    that name, or ``**kwargs``.

    Provenance is an optional extension of the `Backend` protocol, asked for one
    keyword at a time: a backend that predates an extension is called exactly as it
    was, and one that wants only part of it gets only that part. An un-introspectable
    callable is treated as accepting nothing (safe)."""
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.name == keyword or p.kind is inspect.Parameter.VAR_KEYWORD for p in params
    )


#: How many loop iterations a run may take before it is called non-terminating.
#: One iteration is one advance + poll, so under fixed-interval polling this also bounds
#: the makespan a run can reach: `max_ticks * poll_interval` time units. `None` lifts the
#: limit entirely, for a long virtual run against a backend whose clock does advance.
DEFAULT_MAX_TICKS = 100_000



def _normalize_mode_ids(environment: dict) -> dict:
    """Return a copy of `environment` with an explicit `id` on every process mode
    that lacks one.

    This must happen before any reduction: dropping a mode renumbers the remaining
    position-based ids, so a reduced-env plan's mode id would no longer map to the
    same physical mode in the backend's full environment. Pinning ids up front
    keeps the id -> mode mapping stable across reduction (D21). Ids are `m<i>`
    rather than the bare position, because a mode id must be a v0 identifier
    (§8.1) and so cannot start with a digit.

    A generated id must not collide with a *user-supplied* id in the same process
    (modes are keyed by id downstream, so a collision silently shadows a mode):
    the fill skips any `m<i>` a user already used.
    """
    env = copy.deepcopy(environment)
    for process in (env.get("processes") or {}).values():
        modes = process.get("modes") or []
        used = {m.get("id") for m in modes if m.get("id") is not None}
        counter = 0
        for mode in modes:
            if mode.get("id") is None:
                while f"m{counter}" in used:
                    counter += 1
                mode["id"] = f"m{counter}"
                used.add(f"m{counter}")
                counter += 1
    return env


class DownScope(str, Enum):
    """How a down device is reduced out of the environment handed to the scheduler.

    The scope selects axes for a down **device**. A down **transporter** has no scope:
    carrying material is its only capability, so when it is down its transports are
    dropped whatever the scope says (see `_reduce_environment`).

    A down device can be made unschedulable along two independent axes: its process
    modes (processing) and the transports touching its spots (moving material on/off
    it). Which axes apply is the deactivation's *scope*:

    * ``BOTH`` (default) -- the device is fully unreachable: its modes *and* its
      spots' transports are dropped. The safe choice for real hardware, where a
      safety-stopped or disconnected device rejects transports too; a plan that
      still routed material onto it would fail at dispatch.
    * ``PROCESSING`` -- drop only the modes; keep the transports. Material can still
      be moved off the down device (the classic re-route). Valid *only* with a
      backend that physically permits transports to/from a down device -- the
      `Simulator` does (D21); a real safety-stopped device does not.
    * ``TRANSPORT`` -- drop only the transports; keep the modes. For completeness (a
      device reachable for processing but not for material movement); rarely needed.

    The scope must match what the backend actually allows, or a planned-then-rejected
    transport crashes the dispatch -- which is why ``BOTH`` is the safe default.
    """

    BOTH = "both"
    PROCESSING = "processing"
    TRANSPORT = "transport"


def _reduce_environment(
    environment: dict, down: set[str], scope: DownScope = DownScope.BOTH
) -> dict:
    """Return a copy of `environment` with a down machine made unschedulable, along
    the axes selected by `scope` (spec §7, D21/D39; see `DownScope`).

    `down` holds ids of machines that cannot be used -- devices, transporters and
    replenishers alike (`Backend.down_devices`). For a **device**: with `scope`
    covering processing, every process mode using it is removed; with `scope`
    covering transport, every transport touching one of its spots is removed, and so
    is every refill of it -- putting stock into a device is material movement, so it
    goes with the transports rather than with the modes. For a **transporter**: every
    transport it carries is removed *in every scope*, since carrying is the only
    thing a transporter does and there is no axis to select (D39). For a
    **replenisher**: every refill it performs is removed in every scope, for the same
    reason. Device / spot / transporter / replenisher definitions are always kept (an
    isolated spot the scheduler simply never routes to).

    Only new scheduling is affected -- transports already committed in the history are
    untouched. Recovery is automatic: the reduction is recomputed from the full
    environment each replan, so a machine no longer in `down` returns with its modes
    and transports.

    An id names one machine here because the environment definition makes sure of
    it: devices, transporters and replenishers share one id space, and a collision in
    it is rejected by the environment validator (`machine_id_conflict`,
    ofplang-schedule >= 0.2.0; it was `device_transporter_id_conflict` from 0.1.5,
    before replenishers existed). Under an older schedule the collision is only a
    warning, and `down` cannot tell the machines apart -- downing one downs the other.
    """
    reduced = copy.deepcopy(environment)
    # A down transporter cannot carry anything, whatever the scope (D39). A down
    # replenisher likewise cannot refill anything: performing refills is all it does.
    if reduced.get("transports"):
        reduced["transports"] = [
            transport
            for transport in reduced["transports"]
            if transport.get("transporter") not in down
        ]
    if reduced.get("replenishments"):
        reduced["replenishments"] = [
            entry for entry in reduced["replenishments"] if entry.get("replenisher") not in down
        ]
    if scope in (DownScope.BOTH, DownScope.PROCESSING):
        for process in (reduced.get("processes") or {}).values():
            process["modes"] = [
                mode
                for mode in (process.get("modes") or [])
                if not (set(mode.get("devices") or []) & down)
            ]
    if scope in (DownScope.BOTH, DownScope.TRANSPORT):
        # Qualified spots ("<device>.<spot>", §8.2) owned by a down device: a
        # transport with either endpoint here can no longer be planned.
        down_spots = {
            f"{device['id']}.{spot}"
            for device in (reduced.get("devices") or [])
            if device.get("id") in down
            for spot in (device.get("spots") or [])
        }
        if down_spots:
            reduced["transports"] = [
                transport
                for transport in (reduced.get("transports") or [])
                if transport.get("from") not in down_spots
                and transport.get("to") not in down_spots
            ]
        # Refilling a down device is material movement onto it, so it is withdrawn on
        # the same axis as the transports that would reach it.
        if reduced.get("replenishments"):
            reduced["replenishments"] = [
                entry for entry in reduced["replenishments"] if entry.get("device") not in down
            ]
    return reduced


def _load_request(request: JobRequest) -> dict:
    """The workflow document one job request names, however it was given.

    A request carries either a path to a workflow YAML file or an already-loaded
    mapping (a caller that rewrote one in memory need not round-trip it through a temp
    file). Everything downstream -- dataflow, contracts, process defs, and the
    scheduler on each replan -- takes the document directly, so it is read at most once
    and never re-serialized.

    A function rather than a step of `__init__` because the jobs a run *begins* with are
    not the only ones it may have: `admit` reads a job's workflow exactly the same way,
    mid-run, and the two must agree on what a request may hold and what it may not.
    """
    if isinstance(request.workflow, dict):
        doc = request.workflow
    elif isinstance(request.workflow, (str, Path)):
        doc = load_document(request.workflow)
    else:
        raise RunnerError("workflow must be a mapping or a path")
    if not isinstance(doc, dict):
        raise RunnerError("workflow must be a mapping")
    return doc


class RollingRunner:
    """Drives workflow + environment (+ boundary) to completion by replanning.

    `boundary` is the run I/O document (D28): a `boundary:` mapping with per-port
    `{spot, view}` descriptors for the workflow's entry inputs and final outputs.
    The runner projects it into the scheduler `interface` (spots only), the seed
    `job` (input views), and the pinned output spots (checked at run end). None
    means no boundary (no Object placement, all entry inputs defaulted)."""

    def __init__(
        self,
        workflow,
        environment_path,
        boundary: dict | None = None,
        *,
        device_model=None,
        backend_factory: Callable[[dict], Backend] | None = None,
        running_task_margin: int = 0,
        random_seed: int | None = None,
        poll_interval: int | None = 1,
        duration_model=None,
        contract_observer=None,
        max_ticks: int | None = DEFAULT_MAX_TICKS,
        down_scope: DownScope = DownScope.BOTH,
        observe: bool = False,
        observation_out: str | None = None,
        ignore_resources: bool = False,
        max_transport_legs: int = 1,
        inventories: dict | None = None,
        occupied: list[dict] | None = None,
        on_job_failure: str = "continue",
    ):
        # The workflow as an in-memory document. `workflow` is either a path to a
        # workflow YAML file (loaded once here) or an already-loaded mapping (e.g. a
        # caller that rewrote it in memory, so it need not round-trip through a temp
        # file). Every collaborator below -- dataflow, contracts, process defs, and the
        # scheduler on each replan -- accepts the document directly, so it is read at
        # most once and never re-serialized.
        #
        # For a run of several jobs (§6.11) `workflow` is instead a list of
        # `JobRequest`s -- id, workflow, boundary, release -- one per job, each read
        # the same way. The boundary is per job because it says where *that* job's
        # material sits, so the run-level `boundary` argument is meaningless then and
        # supplying it is an error rather than a silently ignored value.
        requests: list[JobRequest]
        if isinstance(workflow, list):
            if boundary is not None:
                raise RunnerError(
                    "a run of named jobs carries a boundary per job; put it on the "
                    "JobRequest rather than on the run"
                )
            requests = list(workflow)
        else:
            requests = [JobRequest(id="", workflow=workflow, boundary=boundary)]
        loaded = [(request, _load_request(request)) for request in requests]
        # `environment_path` is likewise a path or an already-loaded environment document
        # (a caller that reads it for its own reasons -- a dialect front door inspecting
        # `x-` keys -- need not have it read a second time here). The document is copied
        # before it is touched (`_normalize_mode_ids`), so the caller's is left alone.
        # Where the environment came from, for the plan's `meta` provenance: the path
        # it was read from, or None when it was handed over as a document (there is no
        # path to name, and the scheduler records `<in-memory>` for that).
        self.environment_path = (
            None if isinstance(environment_path, dict) else str(environment_path)
        )
        # How a down device is reduced out of the scheduling environment (D21): by
        # default fully unreachable (modes + its spots' transports), the safe choice
        # for real hardware. `DownScope.PROCESSING` keeps the transports for the
        # classic re-route (valid only with a backend that permits them, e.g. the
        # Simulator). See `DownScope` and `_reduce_environment`.
        self._down_scope = down_scope
        # Keep the environment as a dict too: when devices go down we schedule
        # against a reduced copy of it (D21), while the backend keeps the full one.
        # Mode ids are pinned up front so they stay stable when modes are dropped.
        self._environment = _normalize_mode_ids(
            environment_path
            if isinstance(environment_path, dict)
            else load_document(environment_path)
        )
        # The consumable model (SPEC §4.7) switched off for this run: the scheduler
        # shape-checks the environment's resource declarations but applies none of
        # them, so a lab that declares stocks runs without the boundary stating what
        # it started with. Off is always a relaxation.
        self._ignore_resources = ignore_resources
        # How many transport activities one Object-bearing arc may be carried in
        # (SPEC §6.4.1). The scheduler's to offer rather than the document's, so it
        # arrives here and is handed to every replan; one is the single hop this has
        # always planned, and raising it only finds routes a lower setting called
        # unreachable.
        self._max_transport_legs = max_transport_legs
        # The backend reads the environment itself. By default it is the built-in
        # `VirtualTimeSimulator`, with an optional device model (D27 F4b) that computes
        # outputs from inputs; without one the built-in `script_device_model` is used
        # (it runs a v0 §22 `script` process, and otherwise falls back to
        # `default_device_model`: type defaults + `objects.map` object carry). A scenario
        # concern injected from Python, like `duration_model`.
        #
        # An alternative backend (e.g. one driving real hardware) is injected as a
        # `backend_factory(environment) -> Backend`: the runner calls it with its own
        # mode-id-normalized environment, so a custom backend sees the same stable
        # mode ids the scheduler does across reduction/replan (why a factory, not a
        # pre-built instance -- the caller does not have the normalized env). The
        # runner drives any backend only through the `Backend` protocol. `device_model`
        # is the default virtual-time Simulator's concern, so pairing it with a custom factory is a
        # usage error rather than a silent no-op.
        if backend_factory is not None:
            if device_model is not None:
                raise RunnerError(
                    "device_model applies only to the default virtual-time Simulator "
                    "backend; a backend_factory must configure its own value model"
                )
            self.sim: Backend = backend_factory(self._environment)
        else:
            self.sim = VirtualTimeSimulator(self._environment, device_model=device_model)

        # Whether this backend's `dispatch_processing` accepts the workflow provenance
        # (`node`). It is an optional extension of the `Backend` protocol, so a backend
        # that predates it (or a minimal one) is driven exactly as before -- the runner
        # passes `node` only when the backend opts in (a `node` parameter or `**kwargs`).
        self._sim_accepts_node = _accepts(self.sim.dispatch_processing, "node")
        # ... and which job of a joint run it belongs to (§6.11). Asked for separately
        # because it is a separate extension: a backend that wanted the node before this
        # existed goes on getting the node and nothing else. A transport is asked too --
        # two jobs of one workflow can move between the same pair of spots, so a record
        # of the moves alone cannot say whose plate went where.
        self._processing_accepts_job = _accepts(self.sim.dispatch_processing, "job")
        self._transport_accepts_job = _accepts(self.sim.dispatch_transport, "job")

        # One job -- one workflow being run, with its dataflow, resolved contracts,
        # boundary and values (`job.py`). A rolling run is a run of a *laboratory*
        # rather than of a workflow, so everything derived from a workflow lives on
        # the job and the runner holds a list of them. These are the jobs the run
        # *begins* with, and not necessarily the ones it ends with: `build_job` reads
        # nothing but its arguments, so a job arriving mid-run (`admit`) is the same
        # call made later and an append to this list.
        self._jobs: list[Job] = [
            build_job(doc, request.boundary, id=request.id, release=request.release)
            for request, doc in loaded
        ]
        self._by_id: dict[str, Job] = {job.id: job for job in self._jobs}
        if len(self._by_id) != len(self._jobs):
            # The id keys the roster, the plan's activities and the provenance of every
            # commit, so two jobs sharing one would have each other's work committed
            # against them. Caught here rather than diagnosed later from the symptom.
            raise RunnerError("job ids must be distinct")
        # Whether this run carries *named* jobs. A single unnamed workflow is planned
        # by the entry point it always used, with a top-level `interface` and no
        # roster, so its plan is byte-for-byte what it was.
        self._named = any(job.id for job in self._jobs)
        # What one job's failure does to the rest (SPEC §6.11). `continue` isolates
        # it: that job stops and the others carry on, which is why they were planned
        # together in the first place -- a laboratory does not down tools because one
        # plate cracked. `stop` is the older, blunter reading, kept for a run where the
        # jobs are parts of one experiment rather than independent work.
        #
        # A run of one workflow is a run of one job, so the two are indistinguishable
        # there: the only job stopping stops the run either way.
        if on_job_failure not in ("continue", "stop"):
            raise RunnerError(
                f"on_job_failure must be 'continue' or 'stop', got {on_job_failure!r}"
            )
        self._on_job_failure = on_job_failure
        # Spots the laboratory is already holding (§6.12): declared at run start, and
        # added to when a job stops leaving material behind.
        # `since` -- when the spot became occupied -- is required by the document
        # (§6.12), and for a spot the laboratory was already holding when the run
        # began the answer is 0. Defaulted rather than demanded: "occupied from the
        # beginning" is the only thing a run's opening state can mean, and a caller
        # writing it out would be answering a question that has one answer.
        declared_occupied: list[dict] = [{"since": 0, **entry} for entry in occupied or []]
        # Jobs the caller has asked to take out of the plan (SPEC §6.11), and the ids
        # of those already taken out. Requested between ticks and enacted by the next
        # replan, because leaving is something the *scheduler* does: it is the one that
        # drops the entry and moves the stock baseline the departing history was part
        # of, and the runner's job is to ask for it and then adopt the answer.
        self._withdrawing: set[str] = set()
        self._withdrawn: set[str] = set()
        # The job an `admit` call is currently checking, while it is still a *candidate*
        # rather than one of this run's jobs. Its run-start checks reach for `_stop_job`
        # and `_record_failure` like any others, and both consult this to tell a
        # candidate from a job: a violation refuses the call and leaves the run exactly
        # as it was, rather than recording a failure of a run nothing has happened to.
        self._admitting: Job | None = None
        # What every stock held when this run began (§6.10). One per run however many
        # jobs draw on it, which is why it is here and not on the job. Carried into
        # the status unchanged on every tick -- the level at `now` is the scheduler's
        # to replay from this plus the `consumption` each fixed activity echoes
        # (§4.7.2), so nothing here ever recomputes it.
        # Stated for the run when there is a run to state it for (the run document,
        # §6.11), and otherwise read off the single job's boundary, where it has always
        # lived. Both are kept: the boundary form is what every existing run uses.
        if inventories is not None:
            declared_inventories = inventories
        else:
            declared = [job.boundary.inventories for job in self._jobs if job.boundary.inventories]
            if any(other != declared[0] for other in declared[1:]):
                raise RunnerError(
                    "jobs declare conflicting starting inventories; the stock is the "
                    "laboratory's, so state it once for the run"
                )
            declared_inventories = declared[0] if declared else {}

        # 🔴 The document this run carries (D48/D49). Opened from what the caller asked
        # for and replaced by every plan the scheduler answers with, so the roster's
        # promises, the moment the levels are stated for and the shape of every
        # multi-leg move are the scheduler's own words from here on -- never rewritten
        # here, and so never lost by failing to rewrite them.
        self._echo = Echo(
            interface=None if self._named else (self._only_job.interface or None),
            inventories=declared_inventories or None,
            occupied=declared_occupied or None,
            jobs=[job.roster_entry() for job in self._jobs] if self._named else None,
        )
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
        # The non-termination guard, in iterations (see `DEFAULT_MAX_TICKS`). None means
        # no limit: the caller has taken on the risk that a backend whose clock does not
        # advance would loop forever.
        self.max_ticks = max_ticks

        # Variance is only coherent under fixed-interval polling (an off-plan finish
        # cannot be observed by event-boundary advance), and needs a positive
        # running-task margin so a successor of an overrunning operation is not
        # dispatched onto a still-busy resource (D23). The margin is the caller's to
        # set (ideally >= poll_interval); the runner only validates it.
        if duration_model is not None:
            if poll_interval is None:
                raise RunnerError(
                    "duration variance requires poll_interval (fixed-interval polling)"
                )
            if running_task_margin < 1:
                raise RunnerError(
                    "duration variance requires running_task_margin >= 1 "
                    "(ideally >= poll_interval, so an overrun defers its successors)"
                )

        self.log = CommitLog()
        self.now = 0
        self.ticks = 0  # loop iterations: one advance + poll each (a test asserts >1)
        self.replans = 0  # how many of those ticks actually called the scheduler (D41)
        self._last_time = None  # `time` section echoed from the most recent plan

        # Replan skipping (D41). `_observed_change` says an operation was seen to
        # finish or fail since the last replan, so the committed history the scheduler
        # fixes has changed; it starts True because the first tick has no plan yet.
        # `_down_at_replan` is the down-machine set the cached plan was built against,
        # so a machine going down (or coming back) shows up as a difference.
        self._observed_change = True
        self._down_at_replan: set[str] = set()

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
        # Warnings the scheduler raised, first occurrence only (a replan repeats the
        # same ones every tick, and the deprecation notices among them are worth
        # exactly one line each). A caller reports them; the runner does not print.
        self.scheduler_warnings: list = []
        self._warned_codes: set = set()

        # Observation document (D38): an optional value-layer record of completed
        # activities' I/O views, off by default (zero overhead). `observe` accumulates
        # entries in memory (for `self.observations` and the render scripts);
        # `observation_out` additionally streams them to a file. A path implies
        # accumulation. `_pending_capture` holds the dispatch-time inputs / moved view
        # of each in-flight op, keyed by backend uuid and popped when it is recorded
        # (so the record is faithful to dispatch, not a post-hoc recompute).
        self._pending_capture: dict[str, dict] = {}
        if observe or observation_out is not None:
            self._obs: ObservationRecorder | None = ObservationRecorder(
                path=observation_out,
                interface=None if self._named else (self._only_job.interface or None),
            )
        else:
            self._obs = None

    def _place_released(self) -> None:
        """Put each job's entry material on its spots once its release has come.

        Entry material is *there*, given, from the job's release (§6.8) — so the
        moment the clock reaches it is the moment the Objects appear. Called every
        tick, and idempotent: a job is placed once.

        🔴 This is the seam a job arriving mid-run (`admit`) slots into. Arriving *is*
        this: entering the roster and having your material appear. Nothing else about
        the loop assumes the job set was known at the start.

        A backend that refuses a placement -- something is already on the spot -- stops
        **that job**, here and at run start alike. It is saying the world is not as the
        plan believed, which is not a disagreement this runner can resolve on the job's
        behalf; but recording it is not resolving it, and a job whose material has
        nowhere to go is one job's misfortune, not the laboratory's. No activity is
        marked failed because none ran -- the same shape as a job whose whole-workflow
        `requires` is violated (`_check_entry_requires`), and the same at run start,
        where that check already stops a job gracefully rather than raising.

        The spots already placed for that job are left placed: entry material is
        *there*, given, from the job's release (§6.8), so the ones that succeeded are
        telling the truth. What the stopped job is holding is derived from its being
        stopped (§6.12), which over-claims here rather than under-claiming -- the safe
        direction.

        `admit` refuses up front the case the runner can see for itself (a spot it is
        already accounting for as occupied), which is the better answer while the job
        is still only a candidate.
        """
        for job in self._jobs:
            if job.placed or job.stopped or job.release > self.now:
                continue
            for _port, spot in (job.interface.get("inputs") or {}).items():
                try:
                    self.sim.place(spot)
                except BackendRefused as exc:
                    subject = self._subject("main", job)
                    self._record_failure(
                        job,
                        "backend_refused_placement",
                        f"{subject}: the backend refused to put this job's entry "
                        f"material on {spot!r} ({type(exc).__name__}: {exc}). "
                        f"Something the plan does not account for is already there",
                        subject,
                    )
                    self._stop_job(job)
                    break
            # Marked placed even where a refusal broke out of the loop: what did go on
            # its spot is there and must not be placed a second time. The job is
            # stopped either way, so the next tick passes over it regardless.
            job.placed = True

    def admit(self, request: JobRequest) -> None:
        """Take a new job **into** a run that is already under way (SPEC §6.11).

        The mirror of `withdraw`. A job is everything derived from one workflow, and
        `build_job` derives it from its arguments alone -- not from this runner, the
        clock, or the other jobs -- so admitting one is appending to the list and
        letting the tick find it there. What makes it *arrive* rather than merely exist
        is what the tick then does with it: `_place_released` puts its entry material
        on its spots, and the next status hands the scheduler a roster with one more
        entry in it.

        It joins at the **end** of the roster, which is the end of the priority order
        (§6.11): priority is what a later arrival owes the jobs already being planned,
        so the newcomer is fitted around their promises rather than the other way
        round. It is promised a completion of its own by the first plan that includes
        it, and from then on is as undisturbable as they are.

        🔴 **The release is this runner's to set, not the scheduler's.** The scheduler
        does default an arriving job to `now` -- but it recognises an arrival as a job
        *absent from the roster* the document carries (§6.11), and by the time this
        runner replans, the newcomer is in the roster it builds from its own job list.
        So that default never fires here, and a job admitted with none stated would be
        released at 0: free, as far as the document says, to have started before it
        existed. Nothing infeasible follows (pending work cannot start before `now`
        anyway), but two jobs admitted at different moments would carry the same
        `release` -- one of the five conditions under which the scheduler may call two
        jobs interchangeable and fix their order for itself. A release stated earlier
        than `now` is raised to it for the same reason, which also makes the unstated
        case (a request's default 0) fall out of the same expression.

        Refused here rather than by the scheduler, exactly as withdrawal is: a replan
        that fails stops every job of a run of named jobs, so a mistaken call left to
        be caught there would take the whole run down with it. The job's own
        preconditions are part of that -- an arrival that cannot run is refused, not
        admitted and then stopped -- so a refused call leaves the run as it was, with
        nothing recorded against it.
        """
        if not self._named:
            raise RunnerError(
                "admit needs a run of named jobs: a single workflow is the whole run, "
                "and it carries no job identity for an arrival to be told apart by"
            )
        if not request.id:
            raise RunnerError(
                "an admitted job needs an id: it is how the roster, the plan's "
                "activities and every commit name it"
            )
        if request.id in self._by_id:
            raise RunnerError(f"cannot admit {request.id!r}: this run already plans it")
        if request.id in self._withdrawn:
            raise RunnerError(
                f"cannot admit {request.id!r}: a job of that id has already left this "
                f"run, taking its history with it. A second job of the same name would "
                f"be a different job wearing the name of one the plan has finished with"
            )
        if self._stopping:
            raise RunnerError(
                f"cannot admit {request.id!r}: every job of this run has stopped, and "
                f"it is only waiting out what is still running. Nothing more is "
                f"dispatched, so an arriving job would never run"
            )
        # A release earlier than `now` is raised to it, and a request that states none
        # carries 0, which is that case (see above).
        job = build_job(
            _load_request(request),
            request.boundary,
            id=request.id,
            release=max(request.release, self.now),
        )
        if job.boundary.inventories:
            # The stock is the laboratory's and the levels are the ones the run *began*
            # with (§6.10): by now they have been drawn on and refilled, and the
            # scheduler replays them from the opening baseline this runner states once.
            # A newcomer restating them would be describing a moment that has passed.
            raise RunnerError(
                f"cannot admit {request.id!r}: its boundary states starting "
                f"inventories, and the stock a run started with is not a job's to "
                f"state -- least of all a job arriving after the run began"
            )
        # 🔴 The one placement conflict this runner can see for itself. It never asks
        # the backend what is on a spot (the backend is told, not asked), so what it can
        # check is what the document accounts for: the freezes it states (§6.12) and the
        # ones it implies. The implied half is **asked of the scheduler**
        # (`derived_holds`) rather than worked out here -- what a stopped job is still
        # holding is derived on every solve, and a second derivation living in the
        # runner is how the two came to disagree before.
        #
        # Checked here at all because a replan that fails stops every job of the run, so
        # an admission the scheduler would refuse has to be refused while refusing is
        # still just an error to the caller.
        #
        # A spot two jobs' *interfaces* share is deliberately not refused: whether the
        # first job's material is still on it is what the history says and not what the
        # bindings say, which is why the scheduler states that as a warning
        # (`interface_shared_input_spot`, §6.11) rather than a refusal.
        #
        # Stamped first so the question is about the laboratory as it is now, not as it
        # was at the last replan -- a job may have stopped since. Writing history down
        # is not a change to the run, so it is no less true if this call then refuses.
        self._echo.stamp(
            self.log.records(), self.now, {j.id for j in self._jobs if j.stopped}
        )
        held = {entry.get("spot") for entry in self._echo.occupied}
        held |= {entry["spot"] for entry in derived_holds(self._echo.document)}
        clash = sorted(
            spot for spot in (job.interface.get("inputs") or {}).values() if spot in held
        )
        if clash:
            raise RunnerError(
                f"cannot admit {request.id!r}: its entry material would be placed on "
                f"{clash}, which this run is already holding something on (§6.12)"
            )

        # The value layer is information -- known from the moment the job is described,
        # read by nobody before the job runs -- so all of it is seeded now, exactly as
        # run start does for the jobs the run began with. Where the Objects physically
        # sit is the tick's business, not this call's.
        seed_entry(job.dataflow, job.contracts, job.values, job.entry_values)
        # The three checks run start makes, for this job alone and in run start's
        # order: the whole-workflow envelope (D33), the atomic preconditions knowable
        # before any dispatch (D37), and any nested composite whose values are already
        # available (D34).
        #
        # 🔴 A violation **refuses the admission** rather than stopping a newly stopped
        # job, which is why they are run while the job is still a candidate
        # (`_admitting`, read by `_stop_job` and `_record_failure`). Refusing is both
        # the more useful answer -- the caller hears why, and can fix the boundary and
        # offer the job again -- and the only one that leaves the run untouched: a job
        # stopped before it has any history has no terminal activity to tell the
        # scheduler it stopped, so its work stays pending in every later plan while
        # this runner declines to dispatch it.
        self._admitting = job
        try:
            self._check_entry_requires(job)
            if not job.stopped:
                self._preflight_job(job)
            if not job.stopped:
                self._check_job_composites(job)
        finally:
            self._admitting = None
        if job.stopped:
            reason = job.failure.detail if job.failure is not None else "a precondition"
            raise RunnerError(
                f"cannot admit {request.id!r}: it does not satisfy its own "
                f"preconditions ({reason})"
            )

        self._jobs.append(job)
        self._by_id[job.id] = job
        # 🔴 And on the roster of the document this run carries. This is the one place
        # the runner adds to a roster the scheduler otherwise owns, and the release goes
        # with it: the scheduler's default for an arrival recognises a job the roster
        # does not name, but a job with boundary material has to be named, since its
        # `interface` is read from its entry. Leaving the release out was measured to
        # release a job admitted at 23 at 24 -- the moment of the next replan, which is
        # not when it arrived.
        self._echo.admit(job.roster_entry())
        # There is a job more to plan, so the scheduler's answer differs and the plan
        # from the last replan must not be carried into the next tick (D41).
        self._observed_change = True

    def free_spot(self, spot: str) -> None:
        """Say that a spot this run was keeping frozen may be used again (SPEC §6.12).

        The mirror of the `occupied` a run opens with. A freeze says a spot is **not to
        be used**, not that something is on it -- a failed transport freezes both of its
        ends, though its plate is at one of them -- so lifting one is a declaration
        rather than a report, and why (it was collected, it was never there, it was
        looked at and found fine) is not the document's business. That is also why this
        is named for what it does instead of for what happened to the material.

        🔴 **Only a freeze the document states can be lifted, and a stopped job's
        residue is not one of them.** What such a job is holding follows from its own
        history, which the document carries, so the scheduler derives it afresh on every
        solve; there is no entry to remove, and removing one would not stop the next
        solve deriving it again. What ends that hold is the job leaving the plan
        (`withdraw`), which is the moment the history goes and the hold is written down
        for the first time.

        The backend is told too, because this is a statement about the world and the
        world is what the backend models: a spot left holding something the plan
        believes free is how a later delivery comes to be refused for a place that is
        full. `clear` is tolerant of a spot that holds nothing, which is the same
        tolerance this call needs -- the freeze may never have had material under it.

        Takes effect on the next replan. Raises if no such freeze is stated, because a
        caller freeing a spot that is not frozen has something else in mind.
        """
        if not self._echo.free_spot(spot):
            stated = sorted(entry["spot"] for entry in self._echo.occupied)
            raise RunnerError(
                f"cannot free {spot!r}: this run states no freeze on it (frozen: "
                f"{stated}). What a stopped job is holding is derived from its history "
                f"rather than stated, and is released by the job leaving the plan"
            )
        self.sim.clear(spot)
        # The scheduler's answer differs now, so the plan from the last replan must not
        # be carried into the next tick (D41).
        self._observed_change = True

    def withdraw(self, job_id: str) -> None:
        """Ask for `job_id` to be taken **out** of the plan (SPEC §6.11).

        The roster is the set of jobs something of which is still in the laboratory --
        unfinished work, or material nobody has collected -- so an entry is removed
        when neither is true. Whether the room is actually clear is not something this
        runner can see, which is why leaving is asked for here rather than inferred
        from a job going quiet: calling this *is* the statement that what the job
        bound has been collected.

        Takes effect on the next replan, not now. The scheduler is the one that drops
        the entry and carries the stock levels forward past the departing history
        (`inventories.at`, SPEC §6.10), so the request is recorded and handed over with
        the next status; `_enact_withdrawals` then adopts what comes back.

        🔴 **Refused here rather than by the scheduler.** A withdrawal the scheduler
        rejects makes the replan fail, and a failed replan in a run of named jobs stops
        every job in it -- so a mistaken call would take the whole run down. These are
        the same conditions it would refuse on, checked while refusing is still just an
        error to the caller.

        🔴 **A job that has not started yet may leave**, and deliberately so: SPEC
        §6.11 allows it, and this runner is held to the same rule rather than a
        stricter one of its own. It is the cheapest possible withdrawal -- nothing of
        the job is in the laboratory, so there is nothing to collect -- and the work it
        had ahead of it simply is not planned. Before the first replan there is no
        pending work to see either way, which is why the checks below cannot tell such
        a call from a finished job's and do not try.
        """
        if not self._named:
            raise RunnerError(
                "withdraw needs a run of named jobs: a single workflow is the whole "
                "run, and a run with no jobs left is not a run"
            )
        job = self._by_id.get(job_id)
        if job is None:
            known = sorted(j.id for j in self._jobs)
            raise RunnerError(
                f"cannot withdraw {job_id!r}: this run plans {known}"
                + (f" (already withdrawn: {sorted(self._withdrawn)})"
                   if self._withdrawn else "")
            )
        busy = sorted(
            {rec.activity.get("node") and "running" or "running"
             for rec in self.log.running() if self._job_of(rec.activity) is job}
        )
        if busy:
            raise RunnerError(
                f"cannot withdraw {job_id!r}: it still has work running. A job leaves "
                f"the plan when there is nothing of it left to do"
            )
        if any(self._job_of(a) is job for a in self._undispatched()):
            raise RunnerError(
                f"cannot withdraw {job_id!r}: it still has work to do. A job leaves "
                f"the plan when there is nothing of it left to do"
            )
        if not ({j.id for j in self._jobs} - self._withdrawing - {job_id}):
            raise RunnerError(
                f"cannot withdraw {job_id!r}: it is the last job of the run, and a "
                f"plan of no jobs is not a plan"
            )
        self._withdrawing.add(job_id)
        # The scheduler's answer will differ, so a replan is due whatever else the
        # tick finds (D41) -- without this the request could sit unenacted.
        self._observed_change = True

    def _enact_withdrawals(self) -> None:
        """Take the jobs that have just left out of this runner's own accounting.

        Called after a replan that carried the request, because the document that
        carries it must still describe them: the scheduler reads a departing job's draws
        from the `consumption` echoes of its activities (SPEC §4.7.2), so dropping them
        any earlier would give the stock back everything that job drew.

        🔴 **Almost nothing is left to do here, and that is the point.** What the job was
        holding, what the stocks hold now and the moment they are stated for are all in
        the plan just adopted -- the scheduler derived them, wrote them and handed them
        over, and the runner carries that document rather than copying pieces out of it.
        Each of those copies was a round trip that could be dropped; two of them were
        dropped, at different times, and the stock quietly gained back everything the
        departing job had drawn.

        The two things that remain are the two the document cannot do:

        1. the spots the caller **bound** are cleared in the backend. They said where
           the material would be and leaving says they collected it, so the world has to
           be told -- otherwise the next delivery to one of them is refused for material
           nobody has. What was *not* bound stays put: the schedule chose where it came
           to rest, so nobody was told, and the plan freezes it instead (§6.12);
        2. the job stops being one of this run's jobs.
        """
        for job in [job for job in self._jobs if job.id in self._withdrawing]:
            for spot in sorted((job.boundary.output_spots or {}).values()):
                self.sim.clear(spot)
            self._jobs.remove(job)
            self._by_id.pop(job.id, None)
            self._withdrawn.add(job.id)
        self._withdrawing.clear()

    def _subject(self, name: str, job: Job) -> str:
        """How a diagnostic names something. In a run of several jobs the same node
        path belongs to several of them, so the job comes too."""
        return f"{job.id}:{name}" if job.id else name

    def _job_of(self, activity: dict) -> Job | None:
        """The job an activity belongs to (§6.11), or None for one that belongs to no
        job -- a replenishment, which the scheduler decided to run and which commonly
        serves several.

        A single-workflow run carries no `job` on anything and its one job is named by
        the empty string, so the same lookup serves both.
        """
        if activity.get("kind") == "replenishment":
            return None
        return self._by_id.get(activity.get("job", ""))

    @property
    def _only_job(self) -> Job:
        """The one job of a single-workflow run.

        What is left of a seam that used to run through the whole class: every place
        the loop needed a job now takes one, and these are the survivors -- the public
        accessors below, which answer for *the* job because that is what a caller with
        one workflow means. A run of several has no such job, and says so rather than
        answering for whichever happens to be first.
        """
        if len(self._jobs) != 1:
            raise RunnerError(
                "this run carries several jobs; ask the job you mean rather than the run"
            )
        return self._jobs[0]

    @property
    def jobs(self) -> list[Job]:
        return list(self._jobs)

    @property
    def inventories(self) -> dict:
        """What the stocks hold, as of the moment the carried document states (§6.10).

        Read out of that document rather than held here. The levels at `now` are the
        scheduler's to replay (§4.7.2) and the moment they are stated for is the
        scheduler's to move when asked, so a copy kept here would be a second answer to
        a question that has one -- which is exactly what a departing job's draws used to
        get lost in.
        """
        return self._echo.inventories

    @property
    def occupied(self) -> list[dict]:
        """The spots this document *states* are frozen (§6.12).

        Not the ones it implies. What a stopped job is still holding follows from its
        own history and is derived by the scheduler on every solve; this section is for
        a hold nothing can derive -- what the laboratory was already holding when the
        run began, and what a person has put somewhere since. Ask `derived_holds` for
        the other half.
        """
        return self._echo.occupied

    @property
    def dataflow(self):
        return self._only_job.dataflow

    @property
    def contracts(self):
        return self._only_job.contracts

    @property
    def boundary(self):
        return self._only_job.boundary

    @property
    def interface(self) -> dict:
        return self._only_job.interface

    @property
    def values(self):
        return self._only_job.values

    @property
    def outputs(self) -> dict:
        return self._only_job.outputs

    @property
    def observations(self) -> list[dict]:
        """The accumulated observation entries (D38); empty when observation is off."""
        return self._obs.entries if self._obs is not None else []

    @staticmethod
    def _fmt_node(node) -> str:
        """A node path rendered as a readable label for reasons / traces; the empty
        path (the workflow boundary) is `main`."""
        return "/".join(node) if node else "main"

    @staticmethod
    def _activity_subject(activity: dict) -> str:
        """A readable subject label for a failed activity (D36): a processing's node
        path, a transport's `from_spot -> to_spot`, or a refill's
        `replenisher -> device`."""
        node = activity.get("node")
        if node is not None:
            return "/".join(node) if node else "main"
        # A refill has no node and no spots -- it is a visit, and what identifies it is
        # who filled what. Named here because a refill's failure stops every job of the
        # run, so its reason is the one a reader most needs to be able to act on; it
        # used to render as `None -> None`, the two spots a refill does not have.
        if activity.get("kind") == "replenishment":
            return f"{activity.get('replenisher')} -> {activity.get('device')}"
        return f"{activity.get('from_spot')} -> {activity.get('to_spot')}"

    def _record_failure(
        self, job: Job | None, kind: str, detail: str, subject: str
    ) -> None:
        """Record why something failed (D36), first failure wins at each level (later
        ones are the cascade of the first). Does not itself stop anything -- the caller
        calls `_stop_job`.

        Two levels, because a run of several jobs can fail in several unrelated ways:
        the run keeps the first failure of the whole run (what the CLI has always
        printed), and the job keeps the one that stopped *it*. `job` is None for a
        failure that belongs to no job -- a refill's.
        """
        failure = Failure(kind=kind, detail=detail, subject=subject, now=self.now)
        if job is not None and job is self._admitting:
            # A candidate for admission (`admit`), not a job of this run. The caller is
            # told why their call was refused -- through this record, which is where
            # `admit` reads the reason from -- and the run, which nothing has happened
            # to, keeps its own failure record empty.
            if job.failure is None:
                job.failure = failure
            return
        if job is not None and job.failure is None:
            job.failure = failure
        if self.failure is None:
            self.failure = failure

    def _stop_job(self, job: Job | None) -> None:
        """Stop `job`: dispatch no more of its work, and decide whether that stops
        the run (D25, per job since SPEC §6.11).

        Everything stops when `job` is None -- a refill's failure, or a replan nothing
        can be planned from: neither can be attributed, and a refill that failed leaves
        the stock it was topping up short. Everything stops under
        `on_job_failure="stop"` too, which is what that policy means.

        🔴 `_stopping` is *derived* from the jobs rather than set alongside them, so the
        two can never disagree. That matters: a run whose jobs were left un-stopped
        while the run stopped would, at the end, check their boundary deliveries and
        echo their outputs -- reporting as delivered the work it had just abandoned.

        A run of one workflow is a run of one job, so `all(...)` is true the moment that
        job stops: `_stopping` is set exactly when it always was, and every
        single-workflow run behaves identically. What changes is only that a run with
        other jobs left, and the policy to use them, carries on.
        """
        if job is not None and job is self._admitting:
            # A candidate for admission (`admit`) failing its own run-start checks. It
            # is not one of this run's jobs: nothing of it has run, nothing of it is
            # planned, and `on_job_failure` has no business deciding what a *refused*
            # call does to the others. Marked so `admit` sees the violation, and
            # nothing else told.
            job.stopped = True
            return
        self.failed = True
        if job is None or self._on_job_failure == "stop":
            targets: list[Job] = list(self._jobs)
        else:
            targets = [job]
        for target in targets:
            if target.stopped:
                continue
            target.stopped = True
        # 🔴 Nothing about what was abandoned, or about what the job is left holding, is
        # recorded here any more. The work it never got to is in the document this run
        # carries, still pending, and the next stamp marks it `cancelled` in place --
        # which is also what tells the scheduler the job stopped, and therefore what
        # makes its residue derivable (§6.12). The spots the failing activity touched
        # come from that activity's own `failed` status, read on every solve.
        #
        # But the stamp has to be *asked for*: without this a tick may decide it need
        # not replan, and the plan still lists that job's work as dispatchable.
        self._observed_change = True
        self._stopping = all(target.stopped for target in self._jobs)

    def run(self) -> dict:
        """Drive to completion and return the final execution status (§6/§7).

        Thin wrapper over `_run_impl` guaranteeing the observation stream (D38) is
        closed even if the run raises -- leaving a trailer-less (= incomplete)
        stream, the crash signal a consumer relies on."""
        try:
            return self._run_impl()
        finally:
            if self._obs is not None:
                self._obs.close()

    def _run_impl(self) -> dict:
        """Drive to completion and return the final execution status (§6/§7). Raises
        `RunnerError` if a replan produces no plan (infeasible) or the run cannot
        progress.

        On an activity failure the run stops rather than raising: it dispatches no
        more work, waits for what is still running to finish, and returns a final
        status with the failed activity `failed` and the abandoned work `cancelled`
        (D25). `self.failed` records that this happened (the CLI maps it to exit 1).

        A backend that **refuses** a dispatch or a placement (`BackendRefused`) is one
        such failure and no longer propagates: the world is not as the plan believed,
        which is a fact about the laboratory rather than a fault in this program, so it
        is recorded and the job it belongs to stops (`_refused_dispatch`). What still
        propagates is a backend error that says the *plan* or this runner is wrong -- an
        unknown reference, a dispatched relay, a clock run backwards -- which no status
        document should be made to look like an ordinary mishap."""
        # Seed every job's boundary inputs: each entry input port gets its view value
        # from the boundary (contract-checked) or a typed default (D27 F4).
        #
        # 🔴 Seeding and *placing* part company here. A view value is information, known
        # from the start and read by nobody before the job runs, so all of it is seeded
        # now. Where the Object physically sits is not: entry material is there, given,
        # from the job's release (§6.8), and putting it on a spot before then would have
        # the scheduler plan around a place it believes free while it is full.
        # `_place_released` does that, each tick, and it is exactly what a job arriving
        # mid-run (`admit`) does.
        for job in self._jobs:
            seed_entry(job.dataflow, job.contracts, job.values, job.entry_values)
        # Spots the laboratory was already holding (§6.12) are occupied in the backend
        # too, so an operation that should never have been planned onto one fails loudly
        # instead of quietly succeeding against a world the plan disagrees with.
        for held in self.occupied:
            spot = held.get("spot")
            if spot:
                self.sim.place(spot)
        self._place_released()

        # Whole-workflow precondition contracts (v0 §9 `requires` on the entry composite,
        # D32 Phase 1): checked once the boundary inputs are seeded, before any work is
        # dispatched. A violation stops the run before it starts (graceful, D25): no
        # activity runs, `self.failed`/`_stopping` are set, and the loop below breaks
        # immediately (nothing is running), so the final status is emptily terminal.
        for job in self._jobs:
            self._check_entry_requires(job)

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
            # A job whose release has come now has its material on its spots. At run
            # start this placed everything released at 0; from here on it is what makes
            # a later release real, and what makes an arrival real (`admit`).
            self._place_released()
            if self.max_ticks is not None and self.ticks > self.max_ticks:
                # One iteration per poll interval, so this is reached either because the run
                # is genuinely longer than the limit or because time is not moving. Only the
                # caller can say which, so name the knob and what it bounds.
                raise RunnerError(
                    f"exceeded max ticks ({self.max_ticks}): the run either needs a higher "
                    "limit (--max-ticks N, or 0 for none -- one tick is one poll interval, "
                    "so the limit caps the makespan at max_ticks * poll_interval) or is not "
                    "progressing (a backend whose clock does not advance)"
                )

            if not self._stopping:
                # Which machines are down right now (D21/D39). Polled every tick, as
                # before -- for a real backend this call is also where availability
                # probing happens, on its own policy's cadence -- and it is one of the
                # three things that can make the scheduler's answer differ.
                down = set(self.sim.down_devices())
                if self._needs_replan(down):
                    # Replan and dispatch what can start now.
                    pending = self._replan_and_dispatch(down)
                else:
                    # Nothing the scheduler reads has changed: the plan from the last
                    # replan still stands, so carry it into this tick untouched. Nothing
                    # is dispatched -- `_needs_replan` returns True for the tick a
                    # pending activity comes due, so a dispatch always follows a fresh
                    # plan (D9: pending identities are only stable within one plan).
                    pending = self._dispatchable()
                # The run is done when there is neither unstarted work nor anything
                # still running. `_needs_replan` asks for a fresh plan before this is
                # ever read as empty, so the completion test never rests on a stale one.
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
            # flip `_stopping`. `advance` returns the time actually reached and the
            # runner adopts it as `now` (the `Backend` contract): for the virtual-time
            # Simulator that is exactly the requested time, but a real backend may
            # overshoot while sleeping out the interval, and `now` must track the real clock.
            self.now = self.sim.advance(self._next_time(pending))
            self._poll()
            # A poll may have recorded the last value a nested composite's contract was
            # waiting on -- check any that just became ready (D34).
            self._check_ready_composites()

        # Assemble the whole-workflow outputs from the produced values (D26); exposed
        # via `job.outputs` and `job.values.snapshot()` (v0-lite: a runner-internal
        # channel, not the §6/§7 document).
        for job in self._jobs:
            job.outputs = collect_outputs(job.dataflow, job.values)

        # For each job that got there, verify each pinned Object output was actually
        # delivered (P3, D28) -- read off this run's own completed moves, not off the
        # backend: where material is is not something the runner asks (D15).
        for job in self._jobs:
            # A stopped job delivered nothing and promised nothing: its outputs never
            # reached their spots (that is what stopping means) and its postcondition
            # is about a result it does not have. Checking either would report a
            # failure that is the first one's shadow.
            if job.stopped:
                continue
            self._check_output_spots(job)
            # Whole-workflow postcondition contracts (v0 §9 `ensures` on the entry
            # composite, D32 Phase 1): checked once the outputs are assembled, over the
            # boundary inputs and produced outputs. A violation is a runtime contract
            # violation (v0 §9.3): set `self.failed` (exit 1). The activities stay
            # `completed` -- the failure is at the whole-workflow boundary, not any one
            # activity. Only checked for a job that ran to the end (the guard above).
            entry = job.contracts.entry
            if (
                job.entry_is_composite
                and entry is not None
                and job.contract_asts.get(entry, {}).get("ensures")
                and self._violated_contract(
                    job, entry, "ensures", self._main_contract_inputs(job), job.outputs,
                    self._subject("main", job),
                )
                is not None
            ):
                self.failed = True
        # Echo the produced output views back into a result document of the same
        # boundary schema (D28), for `--boundary-out`. A run of named jobs keys them by
        # job -- one file still, like the status and the observation document -- while
        # a single unnamed workflow returns the document it always did.
        if self._named:
            # A stopped job is left out: the boundary document is the claim that these
            # outputs were produced and delivered, and a job that stopped delivered
            # nothing. What it did produce before it stopped is in the observation
            # document, which claims only to record what was observed.
            self.result_boundary = {
                "jobs": {
                    job.id: job.boundary.result(job.outputs)
                    for job in self._jobs
                    if not job.stopped
                }
            }
        else:
            job = self._only_job
            self.result_boundary = job.boundary.result(job.outputs)

        # A stopped run reports the work that never ran as cancelled (D25). The failure
        # reason (D36) is NOT put in the status -- it stays a valid §6 document -- but is
        # exposed via `self.failure` (and printed by the CLI).
        # Finalise the observation stream (D38): the trailer records the final time
        # and outcome. The aborted-run case (an exception) is handled by run()'s
        # finally, which closes without a trailer.
        if self._obs is not None:
            self._obs.finish(
                self.now,
                "failed" if self.failed else "completed",
                time_section=self._last_time,
            )

        # One last stamp, for the history made since the final replan -- and, when the
        # run stopped, for the work that never started: a stopped job's pending entries
        # become `cancelled` here exactly as they would have on the next replan there
        # was not going to be (D25).
        self._echo.stamp(
            self.log.records(), self.now, {job.id for job in self._jobs if job.stopped}
        )
        return self._echo.result()

    def _check_output_spots(self, job: Job) -> None:
        """Verify every pinned Object output of `job` was actually delivered (P3, D28).

        The evidence is the delivery itself: a boundary arc (SPEC §6.8) is carried by
        an ordinary transport, so a bound output reached its spot exactly when some
        completed move of this job arrived there. Where the move took several legs
        (§6.4.1) it is the last of them that arrives, which is why the spot rather
        than the leg is what this matches on.

        🔴 **It asks the records, not the laboratory.** Reading the spot would be
        asking a backend where material is, which the runner does not do (D15,
        `..backend`) and a real backend could not answer. So this is a consistency
        assertion over the runner's own bookkeeping, and it cannot fire in ordinary
        operation: a job that did not stop completed all of its work, the delivery
        included. What it would catch is a binding with no delivery behind it -- and
        it says so in those terms, which "the spot is empty" could not.
        """
        delivered = {
            record.activity.get("to_spot")
            for record in self.log.records()
            if record.status == "completed"
            and record.activity.get("kind") == "transport"
            and self._job_of(record.activity) is job
        }
        for port, spot in job.boundary.output_spots.items():
            if spot not in delivered:
                raise RunnerError(
                    f"boundary output {self._subject(port, job)!r} was not delivered "
                    f"to its declared spot {spot!r}: no completed move of this job "
                    f"arrived there"
                )

    def _check_entry_requires(self, job: Job) -> None:
        """One job's whole-workflow precondition (v0 §9 `requires` on the entry
        composite, D32 Phase 1 / D33), checked once its boundary inputs are seeded and
        before any of its work is dispatched.

        A violation stops the job gracefully (D25) at the workflow boundary: no
        activity is marked failed, because none ran. Per job rather than per run
        because that is what the check is about -- one workflow's envelope -- and
        because a job that arrives mid-run (`admit`) has to be held to it too, at the
        moment it is admitted rather than at a run start it missed.
        """
        entry = job.contracts.entry
        if (
            job.entry_is_composite
            and entry is not None
            and job.contract_asts.get(entry, {}).get("requires")
            and self._violated_contract(
                job, entry, "requires", self._main_contract_inputs(job), {},
                self._subject("main", job),
            )
            is not None
        ):
            self._stop_job(job)

    def _preflight_atomic_requires(self) -> None:
        """Run-start preflight (D37): check each atomic invocation's phase-hoisted
        preconditions (`requires_preflight` -- those over run/graph-phase inputs alone)
        before any work is dispatched, over the run-start-available values. A violation
        stops the run before it starts (like an atomic `requires`, but caught up front,
        so no dependent work runs). Skipped once the run is already stopping."""
        for job in self._jobs:
            if self._stopping:
                return
            if not job.stopped:
                self._preflight_job(job)

    def _preflight_job(self, job: Job) -> None:
        """`_preflight_atomic_requires` for one job."""
        for node, process in job.dataflow.process_of.items():
            # Only the preflight candidates whose referenced inputs are *actually*
            # fixed at run start for this node (boundary / literal / unconnected) are
            # checked here; any candidate reading a producer-fed input is deferred to
            # dispatch (checked once the producer has run), so we never evaluate a
            # requires against a not-yet-produced input's typed default.
            checkable, _deferred = self._split_preflight(job, node, process)
            if not checkable:
                continue
            inputs = assemble_inputs(job.dataflow, job.contracts, job.values, node)
            if (
                self._violated_exprs(
                    job, process, "requires_preflight", checkable, inputs, {},
                    self._subject(self._fmt_node(node), job),
                )
                is not None
            ):
                self._stop_job(job)
                return

    def _main_contract_inputs(self, job: Job) -> dict:
        """The entry composite's input view values, for its own whole-workflow contract
        checks (v0 §9 on `main`, D32 Phase 1): each declared entry input read from the
        boundary-seeded value store. Every entry input is seeded at run start
        (`seed_entry`), so all are present."""
        entry = job.contracts.entry
        return {
            port: job.values.get((), port)
            for port in job.contracts.processes[entry].inputs
        }

    def _contract_resolver(self, job: Job, process: str, inputs: dict, outputs: dict):
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
                job.contracts.input_type(process, port)
                if scope == "inputs"
                else job.contracts.output_type(process, port)
            )
            if isinstance(rtype, ArrayType):
                return len(value)  # the only Array view field is `length`
            return value[fields[0]]  # nominal view record field

        return resolve

    def _input_available_at_start(self, job: Job, node, port: str) -> bool:
        """Whether input `port` of `node` has a value fixed at run start.

        True when the port is fed by the boundary (a seeded entry input), bound to a
        static literal, or unconnected (a typed default) -- all fixed before any work
        runs. False when a producing node feeds it: that value is not known until the
        producer completes, so a `requires` over it cannot be preflighted (D37 assumed
        run-phase inputs are always boundary/literal; a legal run->run producer output
        breaks that assumption). `input_source` uses `()` for the boundary node."""
        source = job.dataflow.input_source.get((tuple(node), port))
        if source is not None:
            return source[0] == ()
        return True  # a static literal or an unconnected input: fixed at run start

    def _split_preflight(self, job: Job, node, process: str):
        """Partition a process's preflight-candidate `requires` at `node` into those
        actually checkable at run start (every referenced input fixed at run start)
        and those deferred to dispatch (a referenced input is producer-fed). The split
        is a static property of the dataflow, so it is the same at preflight and at
        dispatch -- guaranteeing each expression is checked exactly once."""
        candidates = job.contract_asts.get(process, {}).get("requires_preflight") or []
        checkable, deferred = [], []
        for pair in candidates:
            _expr, ast = pair
            if all(
                self._input_available_at_start(job, node, port)
                for _s, port in referenced_ports(ast)
            ):
                checkable.append(pair)
            else:
                deferred.append(pair)
        return checkable, deferred

    def _violated_contract(
        self, job: Job, process: str, section: str, inputs: dict, outputs: dict, subject: str
    ):
        """Evaluate `process`'s stored `section` (requires / ensures) contracts for
        `subject`; see `_violated_exprs`."""
        return self._violated_exprs(
            job,
            process,
            section,
            job.contract_asts.get(process, {}).get(section) or [],
            inputs,
            outputs,
            subject,
        )

    def _violated_exprs(
        self, job: Job, process: str, section: str, exprs, inputs: dict, outputs: dict,
        subject: str,
    ):
        """Evaluate an explicit list of `(expr, ast)` contracts for `subject` and return
        the first violated expression, or None if all hold.

        Every expression is evaluated (so the optional `contract_observer` sees each
        one, held or violated, D36; v0 §9.2 permits evaluating all at runtime), and the
        first violation is recorded as the run's failure reason under `section`. A
        contract that evaluates false -- or whose runtime evaluation errors (v0 §9.2) --
        is a runtime contract violation (v0 §9.3)."""
        if not exprs:
            return None
        resolve = self._contract_resolver(job, process, inputs, outputs)
        first_violation = None
        for expr, ast in exprs:
            try:
                held = bool(evaluate(ast, resolve))
            except (ArithmeticError, TypeError):
                # A genuine runtime *evaluation* error over the view values -- e.g.
                # division by zero (ArithmeticError) or an operand-type mismatch
                # (TypeError) -- counts as a contract violation (v0 §9.2/§9.3).
                # Structural / lookup errors (a missing port in `resolve`, i.e. a
                # runner bug) are NOT swallowed here: they propagate rather than being
                # mis-reported as a user-facing contract violation.
                held = False
            if self._contract_observer is not None:
                self._contract_observer(
                    {"subject": subject, "process": process, "section": section,
                     "expr": expr, "held": held, "now": self.now}
                )
            if not held and first_violation is None:
                first_violation = expr
        if first_violation is not None:
            self._record_failure(
                job, f"contract_{section}", f"{subject}: {first_violation}", subject
            )
        return first_violation

    def _composite_ready(self, job: Job, mapping: dict) -> bool:
        """Whether every value-store key in `mapping` (a composite's inputs or outputs,
        port -> (node, port)) has been produced / seeded. Literal-bound ports are not
        in `mapping`, so they never gate readiness (their value is always available)."""
        return all(job.values.has(node, port) for (node, port) in mapping.values())

    def _composite_values(self, job: Job, mapping: dict, literals: dict) -> dict:
        """A composite's port -> view value map: each routed port read from the value
        store, plus each literal-bound port's constant."""
        store = job.values
        values = {
            cport: store.get(node, port) for cport, (node, port) in mapping.items()
        }
        values.update(literals)
        return values

    def _check_ready_composites(self) -> None:
        """Check each nested composite invocation's contracts (v0 §9 / D34) as soon as its
        values are available: `requires` once all its inputs are present, `ensures`
        once all its inputs and outputs are. Each invocation is checked once (tracked in
        `_checked_requires` / `_checked_ensures`). A violation stops the run gracefully
        (D25) at the composite boundary -- no single activity is marked failed, like the
        entry composite (D33). Skipped once the run is already stopping."""
        for job in self._jobs:
            if self._stopping:
                return
            if not job.stopped:
                self._check_job_composites(job)

    def _check_job_composites(self, job: Job) -> None:
        """`_check_ready_composites` for one job."""
        for path, b in job.composites.items():
            asts = job.contract_asts.get(b.process)
            if not asts:
                continue  # this composite declares no contracts
            # `requires`: over the composite's inputs, checked before its body's
            # input-dependent activities can run (they wait on the same values).
            if (
                "requires" in asts
                and path not in job.checked_requires
                and self._composite_ready(job, b.inputs)
            ):
                job.checked_requires.add(path)
                inputs = self._composite_values(job, b.inputs, b.input_literals)
                if (
                    self._violated_contract(
                        job, b.process, "requires", inputs, {},
                        self._subject(self._fmt_node(path), job),
                    )
                    is not None
                ):
                    self._stop_job(job)
                    return
            # `ensures`: over the composite's inputs and outputs, checked once its
            # outputs exist (before any downstream consumer of them runs).
            if (
                "ensures" in asts
                and path not in job.checked_ensures
                and self._composite_ready(job, b.inputs)
                and self._composite_ready(job, b.outputs)
            ):
                job.checked_ensures.add(path)
                inputs = self._composite_values(job, b.inputs, b.input_literals)
                outputs = self._composite_values(job, b.outputs, b.output_literals)
                if (
                    self._violated_contract(
                        job, b.process, "ensures", inputs, outputs,
                        self._subject(self._fmt_node(path), job),
                    )
                    is not None
                ):
                    self._stop_job(job)
                    return

    def _requires_gate_open(self, job: Job, node) -> bool:
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
        for path, boundary in job.composites.items():
            if (
                len(path) < len(node)
                and node[: len(path)] == path
                and "requires" in (job.contract_asts.get(boundary.process) or {})
                and path not in job.checked_requires
            ):
                return False
        return True

    def _needs_replan(self, down: set[str]) -> bool:
        """Whether this tick has to call the scheduler, or the last plan still stands.

        What the scheduler answers is a function of the workflow, the environment it
        is handed, the committed history and `now` (D9). Between two ticks only three
        of those can move, so those are the questions asked here: did an operation
        finish, did a machine go down, has a pending activity come due. A tick where
        none of them happened would put the same question to CP-SAT and get the same
        plan back, at the cost of a full solve -- with `poll_interval=1` that is one
        solve per time unit, i.e. as many solves as the makespan is long.

        This decides only whether to *replan*. The clock still advances and the backend
        is still polled every tick, so completion times -- which in fixed-interval mode
        are the poll at which completion was first seen (D22) -- are exactly what they
        were.
        """
        # An operation finished or failed. The history the scheduler pins has changed,
        # and a completion observed earlier than planned is precisely what lets the
        # remaining work move up -- so this must replan, not wait for the next due time.
        if self._observed_change:
            return True

        # A machine went down or came back: the environment being scheduled against
        # differs, and pending work may have to be re-routed around it (D21).
        if down != self._down_at_replan:
            return True

        # Work has come due. A body activity still gated on its composite's `requires`
        # (D34) does not count: replanning cannot open that gate, and the poll that
        # eventually does sets `_observed_change` anyway -- counting it would replan on
        # every tick until it opened.
        undispatched = self._dispatchable()
        for activity in undispatched:
            if int(activity["start"]) > self.now:
                continue
            if activity.get("kind") == "processing":
                job = self._job_of(activity)
                if job is not None and not self._requires_gate_open(job, activity["node"]):
                    continue
            return True

        # Nothing left to dispatch and nothing running: ask once more, so the loop's
        # completion test reads a plan that is current.
        return not undispatched and not self.log.running()

    def _dispatchable(self) -> list[dict]:
        """The undispatched work the run may still start: `_undispatched` minus what
        belongs to a job that has stopped.

        Kept apart from `_undispatched` because the two are asked different questions.
        This one drives dispatch, "is a replan due", and "is the run over"; the raw one
        answers "what never ran", which for a stopped job is exactly the work this
        filters out -- so cancelling from a filtered list would report nothing.
        """
        return [a for a in self._undispatched() if not self._is_stopped(a)]

    def _is_stopped(self, activity: dict) -> bool:
        """Whether this activity belongs to a job that has stopped. A refill belongs to
        no job, so it is never stopped by one -- it serves whoever is left."""
        job = self._job_of(activity)
        return job is not None and job.stopped

    def _undispatched(self) -> list[dict]:
        """The last plan's pending activities that have not been dispatched.

        Matched by provenance rather than identity: pending identities are regenerated
        on every replan, but a `node` path / arc endpoint pair is not (D9), so this is
        what lines a plan entry up against the committed record that started it.
        """
        committed = {self._provenance_key(r.activity) for r in self.log.records()}
        return [a for a in self._last_pending if self._provenance_key(a) not in committed]

    def _replan_and_dispatch(self, down: set[str]) -> list[dict]:
        """One replanning tick: build the status from committed history, replan, and
        dispatch every pending activity that can start now. Returns the plan's
        pending (non-relay) activities (also remembered for cancellation)."""
        # Schedule against the normalized environment reflecting the machines that are
        # down (discovered by the caller, which needs the same set to decide whether a
        # replan is due at all): the full env when nothing is down, or a reduced copy
        # (the down machines' `down_scope` dropped -- modes and/or their spots'
        # transports) that triggers a re-route (D21).
        # Always the normalized dict, so the scheduler and the backend agree on mode
        # ids. Committed history is fed back so it is fixed and the rest re-optimised.
        # Recorded as of this plan: the next tick compares against these to see whether
        # anything the scheduler reads has moved (D41).
        self.replans += 1
        self._observed_change = False
        self._down_at_replan = down
        environment = (
            _reduce_environment(self._environment, down, self._down_scope)
            if down
            else self._environment
        )
        # 🔴 The document handed over is the last plan, with this run's history stamped
        # onto it and nothing else changed (D48/D49). A job on its way out is still in
        # it, roster entry and activities both: the scheduler reads what the job drew
        # from its `consumption` echoes so the stock is not given back (SPEC §4.7.2),
        # and it is the entry the scheduler removes. Only once it has answered are they
        # dropped here.
        #
        # The work of a job that has **stopped** is stamped `cancelled` by the same
        # call, which is how the scheduler learns it stopped at all -- and therefore how
        # what it left behind becomes derivable (§6.12).
        self._echo.stamp(
            self.log.records(), self.now, {job.id for job in self._jobs if job.stopped}
        )
        # ... and the objective, which is the one planning input restated rather than
        # echoed: with nothing declared it is a function of how many jobs there are
        # (§4.8), and the roster is this runner's. The count is the one the scheduler
        # will see -- the workflows handed over below, which is 0 for a single unnamed
        # workflow, since such a plan has no roster at all.
        planned = [job for job in self._jobs if job.id not in self._withdrawing]
        self._echo.declare_objective(default_objective(len(planned) if self._named else 0))
        status_doc = self._echo.document
        # 🔴 And a job on its way out is **not** among the workflows. The two are
        # deliberately asymmetric: the status describes what has happened, so it keeps
        # the departing job; the workflows are what there is left to plan, and there is
        # nothing left to plan for a job that is leaving -- handing one over as well is
        # refused as a contradiction (`unknown_withdrawal`, SPEC §6.11).
        report = replan(
            [(job.id, job.workflow) for job in planned]
            if self._named
            else self._only_job.workflow,
            environment,
            status_doc,
            withdraw=sorted(self._withdrawing),
            # 🔴 Leaving no longer moves the moment the levels are stated for by itself
            # (SPEC §6.10): the scheduler works the levels out, and whether the history
            # before that moment may be let go of is the caller's call. A departing job
            # drew on stocks since, so the answer here is yes -- and saying so is what
            # keeps its draws from being handed back.
            carry_levels_to_now=bool(self._withdrawing),
            running_task_margin=self.margin,
            random_seed=self.seed,
            environment_source=self.environment_path,
            ignore_resources=self._ignore_resources,
            max_transport_legs=self._max_transport_legs,
        )
        self._collect_warnings(report)
        if not report.ok:
            # 🔴 A run of one workflow raises, as it always has: there is one job, its
            # work is the whole run, and an unplannable run has nothing to report but
            # the exception.
            #
            # A run of several does not. One job's residue can make the rest
            # unplannable -- that is the cost of declaring it (`_occupied_now`) --
            # and killing the run by exception would throw away the status describing
            # everything the other jobs did complete. So every job stops, what is
            # running is waited out, and the run ends the way any other failure ends
            # it: a final status, a reason, exit 1.
            if not self._named:
                raise RunnerError(self._failure_message(report))
            message = self._failure_message(report)
            self._record_failure(None, "replan_infeasible", message, "run")
            self._stop_job(None)
            return []
        plan = report.plan
        # 🔴 The answer becomes the question. Everything the runner used to copy back
        # out of the plan -- each job's promise and fingerprint, the stock baseline a
        # withdrawal moved, the occupancies it wrote, the shape of every grown transport
        # chain -- is simply carried, because the document carried *is* the plan.
        self._echo.adopt(plan)
        if self._withdrawing:
            self._enact_withdrawals()
        self._last_time = plan.get("time")

        # Pending work is what carries no status. A relay is excluded whatever its
        # status: it is an instantaneous scheduling junction with no physical operation
        # behind it, so there is nothing to dispatch and nothing to commit.
        pending = self._echo.pending()
        self._last_pending = pending

        # Dispatch everything that can start now. Pending is optimised at/after
        # `now`, so these are the entries at exactly `now`; their predecessors
        # finished by now (we polled on the previous advance), so the backend's
        # preconditions hold.
        for act in pending:
            # A `requires` violation in `_commit_start` can stop the whole run (D32);
            # stop dispatching the rest of this tick's pending work at once (D25).
            if self._stopping:
                break
            # ... and when it stopped only one job, the rest of that job's work in this
            # (now stale) plan is skipped while everyone else's is dispatched.
            if self._is_stopped(act):
                continue
            if int(act["start"]) <= self.now:
                # Gate a body activity on its composite's `requires` being checked
                # (v0 §9 / D34 gap): an input-independent body node must not run before
                # its composite's precondition is verified. Deferred (not failed): a
                # later tick redispatches it once the `requires` is checked. Only
                # process invocations are gated; a transport is ordered by the Object it
                # moves, which a gated producer has not yet created.
                if act.get("kind") == "processing":
                    job = self._job_of(act)
                    if job is not None and not self._requires_gate_open(job, act["node"]):
                        continue
                # A backend that refuses the dispatch is saying the laboratory is not
                # in the state the plan believed (`..backend.BackendRefused`), and
                # nothing was started. Caught here rather than around each
                # `dispatch_*` because this is the only place a dispatch begins, and
                # because `_commit_start` leaves nothing half-done when one raises:
                # the committed record and the observation capture are both written
                # only once the backend has taken the operation.
                try:
                    self._commit_start(act)
                except BackendRefused as exc:
                    self._refused_dispatch(act, exc)
        return pending

    def _job_provenance(self, activity: dict) -> dict:
        """`{"job": id}` for an activity that belongs to one, and `{}` otherwise.

        🔴 **Only where there is a job to name.** A single-workflow run calls its one
        job by the empty string (`_job_of`), and a run of one workflow is not a run of a
        laboratory: passing that would tell a backend about a distinction this run does
        not have, and would change what a backend keyed on provenance produces for every
        run that has always been a single workflow. A refill belongs to no job at all.
        """
        job = self._job_of(activity)
        return {"job": job.id} if job is not None and job.id else {}

    def _transported_view(self, activity: dict):
        """The view value of the Object a transport leg carries: the producing arc
        endpoint's stored output (D26), passed to the backend so a transport-running
        backend can act on what it moves. Best-effort -- None when the arc endpoint or
        its value is not resolvable; a transport does not change the view (physical
        move preserves identity), so this is read-only context, never written back."""
        job = self._job_of(activity)
        if job is None:
            return None
        arc = activity.get("arc") or {}
        src = arc.get("from") or {}
        node, port = src.get("node"), src.get("port")
        if node is not None and port is not None and job.values.has(node, port):
            return job.values.get(node, port)
        return None

    def _record_observation(self, rec: Committed, outputs: dict | None) -> None:
        """Append a completed activity's I/O views to the observation record (D38).
        Inputs (processing) / moved view (transport) come from the dispatch-time
        stash, popped here; outputs are the values recorded at completion. Values are
        deep-copied by the recorder, so this is faithful to the moment of completion.

        The kinds are named rather than split as "transport or else", so a kind added
        later has to say what it observes instead of being filed as a processing with
        no inputs and no outputs -- a record that looks like an answer and is not.

        A refill records **nothing**. The observation document is the value layer's
        companion: what each activity's ports held (D38). A refill has no ports and
        no views; what it did is a level, and a level is derived from the status, not
        observed (§4.7.2). An empty record would only say "a refill happened", which
        the status already says with times."""
        assert self._obs is not None
        cap = self._pending_capture.pop(rec.uuid, {}) if rec.uuid is not None else {}
        if rec.kind == "transport":
            self._obs.record(rec, moved=cap.get("moved"), time_section=self._last_time)
        elif rec.kind == "processing":
            self._obs.record(
                rec,
                inputs=cap.get("inputs") or {},
                outputs=outputs or {},
                time_section=self._last_time,
            )
        elif rec.kind != "replenishment":
            raise RunnerError(f"cannot observe an activity of kind {rec.kind!r}")

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
        # A processing duration must stay positive (§5.5), and so must a refill's
        # (§5.7); only a transport may be zero, a same-spot hop being a real no-op.
        if self.duration_model is None:
            actual = planned
        else:
            floor = 0 if kind == "transport" else 1
            actual = max(floor, int(self.duration_model(activity, planned)))
        end = start + planned

        if kind == "processing":
            # Every connected input must already carry a real value: the plan orders
            # this activity after its producers, so a missing one means it was started
            # while a predecessor was still running. `assemble_inputs` cannot see that
            # -- it would hand the op a typed default and the run would compute on a
            # value that is not the workflow's -- so it is caught here instead, and
            # gracefully (D25): the activity is marked failed, its successors are
            # cancelled and the reason is recorded, rather than the run continuing on
            # made-up data.
            #
            # The reachable cause is a running-task margin of 0 against a backend whose
            # operations can finish later than planned (a wall-clock / real backend, or
            # a `duration_model`): the scheduler pins a running activity to end at
            # `max(reported end, now + margin)`, so with margin 0 a successor may be
            # planned at `now` while its predecessor is still running. A Pure Data
            # successor holds no spot or device, so nothing else would stop it. Hence
            # the hint: a margin of 0 is not a usable setting for such a backend.
            job = self._job_of(activity)
            assert job is not None  # a processing activity always names its job
            unproduced = unproduced_inputs(job.dataflow, job.values, activity["node"])
            if unproduced:
                subject = self._subject(self._fmt_node(tuple(activity["node"])), job)
                self.log.add(Committed(activity, kind, "failed", start, start, uuid=None))
                self._record_failure(
                    job,
                    "input_not_produced",
                    f"{subject}: input(s) {unproduced} have no value because their "
                    f"producer has not completed; the activity was dispatched while a "
                    f"predecessor was still running. Set a running-task margin of at "
                    f"least the poll interval (--margin) so an overrunning operation "
                    f"defers its successors.",
                    subject,
                )
                self._stop_job(job)
                return
            # Pass the output value signature (D26/D27) so the backend generates a
            # typed value for each output port at completion, and the assembled input
            # values (F4; routed from upstream / the seeded boundary). The backend
            # records inputs but does not yet use them (F4b).
            output_schema = job.output_schemas.get(activity["process"], {})
            inputs = assemble_inputs(
                job.dataflow, job.contracts, job.values, activity["node"]
            )
            # Precondition contracts (v0 §9 `requires`, D32): checked before the op runs,
            # over its assembled inputs. A violation must prevent the op from running,
            # so the activity is recorded `failed` and never dispatched, stopping the
            # run gracefully (D25) -- like an observed activity failure, but caught up
            # front. `requires` may reference only inputs (v0 §9.1), so outputs is empty.
            # Besides the data-phase `requires` (D32), also check any preflight
            # candidate deferred from run start because it reads a producer-fed input
            # (now available) -- so it is verified here rather than skipped (D37 gap).
            proc = activity["process"]
            _checkable, deferred = self._split_preflight(job, activity["node"], proc)
            requires = (job.contract_asts.get(proc, {}).get("requires") or []) + deferred
            violated = self._violated_exprs(
                job, proc, "requires", requires, inputs, {},
                self._subject(self._fmt_node(tuple(activity["node"])), job),
            )
            if violated is not None:
                self.log.add(Committed(activity, kind, "failed", start, start, uuid=None))
                self._stop_job(job)
                return
            # Pass the workflow provenance (`node`) only to a backend that opts in, so a
            # backend predating this extension is driven unchanged (backward compatible).
            provenance = {"node": activity["node"]} if self._sim_accepts_node else {}
            if self._processing_accepts_job:
                provenance.update(self._job_provenance(activity))
            uuid = self.sim.dispatch_processing(
                activity["process"], activity["mode"], duration=actual,
                output_schema=output_schema, inputs=inputs,
                definition=job.process_defs.get(activity["process"]),
                **provenance,
            )
            # Stash the assembled inputs for the observation record (D38): faithful to
            # what this op actually consumed at dispatch, popped when it completes.
            if self._obs is not None:
                self._pending_capture[uuid] = {"inputs": inputs}
        elif kind == "transport":
            view = self._transported_view(activity)
            uuid = self.sim.dispatch_transport(
                activity.get("transporter"),
                activity["from_spot"],
                activity["to_spot"],
                duration=actual,
                view=view,
                **(self._job_provenance(activity) if self._transport_accepts_job else {}),
            )
            if self._obs is not None:
                self._pending_capture[uuid] = {"moved": view}
        elif kind == "replenishment":
            # A refill carries no workflow provenance and no values: the scheduler
            # placed it because a stock would otherwise run out, and `amounts` is
            # what it derived the visit puts in. Nothing is captured for the
            # observation record -- there is no view to observe (see
            # `_record_observation`).
            uuid = self.sim.dispatch_replenishment(
                activity["replenisher"],
                activity["device"],
                activity.get("amounts") or {},
                duration=actual,
            )
        else:  # pragma: no cover - schema guarantees the kinds above, or relay
            raise RunnerError(f"unknown activity kind: {kind!r}")
        self.log.add(Committed(activity, kind, "running", start, end, uuid=uuid))

    def _refused_dispatch(self, activity: dict, exc: BackendRefused) -> None:
        """The backend would not start `activity`: record it `failed` and stop its job.

        The translation of a physical fact into the planning layer's words, which is
        the runner's job and not the scheduler's: the backend has said the laboratory
        is not in the state the plan believed -- a spot really full, a source really
        empty, a machine really busy or really down -- and the only word the plan has
        for that is an activity that **failed**. So that is what is written, exactly as
        for an operation observed to fail once running (`_poll`) and for one refused by
        its own preconditions (`input_not_produced`, above): zero-length at `now`, with
        no handle, because nothing ran.

        🔴 **Nothing else has to be said.** What the refusal leaves stranded follows
        from the record: a failed activity claims every spot it touched, the scheduler
        derives that from this document on its next solve (SPEC §6.12), and a failed
        transport claims *both* ends because no one can say which one still holds the
        Object. Declaring it here as well would be the second derivation that §6.12
        exists to remove.

        Before this, the exception escaped `run()`. In a run of one job that lost only
        the reason; in a run of a laboratory it threw away the status of every *other*
        job -- work that had completed perfectly well -- which is the opposite of what
        planning them together is for.

        A refill belongs to no job (`_job_of` answers None), so its refusal stops the
        run, as a refill's failure always has: it cannot be attributed, and the stock
        it was topping up is still short.
        """
        kind = activity["kind"]
        job = self._job_of(activity)
        subject = self._activity_subject(activity)
        self.log.add(Committed(activity, kind, "failed", self.now, self.now, uuid=None))
        self._record_failure(
            job,
            "backend_refused_dispatch",
            f"{subject}: the backend refused to start this {kind} "
            f"({type(exc).__name__}: {exc}). The laboratory is not in the state the "
            f"plan was built on; nothing was started, and the spots this activity "
            f"touches are held until someone clears them",
            subject,
        )
        self._stop_job(job)

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
            if observed != "running":
                # This record leaves the running set, so the committed history the
                # scheduler fixes differs from the one the cached plan was built on: the
                # next tick has to replan (D41). Set before the branches below, so it
                # covers a completion, a failure, and any future terminal state alike.
                self._observed_change = True
            if observed == "completed":
                rec.status = "completed"
                rec.end = self.now
                # Record the values the backend produced (D26); only value-carrying
                # processing ops report `outputs`, keyed here by their node path. Each
                # output is contract-checked against its port type (D27 F4): the F2
                # defaults always conform, but a future device model / real backend
                # (F4b) could emit a non-conformant value, caught here.
                outputs = observed_state.get("outputs")
                job = self._job_of(rec.activity)
                if outputs is not None and job is not None:
                    process = rec.activity["process"]
                    # Every declared output port must carry a value. A workflow cannot
                    # give an output port a default, so an unset one has no value at all
                    # and nothing downstream can be computed from it -- a device model
                    # that returns only some of them is a backend contract violation,
                    # not something to paper over. Stop gracefully here, where the
                    # under-producing model can be named, rather than leaving the
                    # consumer to discover a value it can only describe as absent. The
                    # built-in models fill every port, so only an injected
                    # `device_model` / `backend_factory` can trip this.
                    schema = job.output_schemas.get(process, {})
                    absent = sorted(set(schema) - set(outputs))
                    if absent:
                        subject = self._subject(self._fmt_node(tuple(rec.activity["node"])), job)
                        rec.status = "failed"
                        self._stop_job(job)
                        self._record_failure(
                            job,
                            "backend_output_missing",
                            f"{subject}: the device model produced no value for declared "
                            f"output(s) {absent}",
                            subject,
                        )
                        continue
                    normalized: dict = {}
                    nonconformant = None
                    for port, value in outputs.items():
                        resolved = job.contracts.output_type(process, port)
                        if not conforms(value, resolved):
                            nonconformant = port
                            break
                        # Project type-level static view values onto the produced value
                        # (D35): the stored / downstream / contract-visible value carries
                        # the static constant even if the backend emitted a default (or a
                        # differing value, option A).
                        normalized[port] = with_static_views(value, resolved)
                    if nonconformant is not None:
                        # A backend / device-model output that does not conform to its
                        # declared type is a runtime verification failure (§22.2 / §9.3).
                        # Stop gracefully (D25) -- mark the (physically completed)
                        # activity failed and record why -- symmetric with a script's
                        # graceful failure, rather than raising a hard RunnerError.
                        subject = self._subject(
                            self._fmt_node(tuple(rec.activity["node"])), job
                        )
                        rec.status = "failed"
                        self._stop_job(job)
                        self._record_failure(
                            job,
                            "backend_output_type",
                            f"{subject}: output {nonconformant!r} does not conform "
                            "to its declared type",
                            subject,
                        )
                        continue
                    record_outputs(job.values, tuple(rec.activity["node"]), normalized)
                    outputs = normalized
                # Postcondition contracts (v0 §9 `ensures`, D32): checked once the outputs
                # exist, over this invocation's assembled inputs and produced outputs. A
                # violation is a runtime contract violation (v0 §9.3): mark the (physically
                # completed) activity `failed` and stop the run gracefully (D25). Only
                # processing activities carry a `process` (a transport leg does not).
                process = rec.activity.get("process")
                if (
                    process is not None
                    and job is not None
                    and job.contract_asts.get(process, {}).get("ensures")
                ):
                    inputs = assemble_inputs(
                        job.dataflow, job.contracts, job.values, rec.activity["node"]
                    )
                    subject = self._subject(self._fmt_node(tuple(rec.activity["node"])), job)
                    if (
                        self._violated_contract(
                            job, process, "ensures", inputs, outputs or {}, subject
                        )
                        is not None
                    ):
                        rec.status = "failed"
                        self._stop_job(job)
                        # Withdraw this invocation's outputs (review #5): they were
                        # recorded above so `ensures` could read them, but a value that
                        # failed its postcondition must not surface as a produced
                        # workflow output via collect_outputs / the result boundary.
                        node = tuple(rec.activity["node"])
                        for port in outputs or {}:
                            job.values.discard(node, port)
                # Record the completed activity in the observation document (D38) --
                # only if it survived its `ensures` (a violated postcondition flipped
                # it to `failed` just above and discarded its outputs).
                if self._obs is not None and rec.status == "completed":
                    self._record_observation(rec, outputs)
            elif observed == "failed":
                rec.status = "failed"
                rec.end = self.now
                # Record why (D36): a model-driven failure carries a (code, message)
                # reason (e.g. a script error, v0 §22.2); an injected D25 failure has none,
                # so it is reported generically against the activity's subject.
                # 🔴 A refill belongs to no job (`_job_of` answers None), so its
                # failure stops the run: nothing else can be attributed, and the stock
                # it was topping up is now short.
                failed_job = self._job_of(rec.activity)
                subject = self._activity_subject(rec.activity)
                reason = observed_state.get("reason")
                if reason is not None:
                    self._record_failure(failed_job, reason[0], reason[1], subject)
                else:
                    self._record_failure(
                        failed_job, "activity_failed", f"activity {subject} failed", subject
                    )
                self._stop_job(failed_job)

    @staticmethod
    def _spots_of(activity: dict) -> set[str]:
        """Every spot an activity touches: a processing's input and output spots, a
        transport's two ends. What it could be holding, in other words."""
        spots = set((activity.get("input_spots") or {}).values())
        spots |= set((activity.get("output_spots") or {}).values())
        spots |= {activity[k] for k in ("from_spot", "to_spot") if activity.get(k)}
        return spots

    @staticmethod
    def _provenance_key(activity: dict):
        """This activity's identity across replans (`provenance.activity_key`).

        🔴 Lives next to `arc_key` rather than here, because the runner is no longer the
        only thing asking: the document it carries is the last plan, and stamping this
        run's history onto it is the same question -- which entry did this record come
        from? One answer, one place.
        """
        try:
            return activity_key(activity)
        except UnknownActivityKind as exc:
            raise RunnerError(str(exc)) from exc

    def _collect_warnings(self, report) -> None:
        """Keep the scheduler's warnings, one line per distinct code.

        Every replan re-derives the same warnings from the same environment, so
        recording each occurrence would give one copy per tick. The first is the one
        worth reporting: what they say -- a deprecated section, a model switched off
        -- is a property of the inputs, not of the moment. An error needs no place
        here; it stops the run and `_failure_message` names it.
        """
        for diag in getattr(report, "diagnostics", None) or []:
            if getattr(diag, "severity", "error") != "warning":
                continue
            code = getattr(diag, "code", None)
            if code in self._warned_codes:
                continue
            self._warned_codes.add(code)
            self.scheduler_warnings.append(diag)

    @staticmethod
    def _failure_message(report) -> str:
        codes = ", ".join(str(getattr(d, "code", d)) for d in report.diagnostics)
        detail = f" ({codes})" if codes else ""
        return f"scheduler produced no plan; outcome={report.outcome}{detail}"
